//! # `wfspsc` — Wait-Free Producer-Priority Overwriting SPSC RingBuffer
//!
//! A high-performance Single-Producer Single-Consumer (SPSC) ring buffer
//! engineered specifically for **minimum producer write overhead** (`instructions/write`
//! and `cycles/write`).
//!
//! ## Core Design Principles
//!
//! 1. **Zero Consumer Coordination on Producer Path (True Wait-Free $O(1)$)**:
//!    - The producer **never reads** any consumer index (`read_pos` / `tail`) and
//!      **never executes branches or CAS/RMW** to check whether the buffer is full.
//!    - The consumer **never writes** to any cache line read or written by the producer,
//!      keeping producer cache lines in Exclusive/Modified (`E`/`M`) state in L1D.
//!
//! 2. **Single-Pointer Unified Header + Ring Layout (`128B` Cache-Line Aligned)**:
//!    - `write_seq` sits at byte offset `0` (`base_ptr[0]`) in its own dedicated 128-byte
//!      cache line (matching Apple Silicon M-series and modern CPU L2 coherence granules).
//!    - The slot array starts immediately at byte offset `128` (`base_ptr[16]`).
//!    - On ARM64 (ARMv8.4+ / Apple M1–M4), the `+128` byte offset folds directly into
//!      the 9-bit signed immediate of `STLUR` (`stlur x_val, [x_addr, #128]`) and `LDAPUR`
//!      (`ldapur x_val, [x_addr, #128]`), allowing both producer and consumer to address
//!      `write_seq` and `slots` from a **single register (`base_ptr`)**.
//!
//! 3. **Dual-Role `write_seq` Release Protocol (Half the Atomic Stores of a Seqlock)**:
//!    - In a ring buffer of $N = 2^k$ physical slots ($N - 1$ usable slots + 1 active
//!      overwrite guard slot), `write_seq.store(w, Release)` at the end of write $w - 1$
//!      simultaneously:
//!      1) **Commits** slot $(w - 1) \bmod N$ for consumer reads, and
//!      2) **Invalidates** slot $w \bmod N$ (which holds old sequence $w - N$) *before*
//!         write $w$ begins overwriting it.
//!    - Combined with `Release` atomic slot stores (`STLR`/`STLUR`) and `Acquire` slot loads
//!      (`LDAR`/`LDAPUR`), this forms a formally proven `happens-before` chain in the
//!      C++20/Rust memory model with **zero non-atomic data races** and **avoids Apple
//!      Silicon's mixed `STR`+`STLR` write-streaming stall** across large capacities.

pub mod pmu;

use std::alloc::{Layout, alloc_zeroed, dealloc, handle_alloc_error};
use std::marker::PhantomData;
use std::ptr::{self, NonNull};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

/// Cache line / L2 sector alignment (128 bytes for Apple Silicon & modern x86_64).
pub const CACHE_LINE_SIZE: usize = 128;

/// Number of `u64` words in the 128-byte header preceding slot 0 (`16 * 8 = 128` bytes).
/// Word 0 is `write_seq`; words 1..16 are cache-line padding.
const HEADER_WORDS: usize = CACHE_LINE_SIZE / core::mem::size_of::<AtomicU64>();

/// Returns the number of `AtomicU64` words required to store `T`.
#[inline(always)]
const fn words_per_slot<T>() -> usize {
    let sz = core::mem::size_of::<T>();
    if sz == 0 { 0 } else { sz.div_ceil(8) }
}

/// Stores `val` into the slot at index `idx` using `Ordering::Release` word stores.
#[inline(always)]
unsafe fn store_slot_release<T: Copy>(base_ptr: *const AtomicU64, idx: usize, val: T) {
    let words = words_per_slot::<T>();
    if words == 0 {
        return;
    }
    let slot_ptr = unsafe { base_ptr.add(HEADER_WORDS + idx * words) };
    if core::mem::size_of::<T>() == 8 {
        let bits = unsafe { ptr::read_unaligned((&raw const val).cast::<u64>()) };
        unsafe { (*slot_ptr).store(bits, Ordering::Release) };
    } else if core::mem::size_of::<T>() == 4 {
        let bits = unsafe { ptr::read_unaligned((&raw const val).cast::<u32>()) };
        unsafe { (*slot_ptr).store(bits as u64, Ordering::Release) };
    } else if core::mem::size_of::<T>() == 2 {
        let bits = unsafe { ptr::read_unaligned((&raw const val).cast::<u16>()) };
        unsafe { (*slot_ptr).store(bits as u64, Ordering::Release) };
    } else if core::mem::size_of::<T>() == 1 {
        let bits = unsafe { ptr::read_unaligned((&raw const val).cast::<u8>()) };
        unsafe { (*slot_ptr).store(bits as u64, Ordering::Release) };
    } else if core::mem::size_of::<T>().is_multiple_of(8) {
        let src = (&raw const val).cast::<u64>();
        let mut i = 0;
        while i < words {
            let w = unsafe { ptr::read_unaligned(src.add(i)) };
            unsafe { (*slot_ptr.add(i)).store(w, Ordering::Release) };
            i += 1;
        }
    } else {
        let src = (&raw const val).cast::<u8>();
        let total = core::mem::size_of::<T>();
        let mut i = 0;
        while i < words {
            let mut w = 0u64;
            let rem = total - i * 8;
            let copy_len = if rem >= 8 { 8 } else { rem };
            unsafe {
                ptr::copy_nonoverlapping(src.add(i * 8), (&raw mut w).cast::<u8>(), copy_len);
                (*slot_ptr.add(i)).store(w, Ordering::Release);
            }
            i += 1;
        }
    }
}

/// Loads `T` from the slot at index `idx` using `Ordering::Acquire` word loads.
#[inline(always)]
unsafe fn load_slot_acquire<T: Copy>(base_ptr: *const AtomicU64, idx: usize) -> T {
    let words = words_per_slot::<T>();
    if words == 0 {
        return unsafe { core::mem::zeroed() };
    }
    let slot_ptr = unsafe { base_ptr.add(HEADER_WORDS + idx * words) };
    if core::mem::size_of::<T>() == 8 {
        let bits = unsafe { (*slot_ptr).load(Ordering::Acquire) };
        unsafe { ptr::read_unaligned((&raw const bits).cast::<T>()) }
    } else if core::mem::size_of::<T>() == 4 {
        let bits = unsafe { (*slot_ptr).load(Ordering::Acquire) as u32 };
        unsafe { ptr::read_unaligned((&raw const bits).cast::<T>()) }
    } else if core::mem::size_of::<T>() == 2 {
        let bits = unsafe { (*slot_ptr).load(Ordering::Acquire) as u16 };
        unsafe { ptr::read_unaligned((&raw const bits).cast::<T>()) }
    } else if core::mem::size_of::<T>() == 1 {
        let bits = unsafe { (*slot_ptr).load(Ordering::Acquire) as u8 };
        unsafe { ptr::read_unaligned((&raw const bits).cast::<T>()) }
    } else if core::mem::size_of::<T>().is_multiple_of(8) {
        let mut out = core::mem::MaybeUninit::<T>::uninit();
        let dst = out.as_mut_ptr().cast::<u64>();
        let mut i = 0;
        while i < words {
            let w = unsafe { (*slot_ptr.add(i)).load(Ordering::Acquire) };
            unsafe { ptr::write_unaligned(dst.add(i), w) };
            i += 1;
        }
        unsafe { out.assume_init() }
    } else {
        let mut out = core::mem::MaybeUninit::<T>::uninit();
        let dst = out.as_mut_ptr().cast::<u8>();
        let total = core::mem::size_of::<T>();
        let mut i = 0;
        while i < words {
            let w = unsafe { (*slot_ptr.add(i)).load(Ordering::Acquire) };
            let rem = total - i * 8;
            let copy_len = if rem >= 8 { 8 } else { rem };
            unsafe {
                ptr::copy_nonoverlapping((&raw const w).cast::<u8>(), dst.add(i * 8), copy_len);
            }
            i += 1;
        }
        unsafe { out.assume_init() }
    }
}

/// Outcome of a consumer `pop_with_status` operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PopResult<T> {
    /// No unread items are currently available in the ring buffer.
    Empty,
    /// Successfully read the next sequential item without any data loss.
    Ok(T),
    /// The producer overwrote `skipped` unread items; returns the oldest valid item
    /// remaining in the ring buffer.
    Lagged { value: T, skipped: u64 },
}

impl<T> PopResult<T> {
    /// Extracts the value if `Ok` or `Lagged`, discarding the lag count.
    #[inline]
    pub fn into_option(self) -> Option<T> {
        match self {
            PopResult::Empty => None,
            PopResult::Ok(v) | PopResult::Lagged { value: v, .. } => Some(v),
        }
    }
}

/// Contiguous 128-byte-aligned backing storage for dynamic-capacity `RingBuffer<T>`.
struct RawStorage {
    base_ptr: NonNull<AtomicU64>,
    layout: Layout,
}

unsafe impl Send for RawStorage {}
unsafe impl Sync for RawStorage {}

impl RawStorage {
    fn allocate<T: Copy>(slot_count: usize) -> Self {
        assert!(
            slot_count >= 2 && slot_count.is_power_of_two(),
            "slot_count must be a power of two >= 2"
        );
        let total_words = HEADER_WORDS + slot_count * words_per_slot::<T>();
        let byte_size = (total_words * core::mem::size_of::<AtomicU64>()).max(CACHE_LINE_SIZE);
        let layout = Layout::from_size_align(byte_size, CACHE_LINE_SIZE)
            .expect("invalid layout for RingBuffer");
        let raw = unsafe { alloc_zeroed(layout) };
        let Some(base_ptr) = NonNull::new(raw.cast::<AtomicU64>()) else {
            handle_alloc_error(layout);
        };
        Self { base_ptr, layout }
    }
}

impl Drop for RawStorage {
    fn drop(&mut self) {
        unsafe {
            dealloc(self.base_ptr.as_ptr().cast::<u8>(), self.layout);
        }
    }
}

/// Factory for creating a dynamic-capacity wait-free overwriting SPSC ring buffer.
pub struct RingBuffer<T: Copy + Send> {
    _marker: PhantomData<T>,
}

impl<T: Copy + Send> RingBuffer<T> {
    /// Creates a new SPSC ring buffer with `slot_count` physical slots, rounded up to
    /// the next power of two `>= 2`.
    ///
    /// Note: Out of `N` physical slots, `N - 1` slots hold unread items while 1 slot
    /// serves as the wait-free overwrite guard slot. Use [`Self::with_usable_capacity`]
    /// if you need at least `min_usable` unread items retained before overwrite.
    pub fn new(slot_count: usize) -> (Producer<T>, Consumer<T>) {
        let cap = slot_count.next_power_of_two().max(2);
        let storage = Arc::new(RawStorage::allocate::<T>(cap));
        let base_ptr = storage.base_ptr.as_ptr();
        let mask = cap - 1;
        let producer = Producer {
            write_pos: 0,
            base_ptr,
            mask,
            _storage: Arc::clone(&storage),
            _marker: PhantomData,
        };
        let consumer = Consumer {
            read_pos: 0,
            cached_write_seq: 0,
            base_ptr,
            mask,
            skipped_count: 0,
            _storage: storage,
            _marker: PhantomData,
        };
        (producer, consumer)
    }

    /// Creates a new SPSC ring buffer guaranteed to retain at least `min_usable`
    /// unread items without overwrite (`physical_slots = (min_usable + 1).next_power_of_two()`).
    pub fn with_usable_capacity(min_usable: usize) -> (Producer<T>, Consumer<T>) {
        let needed_slots = min_usable.saturating_add(1).next_power_of_two().max(2);
        Self::new(needed_slots)
    }
}

/// Wait-free Producer handle for a dynamic-capacity overwriting SPSC ring buffer.
///
/// Field layout is `#[repr(C, align(128))]` with hot fields (`write_pos`, `base_ptr`, `mask`)
/// placed at offsets `0`, `8`, `16` so ARM64 loads them with a single `ldp` + `ldr` pair.
#[repr(C, align(128))]
pub struct Producer<T: Copy + Send> {
    write_pos: u64,
    base_ptr: *const AtomicU64,
    mask: usize,
    _storage: Arc<RawStorage>,
    _marker: PhantomData<T>,
}

unsafe impl<T: Copy + Send> Send for Producer<T> {}

impl<T: Copy + Send> Producer<T> {
    /// Writes `value` to the ring buffer unconditionally in wait-free $O(1)$ time.
    /// If the buffer is full, the oldest unread slot is immediately overwritten.
    #[inline(always)]
    pub fn write(&mut self, value: T) {
        let w = self.write_pos;
        let idx = (w as usize) & self.mask;
        let next = w.wrapping_add(1);
        self.write_pos = next;
        unsafe {
            store_slot_release(self.base_ptr, idx, value);
            (*self.base_ptr).store(next, Ordering::Release);
        }
    }

    /// Alias for [`Self::write`].
    #[inline(always)]
    pub fn push(&mut self, value: T) {
        self.write(value);
    }

    /// Opens a register-pinned RAII [`WriteSession`] for tight loops or bursts.
    ///
    /// While a `WriteSession` is active, `write_pos`, `base_ptr`, and `mask` stay in
    /// CPU registers (local SSA variables) without writing `self.write_pos` back to
    /// memory until the session is dropped.
    #[inline(always)]
    pub fn session(&mut self) -> WriteSession<'_, T> {
        WriteSession {
            write_pos: self.write_pos,
            base_ptr: self.base_ptr,
            mask: self.mask,
            producer: self,
        }
    }

    /// Writes a slice of items into the ring buffer, publishing each item immediately.
    #[inline]
    pub fn write_slice(&mut self, values: &[T]) {
        let mut s = self.session();
        for &v in values {
            s.write(v);
        }
    }

    /// Returns the total number of items written by this producer so far.
    #[inline(always)]
    pub fn write_pos(&self) -> u64 {
        self.write_pos
    }

    /// Returns the total number of physical slots $N = 2^k$ in the ring buffer.
    #[inline(always)]
    pub fn capacity(&self) -> usize {
        self.mask + 1
    }

    /// Returns the maximum number of unread items ($N - 1$) that can sit in the
    /// buffer simultaneously before the oldest item enters the overwrite guard slot.
    #[inline(always)]
    pub fn usable_capacity(&self) -> usize {
        self.mask
    }
}

/// Register-pinned RAII writer session created by [`Producer::session`].
pub struct WriteSession<'a, T: Copy + Send> {
    write_pos: u64,
    base_ptr: *const AtomicU64,
    mask: usize,
    producer: &'a mut Producer<T>,
}

impl<'a, T: Copy + Send> WriteSession<'a, T> {
    /// Writes `value` into the ring buffer and immediately publishes `write_seq`.
    /// Compiles to only 2 `STLR`/`STLUR` stores with zero intermediate `self.write_pos` memory writes.
    #[inline(always)]
    pub fn write(&mut self, value: T) {
        let w = self.write_pos;
        let idx = (w as usize) & self.mask;
        let next = w.wrapping_add(1);
        self.write_pos = next;
        unsafe {
            store_slot_release(self.base_ptr, idx, value);
            (*self.base_ptr).store(next, Ordering::Release);
        }
    }

    /// Alias for [`Self::write`].
    #[inline(always)]
    pub fn push(&mut self, value: T) {
        self.write(value);
    }

    /// Writes `value` into its slot without updating `write_seq` on every item;
    /// `write_seq` is published when [`Self::publish`] is called or when the session drops.
    #[inline(always)]
    pub fn write_deferred(&mut self, value: T) {
        let w = self.write_pos;
        let idx = (w as usize) & self.mask;
        self.write_pos = w.wrapping_add(1);
        unsafe {
            store_slot_release(self.base_ptr, idx, value);
        }
    }

    /// Publishes the current `write_pos` to `write_seq` with `Ordering::Release`.
    #[inline(always)]
    pub fn publish(&self) {
        unsafe {
            (*self.base_ptr).store(self.write_pos, Ordering::Release);
        }
    }

    /// Returns the current monotonic write position including writes in this session.
    #[inline(always)]
    pub fn write_pos(&self) -> u64 {
        self.write_pos
    }
}

impl<'a, T: Copy + Send> Drop for WriteSession<'a, T> {
    #[inline(always)]
    fn drop(&mut self) {
        if self.producer.write_pos != self.write_pos {
            unsafe {
                (*self.base_ptr).store(self.write_pos, Ordering::Release);
            }
            self.producer.write_pos = self.write_pos;
        }
    }
}

/// Lock-free optimistic Consumer handle for a dynamic-capacity overwriting SPSC ring buffer.
#[repr(C, align(128))]
pub struct Consumer<T: Copy + Send> {
    read_pos: u64,
    cached_write_seq: u64,
    base_ptr: *const AtomicU64,
    mask: usize,
    skipped_count: u64,
    _storage: Arc<RawStorage>,
    _marker: PhantomData<T>,
}

unsafe impl<T: Copy + Send> Send for Consumer<T> {}

impl<T: Copy + Send> Consumer<T> {
    /// Pops the next valid unread item, automatically skipping any older items
    /// that were overwritten by the producer.
    #[inline]
    pub fn pop(&mut self) -> Option<T> {
        self.pop_with_status().into_option()
    }

    /// Pops the next valid unread item and reports whether unread items were
    /// skipped due to producer overwrites.
    ///
    /// Uses `cached_write_seq` to avoid redundant pre-read `Acquire` loads of `write_seq`
    /// when known unread items are already buffered, cutting cross-core coherence traffic
    /// on `write_seq` in half.
    #[inline]
    pub fn pop_with_status(&mut self) -> PopResult<T> {
        let cap = (self.mask + 1) as u64;
        let usable = self.mask as u64;
        let mut w1 = self.cached_write_seq;
        if self.read_pos >= w1 {
            w1 = unsafe { (*self.base_ptr).load(Ordering::Acquire) };
            self.cached_write_seq = w1;
            if self.read_pos >= w1 {
                return PopResult::Empty;
            }
        }
        let mut total_skipped = 0u64;

        loop {
            let mut r = self.read_pos;
            if r >= w1 {
                return PopResult::Empty;
            }

            let oldest = w1.saturating_sub(usable);
            if r < oldest {
                total_skipped = total_skipped.wrapping_add(oldest - r);
                r = oldest;
            }

            let idx = (r as usize) & self.mask;
            let val = unsafe { load_slot_acquire::<T>(self.base_ptr, idx) };
            let w2 = unsafe { (*self.base_ptr).load(Ordering::Acquire) };
            self.cached_write_seq = w2;

            if w2.wrapping_sub(r) < cap {
                self.read_pos = r.wrapping_add(1);
                if total_skipped > 0 {
                    self.skipped_count = self.skipped_count.wrapping_add(total_skipped);
                    return PopResult::Lagged {
                        value: val,
                        skipped: total_skipped,
                    };
                }
                return PopResult::Ok(val);
            }

            // Slot was overwritten while we were reading it; advance to `r` and retry with `w2`.
            self.read_pos = r;
            w1 = w2;
        }
    }

    /// Jumps directly to the most recently written item (`write_seq - 1`), skipping all
    /// older unread items in the ring buffer. Ideal for real-time latest-state tickers.
    #[inline]
    pub fn pop_latest(&mut self) -> Option<T> {
        let cap = (self.mask + 1) as u64;
        let mut w1 = unsafe { (*self.base_ptr).load(Ordering::Acquire) };
        self.cached_write_seq = w1;

        loop {
            if self.read_pos >= w1 {
                return None;
            }
            let target = w1 - 1;
            let idx = (target as usize) & self.mask;
            let val = unsafe { load_slot_acquire::<T>(self.base_ptr, idx) };
            let w2 = unsafe { (*self.base_ptr).load(Ordering::Acquire) };
            self.cached_write_seq = w2;

            if w2.wrapping_sub(target) < cap {
                let skipped = target.saturating_sub(self.read_pos);
                self.skipped_count = self.skipped_count.wrapping_add(skipped);
                self.read_pos = target.wrapping_add(1);
                return Some(val);
            }
            w1 = w2;
        }
    }

    /// Reads up to `out.len()` valid items into `out` and validates the entire batch
    /// with a single post-read `write_seq` check, drastically reducing cross-core
    /// coherence traffic on `write_seq`.
    #[inline]
    pub fn read_batch(&mut self, out: &mut [T]) -> usize {
        if out.is_empty() {
            return 0;
        }
        let cap = (self.mask + 1) as u64;
        let usable = self.mask as u64;
        let mut w1 = self.cached_write_seq;
        if self.read_pos >= w1 {
            w1 = unsafe { (*self.base_ptr).load(Ordering::Acquire) };
            self.cached_write_seq = w1;
            if self.read_pos >= w1 {
                return 0;
            }
        }

        loop {
            let mut r = self.read_pos;
            if r >= w1 {
                return 0;
            }
            let oldest = w1.saturating_sub(usable);
            if r < oldest {
                self.skipped_count = self.skipped_count.wrapping_add(oldest - r);
                r = oldest;
            }

            let avail = (w1 - r) as usize;
            let count = out.len().min(avail);
            for (i, slot) in out[..count].iter_mut().enumerate() {
                let idx = ((r as usize) + i) & self.mask;
                *slot = unsafe { load_slot_acquire::<T>(self.base_ptr, idx) };
            }

            let w2 = unsafe { (*self.base_ptr).load(Ordering::Acquire) };
            self.cached_write_seq = w2;

            if w2.wrapping_sub(r) < cap {
                self.read_pos = r.wrapping_add(count as u64);
                return count;
            }

            // Overwritten during batch read; advance `read_pos` to `r` and retry with `w2`.
            self.read_pos = r;
            w1 = w2;
        }
    }

    /// Returns the cumulative number of overwritten/skipped items observed by this consumer.
    #[inline(always)]
    pub fn skipped(&self) -> u64 {
        self.skipped_count
    }

    /// Returns the consumer's current read sequence position.
    #[inline(always)]
    pub fn read_pos(&self) -> u64 {
        self.read_pos
    }

    /// Returns the approximate number of unread valid items currently available.
    #[inline]
    pub fn pending(&self) -> usize {
        let w = unsafe { (*self.base_ptr).load(Ordering::Acquire) };
        if self.read_pos >= w {
            return 0;
        }
        let usable = self.mask as u64;
        let oldest = w.saturating_sub(usable);
        let effective_r = self.read_pos.max(oldest);
        (w - effective_r) as usize
    }

    /// Returns `true` if no unread items are currently available.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.pending() == 0
    }

    /// Returns the total number of physical slots $N = 2^k$.
    #[inline(always)]
    pub fn capacity(&self) -> usize {
        self.mask + 1
    }

    /// Returns the usable capacity $N - 1$.
    #[inline(always)]
    pub fn usable_capacity(&self) -> usize {
        self.mask
    }
}

// ============================================================================
// Compile-Time Capacity Variant: StaticRingBuffer<T, N>
// ============================================================================

/// Compile-time power-of-2 SPSC ring buffer where the index mask `(N - 1)` is
/// encoded directly as an immediate constant in the CPU's `AND` instruction,
/// reducing [`StaticProducer`] to just two 64-bit fields (`write_pos` + `base_ptr`)
/// loaded in a single ARM64 `LDP` instruction.
pub struct StaticRingBuffer<T: Copy + Send, const N: usize> {
    _marker: PhantomData<T>,
}

impl<T: Copy + Send, const N: usize> StaticRingBuffer<T, N> {
    /// Creates a compile-time power-of-2 SPSC ring buffer with `N` physical slots
    /// (`N - 1` usable capacity).
    ///
    /// # Panics
    /// Panics at compile time (or instantiation) if `N < 2` or `!N.is_power_of_two()`.
    pub fn new() -> (StaticProducer<T, N>, StaticConsumer<T, N>) {
        const {
            assert!(
                N >= 2 && N.is_power_of_two(),
                "StaticRingBuffer capacity N must be a power of two >= 2"
            );
        }
        let storage = Arc::new(RawStorage::allocate::<T>(N));
        let base_ptr = storage.base_ptr.as_ptr();
        let producer = StaticProducer {
            write_pos: 0,
            base_ptr,
            _storage: Arc::clone(&storage),
            _marker: PhantomData,
        };
        let consumer = StaticConsumer {
            read_pos: 0,
            cached_write_seq: 0,
            base_ptr,
            skipped_count: 0,
            _storage: storage,
            _marker: PhantomData,
        };
        (producer, consumer)
    }
}

/// Wait-free Producer handle with compile-time power-of-2 capacity `N`.
///
/// Because `N - 1` is a compile-time immediate constant, the hot path needs only
/// two fields (`write_pos` at offset 0 and `base_ptr` at offset 8), which ARM64
/// loads in a single `ldp x8, x9, [x0]` instruction.
#[repr(C, align(128))]
pub struct StaticProducer<T: Copy + Send, const N: usize> {
    write_pos: u64,
    base_ptr: *const AtomicU64,
    _storage: Arc<RawStorage>,
    _marker: PhantomData<T>,
}

unsafe impl<T: Copy + Send, const N: usize> Send for StaticProducer<T, N> {}

impl<T: Copy + Send, const N: usize> StaticProducer<T, N> {
    /// Writes `value` unconditionally in wait-free $O(1)$ time using immediate bitmasking.
    #[inline(always)]
    pub fn write(&mut self, value: T) {
        const {
            assert!(N >= 2 && N.is_power_of_two());
        }
        let w = self.write_pos;
        let idx = (w as usize) & (N - 1);
        let next = w.wrapping_add(1);
        self.write_pos = next;
        unsafe {
            store_slot_release(self.base_ptr, idx, value);
            (*self.base_ptr).store(next, Ordering::Release);
        }
    }

    /// Alias for [`Self::write`].
    #[inline(always)]
    pub fn push(&mut self, value: T) {
        self.write(value);
    }

    /// Opens a register-pinned RAII [`StaticWriteSession`] for tight loops or bursts.
    #[inline(always)]
    pub fn session(&mut self) -> StaticWriteSession<'_, T, N> {
        StaticWriteSession {
            write_pos: self.write_pos,
            base_ptr: self.base_ptr,
            producer: self,
        }
    }

    /// Writes a slice of items using a register-pinned session.
    #[inline]
    pub fn write_slice(&mut self, values: &[T]) {
        let mut s = self.session();
        for &v in values {
            s.write(v);
        }
    }

    /// Returns the total number of items written so far.
    #[inline(always)]
    pub fn write_pos(&self) -> u64 {
        self.write_pos
    }

    /// Returns the compile-time physical slot count `N`.
    #[inline(always)]
    pub const fn capacity(&self) -> usize {
        N
    }

    /// Returns the compile-time usable capacity `N - 1`.
    #[inline(always)]
    pub const fn usable_capacity(&self) -> usize {
        N - 1
    }
}

/// Register-pinned RAII writer session for [`StaticProducer`].
pub struct StaticWriteSession<'a, T: Copy + Send, const N: usize> {
    write_pos: u64,
    base_ptr: *const AtomicU64,
    producer: &'a mut StaticProducer<T, N>,
}

impl<'a, T: Copy + Send, const N: usize> StaticWriteSession<'a, T, N> {
    /// Writes `value` using compile-time immediate masking and register-pinned state.
    #[inline(always)]
    pub fn write(&mut self, value: T) {
        const {
            assert!(N >= 2 && N.is_power_of_two());
        }
        let w = self.write_pos;
        let idx = (w as usize) & (N - 1);
        let next = w.wrapping_add(1);
        self.write_pos = next;
        unsafe {
            store_slot_release(self.base_ptr, idx, value);
            (*self.base_ptr).store(next, Ordering::Release);
        }
    }

    /// Alias for [`Self::write`].
    #[inline(always)]
    pub fn push(&mut self, value: T) {
        self.write(value);
    }

    /// Writes `value` into its slot without immediately updating `write_seq`,
    /// deferring the `write_seq` commit until [`Self::publish`] or `Drop`.
    ///
    /// Note: Use this for burst writes when you want `write_seq` updated once per burst
    /// (~1.2 cycles/write on Apple M4). If a consumer is concurrently reading and the
    /// burst size approaches `N`, prefer [`Self::write`].
    #[inline(always)]
    pub fn write_deferred(&mut self, value: T) {
        const {
            assert!(N >= 2 && N.is_power_of_two());
        }
        let w = self.write_pos;
        let idx = (w as usize) & (N - 1);
        self.write_pos = w.wrapping_add(1);
        unsafe {
            store_slot_release(self.base_ptr, idx, value);
        }
    }

    /// Publishes the current `write_pos` to `write_seq` with `Ordering::Release`.
    #[inline(always)]
    pub fn publish(&self) {
        unsafe {
            (*self.base_ptr).store(self.write_pos, Ordering::Release);
        }
    }

    /// Returns the current monotonic write position.
    #[inline(always)]
    pub fn write_pos(&self) -> u64 {
        self.write_pos
    }
}

impl<'a, T: Copy + Send, const N: usize> Drop for StaticWriteSession<'a, T, N> {
    #[inline(always)]
    fn drop(&mut self) {
        if self.producer.write_pos != self.write_pos {
            unsafe {
                (*self.base_ptr).store(self.write_pos, Ordering::Release);
            }
            self.producer.write_pos = self.write_pos;
        }
    }
}

/// Lock-free optimistic Consumer handle with compile-time power-of-2 capacity `N`.
#[repr(C, align(128))]
pub struct StaticConsumer<T: Copy + Send, const N: usize> {
    read_pos: u64,
    cached_write_seq: u64,
    base_ptr: *const AtomicU64,
    skipped_count: u64,
    _storage: Arc<RawStorage>,
    _marker: PhantomData<T>,
}

unsafe impl<T: Copy + Send, const N: usize> Send for StaticConsumer<T, N> {}

impl<T: Copy + Send, const N: usize> StaticConsumer<T, N> {
    /// Pops the next valid unread item, automatically skipping overwritten items.
    #[inline]
    pub fn pop(&mut self) -> Option<T> {
        self.pop_with_status().into_option()
    }

    /// Pops the next valid unread item and reports whether unread items were skipped.
    #[inline]
    pub fn pop_with_status(&mut self) -> PopResult<T> {
        const {
            assert!(N >= 2 && N.is_power_of_two());
        }
        let cap = N as u64;
        let usable = (N - 1) as u64;
        let mut w1 = self.cached_write_seq;
        if self.read_pos >= w1 {
            w1 = unsafe { (*self.base_ptr).load(Ordering::Acquire) };
            self.cached_write_seq = w1;
            if self.read_pos >= w1 {
                return PopResult::Empty;
            }
        }
        let mut total_skipped = 0u64;

        loop {
            let mut r = self.read_pos;
            if r >= w1 {
                return PopResult::Empty;
            }

            let oldest = w1.saturating_sub(usable);
            if r < oldest {
                total_skipped = total_skipped.wrapping_add(oldest - r);
                r = oldest;
            }

            let idx = (r as usize) & (N - 1);
            let val = unsafe { load_slot_acquire::<T>(self.base_ptr, idx) };
            let w2 = unsafe { (*self.base_ptr).load(Ordering::Acquire) };
            self.cached_write_seq = w2;

            if w2.wrapping_sub(r) < cap {
                self.read_pos = r.wrapping_add(1);
                if total_skipped > 0 {
                    self.skipped_count = self.skipped_count.wrapping_add(total_skipped);
                    return PopResult::Lagged {
                        value: val,
                        skipped: total_skipped,
                    };
                }
                return PopResult::Ok(val);
            }

            self.read_pos = r;
            w1 = w2;
        }
    }

    /// Jumps directly to the most recently written item (`write_seq - 1`).
    #[inline]
    pub fn pop_latest(&mut self) -> Option<T> {
        const {
            assert!(N >= 2 && N.is_power_of_two());
        }
        let cap = N as u64;
        let mut w1 = unsafe { (*self.base_ptr).load(Ordering::Acquire) };
        self.cached_write_seq = w1;

        loop {
            if self.read_pos >= w1 {
                return None;
            }
            let target = w1 - 1;
            let idx = (target as usize) & (N - 1);
            let val = unsafe { load_slot_acquire::<T>(self.base_ptr, idx) };
            let w2 = unsafe { (*self.base_ptr).load(Ordering::Acquire) };
            self.cached_write_seq = w2;

            if w2.wrapping_sub(target) < cap {
                let skipped = target.saturating_sub(self.read_pos);
                self.skipped_count = self.skipped_count.wrapping_add(skipped);
                self.read_pos = target.wrapping_add(1);
                return Some(val);
            }
            w1 = w2;
        }
    }

    /// Reads up to `out.len()` valid items into `out` and validates the entire batch
    /// with a single post-read `write_seq` check.
    #[inline]
    pub fn read_batch(&mut self, out: &mut [T]) -> usize {
        const {
            assert!(N >= 2 && N.is_power_of_two());
        }
        if out.is_empty() {
            return 0;
        }
        let cap = N as u64;
        let usable = (N - 1) as u64;
        let mut w1 = self.cached_write_seq;
        if self.read_pos >= w1 {
            w1 = unsafe { (*self.base_ptr).load(Ordering::Acquire) };
            self.cached_write_seq = w1;
            if self.read_pos >= w1 {
                return 0;
            }
        }

        loop {
            let mut r = self.read_pos;
            if r >= w1 {
                return 0;
            }
            let oldest = w1.saturating_sub(usable);
            if r < oldest {
                self.skipped_count = self.skipped_count.wrapping_add(oldest - r);
                r = oldest;
            }

            let avail = (w1 - r) as usize;
            let count = out.len().min(avail);
            for (i, slot) in out[..count].iter_mut().enumerate() {
                let idx = ((r as usize) + i) & (N - 1);
                *slot = unsafe { load_slot_acquire::<T>(self.base_ptr, idx) };
            }

            let w2 = unsafe { (*self.base_ptr).load(Ordering::Acquire) };
            self.cached_write_seq = w2;

            if w2.wrapping_sub(r) < cap {
                self.read_pos = r.wrapping_add(count as u64);
                return count;
            }

            self.read_pos = r;
            w1 = w2;
        }
    }

    /// Returns the cumulative number of overwritten/skipped items observed by this consumer.
    #[inline(always)]
    pub fn skipped(&self) -> u64 {
        self.skipped_count
    }

    /// Returns the consumer's current read sequence position.
    #[inline(always)]
    pub fn read_pos(&self) -> u64 {
        self.read_pos
    }

    /// Returns the approximate number of unread valid items currently available.
    #[inline]
    pub fn pending(&self) -> usize {
        let w = unsafe { (*self.base_ptr).load(Ordering::Acquire) };
        if self.read_pos >= w {
            return 0;
        }
        let usable = (N - 1) as u64;
        let oldest = w.saturating_sub(usable);
        let effective_r = self.read_pos.max(oldest);
        (w - effective_r) as usize
    }

    /// Returns `true` if no unread items are currently available.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.pending() == 0
    }

    /// Returns the compile-time physical slot count `N`.
    #[inline(always)]
    pub const fn capacity(&self) -> usize {
        N
    }

    /// Returns the compile-time usable capacity `N - 1`.
    #[inline(always)]
    pub const fn usable_capacity(&self) -> usize {
        N - 1
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    #[test]
    fn test_basic_fifo_no_overwrite() {
        let (mut prod, mut cons) = RingBuffer::<u64>::with_usable_capacity(4);
        assert_eq!(prod.capacity(), 8);
        assert_eq!(prod.usable_capacity(), 7);
        assert!(cons.is_empty());

        for i in 1..=5 {
            prod.write(i * 10);
        }
        assert_eq!(cons.pending(), 5);

        for i in 1..=5 {
            assert_eq!(cons.pop_with_status(), PopResult::Ok(i * 10));
        }
        assert_eq!(cons.pop_with_status(), PopResult::Empty);
        assert_eq!(cons.skipped(), 0);
    }

    #[test]
    fn test_overwrite_drops_oldest() {
        // 4 physical slots -> 3 usable slots + 1 guard slot
        let (mut prod, mut cons) = RingBuffer::<u64>::new(4);
        assert_eq!(prod.capacity(), 4);
        assert_eq!(prod.usable_capacity(), 3);

        // Write 10 items (0..10) without consumer reading
        for i in 0..10 {
            prod.write(i);
        }

        // Only the last 3 items (7, 8, 9) should remain; 7 items (0..7) were overwritten
        assert_eq!(cons.pending(), 3);
        assert_eq!(
            cons.pop_with_status(),
            PopResult::Lagged {
                value: 7,
                skipped: 7
            }
        );
        assert_eq!(cons.pop_with_status(), PopResult::Ok(8));
        assert_eq!(cons.pop_with_status(), PopResult::Ok(9));
        assert_eq!(cons.pop_with_status(), PopResult::Empty);
        assert_eq!(cons.skipped(), 7);
    }

    #[test]
    fn test_pop_latest() {
        let (mut prod, mut cons) = StaticRingBuffer::<u64, 16>::new();
        for i in 1..=100 {
            prod.write(i);
        }
        assert_eq!(cons.pop_latest(), Some(100));
        assert_eq!(cons.pop(), None);
        assert_eq!(cons.skipped(), 99);
    }

    #[test]
    fn test_multi_word_struct_integrity() {
        #[derive(Debug, Clone, Copy, PartialEq, Eq)]
        struct Tick {
            seq: u64,
            bid: u64,
            ask: u64,
            checksum: u64,
        }

        impl Tick {
            fn new(seq: u64) -> Self {
                let bid = seq.wrapping_mul(0x9E3779B97F4A7C15);
                let ask = bid.wrapping_add(100);
                let checksum = seq ^ bid ^ ask;
                Self {
                    seq,
                    bid,
                    ask,
                    checksum,
                }
            }

            fn verify(&self) -> bool {
                (self.seq ^ self.bid ^ self.ask) == self.checksum
                    && self.bid == self.seq.wrapping_mul(0x9E3779B97F4A7C15)
                    && self.ask == self.bid.wrapping_add(100)
            }
        }

        let (mut prod, mut cons) = StaticRingBuffer::<Tick, 64>::new();
        const TOTAL: u64 = 500_000;

        let writer = thread::spawn(move || {
            let mut s = prod.session();
            for i in 1..=TOTAL {
                s.write(Tick::new(i));
            }
        });

        let reader = thread::spawn(move || {
            let mut last_seq = 0u64;
            let mut read_count = 0u64;
            loop {
                match cons.pop() {
                    Some(tick) => {
                        assert!(tick.verify(), "corrupted/torn tick read: {:?}", tick);
                        assert!(
                            tick.seq > last_seq,
                            "non-monotonic sequence: {} <= {}",
                            tick.seq,
                            last_seq
                        );
                        last_seq = tick.seq;
                        read_count += 1;
                        if last_seq == TOTAL {
                            break;
                        }
                    }
                    None => {
                        if last_seq == TOTAL {
                            break;
                        }
                        thread::yield_now();
                    }
                }
            }
            assert_eq!(read_count + cons.skipped(), TOTAL);
        });

        writer.join().unwrap();
        reader.join().unwrap();
    }

    #[test]
    fn test_odd_sized_struct() {
        #[derive(Debug, Clone, Copy, PartialEq, Eq)]
        struct Odd12 {
            a: u32,
            b: u32,
            c: u32,
        }
        let (mut prod, mut cons) = RingBuffer::<Odd12>::new(8);
        for i in 0..20u32 {
            prod.write(Odd12 {
                a: i,
                b: i + 1,
                c: i ^ 0xdeadbeef,
            });
        }
        // Last 7 items: 13..20
        for i in 13..20u32 {
            assert_eq!(
                cons.pop(),
                Some(Odd12 {
                    a: i,
                    b: i + 1,
                    c: i ^ 0xdeadbeef,
                })
            );
        }
        assert_eq!(cons.pop(), None);
    }

    #[test]
    fn test_extreme_wrap_stress_cap4() {
        // Capacity 4 (only 3 usable slots) under heavy concurrent wrap-around
        let (mut prod, mut cons) = StaticRingBuffer::<[u64; 2], 4>::new();
        const N_ITEMS: u64 = 1_000_000;

        let t_prod = thread::spawn(move || {
            let mut s = prod.session();
            for i in 1..=N_ITEMS {
                s.write([i, !i]);
            }
        });

        let t_cons = thread::spawn(move || {
            let mut prev = 0u64;
            let mut count = 0u64;
            while prev < N_ITEMS {
                if let Some([a, b]) = cons.pop() {
                    assert_eq!(b, !a, "torn read on cap=4: a={a:#x}, b={b:#x}");
                    assert!(a > prev, "non-monotonic read: {a} <= {prev}");
                    prev = a;
                    count += 1;
                }
            }
            assert_eq!(count + cons.skipped(), N_ITEMS);
        });

        t_prod.join().unwrap();
        t_cons.join().unwrap();
    }

    #[test]
    fn test_write_slice_and_read_batch() {
        let (mut prod, mut cons) = RingBuffer::<u32>::new(16);
        let input: Vec<u32> = (0..25).collect();
        prod.write_slice(&input);

        let mut out = [0u32; 32];
        let n = cons.read_batch(&mut out);
        assert_eq!(n, 15);
        assert_eq!(&out[..n], &(10..25).collect::<Vec<u32>>()[..]);
        assert_eq!(cons.skipped(), 10);
    }
}
