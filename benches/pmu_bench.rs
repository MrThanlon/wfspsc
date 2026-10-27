//! Hardware PMU Benchmark for `wfspsc`
//!
//! Focuses on Producer-side `instructions/write` and `cycles/write` using hardware
//! performance counters (`thread_selfcounts(THSC_CPI)` on macOS Apple Silicon,
//! `perf_event_open` on Linux).
//!
//! Run with:
//! ```sh
//! cargo bench
//! ```

use std::cell::UnsafeCell;
use std::hint::black_box;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::thread;

use wfspsc::pmu::{PmuReport, ThreadPmu, pin_current_thread_high_perf};
use wfspsc::{Producer, RingBuffer, StaticProducer, StaticRingBuffer};

fn print_header(title: &str) {
    println!("\n================================================================================================");
    println!("  {title}");
    println!("================================================================================================");
    println!(
        "{:<44} | {:>10} | {:>10} | {:>6} | {:>8} | {:>10}",
        "Benchmark Case", "instr/wr", "cycles/wr", "IPC", "ns/wr", "Mops/s"
    );
    println!("---------------------------------------------+------------+------------+--------+----------+------------");
}

fn print_row(name: &str, r: &PmuReport) {
    println!(
        "{:<44} | {:>10.2} | {:>10.2} | {:>6.2} | {:>8.3} | {:>10.1}",
        name,
        r.instructions_per_op(),
        r.cycles_per_op(),
        r.ipc(),
        r.ns_per_op(),
        r.mops_per_sec(),
    );
}

#[inline(never)]
fn bench_static_session_u64<const N: usize>(prod: &mut StaticProducer<u64, N>, iters: u64) {
    let mut s = prod.session();
    for i in 0..iters {
        s.write(i);
    }
}

#[inline(never)]
fn bench_static_deferred_u64<const N: usize>(prod: &mut StaticProducer<u64, N>, iters: u64) {
    let mut s = prod.session();
    // Publish every 64 writes to amortize write_seq updates while keeping consumer fresh
    let chunks = iters / 64;
    let mut v = 0u64;
    for _ in 0..chunks {
        for _ in 0..64 {
            s.write_deferred(v);
            v = v.wrapping_add(1);
        }
        s.publish();
    }
}

#[inline(never)]
fn bench_static_write_direct_u64<const N: usize>(prod: &mut StaticProducer<u64, N>, iters: u64) {
    for i in 0..iters {
        prod.write(i);
    }
}

#[inline(never)]
fn single_call_abi_static<const N: usize>(prod: &mut StaticProducer<u64, N>, val: u64) {
    prod.write(val);
}

#[inline(never)]
fn bench_dyn_session_u64(prod: &mut Producer<u64>, iters: u64) {
    let mut s = prod.session();
    for i in 0..iters {
        s.write(i);
    }
}

#[inline(never)]
fn bench_dyn_write_direct_u64(prod: &mut Producer<u64>, iters: u64) {
    for i in 0..iters {
        prod.write(i);
    }
}

// Baseline 1: Naive UnsafeCell (STR) + write_seq (STLR)
#[inline(never)]
fn bench_naive_unsafecell_str_stlr(
    buf: &[UnsafeCell<u64>],
    mask: usize,
    write_seq: &AtomicU64,
    iters: u64,
) {
    let ptr = buf.as_ptr();
    let mut w = 0u64;
    for i in 0..iters {
        unsafe {
            *(*ptr.add((w as usize) & mask)).get() = i;
        }
        w = w.wrapping_add(1);
        write_seq.store(w, Ordering::Release);
    }
}

// Baseline 2: Classic 2-phase Seqlock Slot (STLR odd -> STR data -> STLR even)
#[repr(C)]
struct SeqlockSlot {
    seq: AtomicU64,
    data: UnsafeCell<u64>,
}

#[inline(never)]
fn bench_seqlock_2phase(
    buf: &[SeqlockSlot],
    mask: usize,
    write_seq: &AtomicU64,
    iters: u64,
) {
    let ptr = buf.as_ptr();
    let mut w = 0u64;
    for i in 0..iters {
        let slot = unsafe { &*ptr.add((w as usize) & mask) };
        let seq_odd = (w << 1) | 1;
        slot.seq.store(seq_odd, Ordering::Release);
        unsafe {
            *slot.data.get() = i;
        }
        slot.seq.store(seq_odd + 1, Ordering::Release);
        w = w.wrapping_add(1);
        write_seq.store(w, Ordering::Release);
    }
}

// Baseline 3: Traditional Head/Tail SPSC that advances consumer `read_pos` via CAS when full
#[repr(C, align(128))]
struct CasOverwritingSpsc {
    write_pos: AtomicU64,
    _pad1: [u8; 120],
    read_pos: AtomicU64,
    _pad2: [u8; 120],
    buf: Vec<AtomicU64>,
}

#[inline(never)]
fn bench_traditional_cas_overwrite(q: &CasOverwritingSpsc, mask: usize, iters: u64) {
    let cap = (mask + 1) as u64;
    let ptr = q.buf.as_ptr();
    let mut w = q.write_pos.load(Ordering::Relaxed);
    for i in 0..iters {
        let r = q.read_pos.load(Ordering::Acquire);
        if w.wrapping_sub(r) >= cap {
            // Buffer full: producer must push consumer tail forward via CAS
            let _ = q.read_pos.compare_exchange_weak(
                r,
                r.wrapping_add(1),
                Ordering::AcqRel,
                Ordering::Relaxed,
            );
        }
        unsafe {
            (*ptr.add((w as usize) & mask)).store(i, Ordering::Release);
        }
        w = w.wrapping_add(1);
        q.write_pos.store(w, Ordering::Release);
    }
}

fn main() {
    pin_current_thread_high_perf();
    let pmu = ThreadPmu::new();

    println!("================================================================================================");
    println!("  wfspsc Hardware PMU Benchmark Suite");
    println!(
        "  PMU Status : {}",
        if pmu.is_available() {
            "ACTIVE (Per-Thread Hardware Counters: PMC0=Instructions, PMC1=Cycles)"
        } else {
            "UNAVAILABLE (Fallback to wall-clock timing only)"
        }
    );
    println!("================================================================================================");

    // Warmup & PMU Calibration
    const CAL_ITERS: u64 = 50_000_000;
    let cal = pmu.measure(CAL_ITERS, || {
        let mut acc = 0u64;
        for i in 0..CAL_ITERS {
            unsafe {
                core::arch::asm!(
                    "add {0}, {0}, {1}",
                    inout(reg) acc,
                    in(reg) i,
                    options(nomem, nostack)
                );
            }
        }
        black_box(acc);
    });
    println!(
        "  PMU Calibration (4-instr loop): {:.2} instr/iter | {:.2} cycles/iter | {:.2} IPC | ~{:.2} GHz",
        cal.instructions_per_op(),
        cal.cycles_per_op(),
        cal.ipc(),
        cal.effective_ghz()
    );

    // =========================================================================
    // Suite 1: Producer Write API Variants (N = 4096, u64 payload, 100% Overwrite)
    // =========================================================================
    const N_OPS: u64 = 100_000_000;
    print_header("Suite 1: Producer API Variants (Capacity = 4096, Payload = u64, Full Overwrite)");

    {
        let (mut prod, _cons) = StaticRingBuffer::<u64, 4096>::new();
        // Warmup
        bench_static_session_u64(black_box(&mut prod), 5_000_000);

        let r_static_deferred = pmu.measure(N_OPS, || {
            bench_static_deferred_u64(black_box(&mut prod), N_OPS);
        });
        print_row(
            "StaticProducer<u64,4096>::session (deferred64)",
            &r_static_deferred,
        );

        let r_static_session = pmu.measure(N_OPS, || {
            bench_static_session_u64(black_box(&mut prod), N_OPS);
        });
        print_row(
            "StaticProducer<u64,4096>::session().write()",
            &r_static_session,
        );

        let r_static_direct = pmu.measure(N_OPS, || {
            bench_static_write_direct_u64(black_box(&mut prod), N_OPS);
        });
        print_row("StaticProducer<u64,4096>::write()", &r_static_direct);

        let r_static_abi = pmu.measure(N_OPS, || {
            let p = black_box(&mut prod);
            for i in 0..N_OPS {
                single_call_abi_static(p, i);
            }
        });
        print_row(
            "StaticProducer<u64,4096>::write (#[inline(never)])",
            &r_static_abi,
        );
    }

    {
        let (mut prod, _cons) = RingBuffer::<u64>::new(4096);
        bench_dyn_session_u64(black_box(&mut prod), 5_000_000);

        let r_dyn_session = pmu.measure(N_OPS, || {
            bench_dyn_session_u64(black_box(&mut prod), N_OPS);
        });
        print_row("Producer<u64>(4096)::session().write()", &r_dyn_session);

        let r_dyn_direct = pmu.measure(N_OPS, || {
            bench_dyn_write_direct_u64(black_box(&mut prod), N_OPS);
        });
        print_row("Producer<u64>(4096)::write()", &r_dyn_direct);
    }

    // =========================================================================
    // Suite 2: Architectural Comparison (wfspsc vs Naive STR+STLR vs Seqlock vs CAS)
    // =========================================================================
    print_header("Suite 2: Architecture Comparison (Capacity = 4096, Payload = u64)");
    {
        const CAP: usize = 4096;
        const MASK: usize = CAP - 1;
        const OPS: u64 = 50_000_000;

        let (mut prod, _cons) = StaticRingBuffer::<u64, CAP>::new();
        let r_wfspsc = pmu.measure(OPS, || {
            bench_static_session_u64(black_box(&mut prod), OPS);
        });
        print_row("1. wfspsc (STLUR slot + STLR write_seq)", &r_wfspsc);

        let r_wfspsc_def = pmu.measure(OPS, || {
            bench_static_deferred_u64(black_box(&mut prod), OPS);
        });
        print_row("2. wfspsc (STLUR slot + amortized write_seq)", &r_wfspsc_def);

        let seqlock_buf: Vec<SeqlockSlot> = (0..CAP)
            .map(|_| SeqlockSlot {
                seq: AtomicU64::new(0),
                data: UnsafeCell::new(0),
            })
            .collect();
        let seq_atomic = AtomicU64::new(0);
        let r_seqlock = pmu.measure(OPS, || {
            bench_seqlock_2phase(
                black_box(&seqlock_buf),
                MASK,
                black_box(&seq_atomic),
                OPS,
            );
        });
        print_row("3. Per-Slot 2-Phase Seqlock (3x STLR/wr)", &r_seqlock);

        let naive_buf: Vec<UnsafeCell<u64>> = (0..CAP).map(|_| UnsafeCell::new(0)).collect();
        let naive_seq = AtomicU64::new(0);
        let r_naive = pmu.measure(OPS, || {
            bench_naive_unsafecell_str_stlr(
                black_box(&naive_buf),
                MASK,
                black_box(&naive_seq),
                OPS,
            );
        });
        print_row("4. Naive UnsafeCell (STR slot + STLR seq)", &r_naive);

        let cas_q = CasOverwritingSpsc {
            write_pos: AtomicU64::new(0),
            _pad1: [0; 120],
            read_pos: AtomicU64::new(0),
            _pad2: [0; 120],
            buf: (0..CAP).map(|_| AtomicU64::new(0)).collect(),
        };
        let r_cas = pmu.measure(OPS, || {
            bench_traditional_cas_overwrite(black_box(&cas_q), MASK, OPS);
        });
        print_row("5. Traditional SPSC + CAS tail on overwrite", &r_cas);
    }

    // =========================================================================
    // Suite 3: Concurrent 2-Thread SPSC (Producer PMU under Active Consumer)
    // =========================================================================
    print_header("Suite 3: Producer PMU under Concurrent Consumer Contention (N = 4096)");
    {
        const OPS: u64 = 50_000_000;

        // Case A: Consumer idle
        let (mut prod, _cons) = StaticRingBuffer::<u64, 4096>::new();
        let r_idle = pmu.measure(OPS, || {
            bench_static_session_u64(black_box(&mut prod), OPS);
        });
        print_row("Producer + Idle Consumer (0 coherence)", &r_idle);

        // Case B: Active sequential Consumer (`pop()` spinning on another core)
        let (mut prod, mut cons) = StaticRingBuffer::<u64, 4096>::new();
        let stop = Arc::new(AtomicBool::new(false));
        let stop_c = Arc::clone(&stop);
        let consumer_handle = thread::spawn(move || {
            pin_current_thread_high_perf();
            let mut sum = 0u64;
            let mut read_cnt = 0u64;
            while !stop_c.load(Ordering::Relaxed) {
                if let Some(v) = cons.pop() {
                    sum = sum.wrapping_add(v);
                    read_cnt += 1;
                }
            }
            (sum, read_cnt, cons.skipped())
        });

        let r_seq_cons = pmu.measure(OPS, || {
            bench_static_session_u64(black_box(&mut prod), OPS);
        });
        stop.store(true, Ordering::Release);
        let (sum_b, read_b, skip_b) = consumer_handle.join().unwrap();
        black_box(sum_b);
        print_row("Producer + Active `pop()` Consumer (2 cores)", &r_seq_cons);
        println!(
            "    -> Consumer stats: read = {} ({:.1}%), overwritten/skipped = {} ({:.1}%)",
            read_b,
            (read_b as f64 / OPS as f64) * 100.0,
            skip_b,
            (skip_b as f64 / OPS as f64) * 100.0
        );

        // Case C: Active ticker Consumer (`pop_latest()` spinning on another core)
        let (mut prod, mut cons) = StaticRingBuffer::<u64, 4096>::new();
        let stop = Arc::new(AtomicBool::new(false));
        let stop_c = Arc::clone(&stop);
        let ticker_handle = thread::spawn(move || {
            pin_current_thread_high_perf();
            let mut latest = 0u64;
            let mut snaps = 0u64;
            while !stop_c.load(Ordering::Relaxed) {
                if let Some(v) = cons.pop_latest() {
                    latest = v;
                    snaps += 1;
                }
            }
            (latest, snaps, cons.skipped())
        });

        let r_latest_cons = pmu.measure(OPS, || {
            bench_static_session_u64(black_box(&mut prod), OPS);
        });
        stop.store(true, Ordering::Release);
        let (latest_c, snaps_c, skip_c) = ticker_handle.join().unwrap();
        black_box((latest_c, skip_c));
        print_row(
            "Producer + Active `pop_latest()` Consumer",
            &r_latest_cons,
        );
        println!("    -> Ticker snapshots taken: {}", snaps_c);

        // Case D: Batch Producer (deferred64) + Batch Consumer (`read_batch(64)`)
        let (mut prod, mut cons) = StaticRingBuffer::<u64, 4096>::new();
        let stop = Arc::new(AtomicBool::new(false));
        let stop_c = Arc::clone(&stop);
        let batch_handle = thread::spawn(move || {
            pin_current_thread_high_perf();
            let mut buf = [0u64; 64];
            let mut sum = 0u64;
            let mut total_read = 0u64;
            while !stop_c.load(Ordering::Relaxed) {
                let n = cons.read_batch(&mut buf);
                if n > 0 {
                    for &v in &buf[..n] {
                        sum = sum.wrapping_add(v);
                    }
                    total_read += n as u64;
                } else {
                    std::hint::spin_loop();
                }
            }
            (sum, total_read, cons.skipped())
        });

        let r_batch_2core = pmu.measure(OPS, || {
            bench_static_deferred_u64(black_box(&mut prod), OPS);
        });
        stop.store(true, Ordering::Release);
        let (sum_d, read_d, skip_d) = batch_handle.join().unwrap();
        black_box(sum_d);
        print_row(
            "Producer(deferred64) + Active `read_batch`",
            &r_batch_2core,
        );
        println!(
            "    -> Batch Consumer: read = {} ({:.1}%), overwritten/skipped = {} ({:.1}%)",
            read_d,
            (read_d as f64 / OPS as f64) * 100.0,
            skip_d,
            (skip_d as f64 / OPS as f64) * 100.0
        );
    }

    // =========================================================================
    // Suite 4: Payload Size & Capacity Scaling Matrix
    // =========================================================================
    print_header("Suite 4: Payload Size & Ring Capacity Scaling");
    {
        const OPS: u64 = 50_000_000;

        for &cap in &[64usize, 256, 1024, 4096, 16384] {
            let (mut prod, _cons) = RingBuffer::<u64>::new(cap);
            let r = pmu.measure(OPS, || {
                bench_dyn_session_u64(black_box(&mut prod), OPS);
            });
            print_row(&format!("Payload   8B (u64)       | Cap = {:>5}", cap), &r);
        }

        #[inline(never)]
        fn bench_struct<T: Copy + Send, const N: usize>(
            prod: &mut StaticProducer<T, N>,
            val: T,
            iters: u64,
        ) {
            let mut s = prod.session();
            for _ in 0..iters {
                s.write(val);
            }
        }

        let (mut p4, _c4) = StaticRingBuffer::<u32, 4096>::new();
        let r4 = pmu.measure(OPS, || bench_struct(black_box(&mut p4), 42u32, OPS));
        print_row("Payload   4B (u32)       | StaticCap = 4096", &r4);

        let (mut p8, _c8) = StaticRingBuffer::<u64, 4096>::new();
        let r8 = pmu.measure(OPS, || bench_struct(black_box(&mut p8), 42u64, OPS));
        print_row("Payload   8B (u64)       | StaticCap = 4096", &r8);

        let (mut p16, _c16) = StaticRingBuffer::<[u64; 2], 4096>::new();
        let r16 = pmu.measure(OPS, || bench_struct(black_box(&mut p16), [1u64, 2], OPS));
        print_row("Payload  16B ([u64; 2])  | StaticCap = 4096", &r16);

        let (mut p32, _c32) = StaticRingBuffer::<[u64; 4], 4096>::new();
        let r32 = pmu.measure(OPS, || bench_struct(black_box(&mut p32), [1u64; 4], OPS));
        print_row("Payload  32B ([u64; 4])  | StaticCap = 4096", &r32);

        let (mut p64, _c64) = StaticRingBuffer::<[u64; 8], 4096>::new();
        let r64 = pmu.measure(OPS, || bench_struct(black_box(&mut p64), [1u64; 8], OPS));
        print_row("Payload  64B ([u64; 8])  | StaticCap = 4096", &r64);
    }

    println!("================================================================================================\n");
}
