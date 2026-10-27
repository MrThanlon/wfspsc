//! Hardware Performance Monitoring Unit (PMU) counters for per-thread
//! instruction and CPU cycle measurement.
//!
//! - **macOS (Apple Silicon / Intel)**: Uses XNU's unprivileged `thread_selfcounts(THSC_CPI)`
//!   kernel interface backed by hardware fixed PMU counters (`PMC0` = retired instructions,
//!   `PMC1` = core cycles) accumulated strictly for the calling thread.
//! - **Linux**: Uses `perf_event_open` with `PERF_COUNT_HW_INSTRUCTIONS` and
//!   `PERF_COUNT_HW_CPU_CYCLES` scoped to the calling thread (`pid = 0, cpu = -1`).

use std::time::{Duration, Instant};

/// Snapshot of per-thread hardware PMU counters.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PmuSnapshot {
    /// Retired hardware instructions on the current thread.
    pub instructions: u64,
    /// Core CPU cycles elapsed on the current thread.
    pub cycles: u64,
}

impl PmuSnapshot {
    /// Computes the delta `self - earlier`.
    #[inline]
    pub fn delta_since(self, earlier: Self) -> PmuDelta {
        PmuDelta {
            instructions: self.instructions.saturating_sub(earlier.instructions),
            cycles: self.cycles.saturating_sub(earlier.cycles),
        }
    }
}

/// Difference between two PMU snapshots.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PmuDelta {
    /// Retired hardware instructions in the measured interval.
    pub instructions: u64,
    /// Core CPU cycles in the measured interval.
    pub cycles: u64,
}

impl PmuDelta {
    /// Instructions per operation.
    #[inline]
    pub fn instructions_per_op(self, ops: u64) -> f64 {
        if ops == 0 {
            0.0
        } else {
            self.instructions as f64 / ops as f64
        }
    }

    /// Core cycles per operation.
    #[inline]
    pub fn cycles_per_op(self, ops: u64) -> f64 {
        if ops == 0 {
            0.0
        } else {
            self.cycles as f64 / ops as f64
        }
    }

    /// Instructions per cycle (IPC).
    #[inline]
    pub fn ipc(self) -> f64 {
        if self.cycles == 0 {
            0.0
        } else {
            self.instructions as f64 / self.cycles as f64
        }
    }
}

/// Complete measurement report for a benchmark run.
#[derive(Debug, Clone, Copy)]
pub struct PmuReport {
    pub ops: u64,
    pub delta: PmuDelta,
    pub elapsed: Duration,
    pub pmu_available: bool,
}

impl PmuReport {
    #[inline]
    pub fn instructions_per_op(&self) -> f64 {
        self.delta.instructions_per_op(self.ops)
    }

    #[inline]
    pub fn cycles_per_op(&self) -> f64 {
        self.delta.cycles_per_op(self.ops)
    }

    #[inline]
    pub fn ipc(&self) -> f64 {
        self.delta.ipc()
    }

    #[inline]
    pub fn ns_per_op(&self) -> f64 {
        if self.ops == 0 {
            0.0
        } else {
            self.elapsed.as_nanos() as f64 / self.ops as f64
        }
    }

    #[inline]
    pub fn mops_per_sec(&self) -> f64 {
        let secs = self.elapsed.as_secs_f64();
        if secs == 0.0 {
            0.0
        } else {
            (self.ops as f64 / secs) / 1e6
        }
    }

    #[inline]
    pub fn effective_ghz(&self) -> f64 {
        let ns = self.elapsed.as_nanos() as f64;
        if ns == 0.0 {
            0.0
        } else {
            self.delta.cycles as f64 / ns
        }
    }
}

/// Requests the OS scheduler to place the current thread on a high-performance core
/// (e.g., Apple Silicon P-core via `QOS_CLASS_USER_INTERACTIVE`).
#[inline]
pub fn pin_current_thread_high_perf() {
    #[cfg(target_os = "macos")]
    unsafe {
        const QOS_CLASS_USER_INTERACTIVE: u32 = 0x21;
        unsafe extern "C" {
            fn pthread_set_qos_class_self_np(qos_class: u32, relative_priority: i32) -> i32;
        }
        let _ = pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
    }
}

/// Per-thread hardware PMU reader.
pub struct ThreadPmu {
    #[cfg(target_os = "linux")]
    fd_instr: i32,
    #[cfg(target_os = "linux")]
    fd_cycles: i32,
    available: bool,
}

impl ThreadPmu {
    /// Initializes per-thread PMU counters for the calling thread.
    pub fn new() -> Self {
        #[cfg(target_os = "macos")]
        {
            let mut buf = [0u64; 2];
            let rc = unsafe { sys_macos::thread_selfcounts(1, buf.as_mut_ptr(), 16) };
            Self {
                available: rc == 0 && (buf[0] != 0 || buf[1] != 0),
            }
        }

        #[cfg(target_os = "linux")]
        {
            let (fd_instr, fd_cycles, available) = sys_linux::open_thread_pmu();
            Self {
                fd_instr,
                fd_cycles,
                available,
            }
        }

        #[cfg(not(any(target_os = "macos", target_os = "linux")))]
        {
            Self { available: false }
        }
    }

    /// Returns `true` if hardware PMU counters are active for this thread.
    #[inline]
    pub fn is_available(&self) -> bool {
        self.available
    }

    /// Reads the current thread's hardware instruction and cycle counters.
    #[inline(always)]
    pub fn read(&self) -> PmuSnapshot {
        #[cfg(target_os = "macos")]
        {
            let mut buf = [0u64; 2];
            unsafe {
                let _ = sys_macos::thread_selfcounts(1, buf.as_mut_ptr(), 16);
            }
            PmuSnapshot {
                instructions: buf[0],
                cycles: buf[1],
            }
        }

        #[cfg(target_os = "linux")]
        {
            if self.available {
                sys_linux::read_thread_pmu(self.fd_instr, self.fd_cycles)
            } else {
                PmuSnapshot::default()
            }
        }

        #[cfg(not(any(target_os = "macos", target_os = "linux")))]
        {
            PmuSnapshot::default()
        }
    }

    /// Measures `f()` over `ops` operations and returns a `PmuReport`.
    #[inline]
    pub fn measure<F: FnOnce()>(&self, ops: u64, f: F) -> PmuReport {
        let start_time = Instant::now();
        let before = self.read();
        f();
        let after = self.read();
        let elapsed = start_time.elapsed();
        PmuReport {
            ops,
            delta: after.delta_since(before),
            elapsed,
            pmu_available: self.available,
        }
    }
}

impl Default for ThreadPmu {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(target_os = "linux")]
impl Drop for ThreadPmu {
    fn drop(&mut self) {
        unsafe {
            if self.fd_instr >= 0 {
                let _ = sys_linux::close(self.fd_instr);
            }
            if self.fd_cycles >= 0 {
                let _ = sys_linux::close(self.fd_cycles);
            }
        }
    }
}

#[cfg(target_os = "macos")]
mod sys_macos {
    unsafe extern "C" {
        /// XNU `thread_selfcounts(int type, void *buf, size_t nbytes)`:
        /// `type = 1` (`THSC_CPI`) fills `[u64; 2]` with `[instructions, cycles]`
        /// from Apple Silicon / Intel fixed PMU counters for the calling thread.
        pub fn thread_selfcounts(kind: i32, buf: *mut u64, size: usize) -> i32;
    }
}

#[cfg(target_os = "linux")]
mod sys_linux {
    use super::PmuSnapshot;

    #[repr(C)]
    #[derive(Default)]
    struct PerfEventAttr {
        type_: u32,
        size: u32,
        config: u64,
        sample_period_or_freq: u64,
        sample_type: u64,
        read_format: u64,
        flags: u64,
        wakeup_events_or_watermark: u32,
        bp_type: u32,
        bp_addr_or_config1: u64,
        bp_len_or_config2: u64,
        branch_sample_type: u64,
        sample_regs_user: u64,
        sample_stack_user: u32,
        clockid: i32,
        sample_regs_intr: u64,
        aux_watermark: u32,
        sample_max_stack: u16,
        __reserved_2: u16,
    }

    const PERF_TYPE_HARDWARE: u32 = 0;
    const PERF_COUNT_HW_CPU_CYCLES: u64 = 0;
    const PERF_COUNT_HW_INSTRUCTIONS: u64 = 1;
    // exclude_kernel (bit 5) | exclude_hv (bit 6)
    const FLAGS_USER_ONLY: u64 = (1 << 5) | (1 << 6);

    unsafe extern "C" {
        fn syscall(num: i64, ...) -> i64;
        pub fn close(fd: i32) -> i32;
        fn read(fd: i32, buf: *mut u64, count: usize) -> isize;
    }

    #[cfg(target_arch = "x86_64")]
    const SYS_PERF_EVENT_OPEN: i64 = 298;
    #[cfg(target_arch = "aarch64")]
    const SYS_PERF_EVENT_OPEN: i64 = 241;

    fn perf_open(config: u64) -> i32 {
        let mut attr = PerfEventAttr {
            type_: PERF_TYPE_HARDWARE,
            size: core::mem::size_of::<PerfEventAttr>() as u32,
            config,
            flags: FLAGS_USER_ONLY,
            ..Default::default()
        };
        unsafe {
            syscall(
                SYS_PERF_EVENT_OPEN,
                &mut attr as *mut PerfEventAttr,
                0i32,  // calling thread
                -1i32, // any CPU
                -1i32, // group_fd
                0u64,  // flags
            ) as i32
        }
    }

    pub fn open_thread_pmu() -> (i32, i32, bool) {
        let fd_i = perf_open(PERF_COUNT_HW_INSTRUCTIONS);
        let fd_c = perf_open(PERF_COUNT_HW_CPU_CYCLES);
        let ok = fd_i >= 0 && fd_c >= 0;
        (fd_i, fd_c, ok)
    }

    #[inline(always)]
    pub fn read_thread_pmu(fd_i: i32, fd_c: i32) -> PmuSnapshot {
        let mut instr = 0u64;
        let mut cycles = 0u64;
        unsafe {
            let _ = read(fd_i, &mut instr, 8);
            let _ = read(fd_c, &mut cycles, 8);
        }
        PmuSnapshot {
            instructions: instr,
            cycles,
        }
    }
}
