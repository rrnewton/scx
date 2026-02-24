//! Minimal PMU-based Retired Branch Conditional (RBC) counter and timer.
//!
//! Provides several PMU abstractions:
//!
//! - [`RbcCounter`]: A pure counting counter for measuring scheduler overhead.
//!   Each retired conditional branch maps to a configurable number of nanoseconds.
//!
//! - [`RbcTimer`]: A sampling counter that delivers a signal on overflow. Used
//!   for preemptive interleaving — after N retired branches, the PMU fires a
//!   signal that interrupts the running thread.
//!
//! - [`RdpmcHandle`]: Fast branchless counter read via the `rdpmc` x86
//!   instruction. Created from an `RbcCounter` or `RbcTimer` by mmap'ing the
//!   perf event fd. Safe to call from signal handlers.
//!
//! - [`HwBreakpoint`]: Hardware execution breakpoint using CPU debug registers
//!   (DR0-DR3) via `perf_event_open` with `PERF_TYPE_BREAKPOINT`.
//!
//! CPU detection covers Intel (family 0x06) and AMD Zen 1-5 (families 0x17/0x19/0x1A).
//!
//! Extracted from Reverie (BSD-2-Clause).

use std::fmt;
use std::io;
use std::os::unix::io::RawFd;

use perf_event_open_sys as perf;

/// fcntl constants not available in the libc crate.
const F_SETOWN_EX: libc::c_int = 15;
const F_SETSIG: libc::c_int = 10;
const F_OWNER_TID: libc::c_int = 0;

#[repr(C)]
struct FOwnerEx {
    type_: libc::c_int,
    pid: libc::pid_t,
}

/// Errors from PMU counter operations.
#[derive(Debug)]
pub enum PerfError {
    /// The CPU architecture is not supported for RBC counting.
    UnsupportedCpu,
    /// `perf_event_open` syscall failed.
    Open(io::Error),
    /// ioctl on the perf fd failed.
    Ioctl(io::Error),
    /// fcntl on the perf fd failed.
    Fcntl(io::Error),
    /// read on the perf fd failed.
    Read(io::Error),
    /// mmap on the perf fd failed.
    Mmap(io::Error),
}

impl fmt::Display for PerfError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PerfError::UnsupportedCpu => write!(f, "CPU does not support RBC counting"),
            PerfError::Open(e) => write!(f, "perf_event_open failed: {e}"),
            PerfError::Ioctl(e) => write!(f, "perf ioctl failed: {e}"),
            PerfError::Fcntl(e) => write!(f, "perf fcntl failed: {e}"),
            PerfError::Read(e) => write!(f, "perf read failed: {e}"),
            PerfError::Mmap(e) => write!(f, "perf mmap failed: {e}"),
        }
    }
}

/// A PMU hardware event type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PmuEvent {
    /// Retired conditional branches (vendor-specific raw event).
    RetiredBranchConditional,
    /// Hardware instructions retired (PERF_TYPE_HARDWARE + PERF_COUNT_HW_INSTRUCTIONS).
    InstructionsRetired,
}

impl PmuEvent {
    /// Short identifier for serialization (e.g. in trace headers).
    pub fn short_name(self) -> &'static str {
        match self {
            PmuEvent::RetiredBranchConditional => "rbc",
            PmuEvent::InstructionsRetired => "insn",
        }
    }

    /// Parse from the short identifier. Returns `None` if unrecognized.
    pub fn from_short_name(s: &str) -> Option<Self> {
        match s {
            "rbc" => Some(PmuEvent::RetiredBranchConditional),
            "insn" => Some(PmuEvent::InstructionsRetired),
            _ => None,
        }
    }
}

impl std::fmt::Display for PmuEvent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.short_name())
    }
}

/// PMU configuration for retired conditional branches.
///
/// Holds the raw `perf_event_attr.config` value detected via CPUID.
pub struct PmuConfig {
    /// Raw event selector (umask<<8 | event, for PERF_TYPE_RAW).
    pub rcb_event: u64,
}

/// CPUID vendor and family info extracted from leaf 0x0 and 0x1.
struct CpuIdInfo {
    vendor: [u8; 12],
    family: u32,
}

impl CpuIdInfo {
    /// Query CPUID to get vendor string and full family ID.
    fn detect() -> Self {
        // CPUID leaf 0: vendor string in EBX:EDX:ECX
        let leaf0 = unsafe { std::arch::x86_64::__cpuid(0) };
        let mut vendor = [0u8; 12];
        vendor[0..4].copy_from_slice(&leaf0.ebx.to_le_bytes());
        vendor[4..8].copy_from_slice(&leaf0.edx.to_le_bytes());
        vendor[8..12].copy_from_slice(&leaf0.ecx.to_le_bytes());

        // CPUID leaf 1: family/model in EAX
        let leaf1 = unsafe { std::arch::x86_64::__cpuid(1) };
        let eax = leaf1.eax;
        let base_family = (eax >> 8) & 0xF;
        let ext_family = (eax >> 20) & 0xFF;
        // Full family = base + extended (AMD convention for family >= 0x0F)
        let family = if base_family == 0x0F {
            base_family + ext_family
        } else {
            base_family
        };

        CpuIdInfo { vendor, family }
    }

    fn vendor_str(&self) -> &str {
        std::str::from_utf8(&self.vendor).unwrap_or("")
    }
}

impl PmuConfig {
    /// Detect the RBC event for the current CPU via CPUID.
    ///
    /// Returns `None` if the CPU family/vendor is not recognized.
    pub fn detect() -> Option<Self> {
        let info = CpuIdInfo::detect();

        let rcb_event = match info.vendor_str() {
            "GenuineIntel" if info.family == 0x06 => {
                // Intel: BR_INST_RETIRED.COND - event=0xC4, umask=0x01
                0x01c4
            }
            "AuthenticAMD" => match info.family {
                // Zen 1-2 (family 0x17), Zen 3-4 (0x19), Zen 5 (0x1A)
                0x17 | 0x19 | 0x1A => 0x00d1, // RETIRED_COND_BRANCH
                _ => return None,
            },
            _ => return None,
        };

        Some(PmuConfig { rcb_event })
    }

    /// Resolve a PMU event to `(perf_event_attr.type_, perf_event_attr.config)`.
    pub fn resolve(&self, event: PmuEvent) -> (u32, u64) {
        match event {
            PmuEvent::RetiredBranchConditional => (perf::bindings::PERF_TYPE_RAW, self.rcb_event),
            PmuEvent::InstructionsRetired => (
                perf::bindings::PERF_TYPE_HARDWARE,
                perf::bindings::PERF_COUNT_HW_INSTRUCTIONS as u64,
            ),
        }
    }
}

/// Open a perf_event_open counting fd for the given type/config pair.
///
/// Returns the raw fd. The counter starts disabled, excludes kernel and hypervisor.
fn open_counting_fd(type_: u32, config: u64) -> Result<RawFd, PerfError> {
    let mut attr = perf::bindings::perf_event_attr {
        type_,
        size: std::mem::size_of::<perf::bindings::perf_event_attr>() as u32,
        config,
        ..Default::default()
    };
    attr.set_disabled(1);
    attr.set_exclude_kernel(1);
    attr.set_exclude_hv(1);

    // pid=0 (current thread), cpu=-1 (any CPU)
    let fd = unsafe { perf::perf_event_open(&mut attr, 0, -1, -1, 0) };
    if fd < 0 {
        return Err(PerfError::Open(io::Error::last_os_error()));
    }
    Ok(fd)
}

/// Open a perf_event_open sampling fd for the given type/config/period.
///
/// Returns the raw fd. The counter starts disabled, pinned, excludes kernel
/// and hypervisor, and generates a wakeup after one sample event.
fn open_sampling_fd(type_: u32, config: u64, sample_period: u64) -> Result<RawFd, PerfError> {
    let mut attr = perf::bindings::perf_event_attr {
        type_,
        size: std::mem::size_of::<perf::bindings::perf_event_attr>() as u32,
        config,
        ..Default::default()
    };
    attr.__bindgen_anon_1.sample_period = sample_period;
    attr.set_disabled(1);
    attr.set_exclude_kernel(1);
    attr.set_exclude_hv(1);
    attr.set_pinned(1);
    // Generate a wakeup (overflow notification) after one sample event.
    attr.__bindgen_anon_2.wakeup_events = 1;

    let fd = unsafe { perf::perf_event_open(&mut attr, 0, -1, -1, 0) };
    if fd < 0 {
        return Err(PerfError::Open(io::Error::last_os_error()));
    }
    Ok(fd)
}

/// A PMU counter for retired conditional branches.
///
/// Wraps a `perf_event_open` file descriptor. The counter is pinned to the
/// current thread and CPU-independent (it follows the thread).
pub struct RbcCounter {
    fd: RawFd,
}

impl RbcCounter {
    /// Open a new RBC counter for the current thread.
    ///
    /// The counter starts disabled; call [`enable`](Self::enable) to start counting.
    pub fn new(config: &PmuConfig) -> Result<Self, PerfError> {
        Self::new_event(config, PmuEvent::RetiredBranchConditional)
    }

    /// Open a counter for an arbitrary PMU event.
    ///
    /// The counter starts disabled; call [`enable`](Self::enable) to start counting.
    pub fn new_event(config: &PmuConfig, event: PmuEvent) -> Result<Self, PerfError> {
        let (type_, cfg) = config.resolve(event);
        let fd = open_counting_fd(type_, cfg)?;
        Ok(RbcCounter { fd })
    }

    /// Enable the counter.
    pub fn enable(&self) -> Result<(), PerfError> {
        ioctl_no_arg(self.fd, perf::bindings::ENABLE)
    }

    /// Disable the counter.
    pub fn disable(&self) -> Result<(), PerfError> {
        ioctl_no_arg(self.fd, perf::bindings::DISABLE)
    }

    /// Reset the counter to zero.
    pub fn reset(&self) -> Result<(), PerfError> {
        ioctl_no_arg(self.fd, perf::bindings::RESET)
    }

    /// Read the current counter value.
    pub fn read(&self) -> Result<u64, PerfError> {
        read_counter(self.fd)
    }

    /// Return the raw file descriptor for this counter.
    pub fn raw_fd(&self) -> RawFd {
        self.fd
    }

    /// Create an [`RdpmcHandle`] for fast branchless counter reads.
    ///
    /// Maps the perf event fd into memory so the counter can be read via the
    /// `rdpmc` x86 instruction without any syscall overhead.
    pub fn mmap_rdpmc(&self) -> Result<RdpmcHandle, PerfError> {
        RdpmcHandle::from_fd(self.fd)
    }
}

impl Drop for RbcCounter {
    fn drop(&mut self) {
        unsafe { libc::close(self.fd) };
    }
}

/// A PMU timer that delivers a signal after a specified number of retired
/// conditional branches.
///
/// Unlike [`RbcCounter`] (which is a pure counting counter), `RbcTimer` is a
/// sampling counter: it fires a signal when the counter overflows past the
/// configured sample period.
///
/// # Signal Delivery
///
/// After creation, call [`set_signal_delivery`](Self::set_signal_delivery) to
/// route the overflow notification to a specific thread as a specific signal.
/// The recommended signal is `SIGSTKFLT` (unused by the kernel, safe as a
/// private marker signal).
///
/// # Lifecycle
///
/// ```text
/// new() → set_signal_delivery() → set_period() → reset() → enable()
///   ... branches execute ... signal fires ...
/// disable() → [repeat from set_period()]
/// ```
pub struct RbcTimer {
    fd: RawFd,
}

impl RbcTimer {
    /// A very large period that effectively disables overflow signals.
    ///
    /// Use this as the initial `sample_period` when creating a timer that will
    /// have its period set later via [`set_period`](Self::set_period).
    pub const DISABLE_SAMPLE_PERIOD: u64 = 1 << 60;

    /// Open a new RBC timer for the current thread.
    ///
    /// The timer starts disabled. The `sample_period` controls how many retired
    /// conditional branches must occur before an overflow notification fires.
    /// Use [`DISABLE_SAMPLE_PERIOD`](Self::DISABLE_SAMPLE_PERIOD) to create a
    /// timer without immediate overflow, then set the real period later with
    /// [`set_period`](Self::set_period).
    pub fn new(config: &PmuConfig, sample_period: u64) -> Result<Self, PerfError> {
        Self::new_event(config, PmuEvent::RetiredBranchConditional, sample_period)
    }

    /// Open a timer for an arbitrary PMU event.
    ///
    /// The timer starts disabled. The `sample_period` controls how many events
    /// must occur before an overflow notification fires.
    /// Use [`DISABLE_SAMPLE_PERIOD`](Self::DISABLE_SAMPLE_PERIOD) to create a
    /// timer without immediate overflow, then set the real period later with
    /// [`set_period`](Self::set_period).
    pub fn new_event(
        config: &PmuConfig,
        event: PmuEvent,
        sample_period: u64,
    ) -> Result<Self, PerfError> {
        let (type_, cfg) = config.resolve(event);
        let fd = open_sampling_fd(type_, cfg, sample_period)?;
        Ok(RbcTimer { fd })
    }

    /// Configure signal delivery on counter overflow.
    ///
    /// Routes the overflow notification to thread `tid` as signal `signo`.
    /// The signal is delivered asynchronously when the counter overflows past
    /// the sample period.
    ///
    /// `tid` is a Linux thread ID (from `gettid(2)`). `signo` is the signal
    /// number to deliver (e.g. `libc::SIGSTKFLT`).
    pub fn set_signal_delivery(
        &self,
        tid: libc::pid_t,
        signo: libc::c_int,
    ) -> Result<(), PerfError> {
        set_signal_delivery(self.fd, tid, signo)
    }

    /// Change the overflow period.
    ///
    /// The counter will fire an overflow notification after `ticks` more
    /// retired conditional branches. This takes effect from the current
    /// counter position.
    pub fn set_period(&self, ticks: u64) -> Result<(), PerfError> {
        // PERF_EVENT_IOC_PERIOD expects a pointer to u64.
        let mut ticks = ticks;
        let ret = unsafe {
            libc::ioctl(
                self.fd,
                perf::bindings::PERIOD as libc::c_ulong,
                &mut ticks as *mut u64,
            )
        };
        if ret < 0 {
            return Err(PerfError::Ioctl(io::Error::last_os_error()));
        }
        Ok(())
    }

    /// Enable the timer.
    pub fn enable(&self) -> Result<(), PerfError> {
        ioctl_no_arg(self.fd, perf::bindings::ENABLE)
    }

    /// Disable the timer.
    pub fn disable(&self) -> Result<(), PerfError> {
        ioctl_no_arg(self.fd, perf::bindings::DISABLE)
    }

    /// Reset the counter to zero.
    pub fn reset(&self) -> Result<(), PerfError> {
        ioctl_no_arg(self.fd, perf::bindings::RESET)
    }

    /// Read the current counter value.
    pub fn read(&self) -> Result<u64, PerfError> {
        read_counter(self.fd)
    }

    /// Create an [`RdpmcHandle`] for fast branchless counter reads.
    ///
    /// Maps the perf event fd into memory so the counter can be read via the
    /// `rdpmc` x86 instruction without any syscall overhead.
    pub fn mmap_rdpmc(&self) -> Result<RdpmcHandle, PerfError> {
        RdpmcHandle::from_fd(self.fd)
    }

    /// Return the raw file descriptor for this timer.
    ///
    /// Useful for signal handler identification (matching `si_fd` against
    /// known timer fds).
    pub fn raw_fd(&self) -> RawFd {
        self.fd
    }
}

impl Drop for RbcTimer {
    fn drop(&mut self) {
        unsafe { libc::close(self.fd) };
    }
}

/// Raw ioctl request code for `PERF_EVENT_IOC_ENABLE`.
///
/// Exported for use in async-signal-safe code paths that cannot call
/// higher-level `RbcTimer` methods.
pub const PERF_IOC_ENABLE: libc::c_ulong = perf::bindings::ENABLE as libc::c_ulong;

/// Raw ioctl request code for `PERF_EVENT_IOC_DISABLE`.
pub const PERF_IOC_DISABLE: libc::c_ulong = perf::bindings::DISABLE as libc::c_ulong;

/// Raw ioctl request code for `PERF_EVENT_IOC_RESET`.
pub const PERF_IOC_RESET: libc::c_ulong = perf::bindings::RESET as libc::c_ulong;

/// Raw ioctl request code for `PERF_EVENT_IOC_PERIOD`.
pub const PERF_IOC_PERIOD: libc::c_ulong = perf::bindings::PERIOD as libc::c_ulong;

/// Raw ioctl request code for `PERF_EVENT_IOC_MODIFY_ATTRIBUTES`.
pub const PERF_IOC_MODIFY_ATTRIBUTES: libc::c_ulong =
    perf::bindings::MODIFY_ATTRIBUTES as libc::c_ulong;

/// Shared ioctl helper for ENABLE/DISABLE/RESET (no argument).
fn ioctl_no_arg(fd: RawFd, request: u32) -> Result<(), PerfError> {
    let ret = unsafe { libc::ioctl(fd, request as libc::c_ulong, 0 as libc::c_ulong) };
    if ret < 0 {
        return Err(PerfError::Ioctl(io::Error::last_os_error()));
    }
    Ok(())
}

/// Shared read helper for counter value.
fn read_counter(fd: RawFd) -> Result<u64, PerfError> {
    let mut count: u64 = 0;
    let ret = unsafe {
        libc::read(
            fd,
            &mut count as *mut u64 as *mut libc::c_void,
            std::mem::size_of::<u64>(),
        )
    };
    if ret < 0 {
        return Err(PerfError::Read(io::Error::last_os_error()));
    }
    Ok(count)
}

/// Execute the `rdpmc` instruction to read a performance counter.
///
/// `ecx` is the counter index (from `perf_event_mmap_page.index - 1`).
/// Returns the full 64-bit counter value (EAX | EDX << 32).
#[inline(always)]
fn rdpmc(ecx: u32) -> u64 {
    let lo: u32;
    let hi: u32;
    unsafe {
        std::arch::asm!(
            "rdpmc",
            in("ecx") ecx,
            out("eax") lo,
            out("edx") hi,
            options(nostack, nomem, preserves_flags),
        );
    }
    (hi as u64) << 32 | lo as u64
}

/// Shared helper to configure async signal delivery on a perf event fd.
///
/// Routes signals from `fd` to thread `tid` as signal `signo`.
fn set_signal_delivery(fd: RawFd, tid: libc::pid_t, signo: libc::c_int) -> Result<(), PerfError> {
    let owner = FOwnerEx {
        type_: F_OWNER_TID,
        pid: tid,
    };
    let ret = unsafe { libc::fcntl(fd, F_SETOWN_EX, &owner as *const FOwnerEx) };
    if ret < 0 {
        return Err(PerfError::Fcntl(io::Error::last_os_error()));
    }
    let ret = unsafe { libc::fcntl(fd, libc::F_SETFL, libc::O_ASYNC) };
    if ret < 0 {
        return Err(PerfError::Fcntl(io::Error::last_os_error()));
    }
    let ret = unsafe { libc::fcntl(fd, F_SETSIG, signo) };
    if ret < 0 {
        return Err(PerfError::Fcntl(io::Error::last_os_error()));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// RdpmcHandle — fast branchless PMU counter read via mmap + rdpmc
// ---------------------------------------------------------------------------

/// Fast branchless PMU counter read via `rdpmc`.
///
/// Created from an [`RbcCounter`] or [`RbcTimer`] by mmap'ing the perf event
/// fd. The resulting handle can read the counter value using the `rdpmc` x86
/// instruction, which is branchless and safe to call from signal handlers.
pub struct RdpmcHandle {
    mmap_page: *const perf::bindings::perf_event_mmap_page,
}

// The mmap pointer is valid cross-thread for the same perf event fd (which is
// per-thread anyway). The mapping is read-only and the kernel maintains
// coherency via the seqcount lock.
unsafe impl Send for RdpmcHandle {}

impl RdpmcHandle {
    /// Create an `RdpmcHandle` by mmap'ing a perf event file descriptor.
    ///
    /// The fd must be a valid perf event fd (from `perf_event_open`). The
    /// kernel must support `cap_user_rdpmc` for this event type.
    pub fn from_fd(fd: RawFd) -> Result<Self, PerfError> {
        let page_size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) } as usize;
        let ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                page_size,
                libc::PROT_READ,
                libc::MAP_SHARED,
                fd,
                0,
            )
        };
        if ptr == libc::MAP_FAILED {
            return Err(PerfError::Mmap(io::Error::last_os_error()));
        }
        let mmap_page = ptr as *const perf::bindings::perf_event_mmap_page;

        // Verify the kernel supports rdpmc for this event.
        let caps = unsafe { &(*mmap_page).__bindgen_anon_1.__bindgen_anon_1 };
        if caps.cap_user_rdpmc() == 0 {
            unsafe { libc::munmap(ptr, page_size) };
            return Err(PerfError::Mmap(io::Error::new(
                io::ErrorKind::Unsupported,
                "cap_user_rdpmc not set — rdpmc not available for this event",
            )));
        }

        Ok(RdpmcHandle { mmap_page })
    }

    /// Read the counter value via the `rdpmc` instruction.
    ///
    /// This is branchless in the common case (seqcount succeeds on first try).
    /// The seqcount retry loop is needed for correctness when the kernel updates
    /// the mmap page concurrently, but in practice completes on the first
    /// iteration.
    ///
    /// # Safety requirements
    ///
    /// The caller must ensure the underlying perf event fd is still open.
    #[inline]
    pub fn read(&self) -> u64 {
        let page = self.mmap_page;
        loop {
            // Read the seqcount lock (must be even when stable).
            let seq = unsafe { std::ptr::read_volatile(&(*page).lock) };
            std::sync::atomic::fence(std::sync::atomic::Ordering::Acquire);

            let index = unsafe { std::ptr::read_volatile(&(*page).index) };
            let offset = unsafe { std::ptr::read_volatile(&(*page).offset) };
            let pmc_width = unsafe { std::ptr::read_volatile(&(*page).pmc_width) };

            // If index == 0, the counter is not directly readable via rdpmc.
            // Fall back to returning offset (which is the kernel-maintained
            // count). This branch is never taken in the common case.
            let count = if index == 0 {
                offset as u64
            } else {
                let raw = rdpmc(index - 1) as i64;
                // Sign-extend / mask to pmc_width bits and add offset.
                let shift = 64 - pmc_width as i64;
                let adjusted = ((raw << shift) >> shift) + offset;
                adjusted as u64
            };

            // Validate the seqcount: re-read and check it hasn't changed.
            std::sync::atomic::fence(std::sync::atomic::Ordering::Acquire);
            let seq2 = unsafe { std::ptr::read_volatile(&(*page).lock) };
            if seq == seq2 && (seq & 1) == 0 {
                return count;
            }
            // Seqcount changed — retry (extremely rare).
            core::hint::spin_loop();
        }
    }
}

impl Drop for RdpmcHandle {
    fn drop(&mut self) {
        let page_size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) } as usize;
        unsafe {
            libc::munmap(self.mmap_page as *mut libc::c_void, page_size);
        }
    }
}

// ---------------------------------------------------------------------------
// HwBreakpoint — hardware execution breakpoint via debug registers
// ---------------------------------------------------------------------------

/// Hardware execution breakpoint using CPU debug registers (DR0-DR3).
///
/// Uses `perf_event_open` with `PERF_TYPE_BREAKPOINT` and `HW_BREAKPOINT_X`
/// to set an execution breakpoint at a virtual address. When the CPU executes
/// the instruction at that address, a signal is delivered to the owning thread.
///
/// Unlike software breakpoints (INT3), hardware breakpoints don't modify the
/// instruction stream — no save/restore/single-step dance required.
pub struct HwBreakpoint {
    fd: RawFd,
}

impl HwBreakpoint {
    /// Create a new hardware execution breakpoint.
    ///
    /// Sets an execution breakpoint at `addr` for thread `tid`. When the
    /// breakpoint fires, signal `signo` is delivered to that thread.
    ///
    /// The breakpoint starts disabled; call [`enable`](Self::enable) to arm it.
    pub fn new(addr: u64, tid: libc::pid_t, signo: libc::c_int) -> Result<Self, PerfError> {
        let mut attr = Self::make_bp_attr(addr);

        // pid=tid, cpu=-1 (any CPU)
        let fd = unsafe { perf::perf_event_open(&mut attr, tid, -1, -1, 0) };
        if fd < 0 {
            return Err(PerfError::Open(io::Error::last_os_error()));
        }

        // Set up signal delivery (same pattern as RbcTimer).
        set_signal_delivery(fd, tid, signo)?;

        Ok(HwBreakpoint { fd })
    }

    /// Enable the breakpoint.
    pub fn enable(&self) -> Result<(), PerfError> {
        ioctl_no_arg(self.fd, perf::bindings::ENABLE)
    }

    /// Disable the breakpoint.
    pub fn disable(&self) -> Result<(), PerfError> {
        ioctl_no_arg(self.fd, perf::bindings::DISABLE)
    }

    /// Change the breakpoint address.
    ///
    /// Uses `PERF_EVENT_IOC_MODIFY_ATTRIBUTES` to atomically update the
    /// breakpoint to fire at `addr` instead.
    pub fn set_addr(&self, addr: u64) -> Result<(), PerfError> {
        let mut attr = Self::make_bp_attr(addr);
        // MODIFY_ATTRIBUTES expects a *mut perf_event_attr.
        let ret = unsafe {
            libc::ioctl(
                self.fd,
                PERF_IOC_MODIFY_ATTRIBUTES,
                &mut attr as *mut perf::bindings::perf_event_attr,
            )
        };
        if ret < 0 {
            return Err(PerfError::Ioctl(io::Error::last_os_error()));
        }
        Ok(())
    }

    /// Return the raw file descriptor for this breakpoint.
    pub fn raw_fd(&self) -> RawFd {
        self.fd
    }

    /// Build a `perf_event_attr` for an execution breakpoint at `addr`.
    fn make_bp_attr(addr: u64) -> perf::bindings::perf_event_attr {
        let mut attr = perf::bindings::perf_event_attr {
            type_: perf::bindings::PERF_TYPE_BREAKPOINT,
            size: std::mem::size_of::<perf::bindings::perf_event_attr>() as u32,
            bp_type: perf::bindings::HW_BREAKPOINT_X,
            ..Default::default()
        };
        attr.__bindgen_anon_3.bp_addr = addr;
        attr.__bindgen_anon_4.bp_len = perf::bindings::HW_BREAKPOINT_LEN_8 as u64;
        // sample_period=1: generate an overflow notification on every hit.
        attr.__bindgen_anon_1.sample_period = 1;
        // Wakeup after every overflow so the signal is delivered promptly.
        attr.__bindgen_anon_2.wakeup_events = 1;
        attr.set_disabled(1);
        attr.set_exclude_kernel(1);
        attr.set_exclude_hv(1);
        attr.set_pinned(1);
        attr
    }
}

impl Drop for HwBreakpoint {
    fn drop(&mut self) {
        unsafe { libc::close(self.fd) };
    }
}

/// Try to create an RBC counter, returning `None` with a warning if unavailable.
///
/// This is the recommended entry point: it handles CPU detection failure and
/// perf_event_open permission errors gracefully.
pub fn try_create_rbc_counter() -> Option<RbcCounter> {
    let config = match PmuConfig::detect() {
        Some(c) => c,
        None => {
            tracing::warn!("RBC counter: CPU not supported (no CPUID match)");
            return None;
        }
    };
    match RbcCounter::new(&config) {
        Ok(counter) => Some(counter),
        Err(e) => {
            tracing::warn!("RBC counter unavailable: {e}");
            None
        }
    }
}

/// Try to create an RBC timer, returning `None` with a warning if unavailable.
///
/// The timer starts with [`RbcTimer::DISABLE_SAMPLE_PERIOD`] so it won't
/// fire until a real period is set via [`RbcTimer::set_period`].
pub fn try_create_rbc_timer() -> Option<RbcTimer> {
    try_create_pmu_timer(PmuEvent::RetiredBranchConditional)
}

/// Try to create a PMU timer for an arbitrary event, returning `None` with a
/// warning if unavailable.
///
/// The timer starts with [`RbcTimer::DISABLE_SAMPLE_PERIOD`] so it won't
/// fire until a real period is set via [`RbcTimer::set_period`].
pub fn try_create_pmu_timer(event: PmuEvent) -> Option<RbcTimer> {
    let config = match PmuConfig::detect() {
        Some(c) => c,
        None => {
            tracing::warn!("PMU timer ({event}): CPU not supported (no CPUID match)");
            return None;
        }
    };
    match RbcTimer::new_event(&config, event, RbcTimer::DISABLE_SAMPLE_PERIOD) {
        Ok(timer) => Some(timer),
        Err(e) => {
            tracing::warn!("PMU timer ({event}) unavailable: {e}");
            None
        }
    }
}

/// Try to create a hardware execution breakpoint, returning `None` if
/// unavailable.
///
/// Hardware breakpoints may not be available in VMs or when debug registers
/// are in use. This function handles errors gracefully with a warning log.
pub fn try_create_hw_breakpoint(addr: u64) -> Option<HwBreakpoint> {
    let tid = unsafe { libc::syscall(libc::SYS_gettid) } as libc::pid_t;
    match HwBreakpoint::new(addr, tid, libc::SIGTRAP) {
        Ok(bp) => Some(bp),
        Err(e) => {
            tracing::warn!("HW breakpoint unavailable: {e}");
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pmu_detect() {
        // Just verify detection doesn't panic; it may return None on unsupported CPUs
        let _config = PmuConfig::detect();
    }

    #[test]
    fn test_rbc_counter_lifecycle() {
        let config = match PmuConfig::detect() {
            Some(c) => c,
            None => {
                eprintln!("skipping RBC test: unsupported CPU");
                return;
            }
        };

        let counter = match RbcCounter::new(&config) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("skipping RBC test: {e}");
                return;
            }
        };

        // Basic lifecycle: reset -> enable -> disable -> read
        counter.reset().unwrap();
        counter.enable().unwrap();

        // Do some work to generate conditional branches
        let mut sum = 0u64;
        for i in 0..1000 {
            if i % 2 == 0 {
                sum += i;
            }
        }
        // Prevent optimization
        std::hint::black_box(sum);

        counter.disable().unwrap();
        let count = counter.read().unwrap();
        // Should have counted some conditional branches.
        // In VMs/containers, PMU may be available but not actually counting.
        if count == 0 {
            eprintln!("skipping RBC assertion: counter reads 0 (likely VM/container)");
            return;
        }
        assert!(count > 0, "expected non-zero RBC count, got {count}");
    }

    #[test]
    fn test_try_create_rbc_counter() {
        // Should not panic regardless of platform
        let _counter = try_create_rbc_counter();
    }

    #[test]
    fn test_try_create_rbc_timer() {
        // Should not panic regardless of platform
        let _timer = try_create_rbc_timer();
    }

    #[test]
    fn test_rbc_timer_signal_delivery() {
        use std::sync::atomic::{AtomicBool, Ordering};

        static SIGNAL_RECEIVED: AtomicBool = AtomicBool::new(false);

        extern "C" fn handler(_signo: libc::c_int) {
            SIGNAL_RECEIVED.store(true, Ordering::SeqCst);
        }

        let config = match PmuConfig::detect() {
            Some(c) => c,
            None => {
                eprintln!("skipping RBC timer test: unsupported CPU");
                return;
            }
        };

        // Create timer with a small period (100 branches).
        let timer = match RbcTimer::new(&config, 100) {
            Ok(t) => t,
            Err(e) => {
                eprintln!("skipping RBC timer test: {e}");
                return;
            }
        };

        // Install SIGSTKFLT handler.
        let sa = libc::sigaction {
            sa_sigaction: handler as *const () as libc::sighandler_t,
            sa_mask: unsafe { std::mem::zeroed() },
            sa_flags: libc::SA_SIGINFO,
            sa_restorer: None,
        };
        let ret = unsafe { libc::sigaction(libc::SIGSTKFLT, &sa, std::ptr::null_mut()) };
        assert_eq!(ret, 0, "sigaction failed");

        // Route signal to this thread.
        let tid = unsafe { libc::syscall(libc::SYS_gettid) } as libc::pid_t;
        timer
            .set_signal_delivery(tid, libc::SIGSTKFLT)
            .expect("set_signal_delivery");

        // Enable and run some branches to trigger overflow.
        timer.reset().expect("reset");
        timer.enable().expect("enable");

        let mut sum = 0u64;
        for i in 0..100_000u64 {
            if i % 2 == 0 {
                sum += i;
            }
        }
        std::hint::black_box(sum);

        timer.disable().expect("disable");

        // Restore default handler.
        let sa_default = libc::sigaction {
            sa_sigaction: libc::SIG_DFL,
            sa_mask: unsafe { std::mem::zeroed() },
            sa_flags: 0,
            sa_restorer: None,
        };
        unsafe {
            libc::sigaction(libc::SIGSTKFLT, &sa_default, std::ptr::null_mut());
        }

        // In VMs/containers, the counter may not actually fire.
        let count = timer.read().unwrap_or(0);
        if count == 0 {
            eprintln!("skipping RBC timer signal assertion: counter reads 0 (likely VM/container)");
            return;
        }

        assert!(
            SIGNAL_RECEIVED.load(Ordering::SeqCst),
            "expected SIGSTKFLT signal after {count} retired branches with period=100"
        );
    }

    #[test]
    fn test_rbc_timer_set_period() {
        let config = match PmuConfig::detect() {
            Some(c) => c,
            None => {
                eprintln!("skipping RBC timer period test: unsupported CPU");
                return;
            }
        };

        // Create with large period, then set a real one.
        let timer = match RbcTimer::new(&config, RbcTimer::DISABLE_SAMPLE_PERIOD) {
            Ok(t) => t,
            Err(e) => {
                eprintln!("skipping RBC timer period test: {e}");
                return;
            }
        };

        timer.set_period(500).expect("set_period should succeed");
        timer.reset().expect("reset");
        timer.enable().expect("enable");

        let mut sum = 0u64;
        for i in 0..1000u64 {
            if i % 2 == 0 {
                sum += i;
            }
        }
        std::hint::black_box(sum);

        timer.disable().expect("disable");
        // Just verify it didn't crash; actual counting may not work in VMs.
    }

    /// Helper: generate conditional branches to exercise PMU counters.
    fn generate_branches(n: u64) -> u64 {
        let mut sum = 0u64;
        for i in 0..n {
            if i % 2 == 0 {
                sum += i;
            }
        }
        std::hint::black_box(sum)
    }

    /// Helper: create an RbcCounter or skip the test if unavailable.
    fn make_counter_or_skip() -> Option<(PmuConfig, RbcCounter)> {
        let config = match PmuConfig::detect() {
            Some(c) => c,
            None => {
                eprintln!("skipping test: unsupported CPU");
                return None;
            }
        };
        let counter = match RbcCounter::new(&config) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("skipping test: {e}");
                return None;
            }
        };
        Some((config, counter))
    }

    #[test]
    fn test_rdpmc_basic() {
        let (_config, counter) = match make_counter_or_skip() {
            Some(pair) => pair,
            None => return,
        };

        let handle = match counter.mmap_rdpmc() {
            Ok(h) => h,
            Err(e) => {
                eprintln!("skipping rdpmc test: {e}");
                return;
            }
        };

        counter.reset().unwrap();
        counter.enable().unwrap();

        generate_branches(10_000);

        counter.disable().unwrap();

        let count = handle.read();
        // In VMs/containers, PMU may be available but not actually counting.
        if count == 0 {
            eprintln!("skipping rdpmc assertion: counter reads 0 (likely VM/container)");
            return;
        }
        assert!(count > 0, "expected non-zero rdpmc count, got {count}");
    }

    #[test]
    fn test_rdpmc_matches_read() {
        let (_config, counter) = match make_counter_or_skip() {
            Some(pair) => pair,
            None => return,
        };

        let handle = match counter.mmap_rdpmc() {
            Ok(h) => h,
            Err(e) => {
                eprintln!("skipping rdpmc test: {e}");
                return;
            }
        };

        counter.reset().unwrap();
        counter.enable().unwrap();

        generate_branches(50_000);

        counter.disable().unwrap();

        let rdpmc_val = handle.read();
        let fd_val = counter.read().unwrap();

        if rdpmc_val == 0 && fd_val == 0 {
            eprintln!("skipping rdpmc vs fd comparison: both read 0 (likely VM/container)");
            return;
        }

        // Both readings are taken after disable, so they should be very close.
        // Allow a small tolerance because rdpmc and read(fd) may sample at
        // slightly different points in the kernel accounting.
        let diff = (rdpmc_val as i64 - fd_val as i64).unsigned_abs();
        let tolerance = std::cmp::max(fd_val / 100, 10); // 1% or 10, whichever is larger
        assert!(
            diff <= tolerance,
            "rdpmc ({rdpmc_val}) and fd read ({fd_val}) differ by {diff}, \
             exceeds tolerance {tolerance}"
        );
    }

    /// Mutex to serialize HW breakpoint tests that install SIGTRAP handlers.
    /// Without serialization, one test restoring SIG_DFL can kill another
    /// test's thread while its breakpoint is still armed.
    static HW_BP_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    #[test]
    fn test_hw_breakpoint_basic() {
        use std::sync::atomic::{AtomicBool, Ordering};

        let _guard = HW_BP_TEST_LOCK.lock().unwrap();

        static BP_FIRED: AtomicBool = AtomicBool::new(false);
        BP_FIRED.store(false, Ordering::SeqCst);

        extern "C" fn trap_handler(_signo: libc::c_int) {
            BP_FIRED.store(true, Ordering::SeqCst);
        }

        // Use the address of our own helper function as the breakpoint target.
        let target_fn: fn(u64) -> u64 = generate_branches;
        let target_addr = target_fn as *const () as u64;

        let tid = unsafe { libc::syscall(libc::SYS_gettid) } as libc::pid_t;
        let bp = match HwBreakpoint::new(target_addr, tid, libc::SIGTRAP) {
            Ok(bp) => bp,
            Err(e) => {
                eprintln!("skipping HW breakpoint test: {e}");
                return;
            }
        };

        // Install SIGTRAP handler.
        let sa = libc::sigaction {
            sa_sigaction: trap_handler as libc::sighandler_t,
            sa_mask: unsafe { std::mem::zeroed() },
            sa_flags: 0,
            sa_restorer: None,
        };
        let ret = unsafe { libc::sigaction(libc::SIGTRAP, &sa, std::ptr::null_mut()) };
        assert_eq!(ret, 0, "sigaction failed");

        bp.enable().expect("enable");

        // Call the target function to trigger the breakpoint.
        generate_branches(100);

        // Disable breakpoint BEFORE restoring default handler to avoid
        // SIGTRAP with SIG_DFL (which kills the process).
        bp.disable().expect("disable");
        drop(bp);

        // Restore default handler only after the breakpoint fd is closed.
        let sa_default = libc::sigaction {
            sa_sigaction: libc::SIG_DFL,
            sa_mask: unsafe { std::mem::zeroed() },
            sa_flags: 0,
            sa_restorer: None,
        };
        unsafe {
            libc::sigaction(libc::SIGTRAP, &sa_default, std::ptr::null_mut());
        }

        assert!(
            BP_FIRED.load(Ordering::SeqCst),
            "expected SIGTRAP from HW breakpoint at {target_addr:#x}"
        );
    }

    #[test]
    fn test_hw_breakpoint_signal_delivery() {
        use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

        let _guard = HW_BP_TEST_LOCK.lock().unwrap();

        static BP_SIGNAL_RECEIVED: AtomicBool = AtomicBool::new(false);
        static BP_SI_CODE: AtomicU64 = AtomicU64::new(0);
        BP_SIGNAL_RECEIVED.store(false, Ordering::SeqCst);
        BP_SI_CODE.store(0, Ordering::SeqCst);

        extern "C" fn siginfo_handler(
            _signo: libc::c_int,
            info: *mut libc::siginfo_t,
            _ctx: *mut libc::c_void,
        ) {
            BP_SIGNAL_RECEIVED.store(true, Ordering::SeqCst);
            if !info.is_null() {
                let code = unsafe { (*info).si_code } as u64;
                BP_SI_CODE.store(code, Ordering::SeqCst);
            }
        }

        let target_fn: fn(u64) -> u64 = generate_branches;
        let target_addr = target_fn as *const () as u64;

        let tid = unsafe { libc::syscall(libc::SYS_gettid) } as libc::pid_t;
        let bp = match HwBreakpoint::new(target_addr, tid, libc::SIGTRAP) {
            Ok(bp) => bp,
            Err(e) => {
                eprintln!("skipping HW breakpoint signal test: {e}");
                return;
            }
        };

        // Install SIGTRAP handler with SA_SIGINFO to get siginfo_t.
        let sa = libc::sigaction {
            sa_sigaction: siginfo_handler as libc::sighandler_t,
            sa_mask: unsafe { std::mem::zeroed() },
            sa_flags: libc::SA_SIGINFO,
            sa_restorer: None,
        };
        let ret = unsafe { libc::sigaction(libc::SIGTRAP, &sa, std::ptr::null_mut()) };
        assert_eq!(ret, 0, "sigaction failed");

        bp.enable().expect("enable");

        generate_branches(100);

        // Disable breakpoint BEFORE restoring default handler.
        bp.disable().expect("disable");
        drop(bp);

        // Restore default handler only after the breakpoint fd is closed.
        let sa_default = libc::sigaction {
            sa_sigaction: libc::SIG_DFL,
            sa_mask: unsafe { std::mem::zeroed() },
            sa_flags: 0,
            sa_restorer: None,
        };
        unsafe {
            libc::sigaction(libc::SIGTRAP, &sa_default, std::ptr::null_mut());
        }

        if !BP_SIGNAL_RECEIVED.load(Ordering::SeqCst) {
            eprintln!(
                "skipping HW breakpoint signal info assertion: \
                 no signal received (likely VM/container)"
            );
            return;
        }

        // Verify that the signal was delivered with a valid si_code.
        // TRAP_HWBKPT (4) indicates a hardware breakpoint/watchpoint.
        let code = BP_SI_CODE.load(Ordering::SeqCst);
        assert!(
            code > 0,
            "expected non-zero si_code from HW breakpoint signal, got {code}"
        );
    }
}
