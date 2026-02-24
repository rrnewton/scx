//! Compact formatting helpers for trace output.
//!
//! Provides [`SimLayer`], a custom tracing subscriber layer that formats events
//! showing simulator virtual time instead of wall-clock time. Unlike the default
//! `tracing_subscriber::fmt` layer, `SimLayer` does NOT reuse a thread-local
//! `String` buffer between events. Instead, each event is formatted into a
//! freshly-allocated `String` and written to stderr via a single `libc::write()`
//! syscall.
//!
//! This avoids a corruption issue under Frida Stalker dynamic binary
//! instrumentation: Stalker's JIT code cache writes stale pointers (8 bytes)
//! into the beginning of the reused thread-local format buffer, producing
//! garbage bytes in log output at preemption points.

use std::fmt;
use std::fmt::Write as _;

use tracing::field::{Field, Visit};
use tracing::{Event, Level, Subscriber};
use tracing_subscriber::fmt::format::Writer;
use tracing_subscriber::fmt::{FmtContext, FormatEvent, FormatFields};
use tracing_subscriber::layer::Context;
use tracing_subscriber::registry::LookupSpan;
use tracing_subscriber::Layer;

use crate::kfuncs::{sim_clock, sim_cpu, sim_cpu_width};
use crate::types::{CpuId, TimeNs};

/// Wrapper that displays large round numbers compactly.
///
/// Exact multiples of powers of 1000 are shortened:
/// - `1_000` → `1K`
/// - `20_000_000` → `20M`
/// - `3_000_000_000` → `3B`
/// - `1_000_000_000_000` → `1T`
///
/// Non-round numbers pass through unchanged: `12345` → `12345`.
pub struct FmtN(pub u64);

impl fmt::Display for FmtN {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let v = self.0;
        const SUFFIXES: &[(u64, &str)] = &[
            (1_000_000_000_000, "T"),
            (1_000_000_000, "B"),
            (1_000_000, "M"),
            (1_000, "K"),
        ];
        for &(divisor, suffix) in SUFFIXES {
            if v >= divisor && v.is_multiple_of(divisor) {
                return write!(f, "{}{}", v / divisor, suffix);
            }
        }
        write!(f, "{v}")
    }
}

/// Whether a timestamp is a per-CPU local time or a global time.
#[derive(Debug, Clone, Copy)]
pub enum TsKind {
    /// Per-CPU local time, optionally carrying the CPU ID and zero-pad width.
    Local(Option<CpuId>, u8),
    Global,
}

/// Timestamp formatter with underscore-grouped digits and `:cpu`/`:G` suffix.
///
/// Formats nanosecond timestamps for trace output with room for 12 digits
/// (up to ~15 seconds), grouped in 3s with underscores, right-aligned.
///
/// When a CPU ID is present, the suffix includes the zero-padded CPU number:
/// - `[  988_779_026:cpu01]` — CPU 1, 2-digit padding
/// - `[  988_779_026:cpu]`   — no CPU context (e.g. timer events)
/// - `[  988_779_026:G]`     — global timestamp
pub struct FmtTs {
    pub ns: TimeNs,
    pub kind: TsKind,
}

impl FmtTs {
    pub fn local(ns: TimeNs, cpu: Option<CpuId>, width: u8) -> Self {
        Self {
            ns,
            kind: TsKind::Local(cpu, width),
        }
    }

    pub fn global(ns: TimeNs) -> Self {
        Self {
            ns,
            kind: TsKind::Global,
        }
    }
}

/// Format a u64 with underscore grouping (groups of 3 from the right).
pub(crate) fn fmt_grouped(v: u64) -> String {
    let digits = v.to_string();
    let len = digits.len();
    if len <= 3 {
        return digits;
    }
    let mut result = String::with_capacity(len + (len - 1) / 3);
    for (i, ch) in digits.chars().enumerate() {
        if i > 0 && (len - i).is_multiple_of(3) {
            result.push('_');
        }
        result.push(ch);
    }
    result
}

impl fmt::Display for FmtTs {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let grouped = fmt_grouped(self.ns);
        match self.kind {
            TsKind::Local(Some(cpu), width) => {
                let w = width as usize;
                write!(f, "{:>15}:cpu{:0>w$}", grouped, cpu.0, w = w)
            }
            TsKind::Local(None, _) => {
                write!(f, "{:>15}:cpu", grouped)
            }
            TsKind::Global => {
                write!(f, "{:>15}:G", grouped)
            }
        }
    }
}

/// Custom event formatter that shows simulator virtual time instead of
/// wall-clock time and uses plain colored text (no italic/background).
pub struct SimFormat;

impl<S, N> FormatEvent<S, N> for SimFormat
where
    S: Subscriber + for<'a> LookupSpan<'a>,
    N: for<'a> FormatFields<'a> + 'static,
{
    fn format_event(
        &self,
        _ctx: &FmtContext<'_, S, N>,
        mut writer: Writer<'_>,
        event: &Event<'_>,
    ) -> fmt::Result {
        // Simulated timestamp
        let clock = sim_clock();
        let cpu = sim_cpu();
        let width = sim_cpu_width();
        write!(writer, "[{}] ", FmtTs::local(clock, cpu, width))?;

        // Level with color (no italic, no background)
        let level = *event.metadata().level();
        if writer.has_ansi_escapes() {
            let color = match level {
                Level::ERROR => "\x1b[31m", // red
                Level::WARN => "\x1b[33m",  // yellow
                Level::INFO => "\x1b[32m",  // green
                Level::DEBUG => "\x1b[34m", // blue
                Level::TRACE => "\x1b[35m", // magenta
            };
            write!(writer, "{color}{level:>5}\x1b[0m ")?;
        } else {
            write!(writer, "{level:>5} ")?;
        }

        // Collect fields and message
        let mut visitor = FieldCollector::default();
        event.record(&mut visitor);

        // Message first
        write!(writer, "{}", visitor.message)?;

        // Then fields as plain key=value (no italic ANSI)
        for (key, value) in &visitor.fields {
            write!(writer, " {key}={value}")?;
        }

        writeln!(writer)
    }
}

/// Visitor that collects the message and key-value fields from a tracing event.
#[derive(Default)]
struct FieldCollector {
    message: String,
    fields: Vec<(String, String)>,
}

impl Visit for FieldCollector {
    fn record_debug(&mut self, field: &Field, value: &dyn fmt::Debug) {
        if field.name() == "message" {
            self.message = format!("{value:?}");
        } else {
            self.fields
                .push((field.name().to_string(), format!("{value:?}")));
        }
    }

    fn record_str(&mut self, field: &Field, value: &str) {
        if field.name() == "message" {
            self.message = value.to_string();
        } else {
            self.fields
                .push((field.name().to_string(), value.to_string()));
        }
    }

    fn record_u64(&mut self, field: &Field, value: u64) {
        self.fields
            .push((field.name().to_string(), value.to_string()));
    }

    fn record_i64(&mut self, field: &Field, value: i64) {
        self.fields
            .push((field.name().to_string(), value.to_string()));
    }

    fn record_bool(&mut self, field: &Field, value: bool) {
        self.fields
            .push((field.name().to_string(), value.to_string()));
    }
}

/// Format a nanosecond duration as a human-readable string.
///
/// Uses the largest unit that fits without a fractional part, with one
/// decimal place for sub-unit remainders:
/// - `0` → `"0ns"`
/// - `1_500` → `"1.5µs"`
/// - `20_000_000` → `"20ms"`
/// - `3_500_000_000` → `"3.5s"`
pub fn fmt_duration_ns(ns: u64) -> String {
    const UNITS: &[(u64, &str)] = &[(1_000_000_000, "s"), (1_000_000, "ms"), (1_000, "µs")];
    for &(divisor, suffix) in UNITS {
        if ns >= divisor {
            let whole = ns / divisor;
            let frac = (ns % divisor) * 10 / divisor;
            return if frac == 0 {
                format!("{whole}{suffix}")
            } else {
                format!("{whole}.{frac}{suffix}")
            };
        }
    }
    format!("{ns}ns")
}

// ---------------------------------------------------------------------------
// SimLayer — Stalker-safe tracing layer
// ---------------------------------------------------------------------------

/// Tracing layer that formats events with simulator virtual time and writes
/// to stderr via raw `libc::write()`, avoiding the thread-local buffer reuse
/// pattern in `tracing_subscriber::fmt` that gets corrupted under Frida Stalker.
///
/// Each event is formatted into a freshly-allocated `String`. The formatted
/// bytes are then written to fd 2 in a single `write()` syscall, guaranteeing
/// atomic output for lines under `PIPE_BUF` (4096 bytes).
///
/// ANSI color codes are enabled when stderr is a terminal.
pub struct SimLayer {
    ansi: bool,
}

impl Default for SimLayer {
    fn default() -> Self {
        Self::new()
    }
}

impl SimLayer {
    /// Create a new `SimLayer`.
    ///
    /// ANSI color output is enabled when stderr is a terminal (detected via
    /// `libc::isatty`).
    pub fn new() -> Self {
        let ansi = unsafe { libc::isatty(2) != 0 };
        Self { ansi }
    }

    /// Format a single event into the provided `String` buffer.
    fn format_event_to_buf(&self, buf: &mut String, event: &Event<'_>) {
        let clock = sim_clock();
        let cpu = sim_cpu();
        let width = sim_cpu_width();

        // Timestamp
        let _ = write!(buf, "[{}] ", FmtTs::local(clock, cpu, width));

        // Level with optional ANSI color
        let level = *event.metadata().level();
        if self.ansi {
            let color = match level {
                Level::ERROR => "\x1b[31m",
                Level::WARN => "\x1b[33m",
                Level::INFO => "\x1b[32m",
                Level::DEBUG => "\x1b[34m",
                Level::TRACE => "\x1b[35m",
            };
            let _ = write!(buf, "{color}{level:>5}\x1b[0m ");
        } else {
            let _ = write!(buf, "{level:>5} ");
        }

        // Collect fields and message
        let mut visitor = FieldCollector::default();
        event.record(&mut visitor);

        // Message first, then fields
        let _ = write!(buf, "{}", visitor.message);
        for (key, value) in &visitor.fields {
            let _ = write!(buf, " {key}={value}");
        }
        buf.push('\n');
    }

    /// Write bytes to stderr via a raw `libc::write()` syscall.
    ///
    /// Bypasses Rust's `io::Stderr` locking, which interacts poorly with
    /// Stalker-translated code. A single `write(2, ...)` is atomic for
    /// payloads under `PIPE_BUF` (4096 bytes on Linux).
    fn write_stderr(data: &[u8]) {
        let mut written = 0;
        while written < data.len() {
            let ret = unsafe {
                libc::write(
                    2,
                    data[written..].as_ptr() as *const libc::c_void,
                    data.len() - written,
                )
            };
            if ret < 0 {
                break; // I/O error, give up
            }
            written += ret as usize;
        }
    }
}

impl<S> Layer<S> for SimLayer
where
    S: Subscriber + for<'a> LookupSpan<'a>,
{
    fn on_event(&self, event: &Event<'_>, _ctx: Context<'_, S>) {
        // Allocate a fresh String for each event to avoid the thread-local
        // buffer reuse that gets corrupted under Frida Stalker DBI.
        let mut buf = String::with_capacity(128);
        self.format_event_to_buf(&mut buf, event);
        // Sanitize: strip bytes that are not printable ASCII, whitespace,
        // or ANSI escape sequences. Frida Stalker DBI can corrupt
        // thread-local state, injecting binary garbage into format buffers.
        let sanitized = sanitize_trace_output(buf.as_bytes());
        Self::write_stderr(&sanitized);
    }
}

/// Sanitize trace output by removing non-printable bytes.
///
/// Replaces any byte that is not printable ASCII (0x20..=0x7E), newline (0x0A),
/// tab (0x09), or carriage return (0x0D) with nothing (strips it), EXCEPT
/// for ESC (0x1B) which starts ANSI escape sequences. ANSI sequences
/// (`ESC [` through the terminating letter) are preserved for color output.
pub fn sanitize_trace_output(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    let mut i = 0;
    while i < data.len() {
        let b = data[i];
        if b == 0x1B {
            // ANSI escape sequence: copy ESC and everything through the
            // terminating byte (an ASCII letter, 0x40..=0x7E for CSI).
            out.push(b);
            i += 1;
            while i < data.len() {
                let c = data[i];
                out.push(c);
                i += 1;
                // CSI sequences (ESC [ ... <letter>) terminate at the
                // first byte in 0x40..=0x7E. For our purposes, any
                // ASCII letter ends the sequence.
                if (0x40..=0x7E).contains(&c) {
                    break;
                }
            }
        } else if b == b'\n' || b == b'\t' || b == b'\r' || (0x20..=0x7E).contains(&b) {
            out.push(b);
            i += 1;
        } else {
            // Non-printable byte (binary garbage): skip it.
            i += 1;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fmt_n() {
        assert_eq!(FmtN(0).to_string(), "0");
        assert_eq!(FmtN(999).to_string(), "999");
        assert_eq!(FmtN(1_000).to_string(), "1K");
        assert_eq!(FmtN(20_000).to_string(), "20K");
        assert_eq!(FmtN(1_500).to_string(), "1500");
        assert_eq!(FmtN(1_000_000).to_string(), "1M");
        assert_eq!(FmtN(20_000_000).to_string(), "20M");
        assert_eq!(FmtN(3_000_000_000).to_string(), "3B");
        assert_eq!(FmtN(1_000_000_000_000).to_string(), "1T");
        assert_eq!(FmtN(12345).to_string(), "12345");
        assert_eq!(FmtN(5_000_000).to_string(), "5M");
        assert_eq!(FmtN(100_000_000).to_string(), "100M");
    }

    #[test]
    fn test_fmt_grouped() {
        assert_eq!(fmt_grouped(0), "0");
        assert_eq!(fmt_grouped(999), "999");
        assert_eq!(fmt_grouped(1_000), "1_000");
        assert_eq!(fmt_grouped(10_000), "10_000");
        assert_eq!(fmt_grouped(20_000_000), "20_000_000");
        assert_eq!(fmt_grouped(999_999_000_000), "999_999_000_000");
        assert_eq!(fmt_grouped(1_234_567), "1_234_567");
    }

    #[test]
    fn test_fmt_ts() {
        use crate::types::CpuId;

        // No CPU context (e.g. timer events)
        assert_eq!(FmtTs::local(0, None, 1).to_string(), "              0:cpu");
        assert_eq!(
            FmtTs::local(10_000, None, 1).to_string(),
            "         10_000:cpu"
        );

        // With CPU ID, width=1
        assert_eq!(
            FmtTs::local(10_000, Some(CpuId(0)), 1).to_string(),
            "         10_000:cpu0"
        );
        assert_eq!(
            FmtTs::local(10_000, Some(CpuId(3)), 1).to_string(),
            "         10_000:cpu3"
        );

        // With CPU ID, width=2
        assert_eq!(
            FmtTs::local(20_000_000, Some(CpuId(1)), 2).to_string(),
            "     20_000_000:cpu01"
        );
        assert_eq!(
            FmtTs::local(20_000_000, Some(CpuId(12)), 2).to_string(),
            "     20_000_000:cpu12"
        );

        // With CPU ID, width=3
        assert_eq!(
            FmtTs::local(0, Some(CpuId(5)), 3).to_string(),
            "              0:cpu005"
        );

        // Global timestamp
        assert_eq!(
            FmtTs::global(999_999_000_000).to_string(),
            "999_999_000_000:G"
        );
    }

    #[test]
    fn test_fmt_duration_ns() {
        assert_eq!(fmt_duration_ns(0), "0ns");
        assert_eq!(fmt_duration_ns(500), "500ns");
        assert_eq!(fmt_duration_ns(1_000), "1µs");
        assert_eq!(fmt_duration_ns(1_500), "1.5µs");
        assert_eq!(fmt_duration_ns(20_000), "20µs");
        assert_eq!(fmt_duration_ns(1_000_000), "1ms");
        assert_eq!(fmt_duration_ns(20_000_000), "20ms");
        assert_eq!(fmt_duration_ns(50_500_000), "50.5ms");
        assert_eq!(fmt_duration_ns(1_000_000_000), "1s");
        assert_eq!(fmt_duration_ns(3_500_000_000), "3.5s");
        assert_eq!(fmt_duration_ns(100_000_000_000), "100s");
    }

    #[test]
    fn test_sanitize_trace_output() {
        // Plain ASCII passes through
        assert_eq!(sanitize_trace_output(b"hello world"), b"hello world");

        // Newlines and tabs pass through
        assert_eq!(sanitize_trace_output(b"a\nb\tc"), b"a\nb\tc");

        // Binary garbage is stripped
        assert_eq!(sanitize_trace_output(b"he\x00llo"), b"hello");
        assert_eq!(sanitize_trace_output(b"\x01\x02ok\x03"), b"ok");
        assert_eq!(sanitize_trace_output(b"ab\x80\xff\xfecde"), b"abcde");

        // ANSI escape sequences are preserved
        let ansi = b"\x1b[32mGREEN\x1b[0m";
        assert_eq!(sanitize_trace_output(ansi), ansi.to_vec());

        // Mixed: ANSI + garbage
        let mixed = b"\x1b[31mRED\x1b[0m\x00\x01tail";
        let expected = b"\x1b[31mRED\x1b[0mtail";
        assert_eq!(sanitize_trace_output(mixed), expected.to_vec());

        // Empty input
        assert_eq!(sanitize_trace_output(b""), b"");

        // DEL (0x7F) is stripped
        assert_eq!(sanitize_trace_output(b"ab\x7fcd"), b"abcd");
    }
}
