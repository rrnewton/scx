//! Preemption trace serialization and deserialization.
//!
//! A [`PreemptionTrace`] groups preemption records by worker for replay.
//! The text format is line-oriented and includes ASLR-resilient RIP offsets
//! so traces recorded on one run can be replayed even when the scheduler
//! .so is loaded at a different base address.

use std::io::{BufRead, Write};

use crate::interleave::WorkerId;
use crate::kfuncs::OpsContext;
use crate::perf::PmuEvent;
use crate::types::CpuId;

use super::{PreemptionRecord, PreemptionRecordStore};

/// A replayable preemption trace, grouped by worker.
///
/// During recording, preemption points are captured globally. For replay,
/// they're reorganized per-worker so each worker thread knows its own
/// sequence of target preemption points.
#[derive(Debug, Clone)]
pub struct PreemptionTrace {
    /// Per-worker preemption points, ordered by sequence number.
    /// Index = worker_id.0
    per_worker: Vec<Vec<PreemptionRecord>>,
    /// Which PMU event was used for preemption timing.
    /// All records in the trace share the same event type.
    break_on: PmuEvent,
}

impl PreemptionTrace {
    /// Build a trace from globally-collected records.
    ///
    /// Records are grouped by `worker_id` and sorted by sequence number
    /// within each group so that replay follows the original ordering.
    pub fn from_records(
        records: &[PreemptionRecord],
        num_workers: usize,
        break_on: PmuEvent,
    ) -> Self {
        let mut per_worker: Vec<Vec<PreemptionRecord>> =
            (0..num_workers).map(|_| Vec::new()).collect();
        for rec in records {
            let wid = rec.worker_id.0;
            if wid < num_workers {
                per_worker[wid].push(*rec);
            }
        }
        // Sort each worker's records by sequence number.
        for bucket in &mut per_worker {
            bucket.sort_by_key(|r| r.sequence);
        }
        PreemptionTrace {
            per_worker,
            break_on,
        }
    }

    /// Build a trace by draining records from a `PreemptionRecordStore`.
    #[allow(dead_code)] // Infrastructure for replay engine.
    pub(crate) fn from_store(
        store: &PreemptionRecordStore,
        num_workers: usize,
        break_on: PmuEvent,
    ) -> Self {
        let records = store.drain();
        Self::from_records(&records, num_workers, break_on)
    }

    /// Get the preemption points for a specific worker.
    pub fn worker_trace(&self, worker: WorkerId) -> &[PreemptionRecord] {
        self.per_worker
            .get(worker.0)
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }

    /// Total number of preemption points across all workers.
    pub fn len(&self) -> usize {
        self.per_worker.iter().map(|v| v.len()).sum()
    }

    /// Whether the trace contains zero preemption points.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Number of workers in this trace.
    pub fn num_workers(&self) -> usize {
        self.per_worker.len()
    }

    /// Which PMU event type was used for preemption timing.
    pub fn break_on(&self) -> PmuEvent {
        self.break_on
    }

    /// Serialize the trace to a line-oriented text format.
    ///
    /// Format:
    /// ```text
    /// # scxsim preemption trace
    /// # workers: 2
    /// # break_on: rbc
    /// # total: 47
    /// seq=0 structop=1:2 rbc=10 timeslice=142 rip=0x7f3a rip_offset=0xc7c cpu=0 worker=0
    /// ```
    ///
    /// `rbc` is the cumulative RBC within the structop. `timeslice` is the
    /// raw PMU count for this preemption (used by the replay engine).
    /// `so_base` is used to compute `rip_offset` (ASLR-resilient).
    pub fn serialize(&self, w: &mut impl Write, so_base: u64) -> std::io::Result<()> {
        let total = self.len();
        writeln!(w, "# scxsim preemption trace")?;
        writeln!(w, "# workers: {}", self.per_worker.len())?;
        writeln!(w, "# break_on: {}", self.break_on.short_name())?;
        writeln!(w, "# total: {total}")?;

        // Flatten and sort by sequence for canonical output order.
        let mut all: Vec<&PreemptionRecord> =
            self.per_worker.iter().flat_map(|v| v.iter()).collect();
        all.sort_by_key(|r| r.sequence);

        for rec in all {
            let rip_offset = if so_base > 0 && rec.instruction_pointer >= so_base {
                rec.instruction_pointer - so_base
            } else {
                rec.instruction_pointer
            };
            let kfn = if rec.kfunc_name.is_empty() {
                "-"
            } else {
                rec.kfunc_name
            };
            writeln!(
                w,
                "seq={} ops={} kfunc={} structop={}:{} rbc={} timeslice={} rip=0x{:x} rip_offset=0x{:x} cpu={} worker={}",
                rec.sequence,
                rec.ops_context.short_name(),
                kfn,
                rec.structop_local,
                rec.structop_global,
                rec.structop_rbc,
                rec.rbc_count,
                rec.instruction_pointer,
                rip_offset,
                rec.cpu_id.0,
                rec.worker_id.0,
            )?;
        }
        Ok(())
    }

    /// Deserialize a trace from the line-oriented text format.
    ///
    /// Parses the header to get the worker count and break_on event type,
    /// then each `seq=...` line into a `PreemptionRecord`. Uses
    /// `rip_offset` + `so_base` to reconstruct absolute RIPs if the
    /// current .so base differs from the recording.
    ///
    /// Old traces without a `# break_on:` header default to `rbc`.
    pub fn deserialize(r: &mut impl BufRead, so_base: u64) -> std::io::Result<Self> {
        let mut num_workers: usize = 0;
        let mut break_on = PmuEvent::RetiredBranchConditional; // default for old traces
        let mut records = Vec::new();

        for line in r.lines() {
            let line = line?;
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            if line.starts_with('#') {
                // Parse header comments.
                if let Some(rest) = line.strip_prefix("# workers: ") {
                    num_workers = rest
                        .trim()
                        .parse()
                        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
                } else if let Some(rest) = line.strip_prefix("# break_on: ") {
                    if let Some(event) = PmuEvent::from_short_name(rest.trim()) {
                        break_on = event;
                    }
                }
                continue;
            }
            // Parse data line.
            let rec = parse_preemption_line(line, so_base).map_err(|e| {
                std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("bad preemption line: {e}: {line}"),
                )
            })?;
            records.push(rec);
        }

        if num_workers == 0 && !records.is_empty() {
            // Infer worker count from max worker_id.
            num_workers = records.iter().map(|r| r.worker_id.0).max().unwrap_or(0) + 1;
        }

        Ok(Self::from_records(&records, num_workers, break_on))
    }
}

/// Parse a single preemption trace line into a `PreemptionRecord`.
///
/// Uses `rip_offset` to reconstruct the absolute RIP relative to the
/// current `so_base`. Falls back to the stored `rip` if offset is missing.
/// Supports both old format (`rbc=` as timeslice) and new format
/// (`timeslice=` for raw count, `rbc=` for cumulative structop RBC).
fn parse_preemption_line(line: &str, so_base: u64) -> Result<PreemptionRecord, String> {
    let mut seq: u64 = 0;
    let mut rbc: Option<u64> = None;
    let mut timeslice: Option<u64> = None;
    let mut rip: u64 = 0;
    let mut rip_offset: Option<u64> = None;
    let mut cpu: u32 = 0;
    let mut worker: usize = 0;
    let mut structop_local: u64 = 0;
    let mut structop_global: u64 = 0;
    let mut structop_rbc: u64 = 0;
    let mut ops_context = OpsContext::None;
    let mut kfunc_name: &'static str = "";

    /// Leak a short parsed string to get a `&'static str`.
    ///
    /// Only used for kfunc names from deserialized trace files; these are
    /// a small, bounded set so the leak is negligible.
    fn leak_str(s: &str) -> &'static str {
        if s == "-" || s.is_empty() {
            return "";
        }
        // Check against known kfunc names to avoid leaking duplicates.
        match s {
            "select_cpu_dfl" => "select_cpu_dfl",
            "select_cpu_and" => "select_cpu_and",
            "dsq_insert" => "dsq_insert",
            "dsq_insert_vtime" => "dsq_insert_vtime",
            "dsq_move_to_local" => "dsq_move_to_local",
            "dsq_nr_queued" => "dsq_nr_queued",
            "now" => "now",
            "get_smp_processor_id" => "get_smp_processor_id",
            "task_cpu" => "task_cpu",
            "ktime_get_ns" => "ktime_get_ns",
            "get_current_task_btf" => "get_current_task_btf",
            "dsq_iter_begin" => "dsq_iter_begin",
            "dsq_iter_next" => "dsq_iter_next",
            "dsq_move" => "dsq_move",
            "kick_cpu" => "kick_cpu",
            "task_running" => "task_running",
            _ => Box::leak(s.to_owned().into_boxed_str()),
        }
    }

    for part in line.split_whitespace() {
        if let Some(val) = part.strip_prefix("seq=") {
            seq = val.parse().map_err(|e| format!("seq: {e}"))?;
        } else if let Some(val) = part.strip_prefix("timeslice=") {
            timeslice = Some(val.parse().map_err(|e| format!("timeslice: {e}"))?);
        } else if let Some(val) = part.strip_prefix("rbc=") {
            rbc = Some(val.parse().map_err(|e| format!("rbc: {e}"))?);
        } else if let Some(val) = part.strip_prefix("rip=") {
            rip = parse_hex_or_dec(val).map_err(|e| format!("rip: {e}"))?;
        } else if let Some(val) = part.strip_prefix("rip_offset=") {
            rip_offset = Some(parse_hex_or_dec(val).map_err(|e| format!("rip_offset: {e}"))?);
        } else if let Some(val) = part.strip_prefix("cpu=") {
            cpu = val.parse().map_err(|e| format!("cpu: {e}"))?;
        } else if let Some(val) = part.strip_prefix("worker=") {
            worker = val.parse().map_err(|e| format!("worker: {e}"))?;
        } else if let Some(val) = part.strip_prefix("structop=") {
            // Parse "L:G" format.
            if let Some((l, g)) = val.split_once(':') {
                structop_local = l.parse().map_err(|e| format!("structop local: {e}"))?;
                structop_global = g.parse().map_err(|e| format!("structop global: {e}"))?;
            }
        } else if let Some(val) = part.strip_prefix("ops=") {
            ops_context = OpsContext::from_short_name(val);
        } else if let Some(val) = part.strip_prefix("kfunc=") {
            kfunc_name = leak_str(val);
        }
    }

    // Reconstruct absolute RIP: prefer rip_offset + so_base for ASLR resilience.
    let instruction_pointer = if let Some(offset) = rip_offset {
        if so_base > 0 {
            so_base + offset
        } else {
            rip // Fallback to stored absolute RIP
        }
    } else {
        rip
    };

    // New format: timeslice= is the raw PMU count, rbc= is cumulative structop RBC.
    // Old format: rbc= is the raw PMU count (no structop fields).
    let rbc_count = timeslice.or(rbc).unwrap_or(0);
    if timeslice.is_some() {
        structop_rbc = rbc.unwrap_or(0);
    }

    Ok(PreemptionRecord {
        sequence: seq,
        rbc_count,
        instruction_pointer,
        cpu_id: CpuId(cpu),
        worker_id: WorkerId(worker),
        structop_local,
        structop_global,
        structop_rbc,
        ops_context,
        kfunc_name,
    })
}

/// Parse a string as hex (0x prefix) or decimal.
fn parse_hex_or_dec(s: &str) -> Result<u64, String> {
    if let Some(hex) = s.strip_prefix("0x").or_else(|| s.strip_prefix("0X")) {
        u64::from_str_radix(hex, 16).map_err(|e| e.to_string())
    } else {
        s.parse::<u64>().map_err(|e| e.to_string())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_record(
        seq: u64,
        rbc: u64,
        rip: u64,
        cpu: u32,
        worker: usize,
        slocal: u64,
        sglobal: u64,
        srbc: u64,
    ) -> PreemptionRecord {
        PreemptionRecord {
            rbc_count: rbc,
            instruction_pointer: rip,
            cpu_id: CpuId(cpu),
            worker_id: WorkerId(worker),
            sequence: seq,
            structop_local: slocal,
            structop_global: sglobal,
            structop_rbc: srbc,
            ops_context: OpsContext::None,
            kfunc_name: "",
        }
    }

    #[test]
    fn test_preemption_trace_grouping() {
        let records = vec![
            make_record(0, 100, 0x1000, 0, 0, 1, 1, 100),
            make_record(1, 200, 0x2000, 1, 1, 1, 2, 200),
            make_record(2, 300, 0x3000, 0, 0, 1, 1, 400),
            make_record(3, 400, 0x4000, 1, 1, 1, 2, 600),
        ];

        let trace = PreemptionTrace::from_records(&records, 2, PmuEvent::RetiredBranchConditional);
        assert_eq!(trace.num_workers(), 2);
        assert_eq!(trace.len(), 4);

        let w0 = trace.worker_trace(WorkerId(0));
        assert_eq!(w0.len(), 2);
        assert_eq!(w0[0].sequence, 0);
        assert_eq!(w0[1].sequence, 2);

        let w1 = trace.worker_trace(WorkerId(1));
        assert_eq!(w1.len(), 2);
        assert_eq!(w1[0].sequence, 1);
        assert_eq!(w1[1].sequence, 3);

        // Out-of-range worker returns empty slice.
        assert!(trace.worker_trace(WorkerId(99)).is_empty());
    }

    #[test]
    fn test_preemption_trace_empty() {
        let trace = PreemptionTrace::from_records(&[], 3, PmuEvent::RetiredBranchConditional);
        assert_eq!(trace.num_workers(), 3);
        assert_eq!(trace.len(), 0);
        assert!(trace.is_empty());
        assert!(trace.worker_trace(WorkerId(0)).is_empty());
    }

    #[test]
    fn test_serialize_roundtrip() {
        let records = vec![
            make_record(0, 100, 0x7f000010c0, 0, 0, 1, 1, 100),
            make_record(1, 200, 0x7f000020d0, 1, 1, 1, 2, 200),
            make_record(2, 300, 0x7f000030e0, 0, 0, 1, 1, 400),
        ];

        let trace = PreemptionTrace::from_records(&records, 2, PmuEvent::RetiredBranchConditional);
        let so_base: u64 = 0x7f00000000;

        // Serialize to buffer.
        let mut buf = Vec::new();
        trace.serialize(&mut buf, so_base).unwrap();
        let text = String::from_utf8(buf.clone()).unwrap();

        // Verify header.
        assert!(text.contains("# scxsim preemption trace"));
        assert!(text.contains("# workers: 2"));
        assert!(text.contains("# break_on: rbc"));
        assert!(text.contains("# total: 3"));

        // Verify structop and rip_offset are present.
        assert!(text.contains("structop="));
        assert!(text.contains("rip_offset="));
        assert!(text.contains("timeslice="));

        // Deserialize with same so_base — should get same absolute RIPs.
        let mut cursor = std::io::Cursor::new(buf);
        let trace2 = PreemptionTrace::deserialize(&mut cursor, so_base).unwrap();
        assert_eq!(trace2.num_workers(), 2);
        assert_eq!(trace2.len(), 3);

        let w0 = trace2.worker_trace(WorkerId(0));
        assert_eq!(w0.len(), 2);
        assert_eq!(w0[0].rbc_count, 100);
        assert_eq!(w0[0].instruction_pointer, 0x7f000010c0);
        assert_eq!(w0[0].structop_local, 1);
        assert_eq!(w0[0].structop_rbc, 100);
        assert_eq!(w0[1].rbc_count, 300);

        let w1 = trace2.worker_trace(WorkerId(1));
        assert_eq!(w1.len(), 1);
        assert_eq!(w1[0].rbc_count, 200);
        assert_eq!(w1[0].instruction_pointer, 0x7f000020d0);
        assert_eq!(w1[0].structop_rbc, 200);
    }

    #[test]
    fn test_aslr_resilience() {
        // Record at one base address, replay at a different one.
        let records = vec![make_record(0, 42, 0x1000_1000, 0, 0, 1, 1, 42)];

        let trace = PreemptionTrace::from_records(&records, 1, PmuEvent::RetiredBranchConditional);
        let record_base: u64 = 0x1000_0000;

        // Serialize with record-time base.
        let mut buf = Vec::new();
        trace.serialize(&mut buf, record_base).unwrap();

        // Deserialize with a different replay-time base.
        let replay_base: u64 = 0x2000_0000;
        let mut cursor = std::io::Cursor::new(buf);
        let trace2 = PreemptionTrace::deserialize(&mut cursor, replay_base).unwrap();

        let w0 = trace2.worker_trace(WorkerId(0));
        assert_eq!(w0.len(), 1);
        // rip_offset = 0x1000_1000 - 0x1000_0000 = 0x1000
        // new rip = 0x2000_0000 + 0x1000 = 0x2000_1000
        assert_eq!(w0[0].instruction_pointer, 0x2000_1000);
    }

    #[test]
    fn test_parse_hex_or_dec() {
        assert_eq!(parse_hex_or_dec("42").unwrap(), 42);
        assert_eq!(parse_hex_or_dec("0xff").unwrap(), 255);
        assert_eq!(parse_hex_or_dec("0XFF").unwrap(), 255);
        assert_eq!(parse_hex_or_dec("0x0").unwrap(), 0);
        assert!(parse_hex_or_dec("not_a_number").is_err());
    }

    #[test]
    fn test_parse_preemption_line_new_format() {
        let line =
            "seq=5 structop=2:3 rbc=100 timeslice=42 rip=0x1000 rip_offset=0x100 cpu=1 worker=0";
        let rec = parse_preemption_line(line, 0x2000).unwrap();
        assert_eq!(rec.sequence, 5);
        assert_eq!(rec.rbc_count, 42); // timeslice takes precedence
        assert_eq!(rec.structop_rbc, 100); // rbc= becomes structop_rbc in new format
        assert_eq!(rec.instruction_pointer, 0x2100); // so_base + rip_offset
        assert_eq!(rec.cpu_id, CpuId(1));
        assert_eq!(rec.worker_id, WorkerId(0));
        assert_eq!(rec.structop_local, 2);
        assert_eq!(rec.structop_global, 3);
    }

    #[test]
    fn test_parse_preemption_line_old_format() {
        // Old format: no timeslice= field, rbc= is the raw count.
        let line = "seq=0 rbc=100 rip=0x1000 cpu=0 worker=0";
        let rec = parse_preemption_line(line, 0).unwrap();
        assert_eq!(rec.rbc_count, 100);
        assert_eq!(rec.instruction_pointer, 0x1000);
    }

    #[test]
    fn test_break_on_insn_roundtrip() {
        let records = vec![
            make_record(0, 500, 0x7f000010c0, 0, 0, 1, 1, 500),
            make_record(1, 1000, 0x7f000020d0, 1, 1, 1, 2, 1000),
        ];

        let trace = PreemptionTrace::from_records(&records, 2, PmuEvent::InstructionsRetired);
        assert_eq!(trace.break_on(), PmuEvent::InstructionsRetired);

        let so_base: u64 = 0x7f00000000;
        let mut buf = Vec::new();
        trace.serialize(&mut buf, so_base).unwrap();
        let text = String::from_utf8(buf.clone()).unwrap();

        // Verify insn header is present.
        assert!(text.contains("# break_on: insn"));

        // Deserialize and verify break_on is preserved.
        let mut cursor = std::io::Cursor::new(buf);
        let trace2 = PreemptionTrace::deserialize(&mut cursor, so_base).unwrap();
        assert_eq!(trace2.break_on(), PmuEvent::InstructionsRetired);
        assert_eq!(trace2.num_workers(), 2);
        assert_eq!(trace2.len(), 2);
    }

    #[test]
    fn test_deserialize_old_trace_defaults_to_rbc() {
        // Old trace without # break_on: header should default to rbc.
        let text = "# scxsim preemption trace\n\
                     # workers: 1\n\
                     # total: 1\n\
                     seq=0 structop=1:1 rbc=50 timeslice=100 rip=0x1000 rip_offset=0x100 cpu=0 worker=0\n";
        let mut cursor = std::io::Cursor::new(text.as_bytes());
        let trace = PreemptionTrace::deserialize(&mut cursor, 0).unwrap();
        assert_eq!(trace.break_on(), PmuEvent::RetiredBranchConditional);
    }

    #[test]
    fn test_pmu_event_short_name_roundtrip() {
        assert_eq!(
            PmuEvent::from_short_name("rbc"),
            Some(PmuEvent::RetiredBranchConditional)
        );
        assert_eq!(
            PmuEvent::from_short_name("insn"),
            Some(PmuEvent::InstructionsRetired)
        );
        assert_eq!(PmuEvent::from_short_name("unknown"), None);
        assert_eq!(PmuEvent::RetiredBranchConditional.short_name(), "rbc");
        assert_eq!(PmuEvent::InstructionsRetired.short_name(), "insn");
    }
}
