//! Lock-free Single-Producer Single-Consumer (SPSC) Shared-Memory Ring Buffer
//! for sub-microsecond communication between the in-process `NCCL_PROFILER_PLUGIN`
//! (`libfabriclock_nccl_profiler.so`) and the out-of-process `fabriclock` Rust daemon.
//!
//! Memory layout is strictly `#[repr(C)]` and cache-line aligned (64 bytes) to prevent
//! false sharing between the PyTorch/NCCL proxy thread and the FabricLock telemetry thread.

use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicU64, AtomicU8, Ordering};

pub const SHM_MAGIC: u64 = 0x4641_4252_4C4F_434B; // "FABRLOCK"
pub const RING_CAPACITY: usize = 4096; // Power of 2 for bitwise modulo masking
pub const RING_MASK: usize = RING_CAPACITY - 1;
pub const DEFAULT_SHM_PATH: &str = "/dev/shm/fabriclock_nccl_ipc.shm";

/// Collective operation type matching NCCL's `ncclFunc_t`.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NcclCollectiveKind {
    Broadcast = 0,
    Reduce = 1,
    AllGather = 2,
    ReduceScatter = 3,
    AllReduce = 4,
    SendRecv = 5,
    Unknown = 255,
}

impl From<u8> for NcclCollectiveKind {
    fn from(v: u8) -> Self {
        match v {
            0 => Self::Broadcast,
            1 => Self::Reduce,
            2 => Self::AllGather,
            3 => Self::ReduceScatter,
            4 => Self::AllReduce,
            5 => Self::SendRecv,
            _ => Self::Unknown,
        }
    }
}

/// Fixed-size 64-byte cache-line-aligned event emitted by `ncclProfiler_v2`
/// upon completion of an RDMA Queue Pair proxy operation (`ncclProfileProxyOp`)
/// or collective kernel step (`ncclProfileColl`).
#[repr(C, align(64))]
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct NcclTelemetryEvent {
    /// Monotonic timestamp (nanoseconds via `clock_gettime(CLOCK_MONOTONIC)`) when op started
    pub start_ts_ns: u64,
    /// Duration of the collective or proxy RDMA step in nanoseconds
    pub duration_ns: u64,
    /// Unique NCCL communicator hash (`commHash`)
    pub comm_hash: u64,
    /// Sequence number / collective step counter
    pub step_seq: u64,
    /// Transmitted or received payload size in bytes
    pub byte_count: u64,
    /// Hardware RDMA Queue Pair number (`qp_num`) used for this transfer
    pub qp_num: u32,
    /// Local GPU rank within the communicator
    pub local_rank: u32,
    /// Remote peer GPU rank in the ring/tree channel
    pub peer_rank: u32,
    /// Encapsulating RoCEv2 UDP source port (`sport`) if known, or 0
    pub udp_sport: u16,
    /// Local NIC rail index (`0..7` corresponding to `mlx5_0..mlx5_7`)
    pub rail_id: u8,
    /// NCCL ring/tree channel ID (`0..31`)
    pub channel_id: u8,
    /// Collective operation kind (`AllReduce`, `ReduceScatter`, etc.)
    pub coll_kind: u8,
    /// 1 if this event is a low-level RDMA ProxyOp (`ncclProfileProxyOp`), 0 if high-level Coll
    pub is_proxy_op: u8,
    /// Reserved padding to maintain exact 64-byte alignment
    pub _pad: [u8; 2],
}

impl Default for NcclTelemetryEvent {
    fn default() -> Self {
        Self {
            start_ts_ns: 0,
            duration_ns: 0,
            comm_hash: 0,
            step_seq: 0,
            byte_count: 0,
            qp_num: 0,
            local_rank: 0,
            peer_rank: 0,
            udp_sport: 0,
            rail_id: 0,
            channel_id: 0,
            coll_kind: NcclCollectiveKind::AllReduce as u8,
            is_proxy_op: 0,
            _pad: [0; 2],
        }
    }
}

/// Active remediation directive written by the `fabriclock` control plane into shared memory
/// and read by the `ncclNet` / `ncclProfiler` plugin at step boundaries.
#[repr(C, align(64))]
#[derive(Debug)]
pub struct RailBypassDirective {
    /// Bitmask of quarantined local NIC rails (bit `i` = 1 means `mlx5_i` is quarantined)
    pub quarantined_rail_mask: AtomicU8,
    /// For each rail `0..7`, the target sibling rail to proxy traffic through via NVSwitch PXN
    pub pxn_sibling_map: [AtomicU8; 8],
    /// Monotonic generation counter incremented whenever a bypass policy changes
    pub policy_generation: AtomicU64,
    /// Target UDP source port offset for ECMP Spine failover (`ibv_modify_qp`)
    pub ecmp_sport_delta: [AtomicU8; 8],
}

/// Complete Shared Memory Region mapped at `/dev/shm/fabriclock_nccl_ipc.shm`.
#[repr(C, align(64))]
pub struct ShmIpcRegion {
    pub magic: AtomicU64,
    /// Written by NCCL plugin thread (Producer)
    pub write_idx: AtomicU64,
    /// Written by FabricLock daemon thread (Consumer)
    pub read_idx: AtomicU64,
    /// Control-plane -> NCCL plugin bypass directives
    pub directive: RailBypassDirective,
    /// Fixed-size ring buffer of 4,096 telemetry events (256 KB total)
    pub events: [NcclTelemetryEvent; RING_CAPACITY],
}

impl ShmIpcRegion {
    /// Initializes a zeroed memory region with the `SHM_MAGIC` header.
    pub fn init_in_place(&mut self) {
        self.write_idx.store(0, Ordering::Release);
        self.read_idx.store(0, Ordering::Release);
        self.directive.quarantined_rail_mask.store(0, Ordering::Release);
        self.directive.policy_generation.store(1, Ordering::Release);
        for i in 0..8 {
            self.directive.pxn_sibling_map[i].store(i as u8, Ordering::Release);
            self.directive.ecmp_sport_delta[i].store(0, Ordering::Release);
        }
        self.magic.store(SHM_MAGIC, Ordering::Release);
    }

    /// Lock-free producer push (called from C/Rust NCCL Profiler callback).
    /// Never blocks the GPU/NCCL proxy thread; overwrites oldest unread slot if full.
    #[inline]
    pub fn push_event(&self, ev: NcclTelemetryEvent) {
        let w = self.write_idx.load(Ordering::Relaxed);
        let slot = (w as usize) & RING_MASK;
        // Safety: SPSC ring buffer slot write before advancing `write_idx` with Release ordering.
        unsafe {
            let ptr = self.events.as_ptr().add(slot) as *mut NcclTelemetryEvent;
            std::ptr::write_volatile(ptr, ev);
        }
        self.write_idx.store(w.wrapping_add(1), Ordering::Release);
    }

    /// Lock-free consumer pop (called by the FabricLock telemetry correlator daemon).
    #[inline]
    pub fn pop_event(&self) -> Option<NcclTelemetryEvent> {
        let r = self.read_idx.load(Ordering::Relaxed);
        let w = self.write_idx.load(Ordering::Acquire);
        if r >= w {
            return None;
        }
        // If producer lapped consumer, jump forward to keep up with real-time tail
        let effective_r = if w.saturating_sub(r) > (RING_CAPACITY as u64) {
            w - (RING_CAPACITY as u64)
        } else {
            r
        };
        let slot = (effective_r as usize) & RING_MASK;
        let ev = unsafe {
            let ptr = self.events.as_ptr().add(slot);
            std::ptr::read_volatile(ptr)
        };
        self.read_idx.store(effective_r.wrapping_add(1), Ordering::Release);
        Some(ev)
    }

    /// Atomically quarantines a degraded host NIC rail and assigns its NVSwitch PXN sibling rail.
    pub fn set_rail_pxn_failover(&self, bad_rail: u8, sibling_rail: u8) {
        if (bad_rail as usize) < 8 && (sibling_rail as usize) < 8 {
            self.directive.pxn_sibling_map[bad_rail as usize].store(sibling_rail, Ordering::Release);
            self.directive
                .quarantined_rail_mask
                .fetch_or(1 << bad_rail, Ordering::AcqRel);
            self.directive.policy_generation.fetch_add(1, Ordering::AcqRel);
        }
    }

    /// Atomically sets an ECMP UDP source-port rotation offset for a rail to bypass a degraded Spine.
    pub fn set_rail_ecmp_sport_delta(&self, rail: u8, sport_delta: u8) {
        if (rail as usize) < 8 {
            self.directive.ecmp_sport_delta[rail as usize].store(sport_delta, Ordering::Release);
            self.directive.policy_generation.fetch_add(1, Ordering::AcqRel);
        }
    }
}

/// Opens or creates the POSIX shared-memory segment (`/dev/shm/fabriclock_nccl_ipc.shm` on Linux)
/// and returns a mutable pointer to `ShmIpcRegion`.
#[cfg(target_os = "linux")]
pub unsafe fn map_or_create_linux_shm(create: bool) -> Option<*mut ShmIpcRegion> {
    use std::ffi::CString;
    let c_path = CString::new(DEFAULT_SHM_PATH).ok()?;
    let flags = if create {
        libc::O_CREAT | libc::O_RDWR
    } else {
        libc::O_RDWR
    };
    let fd = libc::open(c_path.as_ptr(), flags, 0o666);
    if fd < 0 {
        return None;
    }
    let size = std::mem::size_of::<ShmIpcRegion>();
    if create {
        if libc::ftruncate(fd, size as libc::off_t) != 0 {
            libc::close(fd);
            return None;
        }
    }
    let ptr = libc::mmap(
        std::ptr::null_mut(),
        size,
        libc::PROT_READ | libc::PROT_WRITE,
        libc::MAP_SHARED,
        fd,
        0,
    );
    libc::close(fd);
    if ptr == libc::MAP_FAILED || ptr.is_null() {
        return None;
    }
    let region = ptr as *mut ShmIpcRegion;
    if create && (*region).magic.load(Ordering::Acquire) != SHM_MAGIC {
        (*region).init_in_place();
    }
    Some(region)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_event_size_and_lockfree_spsc() {
        assert_eq!(std::mem::size_of::<NcclTelemetryEvent>(), 64);

        let mut boxed_region: Box<ShmIpcRegion> = unsafe {
            let layout = std::alloc::Layout::new::<ShmIpcRegion>();
            let ptr = std::alloc::alloc_zeroed(layout) as *mut ShmIpcRegion;
            Box::from_raw(ptr)
        };
        boxed_region.init_in_place();

        let ev = NcclTelemetryEvent {
            start_ts_ns: 100_000,
            duration_ns: 450_000,
            comm_hash: 0xDEAD_BEEF,
            step_seq: 42,
            byte_count: 67_108_864,
            qp_num: 0x104A,
            local_rank: 0,
            peer_rank: 1,
            udp_sport: 49164,
            rail_id: 0,
            channel_id: 3,
            coll_kind: NcclCollectiveKind::AllReduce as u8,
            is_proxy_op: 1,
            _pad: [0; 2],
        };

        boxed_region.push_event(ev);
        let popped = boxed_region.pop_event().expect("should pop event");
        assert_eq!(popped.step_seq, 42);
        assert_eq!(popped.qp_num, 0x104A);
        assert_eq!(popped.udp_sport, 49164);
        assert!(boxed_region.pop_event().is_none());

        boxed_region.set_rail_pxn_failover(0, 1);
        assert_eq!(boxed_region.directive.quarantined_rail_mask.load(Ordering::Acquire), 1);
        assert_eq!(boxed_region.directive.pxn_sibling_map[0].load(Ordering::Acquire), 1);
    }
}
