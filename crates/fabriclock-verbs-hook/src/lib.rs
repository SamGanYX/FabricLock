//! In-Process RDMA `libibverbs` Queue Pair Tracker & Live `IBV_QP_AV` Flow-Label Rotator.
//!
//! Why this crate is critical for <2ms Spine Optical Gray-Failure Bypass:
//!   - In Linux RDMA (`mlx5_ib`, `bnxt_re`), Queue Pair handles (`struct ibv_qp*`) are owned
//!     by the user-space training process (PyTorch / NCCL).
//!   - An external daemon cannot directly call `ibv_modify_qp()` on another process's `ibv_qp*` pointer.
//!   - By loading `libfabriclock_verbs_hook.so` into the training container (or linking it with
//!     `libfabriclock_nccl_profiler.so`), FabricLock intercepts `ibv_create_qp` and `ibv_modify_qp`,
//!     caches the active Address Vector (`struct ibv_ah_attr`) for every RC Queue Pair, and
//!     automatically rotates `ah_attr.grh.flow_label` via `ibv_modify_qp(qp, &attr, IBV_QP_AV)`
//!     the instant `ShmIpcRegion.directive.policy_generation` increments!

use fabriclock_ipc::ShmIpcRegion;
use fabriclock_remediate::{compute_rocev2_flow_label_for_sport, IbvAhAttr, IBV_QP_AV};
use libc::{c_int, c_void};
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Mutex;

pub const MAX_TRACKED_QPS: usize = 512;
pub const IBV_QPT_RC: u32 = 2; // Reliable Connected Queue Pair used by NCCL RoCEv2/IB

/// Minimal C-ABI representation of `struct ibv_qp` (`<infiniband/verbs.h>`).
#[repr(C)]
#[derive(Debug)]
pub struct IbvQp {
    pub context: *mut c_void,
    pub qp_context: *mut c_void,
    pub pd: *mut c_void,
    pub send_cq: *mut c_void,
    pub recv_cq: *mut c_void,
    pub srq: *mut c_void,
    pub handle: u32,
    pub qp_num: u32,
    pub state: u32,
    pub qp_type: u32,
    pub mutex: [u8; 40],
    pub cond: [u8; 48],
    pub events_completed: u32,
}

/// C-ABI representation of `struct ibv_qp_attr` prefix needed for `IBV_QP_AV` modification.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct IbvQpAttr {
    pub qp_state: u32,
    pub cur_qp_state: u32,
    pub path_mtu: u32,
    pub path_mig_state: u32,
    pub qkey: u32,
    pub rq_psn: u32,
    pub sq_psn: u32,
    pub dest_qp_num: u32,
    pub qp_access_flags: u32,
    pub cap: [u32; 5],
    pub ah_attr: IbvAhAttr,
}

/// Function pointer signature for the underlying `libibverbs` `ibv_modify_qp` implementation.
pub type IbvModifyQpFn =
    unsafe extern "C" fn(qp: *mut IbvQp, attr: *mut IbvQpAttr, attr_mask: c_int) -> c_int;

/// Metadata for a single tracked Reliable Connected (RC) Queue Pair owned by NCCL.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct TrackedQpEntry {
    pub qp_ptr_addr: usize,
    pub qp_num: u32,
    pub dest_qp_num: u32,
    pub rail_id: u8,
    pub port_num: u8,
    pub base_udp_sport: u16,
    pub current_udp_sport: u16,
    pub current_flow_label: u32,
    pub rotation_count: u32,
    #[serde(skip)]
    pub cached_ah_attr: IbvAhAttr,
}

/// Registry of active NCCL RDMA Queue Pairs inside the current training process.
pub struct LiveQpRegistry {
    pub entries: Vec<TrackedQpEntry>,
    pub applied_policy_generation: u64,
    pub total_rotations_executed: u64,
}

impl LiveQpRegistry {
    pub const fn new() -> Self {
        Self {
            entries: Vec::new(),
            applied_policy_generation: 0,
            total_rotations_executed: 0,
        }
    }

    /// Registers or updates a Queue Pair when NCCL transitions it through RTS (`IBV_QP_AV`).
    pub fn register_or_update_qp(
        &mut self,
        qp_ptr: *mut IbvQp,
        qp_num: u32,
        dest_qp_num: u32,
        rail_id: u8,
        ah_attr: IbvAhAttr,
    ) {
        let addr = qp_ptr as usize;
        let sport = 49152 + ((ah_attr.grh.flow_label as u16) & 0x3FFF);
        if let Some(existing) = self.entries.iter_mut().find(|e| e.qp_ptr_addr == addr && e.qp_num == qp_num) {
            existing.dest_qp_num = dest_qp_num;
            existing.cached_ah_attr = ah_attr;
            existing.current_flow_label = ah_attr.grh.flow_label;
            existing.current_udp_sport = sport;
            return;
        }
        if self.entries.len() < MAX_TRACKED_QPS {
            self.entries.push(TrackedQpEntry {
                qp_ptr_addr: addr,
                qp_num,
                dest_qp_num,
                rail_id,
                port_num: ah_attr.port_num.max(1),
                base_udp_sport: sport,
                current_udp_sport: sport,
                current_flow_label: ah_attr.grh.flow_label,
                rotation_count: 0,
                cached_ah_attr: ah_attr,
            });
        }
    }

    /// Checks `ShmIpcRegion.directive` at collective step boundaries; if `policy_generation`
    /// has advanced, calls `modify_fn(qp, &mut attr, IBV_QP_AV)` on all QPs belonging to the
    /// affected rail to rotate their encapsulating UDP source port in <2ms.
    pub fn reconcile_with_shm_directives(
        &mut self,
        shm: &ShmIpcRegion,
        modify_fn: IbvModifyQpFn,
    ) -> usize {
        let gen = shm.directive.policy_generation.load(Ordering::Acquire);
        if gen <= self.applied_policy_generation {
            return 0;
        }

        let mut rotated_qps = 0usize;
        for entry in &mut self.entries {
            let r = (entry.rail_id as usize) & 7;
            let delta = shm.directive.ecmp_sport_delta[r].load(Ordering::Acquire);
            if delta == 0 {
                continue;
            }

            let target_sport = entry.base_udp_sport.wrapping_add(delta as u16);
            if target_sport == entry.current_udp_sport {
                continue;
            }

            let new_fl = compute_rocev2_flow_label_for_sport(target_sport);
            entry.cached_ah_attr.grh.flow_label = new_fl;
            entry.cached_ah_attr.is_global = 1;

            let mut qp_attr = IbvQpAttr {
                ah_attr: entry.cached_ah_attr,
                ..Default::default()
            };

            let rc = unsafe {
                modify_fn(
                    entry.qp_ptr_addr as *mut IbvQp,
                    &mut qp_attr as *mut IbvQpAttr,
                    IBV_QP_AV,
                )
            };

            if rc == 0 {
                entry.current_udp_sport = target_sport;
                entry.current_flow_label = new_fl;
                entry.rotation_count += 1;
                rotated_qps += 1;
                self.total_rotations_executed += 1;
            }
        }

        self.applied_policy_generation = gen;
        rotated_qps
    }
}

static GLOBAL_QP_REGISTRY: Mutex<LiveQpRegistry> = Mutex::new(LiveQpRegistry::new());
static ROTATION_COUNTER: AtomicU64 = AtomicU64::new(0);
static TRACKED_QP_COUNT: AtomicUsize = AtomicUsize::new(0);

/// Default safe hardware callback stub when running outside a physical `libibverbs.so` process.
pub unsafe extern "C" fn mock_ibv_modify_qp_success(
    _qp: *mut IbvQp,
    _attr: *mut IbvQpAttr,
    _attr_mask: c_int,
) -> c_int {
    0
}

/// C-ABI Hook Function: Intercepts `ibv_modify_qp` during NCCL RTR -> RTS setup to record
/// the Queue Pair pointer, `qp_num`, and initial `IbvAhAttr`, and applies any active ECMP sport offset.
#[no_mangle]
pub unsafe extern "C" fn fabriclock_hook_ibv_modify_qp(
    qp: *mut IbvQp,
    attr: *mut IbvQpAttr,
    attr_mask: c_int,
    rail_id: u8,
    real_modify_qp: Option<IbvModifyQpFn>,
) -> c_int {
    if !qp.is_null() && !attr.is_null() && (attr_mask & IBV_QP_AV) != 0 {
        if let Ok(mut reg) = GLOBAL_QP_REGISTRY.lock() {
            reg.register_or_update_qp(
                qp,
                (*qp).qp_num,
                (*attr).dest_qp_num,
                rail_id,
                (*attr).ah_attr,
            );
            TRACKED_QP_COUNT.store(reg.entries.len(), Ordering::Release);
        }
    }

    let target_fn = real_modify_qp.unwrap_or(mock_ibv_modify_qp_success);
    target_fn(qp, attr, attr_mask)
}

/// Exported Linux `LD_PRELOAD` symbol overriding `ibv_modify_qp` in `libibverbs.so.1`.
/// Resolves the real driver function via `dlsym(RTLD_NEXT, "ibv_modify_qp")` and registers
/// every RC Queue Pair with `GLOBAL_QP_REGISTRY`.
#[cfg(target_os = "linux")]
#[no_mangle]
pub unsafe extern "C" fn ibv_modify_qp(
    qp: *mut IbvQp,
    attr: *mut IbvQpAttr,
    attr_mask: c_int,
) -> c_int {
    let sym_name = b"ibv_modify_qp\0";
    let next_ptr = libc::dlsym(libc::RTLD_NEXT, sym_name.as_ptr() as *const libc::c_char);
    let real_fn: Option<IbvModifyQpFn> = if next_ptr.is_null() {
        None
    } else {
        Some(std::mem::transmute::<*mut c_void, IbvModifyQpFn>(next_ptr))
    };
    let rail_guess = if !attr.is_null() {
        (*attr).ah_attr.port_num.saturating_sub(1)
    } else {
        0
    };
    fabriclock_hook_ibv_modify_qp(qp, attr, attr_mask, rail_guess, real_fn)
}

/// Triggered by the FabricLock NCCL Profiler Plugin at the start of each collective step
/// to apply any pending Spine ECMP UDP source-port rotations across all registered QPs.
pub fn poll_and_apply_in_process_qp_rotations(
    shm: &ShmIpcRegion,
    real_modify_qp: Option<IbvModifyQpFn>,
) -> usize {
    let modify_fn = real_modify_qp.unwrap_or(mock_ibv_modify_qp_success);
    if let Ok(mut reg) = GLOBAL_QP_REGISTRY.lock() {
        let count = reg.reconcile_with_shm_directives(shm, modify_fn);
        if count > 0 {
            ROTATION_COUNTER.fetch_add(count as u64, Ordering::Relaxed);
        }
        count
    } else {
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_in_process_qp_intercept_and_live_flowlabel_rotation() {
        let mut shm_box: Box<ShmIpcRegion> = unsafe {
            let layout = std::alloc::Layout::new::<ShmIpcRegion>();
            let ptr = std::alloc::alloc_zeroed(layout) as *mut ShmIpcRegion;
            Box::from_raw(ptr)
        };
        shm_box.init_in_place();

        let mut fake_qp = IbvQp {
            context: std::ptr::null_mut(),
            qp_context: std::ptr::null_mut(),
            pd: std::ptr::null_mut(),
            send_cq: std::ptr::null_mut(),
            recv_cq: std::ptr::null_mut(),
            srq: std::ptr::null_mut(),
            handle: 1,
            qp_num: 0x0041A2,
            state: 3,
            qp_type: IBV_QPT_RC,
            mutex: [0; 40],
            cond: [0; 48],
            events_completed: 0,
        };

        let mut init_attr = IbvQpAttr::default();
        init_attr.dest_qp_num = 0x0099B1;
        init_attr.ah_attr.is_global = 1;
        init_attr.ah_attr.port_num = 1;
        init_attr.ah_attr.grh.flow_label = compute_rocev2_flow_label_for_sport(49164);

        // 1. Simulate NCCL calling `ibv_modify_qp(..., IBV_QP_AV)` on Rail 3
        unsafe {
            let rc = fabriclock_hook_ibv_modify_qp(
                &mut fake_qp as *mut IbvQp,
                &mut init_attr as *mut IbvQpAttr,
                IBV_QP_AV,
                3,
                None,
            );
            assert_eq!(rc, 0);
        }

        // 2. Control Plane detects Spine-07 gray failure on Rail 3 and posts +17 sport delta
        shm_box.set_rail_ecmp_sport_delta(3, 17);

        // 3. Reconcile in-process QPs against shared-memory directive
        let rotated = poll_and_apply_in_process_qp_rotations(&shm_box, None);
        assert_eq!(rotated, 1);

        let reg = GLOBAL_QP_REGISTRY.lock().unwrap();
        let tracked = reg
            .entries
            .iter()
            .find(|e| e.qp_num == 0x0041A2)
            .expect("tracked QP");
        assert_eq!(tracked.base_udp_sport, 49164);
        assert_eq!(tracked.current_udp_sport, 49164 + 17);
        assert_eq!(tracked.rotation_count, 1);
    }
}
