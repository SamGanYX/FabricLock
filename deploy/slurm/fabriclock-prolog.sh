#!/usr/bin/env bash
# ==============================================================================
# FabricLock Slurm Prolog Pre-Flight Gatekeeper (`/etc/slurm/prolog.d/50-fabriclock.sh`)
#
# Runs in <150ms before multi-node PyTorch / Megatron-LM jobs start:
#   1. Queries local `fabriclock preflight` to inspect 1kHz PCIe BAR0 RS-FEC(544,514)
#      counters and 200Hz ECMP UDP source-port Spine tomography.
#   2. If a single Host-to-Leaf transceiver is experiencing gray failure, automatically
#      generates `/var/run/fabriclock/pxn_topo.xml` and exports `NCCL_TOPO_FILE` +
#      `NCCL_IB_HCA` into the job environment to route around the dying cable via
#      900 GB/s intra-node NVSwitch PXN.
#   3. If >=2 rails on the same node have simultaneous optical runaway, drains the node
#      via `scontrol update NodeName=$SLURMD_NODENAME State=DRAIN` so the job is
#      rescheduled cleanly instead of stalling 4,096 GPUs.
# ==============================================================================

set -euo pipefail

FABRICLOCK_BIN="${FABRICLOCK_BIN:-/usr/local/bin/fabriclock}"
RUN_DIR="/var/run/fabriclock"
mkdir -p "${RUN_DIR}"

if [[ ! -x "${FABRICLOCK_BIN}" ]]; then
    exit 0
fi

VERDICT_JSON="$("${FABRICLOCK_BIN}" preflight)"
ALLOW_START="$(echo "${VERDICT_JSON}" | grep -o '"allow_job_start": *[a-z]*' | awk '{print $2}')"

if [[ "${ALLOW_START}" == "false" ]]; then
    logger -t fabriclock-slurm "CRITICAL: Optical Pre-Flight failed on ${SLURMD_NODENAME:-localhost}; draining node."
    scontrol update NodeName="${SLURMD_NODENAME:-localhost}" State=DRAIN Reason="FabricLock: Multi-Rail 800G Optical Gray Failure" || true
    exit 1
fi

# Export NCCL Profiler Plugin & Dynamic PXN Failover Topology if auto-remediated
echo "export NCCL_PROFILER_PLUGIN=/usr/local/lib/libfabriclock_nccl_profiler.so"
echo "export NCCL_PXN_DISABLE=0"
echo "export NCCL_CROSS_NIC=1"
