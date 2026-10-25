#!/usr/bin/env python3
"""
FabricLock Dual-Tier Cluster Simulator & 30-Minute Wedge Audit Demo
Models a 4,096-GPU (512-node x 8-rail) 800G RoCEv2 PAM4 Clos Fabric experiencing:
  1. Tier-1 (Host-to-Leaf) optical transceiver FEC gray failure on gpu-node-042:mlx5_0
  2. Tier-2 (Leaf-to-Spine) optical transceiver gray failure on spine-sw-07:Port22
And executes <2s autonomous remediation without touching switch I2C buses or BGP routing.
"""

from dataclasses import dataclass
from typing import List


@dataclass
class TransceiverFault:
    tier: str
    location: str
    serial_number: str
    vendor_model: str
    pre_fec_ber: float
    ber_2nd_deriv: float
    uncorr_codewords_per_sec: int
    pfc_pause_us_per_step: int
    detection_method: str
    bypass_action: str
    bypass_latency_ms: float
    mfu_recovered_pct: float


def run_simulation() -> None:
    print("=" * 88)
    print(" FABRICLOCK v0.2.0 — 30-MINUTE CLUSTER STRAGGLER & DUAL-TIER OPTICAL AUDIT")
    print(" Target Fabric : 4,096x NVIDIA H100 GPUs (512 HGX Nodes | 8x 800G RoCEv2 Rails)")
    print(" Switch I2C    : 0% Touched (1kHz End-Host PCIe BAR0 + 200Hz ECMP UDP Tomography)")
    print("=" * 88)

    baseline_step_ms = 182.0
    degraded_step_ms = 318.5
    baseline_mfu = 53.8
    degraded_mfu = round(baseline_mfu * (baseline_step_ms / degraded_step_ms), 1)
    hourly_cluster_cost = 4096 * 2.65  # $10,854/hr ($260k/day)
    hourly_waste = round(hourly_cluster_cost * ((baseline_mfu - degraded_mfu) / baseline_mfu), 2)

    print(f"\n[1] SYNCHRONOUS COLLECTIVE HEALTH (NCCL AllReduce Ring/Tree Profiler)")
    print(f"    • Ideal Step Time     : {baseline_step_ms:.1f} ms  ({baseline_mfu:.1f}% MFU)")
    print(f"    • Observed Step Time  : {degraded_step_ms:.1f} ms  ({degraded_mfu:.1f}% MFU)  <-- CRITICAL STRAGGLER STALL")
    print(f"    • Compute Burn Waste  : ${hourly_waste:,.2f}/hour (${hourly_waste * 24:,.0f}/day wasted)")

    faults: List[TransceiverFault] = [
        TransceiverFault(
            tier="TIER-1 (Host-to-Leaf)",
            location="gpu-node-042 :: mlx5_0 (Rail 0 -> tor-leaf-r0-sw03:Port14)",
            serial_number="INNO-800G-DR8-99481A",
            vendor_model="InnoLight T-OSFP-800G-DR8 (PAM4)",
            pre_fec_ber=3.8e-4,
            ber_2nd_deriv=4.72,
            uncorr_codewords_per_sec=412,
            pfc_pause_us_per_step=84_200,
            detection_method="Direct Host PCIe BAR0 Register Read (mlx5 PPCNT @ 1,000 Hz)",
            bypass_action="Dynamic NVSwitch Cross-Rail PXN Failover (GPU0 -> NVSwitch -> GPU1 -> mlx5_1)",
            bypass_latency_ms=185.0,
            mfu_recovered_pct=13.4,
        ),
        TransceiverFault(
            tier="TIER-2 (Leaf-to-Spine)",
            location="tor-leaf-r3-sw02:Port52 -> spine-sw-07:Port22 (Rail 3 Clos Uplink)",
            serial_number="COHR-800G-2FR4-77109C",
            vendor_model="Coherent/Finisar FTCD8527E2PCM (PAM4)",
            pre_fec_ber=2.9e-4,
            ber_2nd_deriv=3.45,
            uncorr_codewords_per_sec=289,
            pfc_pause_us_per_step=52_300,
            detection_method="Bipartite QP Intersection + Active ECMP UDP SrcPort Tomography (Sport=49164)",
            bypass_action="Host RDMA QP UDP SrcPort Hash Rotation (ibv_modify_qp Sport 49164 -> 49181 -> Spine-08)",
            bypass_latency_ms=3.4,
            mfu_recovered_pct=9.1,
        ),
    ]

    print(f"\n[2] PINPOINTED DEGRADING OPTICAL TRANSCEIVERS (2 Gray Failures Isolated in 420ms)")
    print("-" * 88)
    for idx, f in enumerate(faults, 1):
        print(f"  Fault #{idx}: [{f.tier}] {f.location}")
        print(f"    • Transceiver Serial  : {f.serial_number} ({f.vendor_model})")
        print(f"    • Link Status         : LINK UP (Silent Gray Failure — No Switch Port-Down Alarm)")
        print(f"    • Pre-FEC BER         : {f.pre_fec_ber:.2e} (d²BER/dt² = +{f.ber_2nd_deriv:.2f}/s² | {f.uncorr_codewords_per_sec} Uncorr CW/s)")
        print(f"    • Collective Impact   : +{f.pfc_pause_us_per_step / 1000:.1f} ms/step PFC/RNRNAK stall")
        print(f"    • Detection Mechanism : {f.detection_method}")
        print(f"    • Autonomous Bypass   : {f.bypass_action}")
        print(f"    • Bypass Latency      : {f.bypass_latency_ms:.1f} ms (Zero PyTorch Checkpoint Rollback)")
        print("-" * 88)

    remediated_step_ms = 186.2
    remediated_mfu = round(baseline_mfu * (baseline_step_ms / remediated_step_ms), 1)
    recovered_dollars_per_month = round(
        hourly_cluster_cost * ((remediated_mfu - degraded_mfu) / baseline_mfu) * 24 * 30
    )

    print(f"\n[3] POST-BYPASS CLUSTER RECOVERY SUMMARY")
    print(f"    • Step Time Restored  : {degraded_step_ms:.1f} ms  --->  {remediated_step_ms:.1f} ms")
    print(f"    • Cluster MFU Restored: {degraded_mfu:.1f}%      --->  {remediated_mfu:.1f}% (+{remediated_mfu - degraded_mfu:.1f}% MFU)")
    print(f"    • Reclaimed Compute   : ${recovered_dollars_per_month:,.0f} / month saved on this 4,096-GPU cluster")
    print("=" * 88)


if __name__ == "__main__":
    run_simulation()
