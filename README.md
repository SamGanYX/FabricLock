# FabricLock

End-host optical telemetry, ECMP tomography, and collective failover control plane for RoCEv2 and InfiniBand GPU fabrics.

## Problem Statement

On 400G and 800G PAM4 links (`RS-FEC(544,514)`), optical transceivers frequently degrade into a "gray failure" regime before triggering a physical link-down (`LOS` / `LOF`) alarm. When the pre-FEC bit error rate (BER) drifts toward the IEEE 802.3ck/df limit ($2.4 \times 10^{-4}$) or experiences thermal/laser SNR micro-bursts, uncorrectable FEC codewords drop RDMA frames on the wire.

On a synchronous GPU training fabric, a single degraded transceiver does not fail loudly. Instead, it triggers microsecond RoCEv2 Priority Flow Control (PFC) pause frames, ECN/CNP congestion throttling, or `roce_adp_retrans` timeouts. Because `AllReduce`, `ReduceScatter`, and `AllGather` operations block on global barriers across every rank, one marginal transceiver on one rail stalls the entire communicator step.

Polling transceiver Digital Diagnostic Monitoring (DDM) registers directly from Top-of-Rack switches over I2C (`xcvrd` / `ethtool -m` over a 100-400 kHz serial bus) at sub-second intervals risks locking up the switch management bus. Conversely, shutting down a degraded switch port at Layer 2/3 triggers BGP EVPN or OpenSM re-convergence and causes in-flight RDMA Queue Pair timeouts.

## Design Overview

FabricLock isolates and routes around both **Tier-1 (Host-to-Leaf)** and **Tier-2 (Leaf-to-Spine)** optical gray failures purely from the GPU worker hosts:

1. **Tier-1 Detection (1 kHz Host ASIC Counter Polling):**
   Reads `rx_fec_corr_blocks`, `rx_fec_uncorr_blocks`, `rx_pcs_symbol_err`, `roce_adp_retrans`, and `rx_prio3_pause_duration` from the host NIC ASIC (`/sys/class/infiniband/mlx5_*/ports/1/hw_counters/*` and netlink `ETHTOOL_MSG_FEC_GET`). Because the Reed-Solomon FEC decoder lives inside the NIC SerDes ASIC rather than the pluggable transceiver MCU, polling these counters over PCIe touches 0% of the switch I2C bus. Local CMIS 5.0 / SFF-8636 optical EEPROM pages (vendor serial number, laser temperature, per-lane Rx power) are queried only on-demand when the second derivative of $\log_{10}(\text{Pre-FEC BER})$ spikes.

2. **Tier-2 Detection (Active ECMP UDP Source-Port Tomography):**
   A host NIC cannot directly read the FEC registers of an upper-tier Leaf-to-Spine optical link. However, RoCEv2 encapsulates RDMA traffic inside UDP (`DstPort = 4791`), and Leaf switches hash the 5-tuple `(SrcIP, DstIP, Proto, SrcPort, DstPort)` across Spine links via ECMP. By sweeping deterministic UDP source ports (`49152..49215`) across host pairs and intersecting `roce_adp_retrans` and RTT spikes against NCCL proxy operation latencies, the bipartite solver isolates degraded Leaf-to-Spine uplinks when the local Host-to-Leaf FEC counters are clean.

3. **Tier-2 Bypass (Sub-2ms RDMA Queue Pair Flow-Label Rotation):**
   When a Leaf-to-Spine link degrades, `fabriclock-verbs-hook` intercepts `libibverbs` (`ibv_modify_qp` with `IBV_QP_AV`) and mutates the 20-bit IPv6/RoCEv2 GRH `flow_label` on affected Reliable Connected (`IBV_QPT_RC`) Queue Pairs. The NIC recomputes the encapsulating UDP source port (`sport`), causing the Leaf switch ECMP hasher to steer subsequent packets onto a healthy Spine switch without modifying switch BGP routing tables.

4. **Tier-1 Bypass (Step-Boundary NVSwitch Cross-Rail PXN Failover):**
   If the transceiver plugged directly into `mlx5_0` (serving `GPU 0`) is degrading, mutating the UDP source port will not bypass the physical cable. Instead, at the next collective step boundary, FabricLock quarantines `mlx5_0` via shared-memory IPC and generates an updated `NCCL_TOPO_FILE` XML topology (`NCCL_PXN_DISABLE=0`, `NCCL_CROSS_NIC=1`) that routes `GPU 0`'s collective traffic over the intra-node 900 GB/s NVSwitch to egress through sibling NIC `mlx5_1`.

```text
                       [ Spine-06 ]     [ Spine-07 (Degraded) ]     [ Spine-08 ]
                            ^                     x                      ^
                            |                     |                      | (2) Tier-2 Bypass:
                            |                     |                      |     ibv_modify_qp(IBV_QP_AV)
                            +----------+----------+----------------------+     rotates GRH flow_label / UDP sport
                                       |                                       to hash onto Spine-08
                              [ Leaf-Rail0-SW03 ]                        [ Leaf-Rail1-SW03 ]
                                       x                                          ^
                                       | (Degraded Host Optic)                    |
   +-----------------------------------|------------------------------------------|---+
   | GPU Worker Node (HGX 8-GPU)       x                                          |   |
   |                              [ mlx5_0 ]                                 [ mlx5_1 ]
   |                                   ^                                          ^   |
   |                                   |         (1) Tier-1 Bypass:               |   |
   |                               [ GPU 0 ] ===> 900 GB/s NVSwitch (PXN) ===> [ GPU 1 ]
   +----------------------------------------------------------------------------------+
```

## Repository Structure

This repository is organized as an 8-crate Cargo workspace:

* `crates/fabriclock-ipc`: Lock-free 64-byte cache-line-aligned SPSC ring buffer (`/dev/shm/fabriclock_nccl_ipc.shm`) for shared-memory IPC between the in-process NCCL profiler / verbs hook and the out-of-process Rust daemon.
* `crates/fabriclock-hw`: Zero-allocation `/sys/class/infiniband/mlx5_*` counter reader, CMIS 5.0 (OSFP/QSFP-DD 800G) and SFF-8636 optical EEPROM parser, Generic Netlink `ETHTOOL_MSG_FEC_GET` frame builder, and IEEE 802.1AB LLDP neighbor parser.
* `crates/fabriclock-nccl-plugin`: Native C-ABI shared library (`cdylib`) exporting `ncclProfiler_v2` (`init`, `startEvent`, `stopEvent`, `recordEventState`, `finalize`). Uses a pre-allocated 1,024-slot event slab to avoid heap allocations on the NCCL proxy thread.
* `crates/fabriclock-tomography`: 32-byte wire probe packet encoder/decoder, UDP source-port sweep prober (`49152..49215`), and bipartite Clos intersection solver.
* `crates/fabriclock-core`: Sliding-window `RS-FEC(544,514)` Pre-FEC BER calculator, first and second derivative ($\frac{d^2\log_{10}\text{BER}}{dt^2}$) runaway detector, and collective straggler correlator.
* `crates/fabriclock-remediate`: RoCEv2 GRH `flow_label` calculator (`compute_rocev2_flow_label_for_sport`) and NVIDIA NCCL XML topology generator (`<system version="1">`) for intra-node NVSwitch PXN cross-rail failover.
* `crates/fabriclock-verbs-hook`: `LD_PRELOAD` shared library (`cdylib`) hooking `ibv_modify_qp` via `dlsym(RTLD_NEXT, "ibv_modify_qp")` on Linux to track live `struct ibv_qp*` handles and apply `IBV_QP_AV` address vector rotations in-process.
* `crates/fabriclock-daemon`: Multi-node telemetry aggregator, Prometheus/OpenMetrics text exporter (`/metrics`), JSON state server (`/api/v1/cluster/status`), and Slurm/Kubernetes pre-flight gatekeeper.
* `crates/fabriclock-cli`: Command-line binary (`fabriclock`) for running hardware audits, generating synthetic `sysfs` counter trees, benchmarking UDP source-port sweeps, and emitting PXN topology XML files.

## Project Status & Hardware Validation

I built this project to explore end-host approaches to PAM4 optical gray failures and RoCEv2 collective stragglers without requiring switch NOS agents.

* **What is implemented and unit-tested today:** The lock-free shared-memory ring buffer, CMIS 5.0 / SFF-8636 EEPROM byte parsers, IEEE 802.1AB LLDP frame parser, `/sys/class/infiniband` counter reader, `ncclProfiler_v2` C ABI entrypoints, `ibv_modify_qp` `LD_PRELOAD` shim, UDP source-port sweep prober, bipartite tomography solver, and NCCL PXN XML topology generator all compile cleanly and are covered by unit tests.
* **What requires hardware validation:** Because I do not have a multi-rack 400G/800G ConnectX-7 RoCEv2 fabric at school, multi-node Clos tomography and PXN failover are currently verified against synthetic `/sys/class/infiniband` counter trees (`fabriclock init-mock-sysfs`) and injected fault traces. If you operate a ConnectX-6/7 or Broadcom Thor 2 RoCEv2 cluster and want to test `fabriclock-hw` or `libfabriclock_nccl_profiler.so` against real hardware counters, issues and PRs are very welcome.

## Building and Running

Requires Rust 1.80+ (`cargo`).

```bash
# Run unit tests across all 8 workspace crates
cargo test

# Generate a synthetic 8-rail ConnectX-7 /sys/class/infiniband counter tree on disk
# and run the hardware counter poller + bipartite audit against it
cargo run -p fabriclock-cli -- init-mock-sysfs ./mock_sysfs
cargo run -p fabriclock-cli -- audit --sysfs-root ./mock_sysfs

# Run a live localhost UDP source-port sweep across 8 ECMP buckets
cargo run -p fabriclock-cli -- bench-tomography

# Print Prometheus / OpenMetrics output
cargo run -p fabriclock-cli -- metrics

# Emit a valid NCCL_TOPO_FILE XML topology quarantining mlx5_0 and routing GPU 0 via mlx5_1
cargo run -p fabriclock-cli -- generate-pxn-topo gpu-node-042 0
```

## Using with NCCL and Slurm / Kubernetes

On Linux hosts with NCCL 2.21+, compile the release shared libraries:

```bash
cargo build --release -p fabriclock-nccl-plugin -p fabriclock-verbs-hook
```

Export the profiler plugin and optional `libibverbs` hook in your training launch script or Slurm prolog (`deploy/slurm/fabriclock-prolog.sh`):

```bash
export NCCL_PROFILER_PLUGIN=/path/to/target/release/libfabriclock_nccl_profiler.so
export LD_PRELOAD=/path/to/target/release/libfabriclock_verbs_hook.so
```

Sample Kubernetes DaemonSet and Slurm prolog manifests are provided under `deploy/`.

## License

Apache License, Version 2.0.
