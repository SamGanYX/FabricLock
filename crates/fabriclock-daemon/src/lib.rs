//! Distributed Multi-Node Cluster Aggregator, Prometheus OpenMetrics Exporter,
//! and Autonomous Pre-Flight Gatekeeper (`fabriclock-daemon`).
//!
//! Responsibilities:
//!   1. Aggregates per-node `Tier1OpticalAnomaly`, `NcclTelemetryEvent`, and `EcmpFlowObservation`
//!      across all GPU worker nodes in a Slurm or Kubernetes cluster over TCP/JSON-RPC.
//!   2. Continuously solves the global Bipartite Clos Graph and issues Dual-Tier Remediation
//!      directives (`PxnCrossRailFailoverResult` and `QpHashRotationResult`).
//!   3. Renders a zero-dependency Prometheus `/metrics` endpoint (`format_prometheus_openmetrics`)
//!      for Grafana/VictoriaMetrics scraping and a REST `/api/v1/cluster/status` JSON payload.
//!   4. Runs the 15-second Pre-Flight Cluster Certification (`evaluate_preflight_certification`)
//!      for Slurm `Prolog` and Kubernetes `Kueue` / `NodeProblemDetector`.

use fabriclock_core::{correlate_nccl_and_fabric, ClusterDiagnosticReport, Tier1OpticalAnomaly};
use fabriclock_ipc::NcclTelemetryEvent;
use fabriclock_remediate::{
    execute_nvswitch_pxn_failover, rotate_qp_ecmp_hash, PxnCrossRailFailoverResult,
    QpHashRotationResult,
};
use fabriclock_tomography::EcmpFlowObservation;
use serde::{Deserialize, Serialize};
use std::io::{Read, Write};
use std::net::TcpListener;

/// Node-to-Coordinator heartbeat & telemetry frame sent every 100ms by worker agents.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerTelemetryHeartbeat {
    pub node_hostname: String,
    pub timestamp_ns: u64,
    pub tier1_anomalies: Vec<Tier1OpticalAnomaly>,
    pub nccl_events: Vec<NcclTelemetryEvent>,
    pub ecmp_observations: Vec<EcmpFlowObservation>,
}

/// Complete Cluster Control-Plane State Snapshot served over HTTP JSON & Prometheus `/metrics`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterControlPlaneState {
    pub cluster_name: String,
    pub total_gpus: usize,
    pub active_worker_nodes: usize,
    pub switch_i2c_polls_executed: u64,
    pub bgp_route_flaps_triggered: u64,
    pub diagnostic: ClusterDiagnosticReport,
    pub active_pxn_failovers: Vec<PxnCrossRailFailoverResult>,
    pub active_qp_hash_rotations: Vec<QpHashRotationResult>,
}

/// Evaluates a multi-node `WorkerTelemetryHeartbeat` batch and computes the global
/// `ClusterControlPlaneState` along with any autonomous bypass actions.
pub fn process_cluster_telemetry_batch(
    cluster_name: &str,
    total_gpus: usize,
    heartbeats: &[WorkerTelemetryHeartbeat],
    baseline_rtt_ns: u64,
) -> ClusterControlPlaneState {
    let mut all_nccl = Vec::new();
    let mut all_tier1 = Vec::new();
    let mut all_ecmp = Vec::new();

    for hb in heartbeats {
        all_nccl.extend_from_slice(&hb.nccl_events);
        all_tier1.extend_from_slice(&hb.tier1_anomalies);
        all_ecmp.extend_from_slice(&hb.ecmp_observations);
    }

    let diagnostic = correlate_nccl_and_fabric(&all_nccl, all_tier1, &all_ecmp, baseline_rtt_ns);

    let mut pxn_failovers = Vec::new();
    for t1 in &diagnostic.tier1_host_anomalies {
        pxn_failovers.push(execute_nvswitch_pxn_failover(t1, None));
    }

    let mut qp_rotations = Vec::new();
    for t2 in &diagnostic.tier2_spine_anomalies {
        qp_rotations.push(rotate_qp_ecmp_hash(t2, None));
    }

    ClusterControlPlaneState {
        cluster_name: cluster_name.to_string(),
        total_gpus,
        active_worker_nodes: heartbeats.len(),
        switch_i2c_polls_executed: 0,
        bgp_route_flaps_triggered: 0,
        diagnostic,
        active_pxn_failovers: pxn_failovers,
        active_qp_hash_rotations: qp_rotations,
    }
}

/// Formats the `ClusterControlPlaneState` into standard Prometheus / OpenMetrics text format
/// for scraping by Prometheus, VictoriaMetrics, Datadog, or Grafana Alloy.
pub fn format_prometheus_openmetrics(state: &ClusterControlPlaneState) -> String {
    let mut out = String::with_capacity(4096);

    out.push_str("# HELP fabriclock_cluster_effective_mfu_percent Effective cluster Model FLOPs Utilization percentage.\n");
    out.push_str("# TYPE fabriclock_cluster_effective_mfu_percent gauge\n");
    out.push_str(&format!(
        "fabriclock_cluster_effective_mfu_percent{{cluster=\"{}\"}} {:.2}\n\n",
        state.cluster_name, state.diagnostic.effective_mfu_pct
    ));

    out.push_str("# HELP fabriclock_nccl_step_latency_ms Synchronous NCCL AllReduce collective step latency in ms.\n");
    out.push_str("# TYPE fabriclock_nccl_step_latency_ms gauge\n");
    out.push_str(&format!(
        "fabriclock_nccl_step_latency_ms{{cluster=\"{}\",quantile=\"ideal\"}} {:.2}\n",
        state.cluster_name, state.diagnostic.ideal_step_ms
    ));
    out.push_str(&format!(
        "fabriclock_nccl_step_latency_ms{{cluster=\"{}\",quantile=\"p99\"}} {:.2}\n\n",
        state.cluster_name, state.diagnostic.p99_step_ms
    ));

    out.push_str("# HELP fabriclock_optical_pre_fec_ber Instantaneous Pre-FEC Bit Error Rate on 400G/800G RNIC port.\n");
    out.push_str("# TYPE fabriclock_optical_pre_fec_ber gauge\n");
    for t1 in &state.diagnostic.tier1_host_anomalies {
        out.push_str(&format!(
            "fabriclock_optical_pre_fec_ber{{node=\"{}\",hca=\"{}\",rail=\"{}\",sn=\"{}\"}} {:.4e}\n",
            t1.node_hostname, t1.hca_name, t1.rail_id, t1.transceiver_sn, t1.instantaneous_ber
        ));
    }
    out.push('\n');

    out.push_str("# HELP fabriclock_spine_ecmp_elevated_rtt_us Elevated Leaf-to-Spine RTT detected via ECMP UDP tomography.\n");
    out.push_str("# TYPE fabriclock_spine_ecmp_elevated_rtt_us gauge\n");
    for t2 in &state.diagnostic.tier2_spine_anomalies {
        out.push_str(&format!(
            "fabriclock_spine_ecmp_elevated_rtt_us{{leaf=\"{}\",rail=\"{}\",sport=\"{}\"}} {:.2}\n",
            t2.leaf_switch_id, t2.rail_id, t2.spine_ecmp_bucket, t2.mean_elevated_rtt_us
        ));
    }
    out.push('\n');

    out.push_str("# HELP fabriclock_autonomous_bypasses_active Total active hitless PXN and QP hash rotations.\n");
    out.push_str("# TYPE fabriclock_autonomous_bypasses_active gauge\n");
    out.push_str(&format!(
        "fabriclock_autonomous_bypasses_active{{cluster=\"{}\",tier=\"tier1_pxn_rail_shift\"}} {}\n",
        state.cluster_name,
        state.active_pxn_failovers.len()
    ));
    out.push_str(&format!(
        "fabriclock_autonomous_bypasses_active{{cluster=\"{}\",tier=\"tier2_qp_sport_rotation\"}} {}\n",
        state.cluster_name,
        state.active_qp_hash_rotations.len()
    ));

    out
}

/// Pre-Flight Certification Decision for Slurm `Prolog` / Kubernetes `Kueue` admission control.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreFlightVerdict {
    pub allow_job_start: bool,
    pub auto_remediated: bool,
    pub recommended_nccl_env: Vec<(String, String)>,
    pub cordoned_nodes: Vec<String>,
    pub summary: String,
}

/// Evaluates whether a newly scheduled Slurm/K8s multi-node GPU training job should proceed,
/// automatically exporting `NCCL_IB_HCA` and `NCCL_TOPO_FILE` overrides if single-rail
/// gray failures are present, or cordoning a node if >=2 rails on the same node are degraded.
pub fn evaluate_preflight_certification(state: &ClusterControlPlaneState) -> PreFlightVerdict {
    let mut env_overrides = Vec::new();
    let mut cordoned_nodes = Vec::new();

    if state.active_pxn_failovers.len() >= 2
        && state.active_pxn_failovers[0].node_hostname == state.active_pxn_failovers[1].node_hostname
    {
        let bad_node = state.active_pxn_failovers[0].node_hostname.clone();
        cordoned_nodes.push(bad_node.clone());
        return PreFlightVerdict {
            allow_job_start: false,
            auto_remediated: false,
            recommended_nccl_env: env_overrides,
            cordoned_nodes,
            summary: format!(
                "PRE-FLIGHT BLOCKED: Node {} has multiple simultaneous degrading 800G transceivers; drain and replace before job start.",
                bad_node
            ),
        };
    }

    if let Some(first_pxn) = state.active_pxn_failovers.first() {
        env_overrides.extend_from_slice(&first_pxn.nccl_env_exports);
    }

    PreFlightVerdict {
        allow_job_start: true,
        auto_remediated: !state.active_pxn_failovers.is_empty()
            || !state.active_qp_hash_rotations.is_empty(),
        recommended_nccl_env: env_overrides,
        cordoned_nodes,
        summary: format!(
            "PRE-FLIGHT PASSED: {} Tier-1 PXN failovers and {} Tier-2 ECMP Spine rotations pre-applied.",
            state.active_pxn_failovers.len(),
            state.active_qp_hash_rotations.len()
        ),
    }
}

/// Serves a single HTTP request on `/metrics` (Prometheus) or `/api/v1/cluster/status` (JSON)
/// so the daemon can be scraped or queried by dashboards and CLI tools.
pub fn serve_single_http_request(listener: &TcpListener, state: &ClusterControlPlaneState) -> std::io::Result<()> {
    let (mut stream, _) = listener.accept()?;
    let mut req_buf = [0u8; 1024];
    let n = stream.read(&mut req_buf)?;
    let req_line = String::from_utf8_lossy(&req_buf[..n]);

    let (content_type, body) = if req_line.contains("GET /api/v1/cluster/status") {
        (
            "application/json",
            serde_json::to_string_pretty(state).unwrap_or_else(|_| "{}".into()),
        )
    } else {
        (
            "text/plain; version=0.0.4; charset=utf-8",
            format_prometheus_openmetrics(state),
        )
    };

    let response = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: {}\r\nContent-Length: {}\r\nAccess-Control-Allow-Origin: *\r\nConnection: close\r\n\r\n{}",
        content_type,
        body.len(),
        body
    );
    stream.write_all(response.as_bytes())?;
    stream.flush()
}

#[cfg(test)]
mod tests {
    use super::*;
    use fabriclock_ipc::NcclCollectiveKind;

    #[test]
    fn test_cluster_batch_prometheus_and_preflight() {
        let hb = WorkerTelemetryHeartbeat {
            node_hostname: "gpu-node-042".into(),
            timestamp_ns: 2_000_000_000,
            tier1_anomalies: vec![Tier1OpticalAnomaly {
                node_hostname: "gpu-node-042".into(),
                hca_name: "mlx5_0".into(),
                pcie_bdf: "0000:10:00.0".into(),
                rail_id: 0,
                transceiver_sn: "INNO99481A0042".into(),
                instantaneous_ber: 3.87e-4,
                d1_log_ber_per_sec: 1.8,
                d2_log_ber_per_sec2: 4.5,
                uncorr_codewords_per_sec: 412.0,
                pfc_pause_rate_us_per_sec: 84_000.0,
                predicted_time_to_link_flap_sec: 12.0,
            }],
            nccl_events: vec![
                NcclTelemetryEvent {
                    duration_ns: 182_000_000,
                    rail_id: 1,
                    coll_kind: NcclCollectiveKind::AllReduce as u8,
                    ..Default::default()
                },
                NcclTelemetryEvent {
                    duration_ns: 318_500_000,
                    rail_id: 0,
                    coll_kind: NcclCollectiveKind::AllReduce as u8,
                    ..Default::default()
                },
            ],
            ecmp_observations: vec![],
        };

        let state = process_cluster_telemetry_batch("us-east-h100-prod-01", 4096, &[hb], 12_000);
        assert_eq!(state.active_pxn_failovers.len(), 1);
        assert_eq!(state.switch_i2c_polls_executed, 0);

        let prom = format_prometheus_openmetrics(&state);
        assert!(prom.contains("fabriclock_optical_pre_fec_ber"));
        assert!(prom.contains("INNO99481A0042"));

        let verdict = evaluate_preflight_certification(&state);
        assert!(verdict.allow_job_start);
        assert!(verdict.auto_remediated);
        assert!(!verdict.recommended_nccl_env.is_empty());
    }
}
