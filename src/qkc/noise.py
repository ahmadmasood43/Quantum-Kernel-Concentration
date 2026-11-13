r"""Calibration snapshots and (scalable) device noise models.

A snapshot is a plain-JSON summary of an IBM backend's calibration at one
moment: per-qubit T1/T2 and readout error, per-gate errors and durations for
the native gates, and the coupling map. Every noisy simulation in this project
is built from a saved snapshot, so it can be re-run offline and audited.

Noise model (per gate instance, following the standard "basic device" recipe):
thermal relaxation over the gate duration, composed with a depolarizing
channel whose strength makes up the rest of the reported gate error; plus a
symmetric readout error. `scale` multiplies the depolarizing and readout
error rates and divides T1/T2 (scale = 0 gives an ideal device).

Depolarizing parameter: for E(rho) = (1-p) rho + p I/d the average gate
infidelity is r = p (d-1)/d, so p = r d / (d-1).
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from qiskit.quantum_info import average_gate_fidelity
from qiskit.transpiler import CouplingMap
from qiskit_aer.noise import NoiseModel, ReadoutError, depolarizing_error, thermal_relaxation_error

ONE_Q_GATES = ("sx", "x")
TWO_Q_GATE = "cz"


def calibration_summary(backend, target=None, properties_datetime=None) -> dict:
    """Extract a JSON-serializable calibration snapshot from a BackendV2.

    `target` overrides backend.target (e.g. a historical target from
    IBMBackend.target_history(datetime=...)); `properties_datetime` selects the
    matching historical calibration timestamp where the backend supports it.
    """
    tgt = target if target is not None else backend.target
    nq = tgt.num_qubits
    qubits = []
    for q in range(nq):
        qp = tgt.qubit_properties[q] if tgt.qubit_properties else None
        meas = tgt["measure"].get((q,)) if "measure" in tgt else None
        row = {
            "t1": getattr(qp, "t1", None),
            "t2": getattr(qp, "t2", None),
            "readout_error": getattr(meas, "error", None),
            "readout_duration": getattr(meas, "duration", None),
        }
        for g in ONE_Q_GATES:
            ip = tgt[g].get((q,)) if g in tgt else None
            row[f"{g}_error"] = getattr(ip, "error", None)
            row[f"{g}_duration"] = getattr(ip, "duration", None)
        qubits.append(row)
    edges = []
    if TWO_Q_GATE in tgt:
        for qargs, ip in tgt[TWO_Q_GATE].items():
            edges.append({"qubits": list(qargs), "error": getattr(ip, "error", None),
                          "duration": getattr(ip, "duration", None)})
    last_update = None
    try:
        props = backend.properties(datetime=properties_datetime) if properties_datetime else backend.properties()
        last_update = props.last_update_date.isoformat() if props is not None else None
    except Exception:
        pass
    return {
        "backend": backend.name,
        "snapshot_utc": datetime.now(timezone.utc).isoformat(),
        "calibration_last_update": last_update,
        "num_qubits": nq,
        "basis_gates": sorted(set(tgt.operation_names)),
        "qubits": qubits,
        "cz_edges": edges,
    }


def save_snapshot(cal: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cal, indent=1))


def load_snapshot(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def coupling_map(cal: dict) -> CouplingMap:
    return CouplingMap([tuple(e["qubits"]) for e in cal["cz_edges"]])


def _relax(t1, t2, dur, scale):
    if t1 is None or t2 is None or dur is None or not dur or scale == 0:
        return None
    t1s, t2s = t1 / scale, t2 / scale
    t2s = min(t2s, 2 * t1s)  # physical constraint required by Aer
    return thermal_relaxation_error(t1s, t2s, dur)


def _gate_error(err, relax, n_qubits, scale):
    d = 2**n_qubits
    relax_infid = 0.0 if relax is None else 1.0 - average_gate_fidelity(relax)
    r_dep = max((err or 0.0) - relax_infid, 0.0) * scale
    p = min(r_dep * d / (d - 1), 1.0)
    dep = depolarizing_error(p, n_qubits) if p > 0 else None
    if dep is None and relax is None:
        return None
    if dep is None:
        return relax
    return dep if relax is None else dep.compose(relax)


def build_noise_model(cal: dict, scale: float = 1.0) -> NoiseModel:
    nm = NoiseModel(basis_gates=[g for g in cal["basis_gates"] if g in ("cz", "rz", "sx", "x", "id", "measure")])
    if scale == 0:
        return nm
    for q, row in enumerate(cal["qubits"]):
        for g in ONE_Q_GATES:
            relax = _relax(row["t1"], row["t2"], row.get(f"{g}_duration"), scale)
            e = _gate_error(row.get(f"{g}_error"), relax, 1, scale)
            if e is not None:
                nm.add_quantum_error(e, g, [q])
        ro = row.get("readout_error")
        if ro is not None:
            p = min(ro * scale, 0.5)
            nm.add_readout_error(ReadoutError([[1 - p, p], [p, 1 - p]]), [q])
    for e in cal["cz_edges"]:
        a, b = e["qubits"]
        ra = _relax(cal["qubits"][a]["t1"], cal["qubits"][a]["t2"], e["duration"], scale)
        rb = _relax(cal["qubits"][b]["t1"], cal["qubits"][b]["t2"], e["duration"], scale)
        # expand(): the error it is called on acts on the lower-index qubit (a)
        relax = None if ra is None or rb is None else ra.expand(rb)
        err = _gate_error(e["error"], relax, 2, scale)
        if err is not None:
            nm.add_quantum_error(err, TWO_Q_GATE, [a, b])
    return nm
