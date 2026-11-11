"""Save a calibration snapshot of a real IBM backend (read-only: no jobs, no QPU time).

    .venv/bin/python experiments/snapshot_backend.py --backend ibm_marrakesh

Writes results/hardware/calibration_<backend>_<UTC timestamp>.json, which
experiments/noise_sim.py uses to build its noise models.
"""

import argparse
from datetime import datetime, timezone

import numpy as np
from qiskit_ibm_runtime import QiskitRuntimeService

from qkc.config import RESULTS
from qkc.noise import calibration_summary, save_snapshot


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backend", default="ibm_marrakesh")
    args = p.parse_args()
    service = QiskitRuntimeService(name="qkc", instance="open-instance")
    be = service.backend(args.backend)
    cal = calibration_summary(be)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = RESULTS / "hardware" / f"calibration_{be.name}_{stamp}.json"
    save_snapshot(cal, out)

    def med(key):
        vals = [q[key] for q in cal["qubits"] if q[key] is not None]
        return float(np.median(vals)) if vals else None

    cz = [e["error"] for e in cal["cz_edges"] if e["error"] is not None]
    print(f"{be.name}: {cal['num_qubits']} qubits, calibration last update {cal['calibration_last_update']}")
    print(f"  median T1 = {med('t1') * 1e6:.0f} us, T2 = {med('t2') * 1e6:.0f} us, "
          f"readout err = {med('readout_error'):.4f}, sx err = {med('sx_error'):.2e}, "
          f"cz err = {np.median(cz):.2e} over {len(cz)} edges")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
