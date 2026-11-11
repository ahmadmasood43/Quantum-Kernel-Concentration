"""Read-only check that saved IBM credentials work. Submits no jobs.

Lists the operational backends the account can see, with qubit count and
pending-job queue length, and writes the list to results/hardware/backends.json
so the backend choice for the hardware experiment is recorded.

Usage:
    .venv/bin/python scripts/check_ibm_access.py
"""

import json
from datetime import datetime, timezone
from pathlib import Path

from qiskit_ibm_runtime import QiskitRuntimeService


def main() -> None:
    service = QiskitRuntimeService(name="qkc")
    rows = []
    for backend in service.backends(operational=True, simulator=False):
        status = backend.status()
        rows.append(
            {
                "name": backend.name,
                "num_qubits": backend.num_qubits,
                "pending_jobs": status.pending_jobs,
            }
        )
    rows.sort(key=lambda r: r["pending_jobs"])
    for r in rows:
        print(f"{r['name']:<20} qubits={r['num_qubits']:<4} pending_jobs={r['pending_jobs']}")

    out = Path(__file__).resolve().parents[1] / "results" / "hardware" / "backends.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {"queried_at_utc": datetime.now(timezone.utc).isoformat(), "backends": rows},
            indent=2,
        )
    )
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
