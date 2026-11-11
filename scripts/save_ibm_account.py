"""Save IBM Quantum Platform credentials locally (run this yourself, once).

The token is read with getpass, so it never appears on screen, in shell
history, or in this repository. Qiskit stores it in ~/.qiskit/qiskit-ibm.json,
which is outside the repo.

Usage:
    .venv/bin/python scripts/save_ibm_account.py
"""

from getpass import getpass

from qiskit_ibm_runtime import QiskitRuntimeService


def main() -> None:
    token = getpass("IBM Quantum API key (input hidden): ").strip()
    if not token:
        raise SystemExit("No key entered; nothing saved.")
    instance = input("Instance CRN (press Enter to let Qiskit pick one): ").strip() or None
    QiskitRuntimeService.save_account(
        channel="ibm_quantum_platform",
        token=token,
        instance=instance,
        name="qkc",
        set_as_default=True,
        overwrite=True,
    )
    print("Saved as account 'qkc' in ~/.qiskit/qiskit-ibm.json.")
    print("Next: .venv/bin/python scripts/check_ibm_access.py")


if __name__ == "__main__":
    main()
