#!/usr/bin/env python3
"""Retire the receipt signing key and start signing with a new one.

    ./bonclay stop
    backend/.venv/bin/python backend/scripts/rotate_receipt_key.py --yes
    ./bonclay start

The current Ed25519 public key is appended to backend/data/receipt_ed25519.retired, a new private key replaces
backend/data/receipt_ed25519.key (mode 600), and from the next start every receipt and withdrawal-log row is signed with
the new key and carries its `key_id`. Old receipts keep verifying: /verify and /.well-known/bonclay-receipt-key know
every retired key (services/provenance.py, "key rotation"). Run it with the API stopped: a running server keeps the old
key in memory and would keep signing with it. When the key comes from BONCLAY_RECEIPT_SIGNING_KEY this script refuses:
generate a new key, set it there, and add the old public key to BONCLAY_RECEIPT_RETIRED_KEYS."""
import argparse
import os
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from services import provenance  # noqa: E402


def api_running(port: str) -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/health", timeout=2) as r:
            return r.status == 200
    except Exception:  # noqa: BLE001
        return False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--yes", action="store_true", help="do it (without this flag the script only shows what would happen)")
    ap.add_argument("--port", default=os.getenv("BONCLAY_API_PORT", "8000"), help="API port to check for a running server")
    a = ap.parse_args()
    current = provenance.public_key_b64()
    print(f"current key: {provenance.key_id(current)}   retired keys: {len(provenance.retired_public_keys())}")
    if api_running(a.port):
        print(f"The API is running on port {a.port}. Stop it first (./bonclay stop): a running server keeps the old key "
              "in memory and would keep signing with it.", file=sys.stderr)
        return 2
    if not a.yes:
        print("Dry run. Add --yes to retire the current key and generate a new one.")
        return 0
    try:
        out = provenance.rotate_signing_key()
    except RuntimeError as e:
        print(f"Not rotated: {e}", file=sys.stderr)
        return 1
    print(f"rotated: {out['retired_key_id']} retired, {out['key_id']} is now current ({out['retired_count']} retired key(s) "
          "known). Start the API again; old receipts still verify.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
