"""The append-only receipt store: legacy migration, appends, and reading lines added by another worker."""
import json
import os

from services import provenance


def test_receipt_store_migrates_appends_and_tails(tmp_path, monkeypatch):
    monkeypatch.setattr(provenance, "_data_dir", str(tmp_path))
    store = provenance._ReceiptStore()
    (tmp_path / "receipts.json").write_text(json.dumps([{"receipt_id": "old-1", "audio_sha256": "aa"}]))

    assert [r["receipt_id"] for r in store.find()] == ["old-1"]
    assert (tmp_path / "receipts.json.migrated").exists() and not (tmp_path / "receipts.json").exists()

    store.add({"receipt_id": "new-1", "audio_sha256": "bb"})
    assert store.find(audio_sha256="bb")[0]["receipt_id"] == "new-1"

    # another worker appends a full line and starts a second one
    with open(tmp_path / "receipts.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({"receipt_id": "other-1"}) + "\n")
        f.write('{"receipt_id": "half')
    assert {r["receipt_id"] for r in store.find()} == {"old-1", "new-1", "other-1"}

    with open(tmp_path / "receipts.jsonl", "a", encoding="utf-8") as f:
        f.write('-done"}\n')
    assert "half-done" in {r["receipt_id"] for r in store.find()}
    assert len(open(tmp_path / "receipts.jsonl").read().splitlines()) == 4
    assert os.path.getsize(tmp_path / "receipts.jsonl") > 0
