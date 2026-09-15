"""Fail-closed publication gates; no render or production files are modified."""
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import finalize_view_consistent_video as finalizer


def test_reject_short_or_duplicate_actor_inventory(tmp_path):
    for ids in [[], ['000899'] * 150, [str(i) for i in range(149)]]:
        with pytest.raises(ValueError, match='150 distinct'):
            finalizer.validate_reviews(tmp_path, ids)


def test_reject_partial_native_review(tmp_path, monkeypatch):
    monkeypatch.setattr(finalizer, 'read', lambda _: {'complete': False, 'records': []})
    with pytest.raises(ValueError, match='native jaw crops'):
        finalizer.validate_reviews(tmp_path, [str(i) for i in range(150)])


def test_check_detects_changed_retained_file(tmp_path, monkeypatch):
    monkeypatch.setattr(finalizer, 'read', lambda _: {'hashes': {'frame.png': 'original'}})
    monkeypatch.setattr(finalizer, 'sha', lambda _: 'changed')
    with pytest.raises(ValueError, match='checksum mismatch'):
        finalizer.check(tmp_path)
