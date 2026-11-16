"""Tests for rl/utils.py helper functions."""
from __future__ import annotations

import json
from pathlib import Path

import pytest


# ------------------------------------------------------------------
# scheduled_int
# ------------------------------------------------------------------


class TestScheduledInt:
    def test_before_start_returns_start_val(self):
        from rl.utils import scheduled_int
        assert scheduled_int(cur_iter=0, start_iter=5, end_iter=10, start_val=100, end_val=200, curve="linear") == 100

    def test_after_end_returns_end_val(self):
        from rl.utils import scheduled_int
        assert scheduled_int(cur_iter=20, start_iter=5, end_iter=10, start_val=100, end_val=200, curve="linear") == 200

    def test_midpoint_linear(self):
        from rl.utils import scheduled_int
        result = scheduled_int(cur_iter=5, start_iter=0, end_iter=10, start_val=0, end_val=100, curve="linear")
        assert result == 50

    def test_at_start_returns_start(self):
        from rl.utils import scheduled_int
        assert scheduled_int(cur_iter=1, start_iter=1, end_iter=10, start_val=10, end_val=50, curve="linear") == 10

    def test_at_end_returns_end(self):
        from rl.utils import scheduled_int
        assert scheduled_int(cur_iter=10, start_iter=1, end_iter=10, start_val=10, end_val=50, curve="linear") == 50

    def test_cosine_curve(self):
        from rl.utils import scheduled_int
        # Cosine should be between start and end at midpoint, but not necessarily the linear midpoint.
        result = scheduled_int(cur_iter=5, start_iter=0, end_iter=10, start_val=0, end_val=100, curve="cosine")
        assert 0 <= result <= 100

    def test_same_start_end_iter(self):
        from rl.utils import scheduled_int
        # Edge case: start_iter == end_iter
        result = scheduled_int(cur_iter=5, start_iter=5, end_iter=5, start_val=10, end_val=90, curve="linear")
        assert result == 90  # should return end_val


# ------------------------------------------------------------------
# merge_counts
# ------------------------------------------------------------------


class TestMergeCounts:
    def test_basic_merge(self):
        from rl.utils import merge_counts
        assert merge_counts({"a": 1, "b": 2}, {"b": 3, "c": 4}) == {"a": 1, "b": 5, "c": 4}

    def test_empty_dicts(self):
        from rl.utils import merge_counts
        assert merge_counts({}, {}) == {}

    def test_one_empty(self):
        from rl.utils import merge_counts
        assert merge_counts({"x": 10}, {}) == {"x": 10}
        assert merge_counts({}, {"y": 5}) == {"y": 5}

    def test_does_not_mutate_inputs(self):
        from rl.utils import merge_counts
        a = {"x": 1}
        b = {"x": 2}
        merge_counts(a, b)
        assert a == {"x": 1}
        assert b == {"x": 2}


# ------------------------------------------------------------------
# split_indices
# ------------------------------------------------------------------


class TestSplitIndices:
    def test_single_part(self):
        from rl.utils import split_indices
        assert split_indices(5, 1) == [[0, 1, 2, 3, 4]]

    def test_two_parts_even(self):
        from rl.utils import split_indices
        result = split_indices(4, 2)
        assert len(result) == 2
        # Interleaved: [0,2] and [1,3]
        all_indices = sorted(idx for chunk in result for idx in chunk)
        assert all_indices == [0, 1, 2, 3]

    def test_two_parts_odd(self):
        from rl.utils import split_indices
        result = split_indices(5, 2)
        all_indices = sorted(idx for chunk in result for idx in chunk)
        assert all_indices == [0, 1, 2, 3, 4]

    def test_more_parts_than_items(self):
        from rl.utils import split_indices
        result = split_indices(2, 5)
        all_indices = sorted(idx for chunk in result for idx in chunk)
        assert all_indices == [0, 1]
        assert all(len(c) <= 1 for c in result)

    def test_zero_items(self):
        from rl.utils import split_indices
        assert split_indices(0, 3) == [[]]


# ------------------------------------------------------------------
# sha256_file
# ------------------------------------------------------------------


class TestSha256File:
    def test_known_content(self, tmp_path):
        from rl.utils import sha256_file
        p = tmp_path / "test.txt"
        p.write_text("hello")
        result = sha256_file(p)
        assert result is not None
        assert len(result) == 64  # hex digest length
        # Known SHA-256 of "hello"
        assert result == "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"

    def test_nonexistent_file(self, tmp_path):
        from rl.utils import sha256_file
        assert sha256_file(tmp_path / "no_such_file.pt") is None

    def test_different_contents_differ(self, tmp_path):
        from rl.utils import sha256_file
        a = tmp_path / "a.bin"
        b = tmp_path / "b.bin"
        a.write_bytes(b"\x00" * 100)
        b.write_bytes(b"\xff" * 100)
        assert sha256_file(a) != sha256_file(b)


# ------------------------------------------------------------------
# append_jsonl
# ------------------------------------------------------------------


class TestAppendJsonl:
    def test_creates_file(self, tmp_path):
        from rl.utils import append_jsonl
        p = tmp_path / "log.jsonl"
        append_jsonl(p, {"a": 1})
        assert p.exists()
        lines = p.read_text().strip().split("\n")
        assert len(lines) == 1
        assert json.loads(lines[0]) == {"a": 1}

    def test_appends_multiple(self, tmp_path):
        from rl.utils import append_jsonl
        p = tmp_path / "log.jsonl"
        append_jsonl(p, {"iter": 1})
        append_jsonl(p, {"iter": 2})
        lines = p.read_text().strip().split("\n")
        assert len(lines) == 2
        assert json.loads(lines[0])["iter"] == 1
        assert json.loads(lines[1])["iter"] == 2

    def test_creates_parent_dirs(self, tmp_path):
        from rl.utils import append_jsonl
        p = tmp_path / "sub" / "dir" / "log.jsonl"
        append_jsonl(p, {"x": 42})
        assert p.exists()

    def test_sorts_keys(self, tmp_path):
        from rl.utils import append_jsonl
        p = tmp_path / "log.jsonl"
        append_jsonl(p, {"z": 1, "a": 2, "m": 3})
        line = p.read_text().strip()
        keys = list(json.loads(line).keys())
        assert keys == sorted(keys)


# ------------------------------------------------------------------
# cleanup_old_datasets
# ------------------------------------------------------------------


class TestCleanupOldDatasets:
    def test_removes_old_datasets(self, tmp_path):
        from rl.utils import cleanup_old_datasets

        # Create fake dataset files for iterations 1-5
        for k in range(1, 6):
            (tmp_path / f"alpha_dataset_iter_{k}.pt").write_text("data")
            shard_dir = tmp_path / f"alpha_dataset_iter_{k}.pt.shards"
            shard_dir.mkdir()
            (shard_dir / "shard0.pt").write_text("shard")

        cleanup_old_datasets(tmp_path, keep_last=2, current_iter=5)

        # Iterations 1,2,3 should be deleted (keep_last=2 means keep 4,5)
        for k in range(1, 4):
            assert not (tmp_path / f"alpha_dataset_iter_{k}.pt").exists()
            assert not (tmp_path / f"alpha_dataset_iter_{k}.pt.shards").exists()
        # 4 and 5 should remain
        for k in range(4, 6):
            assert (tmp_path / f"alpha_dataset_iter_{k}.pt").exists()
            assert (tmp_path / f"alpha_dataset_iter_{k}.pt.shards").exists()

    def test_keep_all(self, tmp_path):
        from rl.utils import cleanup_old_datasets

        for k in range(1, 4):
            (tmp_path / f"alpha_dataset_iter_{k}.pt").write_text("data")

        cleanup_old_datasets(tmp_path, keep_last=10, current_iter=3)

        for k in range(1, 4):
            assert (tmp_path / f"alpha_dataset_iter_{k}.pt").exists()

    def test_noop_when_keep_last_zero_and_iter_zero(self, tmp_path):
        from rl.utils import cleanup_old_datasets
        # Should not raise even with nothing to clean
        cleanup_old_datasets(tmp_path, keep_last=0, current_iter=0)
