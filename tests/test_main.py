"""main.py: one way to play, watch, run matches and replay."""

from __future__ import annotations

import json
import warnings

import matplotlib

matplotlib.use("Agg")

import pytest  # noqa: E402

pytest.importorskip("rs_engine")

import main  # noqa: E402
from game import Game  # noqa: E402


@pytest.fixture(autouse=True)
def _sandbox(tmp_path, monkeypatch):
    """Ratings and saved games go to a temporary folder; the Agg canvas cannot be shown."""
    monkeypatch.setenv("RS_RATINGS_FILE", str(tmp_path / "ratings.jsonl"))
    monkeypatch.chdir(tmp_path)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield tmp_path


def _rated(tmp_path) -> list[dict]:
    path = tmp_path / "ratings.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def test_player_arguments_name_bots_and_people() -> None:
    assert main.player_id("rs:5.0s:4t", "you") == "rs:5s:4t"
    assert main.player_id("human", "fin") == "fin" and main.player_id("alice", "you") == "alice"
    with pytest.raises(ValueError):
        main.player_id("rs:5x", "you")


def test_a_match_without_a_window_swaps_sides_and_is_rated(_sandbox, capsys) -> None:
    main.main(["rs:300", "random", "--games", "2", "--no-display", "--seed", "1"])
    games = _rated(_sandbox)
    assert [(g["p1"], g["p2"]) for g in games] == [("rs:300", "random"), ("random", "rs:300")]
    assert "rs:300 scored 2/2 against random" in capsys.readouterr().out
    assert not list(_sandbox.glob("saved_games*"))  # nothing is saved without a window unless asked


def test_a_bot_against_itself_and_unrated_options_record_nothing(_sandbox) -> None:
    main.main(["rs:300", "rs:300", "--no-display"])
    main.main(["rs:300", "random", "--no-display", "--no-rate", "--save-dir", "kept"])
    assert _rated(_sandbox) == []
    assert len(list((_sandbox / "kept").glob("game_*.json"))) == 1


def test_playing_resigning_saving_and_replaying(_sandbox, monkeypatch, capsys) -> None:
    typed = iter(["0 0 S", "resign"])
    monkeypatch.setattr(Game, "wait_for_move_input", lambda self, prompt: next(typed))
    main.main(["rs:300"])  # one name: you against that bot, you move first
    saved = json.loads((_sandbox / "saved_games_human" / "game_00000.json").read_text())
    assert saved["winner"] == 1 and saved["meta"] == {"players": ["you", "rs:300"], "human_player": 0, "bot_spec": "rs:300", "resigned": 0}
    assert len(saved["moves"]) == 2 and len(saved["evals"]) == 2
    assert [(g["p1"], g["p2"], g["winner"]) for g in _rated(_sandbox)] == [("you", "rs:300", 1)]

    main.main(["--replay", "saved_games_human/game_00000.json", "--delay", "0.01"])
    assert "Game over. Winner: Player 2" in capsys.readouterr().out
