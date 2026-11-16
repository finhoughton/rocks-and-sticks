"""Tests for AlphaBetaPlayerCPP with native NN model."""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from gnn.encode import SAMPLE_ENC
from gnn.model import GNNEval


def _make_native_model(path: Path, seed: int = 0) -> Path:
    from rl.utils import export_native_weights

    pt_path = path.with_suffix(".pt")
    torch.manual_seed(seed)
    node_dim = int(SAMPLE_ENC.data.x.size(1))  # type: ignore[union-attr]
    global_dim = int(SAMPLE_ENC.data.global_feats.size(1))  # type: ignore[union-attr]
    model = GNNEval(node_feat_dim=node_dim, global_feat_dim=global_dim)
    torch.save(model.state_dict(), pt_path)

    bin_path = path.with_suffix(".bin")
    export_native_weights(pt_path, bin_path)
    return bin_path


def test_alphabeta_player_loads_native_model(tmp_path):
    pytest.importorskip("players_ext")
    from players.alphabeta_cpp import AlphaBetaPlayerCPP

    bin_path = _make_native_model(tmp_path / "model")

    # Should not raise
    player = AlphaBetaPlayerCPP(
        0,
        seed=42,
        depth=2,
        use_heuristic=True,
        native_model=str(bin_path),
    )
    assert player is not None


def test_alphabeta_player_rejects_pt_file(tmp_path):
    """Passing a .pt file as native_model should fail (not NNWT format)."""
    pytest.importorskip("players_ext")
    from players.alphabeta_cpp import AlphaBetaPlayerCPP

    pt_path = tmp_path / "model.pt"
    torch.manual_seed(0)
    node_dim = int(SAMPLE_ENC.data.x.size(1))  # type: ignore[union-attr]
    global_dim = int(SAMPLE_ENC.data.global_feats.size(1))  # type: ignore[union-attr]
    model = GNNEval(node_feat_dim=node_dim, global_feat_dim=global_dim)
    torch.save(model.state_dict(), pt_path)

    with pytest.raises(RuntimeError, match="Failed to load native NN weights"):
        AlphaBetaPlayerCPP(
            0,
            seed=42,
            depth=2,
            use_heuristic=True,
            native_model=str(pt_path),
        )


def test_alphabeta_player_rejects_nonexistent_file(tmp_path):
    pytest.importorskip("players_ext")
    from players.alphabeta_cpp import AlphaBetaPlayerCPP

    with pytest.raises(RuntimeError, match="Failed to load native NN weights"):
        AlphaBetaPlayerCPP(
            0,
            seed=42,
            depth=2,
            use_heuristic=True,
            native_model=str(tmp_path / "no_such.bin"),
        )


def test_alphabeta_player_plays_game_with_native_model(tmp_path):
    """AB player with native model can play a short game to completion."""
    pytest.importorskip("players_ext")
    import players_ext

    from game import Game
    from players.alphabeta_cpp import AlphaBetaPlayerCPP
    from players.base import RandomPlayer
    from players.game_total import GameTotal

    bin_path = _make_native_model(tmp_path / "model")

    ab = AlphaBetaPlayerCPP(
        0,
        seed=42,
        depth=2,
        move_cap=48,
        use_heuristic=True,
        native_model=str(bin_path),
        nn_ordering_depth=2,
    )
    rnd = RandomPlayer(1, seed=123)
    players = [ab, rnd]

    game = GameTotal(Game(players=players), players_ext.GameState())

    moves_played = 0
    while game.winner is None and moves_played < 40:
        cp = game.current_player
        mv = game.players[cp].get_move(game)
        game.do_move(cp, mv)
        moves_played += 1

    # Game should have progressed
    assert moves_played > 0
