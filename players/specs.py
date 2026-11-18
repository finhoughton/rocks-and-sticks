"""Bot specifications: short strings that name a bot configuration exactly.

    random               uniformly random legal moves
    ab:<depth>           the older C++ alpha-beta with its handcrafted evaluation
    mcts:<rollouts>      the older C++ MCTS with heuristic leaf values
    rs:<budget>[:<threads>t][:nobook][:ponder][:v<n>]
                         the rs engine. The budget is time per move (5s, 0.5s, 300ms) or
                         positions searched per move (2000, 100k, 1M). Threads default to 1.
                         :ponder keeps searching while the opponent thinks (for games
                         against people; it needs a time budget). :v<n> plays a frozen
                         older engine version (engine_src/versions).

The older bots take options as extra :key=value parts, for the model-backed variants:

    ab:<depth>[:time=<ms>][:gnn=<checkpoint>][:native=<weights.bin>][:order=<depth>]
    mcts:<rollouts>[:gnn=<checkpoint>][:policy=<checkpoint>]
    pymcts:<rollouts>[:time=<seconds>][:gnn=<checkpoint>]     the Python MCTS
    pyab[:gnn=<checkpoint>]                                   the Python alpha-beta
    ppo:<checkpoint>                                          the PPO policy network

Different spellings of the same configuration canonicalise to the same id (rs:0.5s and
rs:500ms are both "rs:500ms"), which the rating system uses as the bot's name.

Bots whose budget is a node count play at the same strength however busy the computer is;
timed bots get weaker when something else is using the CPU.
"""

from __future__ import annotations

import re

from .base import Player, RandomPlayer

# rs bots with a budget below this many positions use a smaller transposition table
_SMALL_TT_NODES = 1_000_000


def _parse_rs(parts: list[str]) -> dict:
    if not parts:
        raise ValueError("rs needs a budget, e.g. rs:5s or rs:100k")
    budget, opts = parts[0], parts[1:]
    cfg = {"time_ms": 0, "nodes": 0, "threads": 1, "book": True, "version": None, "ponder": False}
    m = re.fullmatch(r"(\d+(?:\.\d+)?)(ms|s)", budget)
    if m:
        ms = float(m.group(1)) * (1 if m.group(2) == "ms" else 1000)
        if ms < 1 or ms != int(ms):
            raise ValueError(f"time budget must be a whole number of milliseconds: {budget}")
        cfg["time_ms"] = int(ms)
    else:
        m = re.fullmatch(r"(\d+(?:\.\d+)?)([kM]?)", budget)
        if not m:
            raise ValueError(f"bad rs budget {budget!r}: use e.g. 5s, 500ms, 100k or 1M")
        n = float(m.group(1)) * {"": 1, "k": 1_000, "M": 1_000_000}[m.group(2)]
        if n < 1 or n != int(n):
            raise ValueError(f"node budget must be a whole number: {budget}")
        cfg["nodes"] = int(n)
    for o in opts:
        if re.fullmatch(r"\d+t", o) and int(o[:-1]) >= 1:
            cfg["threads"] = int(o[:-1])
        elif o == "nobook":
            cfg["book"] = False
        elif o == "ponder":
            cfg["ponder"] = True
        elif re.fullmatch(r"v\d+", o):
            cfg["version"] = o[1:]
        else:
            raise ValueError(f"unknown rs option {o!r}: use <n>t, nobook, ponder or v<n>")
    if cfg["ponder"] and not cfg["time_ms"]:
        raise ValueError("ponder needs a time budget (e.g. rs:5s:4t:ponder): with a node budget it changes nothing")
    from .rs_player import ENGINE_VERSION

    if cfg["version"] == ENGINE_VERSION:
        cfg["version"] = None  # the current version needs no tag
    return cfg


def _fmt_time(ms: int) -> str:
    if ms % 1000 == 0:
        return f"{ms // 1000}s"
    if ms < 1000:
        return f"{ms}ms"
    return f"{ms / 1000:g}s"


def _fmt_nodes(n: int) -> str:
    if n % 1_000_000 == 0:
        return f"{n // 1_000_000}M"
    if n % 1000 == 0:
        return f"{n // 1000}k"
    return str(n)


# options the older bots accept, by kind
_OLD_OPTS = {"ab": {"time", "gnn", "native", "order"}, "mcts": {"gnn", "policy"}, "pymcts": {"time", "gnn"}, "pyab": {"gnn"}}


def _parse_old(parts: list[str]) -> tuple[int, dict[str, str]]:
    """(number, options) of an older bot's spec: ab:3, mcts:3000:gnn=x.pt, pyab, ..."""
    kind, rest = parts[0], parts[1:]
    n = 0
    if kind != "pyab":
        if not rest or not rest[0].isdigit() or int(rest[0]) <= 0:
            raise ValueError(f"{kind} needs a positive number, e.g. {kind}:3")
        n, rest = int(rest[0]), rest[1:]
    opts = dict(o.split("=", 1) for o in rest if "=" in o)
    if len(opts) != len(rest) or not set(opts) <= _OLD_OPTS[kind]:
        raise ValueError(f"bad options for {kind}: use {', '.join(sorted(_OLD_OPTS[kind]))} as key=value")
    return n, opts


def canonical(spec: str) -> str:
    """The canonical id of a bot spec. Raises ValueError if it is not one."""
    parts = spec.strip().split(":")
    kind = parts[0]
    if kind == "random" and len(parts) == 1:
        return "random"
    if kind in _OLD_OPTS:
        n, opts = _parse_old(parts)
        return ":".join([kind] + ([str(n)] if n else []) + [f"{k}={v}" for k, v in sorted(opts.items())])
    if kind == "ppo" and len(parts) >= 2 and parts[1]:
        return spec.strip()
    if kind == "rs":
        cfg = _parse_rs(parts[1:])
        out = "rs:" + (_fmt_time(cfg["time_ms"]) if cfg["time_ms"] else _fmt_nodes(cfg["nodes"]))
        if cfg["threads"] != 1:
            out += f":{cfg['threads']}t"
        if not cfg["book"]:
            out += ":nobook"
        if cfg["ponder"]:
            out += ":ponder"
        if cfg["version"]:
            out += f":v{cfg['version']}"
        return out
    raise ValueError(f"not a bot spec: {spec!r}")


def is_bot_spec(spec: str) -> bool:
    try:
        canonical(spec)
        return True
    except ValueError:
        return False


def version(spec: str) -> str | None:
    """Engine version for a bot id. Results from other versions are rated separately."""
    parts = spec.split(":")
    if parts[0] == "rs":
        from .rs_player import ENGINE_VERSION

        return _parse_rs(parts[1:])["version"] or ENGINE_VERSION
    return None


def describe(spec: str) -> str:
    """A short human-readable description of a bot id (or 'human' for anything else)."""
    try:
        spec = canonical(spec)
    except ValueError:
        return "human"
    parts = spec.split(":")
    if parts[0] == "random":
        return "random moves"
    if parts[0] in _OLD_OPTS or parts[0] == "ppo":
        names = {"ab": "old alpha-beta, depth {}", "mcts": "old MCTS, {} rollouts", "pymcts": "Python MCTS, {} rollouts", "pyab": "Python alpha-beta", "ppo": "PPO policy"}
        extra = ", " + ", ".join(parts[2:]) if parts[0] != "ppo" and len(parts) > 2 else ""
        return names[parts[0]].format(parts[1] if len(parts) > 1 else "") + extra
    cfg = _parse_rs(parts[1:])
    if cfg["time_ms"]:
        budget = f"{cfg['time_ms'] / 1000:g} s/move"
    else:
        budget = f"{_fmt_nodes(cfg['nodes'])} positions/move"
    extra = f", {cfg['threads']} threads" if cfg["threads"] != 1 else ""
    extra += "" if cfg["book"] else ", no book"
    extra += ", thinks on your time" if cfg["ponder"] else ""
    if cfg["version"]:
        extra += f", version {cfg['version']}"
    return f"rs engine, {budget}{extra}"


def needs_cpp_state(spec: str) -> bool:
    """The older C++ bots need the game wrapped in players.game_total.GameTotal."""
    return spec.split(":")[0] in ("ab", "mcts")


def _make_old_bot(parts: list[str], number: int, seed: int | None) -> Player:
    """The bots that predate the rs engine, with or without a trained model."""
    kind = parts[0]
    if kind == "ppo":
        import torch

        from gnn.encode import SAMPLE_ENC
        from rl.PPO import PPOGNNPolicy, PPOPlayer

        model = PPOGNNPolicy(node_feat_dim=SAMPLE_ENC.data.x.size(1), global_feat_dim=SAMPLE_ENC.data.global_feats.size(1))  # type: ignore
        model.load_state_dict(torch.load(":".join(parts[1:]), map_location="cpu"))
        model.eval()
        return PPOPlayer(number, model, device="cpu")
    n, opts = _parse_old(parts)
    gnn = opts.get("gnn")
    if kind == "ab":
        from .alphabeta_cpp import AlphaBetaPlayerCPP

        ab = AlphaBetaPlayerCPP(number, seed=seed or 0, depth=n, use_heuristic=gnn is None, time_limit_ms=int(opts.get("time", 0)),
                                native_model=opts.get("native", ""), nn_ordering_depth=int(opts.get("order", 2)))
        if gnn:
            ab.set_model_checkpoint(gnn, device="cpu")
        return ab
    if kind == "mcts":
        from .mcts_cpp import MCTSPlayerCPP

        if gnn:
            p = MCTSPlayerCPP(number, seed=seed or 0, n_rollouts=n, use_nn_value=True)
            p.set_model_checkpoint(gnn, device="cpu")
            if "policy" in opts:
                p.set_policy_checkpoint(opts["policy"], device="cpu")
                p.engine.set_prior_params(0.04, 1.0)
        else:
            p = MCTSPlayerCPP(number, seed=seed or 0, n_rollouts=n, use_nn_value=False, use_heuristic_rollout=True)
        p.set_exploration(dirichlet_alpha=0.0, dirichlet_epsilon=0.0, temperature=0.0, temperature_moves=0)
        return p
    # the Python bots read a globally loaded GNN evaluator
    if gnn:
        from gnn.encode import SAMPLE_ENC
        from gnn.model import load_model

        load_model(gnn, SAMPLE_ENC.data.x.size(1), SAMPLE_ENC.data.global_feats.size(1), device="cpu")  # type: ignore
    if kind == "pymcts":
        from .mcts import MCTSPlayer

        return MCTSPlayer(number, seed=seed, n_rollouts=n, time_limit=float(opts["time"]) if "time" in opts else None,
                          use_gnn=bool(gnn), check_forced_losses=not gnn)
    from .ai import AlphaBetaPlayer

    return AlphaBetaPlayer(number, use_gnn=bool(gnn))


def make_bot(spec: str, number: int, seed: int | None = None, verbose: bool = False) -> Player:
    """Build the bot a spec names, playing as player `number` (0 or 1).

    Old bots (ab, mcts) need the game wrapped in players.game_total.GameTotal.
    """
    spec = canonical(spec)
    parts = spec.split(":")
    kind = parts[0]
    if seed is not None:
        seed %= 2**31 - 1  # the C++ engines take 32-bit seeds
    if kind == "random":
        return RandomPlayer(number, seed=seed)
    if kind in _OLD_OPTS or kind == "ppo":
        return _make_old_bot(parts, number, seed)
    from .rs_player import RSPlayer

    cfg = _parse_rs(parts[1:])
    if cfg["time_ms"]:
        tt_bits = 23 if cfg["threads"] > 1 else 22
    else:
        tt_bits = 20 if cfg["nodes"] < _SMALL_TT_NODES else 22
    return RSPlayer(
        number,
        time_ms=cfg["time_ms"],
        max_nodes=cfg["nodes"],
        threads=cfg["threads"],
        tt_bits=tt_bits,
        # a little root noise so that repeated games differ; far too small to change strength
        root_margin=8,
        seed=seed,
        verbose=verbose,
        book=None if cfg["book"] else "",
        version=cfg["version"],
        ponder=cfg["ponder"],
    )
