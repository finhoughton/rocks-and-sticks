// Iterative-deepening alpha-beta (PVS) for Rocks and Sticks.
//
// Game-specific points:
//  * Passing is legal, so the null move is a real move and null-move pruning has no zugzwang
//    problem. Two passes in a row are scored with the static evaluation.
//  * "Scoring closures" (sticks that complete a polygon) are the captures of this game; the
//    quiescence search resolves them, and a player facing a closure only the opponent can
//    place is treated like being in check (no stand-pat, depth extension).
//  * When player 0 has scored, player 1 must answer with a closure at once or lose, so those
//    nodes only consider closures.
#pragma once

#include <atomic>
#include <chrono>
#include <memory>
#include <cmath>
#include <random>
#include <string>

#include "rs_eval.hpp"

namespace rs
{

constexpr int MATE = 30000;
constexpr int MATE_BOUND = MATE - 512;
constexpr int MAX_PLY = 96;
constexpr int INF = 32000;

struct TTEntry
{
    Move move = MOVE_NONE;
    std::int16_t value = 0;
    std::int8_t depth = -1;
    std::uint8_t flag = 0; // 0 empty, 1 exact, 2 lower, 3 upper
};

// Transposition table shared by all search threads without locks: each slot stores the entry
// and key XOR entry, so a torn write just fails verification.
class TransTable
{
  public:
    explicit TransTable(int bits) : n((size_t)1 << bits), slots(new Slot[(size_t)1 << bits]) {}

    bool probe(std::uint64_t key, TTEntry &out) const
    {
        const Slot &sl = slots[key & (n - 1)];
        std::uint64_t d = sl.data.load(std::memory_order_relaxed);
        std::uint64_t kx = sl.kx.load(std::memory_order_relaxed);
        if ((kx ^ d) != key)
            return false;
        std::memcpy(&out, &d, sizeof(out));
        return out.flag != 0;
    }

    void store(std::uint64_t key, const TTEntry &e)
    {
        std::uint64_t d = 0;
        std::memcpy(&d, &e, sizeof(e));
        Slot &sl = slots[key & (n - 1)];
        sl.data.store(d, std::memory_order_relaxed);
        sl.kx.store(key ^ d, std::memory_order_relaxed);
    }

    void clear()
    {
        for (size_t i = 0; i < n; ++i)
        {
            slots[i].data.store(0, std::memory_order_relaxed);
            slots[i].kx.store(0, std::memory_order_relaxed);
        }
    }

  private:
    struct Slot
    {
        std::atomic<std::uint64_t> kx{0}, data{0};
    };
    size_t n;
    std::unique_ptr<Slot[]> slots;
};
static_assert(sizeof(TTEntry) == 8, "TTEntry must pack into 64 bits");

struct SearchLimits
{
    int max_depth = 64;
    std::int64_t max_nodes = 0; // 0 = unlimited
    int time_ms = 0;            // 0 = unlimited
};

struct SearchResult
{
    Move best = MOVE_PASS;
    int score = 0;
    int depth = 0;
    std::uint64_t nodes = 0;
    std::vector<Move> pv;
    std::vector<std::pair<Move, int>> root_scores; // from the last completed iteration
};

// Search knobs, exposed so variants can be compared in self-play (rs_selfplay --opt/--opt2).
enum SearchOpt
{
    OPT_NMP = 0,        // null-move pruning on/off
    OPT_NMP_R,          // base reduction
    OPT_LMR_X100,       // late-move-reduction divisor x100 (smaller = more reduction); 0 = off
    OPT_THREAT_EXT,     // extend when facing a closure only the opponent can place
    OPT_QEVASION_PLY,   // quiescence plies in which threats must be answered
    OPT_LMR_MIN_MOVE,   // first move index eligible for reduction
    OPT_RFP_MARGIN,     // reverse futility: margin per depth (0 = off)
    OPT_RFP_DEPTH,      // reverse futility applies up to this depth
    N_OPTS
};

inline const char *const OPT_NAMES[N_OPTS] = {"nmp", "nmp_r", "lmr", "threat_ext", "qevasion", "lmr_min", "rfp", "rfp_depth"};

class Searcher
{
  public:
    Weights weights;
    int opt[N_OPTS] = {1, 2, 220, 1, 4, 3, 100, 3};

    bool set_opt(const std::string &name, int value)
    {
        for (int i = 0; i < N_OPTS; ++i)
            if (name == OPT_NAMES[i])
            {
                opt[i] = value;
                return true;
            }
        return false;
    }

    std::uint64_t nodes = 0;
    // Root move selection noise (for self-play variety): pick uniformly among root moves
    // within `root_margin` of the best score when root_margin > 0.
    int root_margin = 0;
    std::mt19937_64 rng{12345};
    std::uint64_t noise_salt = 0;

    // Set by a parallel driver: an external stop flag, and a helper index (0 = main thread)
    // that staggers the depths helpers search.
    std::atomic<bool> *shared_stop = nullptr;
    int helper_id = 0;
    // A request from outside to stop now (ParallelSearcher::request_abort), for every worker.
    std::atomic<bool> *abort_flag = nullptr;

    explicit Searcher(int tt_bits = 20) : tt(std::make_shared<TransTable>(tt_bits)) {}
    explicit Searcher(std::shared_ptr<TransTable> shared) : tt(std::move(shared)) {}

    void clear()
    {
        tt->clear();
        clear_local();
    }

    // Everything except the (possibly shared) transposition table.
    void clear_local()
    {
        std::fill(ec.begin(), ec.end(), ECEntry{});
        std::fill(fc.begin(), fc.end(), FCEntry{});
        std::memset(history, 0, sizeof(history));
    }

    SearchResult search(State &state, const SearchLimits &lim)
    {
        s = &state;
        limits = lim;
        nodes = 0;
        stop = false;
        t_start = std::chrono::steady_clock::now();
        for (auto &k : killers)
            k[0] = k[1] = MOVE_NONE;
        // age history rather than clearing it
        for (auto &h : history)
            for (auto &x : h)
                x /= 4;

        noise_salt = rng();
        SearchResult res;
        std::vector<Move> root_moves;
        s->gen_moves(root_moves, true);
        if (s->winner >= 0 || root_moves.empty())
            return res;
        res.best = root_moves[0];

        std::vector<std::pair<Move, int>> scored;
        for (Move m : root_moves)
            if (move_type(m) != T_ROCK || rock_relevant(move_idx(m)))
                scored.push_back({m, 0});
        order_root(scored);

        int prev_score = 0;
        for (int depth = 1; depth <= limits.max_depth; ++depth)
        {
            // helpers skip some depths so the threads do not all do identical work
            if (helper_id > 0 && depth > 2 && depth < 60 && ((depth + helper_id) % 3 == 0))
                continue;
            int alpha = -INF, beta = INF;
            int window = 40;
            if (depth >= 4 && std::abs(prev_score) < MATE_BOUND)
            {
                alpha = prev_score - window;
                beta = prev_score + window;
            }
            int best_score = -INF;
            Move best_move = MOVE_NONE;
            std::vector<std::pair<Move, int>> iter_scores;
            while (true)
            {
                best_score = -INF;
                best_move = MOVE_NONE;
                iter_scores.clear();
                int a = alpha;
                for (size_t i = 0; i < scored.size(); ++i)
                {
                    Move m = scored[i].first;
                    int v;
                    // Root noise: each move gets a fixed bonus in [0, root_margin), and the search
                    // windows are shifted to match, so the choice is an exact argmax of
                    // (value + bonus) rather than a pick among unverified bounds.
                    const int nz = root_noise(m);
                    bool pass_pass = (move_type(m) == T_PASS && s->last_was_pass());
                    if (pass_pass)
                        v = eval_stm() + nz;
                    else
                    {
                        s->do_move(m);
                        if (i == 0)
                            v = -negamax(depth - 1, -(beta - nz), -(a - nz), 1, true);
                        else
                        {
                            v = -negamax(depth - 1, -(a - nz) - 1, -(a - nz), 1, false);
                            if (v + nz > a && !stop)
                                v = -negamax(depth - 1, -(beta - nz), -(a - nz), 1, true);
                        }
                        s->undo_move();
                        if (std::abs(v) < MATE_BOUND)
                            v += nz;
                    }
                    if (stop)
                        break;
                    iter_scores.push_back({m, v});
                    if (v > best_score)
                    {
                        best_score = v;
                        best_move = m;
                    }
                    if (v > a)
                        a = v;
                    if (a >= beta)
                        break;
                }
                if (stop)
                    break;
                if (best_score <= alpha && alpha > -INF)
                {
                    alpha = (window > 2000) ? -INF : alpha - window * 4;
                    window *= 4;
                    continue;
                }
                if (best_score >= beta && beta < INF)
                {
                    beta = (window > 2000) ? INF : beta + window * 4;
                    window *= 4;
                    // keep the fail-high move first for the re-search
                    promote(scored, best_move);
                    continue;
                }
                break;
            }
            if (stop)
            {
                // A partially searched iteration is still usable if its first move (the previous
                // best) was completed and something beat it.
                if (best_move != MOVE_NONE && !iter_scores.empty() && iter_scores[0].first == res.best && best_score > iter_scores[0].second)
                {
                    res.best = best_move;
                    res.score = best_score;
                }
                break;
            }
            res.best = best_move;
            res.score = best_score;
            res.depth = depth;
            res.root_scores = iter_scores;
            prev_score = best_score;
            promote(scored, best_move);
            // Null-window scores for non-best moves are only bounds; keep ordering stable.
            if (std::abs(best_score) >= MATE_BOUND && depth >= 2)
            {
                // a forced result: no need to look deeper unless we are the losing side
                if (best_score > 0 || depth >= mate_extra_depth(best_score))
                    break;
            }
            if (limits.time_ms > 0 && elapsed_ms() > limits.time_ms * 0.55)
                break;
        }

        res.nodes = nodes;
        res.pv = extract_pv(res.best);
        return res;
    }

    RS_HOT int eval_stm(const std::vector<Closure> *cl = nullptr)
    {
        if (weights.mlp)
        {
            // a network's value depends on everything: cache by the full position key
            const std::uint64_t k = s->key();
            ECEntry &e = ec[k & (ec.size() - 1)];
            if (e.key != k)
            {
                // the board features are shared by every position with these sticks and rocks
                FCEntry &fe = fc[s->hash & (fc.size() - 1)];
                if (fe.key != s->hash)
                {
                    fe.key = s->hash;
                    board_features(*s, cl, fe.f);
                }
                int f[N_FEATS];
                features_from_board(*s, fe.f, f);
                e.key = k;
                e.value[0] = weights.mlp->eval(f);
            }
            return e.value[0];
        }
        // cached by the board alone (rs_eval.hpp board_eval), for both players at once
        const std::uint64_t k = s->hash;
        ECEntry &e = ec[k & (ec.size() - 1)];
        if (e.key != k)
        {
            e.key = k;
            board_eval(*s, weights, cl, e.value);
        }
        return e.value[s->cp] + rest_eval(*s, weights);
    }

    int root_noise(Move m) const
    {
        if (root_margin <= 0)
            return 0;
        return (int)(mix64(noise_salt ^ ((std::uint64_t)m * 0x9E3779B97F4A7C15ULL)) % (std::uint64_t)root_margin);
    }

    // Static helpers usable without a search.
    int static_eval(State &state)
    {
        s = &state;
        return eval_stm();
    }

    int qsearch_value(State &state)
    {
        s = &state;
        stop = false;
        limits = SearchLimits{};
        return qsearch(-INF, INF, 0, 0);
    }

  private:
    State *s = nullptr;
    SearchLimits limits;
    bool stop = false;
    std::chrono::steady_clock::time_point t_start;
    std::shared_ptr<TransTable> tt;

    // Closure areas depend only on the sticks, and most of the tree differs only in rocks or
    // reaches the same stick set by another move order, so cache them by the sticks hash.
    static constexpr int CC_MAX = 40;
    struct CCEntry
    {
        std::uint64_t key = ~0ULL;
        int n = 0;
        std::int32_t v[CC_MAX];
        std::uint8_t d[CC_MAX];
        std::uint16_t area[CC_MAX];
    };
    std::vector<CCEntry> cc = std::vector<CCEntry>(1 << 13);
    struct ECEntry
    {
        std::uint64_t key = ~0ULL;
        int value[2] = {0, 0};
    };
    std::vector<ECEntry> ec = std::vector<ECEntry>(1 << 20);
    // board features by board hash, for the network evaluation (rs_eval.hpp board_features)
    struct FCEntry
    {
        std::uint64_t key = ~0ULL;
        std::int16_t f[N_BOARD_FEATS];
    };
    std::vector<FCEntry> fc = std::vector<FCEntry>(1 << 16);

    RS_HOT void closures(std::vector<Closure> &out)
    {
        CCEntry &e = cc[s->shash & (cc.size() - 1)];
        if (e.key == s->shash)
        {
            out.resize((size_t)e.n);
            for (int i = 0; i < e.n; ++i)
            {
                out[(size_t)i].v = e.v[i];
                out[(size_t)i].d = e.d[i];
                out[(size_t)i].area2 = e.area[i];
            }
        }
        else
        {
            compute_closure_edges(*s, out);
            if ((int)out.size() <= CC_MAX)
            {
                e.key = s->shash;
                e.n = (int)out.size();
                for (int i = 0; i < e.n; ++i)
                {
                    e.v[i] = out[(size_t)i].v;
                    e.d[i] = (std::uint8_t)out[(size_t)i].d;
                    e.area[i] = (std::uint16_t)std::min(out[(size_t)i].area2, 65535);
                }
            }
        }
        finish_closures(*s, out);
    }

    // do_move for a move made at a node whose closure list is `cl`: reuse the known area.
    void make(Move m, const std::vector<Closure> &cl)
    {
        int t = move_type(m);
        if (t < 8)
        {
            int v = move_idx(m), e = v + DOFF[t];
            if (s->sticks[e])
            {
                int cv = v, cd = t;
                if (t >= 4)
                {
                    cv = e;
                    cd = 7 - t;
                }
                for (const Closure &c : cl)
                    if (c.v == cv && c.d == cd)
                    {
                        s->do_move(m, c.area2);
                        return;
                    }
            }
        }
        s->do_move(m);
    }
    Move killers[MAX_PLY][2];
    int history[2][NCELL * 16 / 4]; // indexed by a folded move key
    std::vector<Closure> cl_stack[MAX_PLY + 8];
    std::vector<Move> mv_stack[MAX_PLY + 8];
    std::vector<std::pair<int, Move>> ord_stack[MAX_PLY + 8];

    double elapsed_ms() const
    {
        return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_start).count();
    }

    static int mate_extra_depth(int score)
    {
        // when losing, search a little past the mate distance to look for longer resistance
        return (MATE - std::abs(score)) + 2;
    }

    void check_limits()
    {
        if (limits.max_nodes > 0 && (std::int64_t)nodes >= limits.max_nodes)
            stop = true;
        if (limits.time_ms > 0 && (nodes & 1023) == 0 && elapsed_ms() >= limits.time_ms)
            stop = true;
        if (shared_stop && shared_stop->load(std::memory_order_relaxed))
            stop = true;
        if (abort_flag && abort_flag->load(std::memory_order_relaxed))
            stop = true;
    }

    static size_t hist_index(Move m) { return (size_t)(mix64(m) & (NCELL * 4 - 1)); }

    std::uint64_t node_key() const
    {
        std::uint64_t k = s->key();
        if (s->last_was_pass())
            k ^= 0x9D39247E33776D41ULL;
        return k;
    }

    static int to_tt(int v, int ply)
    {
        if (v >= MATE_BOUND)
            return v + ply;
        if (v <= -MATE_BOUND)
            return v - ply;
        return v;
    }
    static int from_tt(int v, int ply)
    {
        if (v >= MATE_BOUND)
            return v - ply;
        if (v <= -MATE_BOUND)
            return v + ply;
        return v;
    }

    RS_HOT void tt_store(std::uint64_t key, Move m, int value, int depth, int flag, int ply)
    {
        TTEntry e;
        if (tt->probe(key, e))
        {
            if (e.depth > depth && flag != 1)
                return;
            if (m == MOVE_NONE)
                m = e.move;
        }
        e.move = m;
        e.value = (std::int16_t)to_tt(value, ply);
        e.depth = (std::int8_t)std::min(depth, 120);
        e.flag = (std::uint8_t)flag;
        tt->store(key, e);
    }

    static void promote(std::vector<std::pair<Move, int>> &v, Move m)
    {
        for (size_t i = 0; i < v.size(); ++i)
            if (v[i].first == m)
            {
                auto x = v[i];
                v.erase(v.begin() + (long)i);
                v.insert(v.begin(), x);
                return;
            }
    }

    // Cheap static move priority (higher = earlier).
    RS_HOT int move_prior(Move m) const
    {
        int t = move_type(m);
        if (t == T_PASS)
            return -50;
        int v = move_idx(m);
        if (t == T_ROCK)
        {
            if (s->sticks[v])
                return 60 + 10 * State::popcount8(s->sticks[v]);
            int near = 0;
            for (int d = 0; d < 8; ++d)
                if (s->sticks[v + DOFF[d]])
                    near++;
            return 10 * near;
        }
        int e = v + DOFF[t];
        if (s->sticks[e])
            return 40; // joins two existing vertices (spoiler or closure)
        int near = 0;
        for (int d = 0; d < 8; ++d)
            if (s->sticks[e + DOFF[d]])
                near++;
        return 5 * near;
    }

    void order_root(std::vector<std::pair<Move, int>> &scored)
    {
        for (auto &pr : scored)
        {
            int p = move_prior(pr.first);
            if (move_type(pr.first) < 8)
            {
                int v = move_idx(pr.first), d = move_type(pr.first);
                if (s->sticks[v + DOFF[d]] && s->closure_area2(v, d) >= 2)
                    p += 100000;
            }
            pr.second = p;
        }
        std::stable_sort(scored.begin(), scored.end(), [](const auto &a, const auto &b) { return a.second > b.second; });
        TTEntry e;
        if (tt->probe(node_key(), e) && e.move != MOVE_NONE)
            promote(scored, e.move);
    }

    std::vector<Move> extract_pv(Move first)
    {
        std::vector<Move> pv;
        if (first == MOVE_NONE)
            return pv;
        int made = 0;
        Move m = first;
        while (m != MOVE_NONE && made < 24 && s->winner < 0)
        {
            if (!s->move_legal(m, s->cp))
                break;
            pv.push_back(m);
            s->do_move(m);
            made++;
            TTEntry e;
            if (!tt->probe(node_key(), e))
                break;
            m = e.move;
        }
        while (made-- > 0)
            s->undo_move();
        return pv;
    }

    bool rock_relevant(int v) const
    {
        if (s->sticks[v])
            return true;
        for (int d = 0; d < 8; ++d)
            if (s->sticks[v + DOFF[d]])
                return true;
        return false;
    }

    // Can move m possibly answer a scoring closure that only the opponent can place?
    //  * Player 0 must spoil it now (player 1 scoring ends the game): only a stick joining two
    //    existing vertices, or a diagonal (which may cross the closing edge), can do that.
    //  * Player 1 may also build a reply closure, so any stick qualifies.
    // Rocks never help: both ends of the closure are already rocked by the opponent.
    bool is_evasion(Move m, int me) const
    {
        int t = move_type(m);
        if (t >= 8)
            return false;
        if (me == 1)
            return true;
        return IS_DIAG[t] || s->sticks[move_idx(m) + DOFF[t]] != 0;
    }

    // Player 1 to move while behind: the round ends after this move, so only an immediate
    // closure can avoid losing.
    bool must_answer() const { return s->cp == 1 && s->scores[0] > s->scores[1] && s->turn != 0; }

    RS_HOT int qsearch(int alpha, int beta, int ply, int qply)
    {
        if (s->winner >= 0)
            return s->winner == s->cp ? MATE - ply : -(MATE - ply);
        nodes++;
        if ((nodes & 255) == 0)
            check_limits();
        if (stop)
            return 0;
        if (ply >= MAX_PLY - 1)
            return eval_stm();

        const int me = s->cp, opp = 1 - me;
        std::vector<Closure> &cl = cl_stack[ply];
        closures(cl);

        if (must_answer())
        {
            int best = -(MATE - ply - 1);
            for (const Closure &c : cl)
            {
                if (c.pa[me] < 2)
                    continue;
                {
                    Move cm = closure_move(*s, c, me);
                    s->do_move(cm, c.area2);
                }
                int v = -qsearch(-beta, -alpha, ply + 1, qply + 1);
                s->undo_move();
                // cl was overwritten by deeper nodes sharing no storage (different ply), safe
                if (v > best)
                    best = v;
                if (v > alpha)
                    alpha = v;
                if (alpha >= beta)
                    break;
            }
            return best;
        }

        bool have_score = false, threat = false;
        for (const Closure &c : cl)
        {
            if (c.pa[me] >= 2)
                have_score = true;
            else if (c.pa[opp] >= 2)
                threat = true; // only the opponent can score with it
        }
        if (me == 1 && have_score && s->scores[0] == s->scores[1])
            return MATE - ply - 1; // player 1 scores and the round ends

        int best = -INF;
        if (!threat || qply >= opt[OPT_QEVASION_PLY])
        {
            best = eval_stm(&cl);
            if (best >= beta)
                return best;
            if (best > alpha)
                alpha = best;
        }

        // my scoring closures
        if (have_score)
        {
            // copy: deeper plies reuse their own vectors, but keep iteration simple
            for (size_t i = 0; i < cl.size(); ++i)
            {
                const Closure c = cl[i];
                if (c.pa[me] < 2)
                    continue;
                {
                    Move cm = closure_move(*s, c, me);
                    s->do_move(cm, c.area2);
                }
                int v = -qsearch(-beta, -alpha, ply + 1, qply + 1);
                s->undo_move();
                if (stop)
                    return 0;
                if (v > best)
                    best = v;
                if (v > alpha)
                    alpha = v;
                if (alpha >= beta)
                    return best;
            }
        }

        if (threat && qply < opt[OPT_QEVASION_PLY])
        {
            // Only the opponent can place the pending closure, so rocks and passing cannot help.
            std::vector<Move> &mv = mv_stack[ply];
            s->gen_moves(mv, true);
            for (size_t i = 0; i < mv.size(); ++i)
            {
                Move m = mv[i];
                if (!is_evasion(m, me))
                    continue;
                make(m, cl);
                int v = -qsearch(-beta, -alpha, ply + 1, qply + 1);
                s->undo_move();
                if (stop)
                    return 0;
                if (v > best)
                    best = v;
                if (v > alpha)
                    alpha = v;
                if (alpha >= beta)
                    return best;
            }
            if (best == -INF)
                best = -(MATE - ply - 2); // nothing stops the closure
        }
        return best;
    }

    int negamax(int depth, int alpha, int beta, int ply, bool pv)
    {
        if (s->winner >= 0)
            return s->winner == s->cp ? MATE - ply : -(MATE - ply);
        if (depth <= 0)
            return qsearch(alpha, beta, ply, 0);
        nodes++;
        if ((nodes & 255) == 0)
            check_limits();
        if (stop)
            return 0;
        if (ply >= MAX_PLY - 1)
            return eval_stm();

        // mate distance pruning
        alpha = std::max(alpha, -(MATE - ply));
        beta = std::min(beta, MATE - ply - 1);
        if (alpha >= beta)
            return alpha;

        const int me = s->cp, opp = 1 - me;
        const int alpha0 = alpha;
        const std::uint64_t key = node_key();
        TTEntry tte;
        Move tt_move = MOVE_NONE;
        if (tt->probe(key, tte))
        {
            tt_move = tte.move;
            if (!pv && tte.depth >= depth)
            {
                int v = from_tt(tte.value, ply);
                if (tte.flag == 1 || (tte.flag == 2 && v >= beta) || (tte.flag == 3 && v <= alpha))
                    return v;
            }
        }

        std::vector<Closure> &cl = cl_stack[ply];
        closures(cl);

        if (must_answer())
        {
            int best = -(MATE - ply - 1);
            Move best_move = MOVE_NONE;
            for (size_t i = 0; i < cl.size(); ++i)
            {
                const Closure c = cl[i];
                if (c.pa[me] < 2)
                    continue;
                Move m = closure_move(*s, c, me);
                s->do_move(m, c.area2);
                int v = -negamax(depth, -beta, -alpha, ply + 1, pv);
                s->undo_move();
                if (stop)
                    return 0;
                if (v > best)
                {
                    best = v;
                    best_move = m;
                }
                if (v > alpha)
                    alpha = v;
                if (alpha >= beta)
                    break;
            }
            tt_store(key, best_move, best, depth, best >= beta ? 2 : (best > alpha0 ? 1 : 3), ply);
            return best;
        }

        bool have_score = false, threat = false;
        for (const Closure &c : cl)
        {
            if (c.pa[me] >= 2)
                have_score = true;
            else if (c.pa[opp] >= 2)
                threat = true; // only the opponent can score with it
        }
        if (me == 1 && have_score && s->scores[0] == s->scores[1])
            return MATE - ply - 1;

        if (threat && opt[OPT_THREAT_EXT] && ply < 2 * MAX_PLY / 3)
            depth++; // must deal with it now: treat like a check

        const bool prev_pass = s->last_was_pass();
        const int static_eval = eval_stm(&cl);

        // Reverse futility: far enough above beta at shallow depth that a pass already holds it.
        if (opt[OPT_RFP_MARGIN] > 0 && !pv && !threat && depth <= opt[OPT_RFP_DEPTH] && beta > -MATE_BOUND && beta < MATE_BOUND &&
            static_eval - opt[OPT_RFP_MARGIN] * depth >= beta)
            return static_eval;

        // Null move (a real pass) pruning.
        if (opt[OPT_NMP] && !pv && !threat && !prev_pass && depth >= 2 && static_eval >= beta && beta > -MATE_BOUND && beta < MATE_BOUND)
        {
            int R = opt[OPT_NMP_R] + depth / 4;
            s->do_move(MOVE_PASS);
            int v = -negamax(depth - 1 - R, -beta, -beta + 1, ply + 1, false);
            s->undo_move();
            if (stop)
                return 0;
            if (v >= beta)
                return v >= MATE_BOUND ? beta : v;
        }

        // ---- move generation and ordering ----
        std::vector<Move> &mv = mv_stack[ply];
        s->gen_moves(mv, true);
        std::vector<std::pair<int, Move>> &ord = ord_stack[ply];
        ord.clear();
        for (Move m : mv)
        {
            if (threat)
            {
                if (!is_evasion(m, me))
                    continue;
            }
            else if (move_type(m) == T_ROCK && !rock_relevant(move_idx(m)))
                continue; // a rock touching no stick-connected vertex is too slow to matter
            int p;
            if (m == tt_move)
                p = 1 << 30;
            else
            {
                p = move_prior(m) + history[me][hist_index(m)];
                if (m == killers[ply][0])
                    p += 20000;
                else if (m == killers[ply][1])
                    p += 19000;
                int t = move_type(m);
                if (have_score && t < 8)
                {
                    int v = move_idx(m), e = v + DOFF[t];
                    if (s->sticks[e])
                        for (const Closure &c : cl)
                            if (c.area2 >= 2 && ((c.v == v && c.d == t) || (c.v == e && c.d == 7 - t)))
                                p += 1000000 + c.area2;
                }
            }
            ord.push_back({p, m});
        }
        std::sort(ord.begin(), ord.end(), [](const auto &a, const auto &b) { return a.first > b.first; });

        int best = -INF;
        Move best_move = MOVE_NONE;
        int searched = 0;
        for (size_t i = 0; i < ord.size(); ++i)
        {
            const Move m = ord[i].second;
            const int t = move_type(m);
            int v;
            if (t == T_PASS && prev_pass)
            {
                v = static_eval;
            }
            else
            {
                const bool tactical = ord[i].first >= 1000000;
                make(m, cl);
                if (searched == 0)
                {
                    v = -negamax(depth - 1, -beta, -alpha, ply + 1, pv);
                }
                else
                {
                    int r = 0;
                    if (opt[OPT_LMR_X100] > 0 && depth >= 3 && searched >= opt[OPT_LMR_MIN_MOVE] && !tactical && !threat)
                    {
                        r = (int)(0.5 + std::log((double)depth) * std::log((double)searched + 1.0) * 100.0 / (double)opt[OPT_LMR_X100]);
                        if (pv)
                            r -= 1;
                        if (m == killers[ply][0] || m == killers[ply][1])
                            r -= 1;
                        r = std::max(0, std::min(r, depth - 2));
                    }
                    v = -negamax(depth - 1 - r, -alpha - 1, -alpha, ply + 1, false);
                    if (v > alpha && r > 0 && !stop)
                        v = -negamax(depth - 1, -alpha - 1, -alpha, ply + 1, false);
                    if (v > alpha && v < beta && pv && !stop)
                        v = -negamax(depth - 1, -beta, -alpha, ply + 1, true);
                }
                s->undo_move();
            }
            if (stop)
                return 0;
            searched++;
            if (v > best)
            {
                best = v;
                best_move = m;
            }
            if (v > alpha)
                alpha = v;
            if (alpha >= beta)
            {
                if (t != T_PASS)
                {
                    if (killers[ply][0] != m)
                    {
                        killers[ply][1] = killers[ply][0];
                        killers[ply][0] = m;
                    }
                    int &h = history[me][hist_index(m)];
                    h += depth * depth;
                    if (h > 15000)
                        for (auto &x : history[me])
                            x /= 2;
                }
                break;
            }
        }

        if (best == -INF)
            best = -(MATE - ply - 2); // threat node with no evasion at all

        tt_store(key, best_move, best, depth, best >= beta ? 2 : (best > alpha0 ? 1 : 3), ply);
        return best;
    }
};

} // namespace rs
