// Position analysis and static evaluation.
//
// The evaluation works on a small set of features: every minimal scoring polygon (area 1: unit
// square, base-2 triangle, parallelogram) that already has at least one stick, bucketed by how
// many sticks it still needs and how many of its corners each player has rocked; adjacent rock
// pairs by how close they are to an unstoppable closure; rocks in hand; side to move. The
// features are combined either linearly (weights.txt, tools/tune_rs_eval.py) or by a small
// neural network (eval_mlp.txt, tools/train_rs_mlp.py) when one is loaded.
#pragma once

#include <cmath>
#include <fstream>
#include <memory>
#include <string>

#include "rs_state.hpp"

namespace rs
{

struct Closure
{
    int v, d;       // canonical edge (d in 0..3)
    int area2;      // raw smallest-polygon area x2 (1 = half triangle: no score)
    bool can[2];    // can each player place it right now
    int pa[2];      // raw area each player gets from it (0 if they cannot place it)
};

// ---- minimal shapes ----

struct ShapeDef
{
    int vx[4], vy[4]; // corners in cyclic order
    int nchord;
    int ca[2], cb[2]; // chords as corner index pairs
};

constexpr int N_SHAPES = 9;
constexpr ShapeDef SHAPES[N_SHAPES] = {
    {{0, 1, 1, 0}, {0, 0, 1, 1}, 2, {0, 1}, {2, 3}},   // unit square
    {{0, 1, 2, 1}, {0, 0, 0, 1}, 1, {1, 0}, {3, 0}},   // triangle, base horizontal, apex up
    {{0, 1, 2, 1}, {0, 0, 0, -1}, 1, {1, 0}, {3, 0}},  // apex down
    {{0, 0, 0, 1}, {0, 1, 2, 1}, 1, {1, 0}, {3, 0}},   // base vertical, apex right
    {{0, 0, 0, -1}, {0, 1, 2, 1}, 1, {1, 0}, {3, 0}},  // apex left
    {{0, 1, 2, 1}, {0, 0, 1, 1}, 1, {1, 0}, {3, 0}},   // parallelograms
    {{0, 1, 0, -1}, {0, 0, 1, 1}, 1, {0, 0}, {2, 0}},
    {{0, 1, 1, 0}, {0, 1, 2, 1}, 1, {1, 0}, {3, 0}},
    {{0, 1, 1, 0}, {0, -1, 0, 1}, 1, {0, 0}, {2, 0}},
};

struct ShapeEdgeRef
{
    int shape;   // shape type
    int origin;  // idx offset from the canonical edge vertex to the shape origin
    int k;       // which shape edge this is
};

struct ShapeTables
{
    int voff[N_SHAPES][4];        // corner idx offsets from origin
    int edir[N_SHAPES][4];        // direction of edge k (corner k -> corner k+1)
    int chord_dir[N_SHAPES][2];   // direction of chord (corner ca -> corner cb)
    std::vector<ShapeEdgeRef> by_dir[4];

    static int dir_from_delta(int dx, int dy)
    {
        for (int d = 0; d < 8; ++d)
            if (DX[d] == dx && DY[d] == dy)
                return d;
        return -1;
    }

    ShapeTables()
    {
        for (int s = 0; s < N_SHAPES; ++s)
        {
            const ShapeDef &sd = SHAPES[s];
            for (int k = 0; k < 4; ++k)
                voff[s][k] = sd.vy[k] * GRID + sd.vx[k];
            for (int k = 0; k < 4; ++k)
            {
                int a = k, b = (k + 1) & 3;
                int d = dir_from_delta(sd.vx[b] - sd.vx[a], sd.vy[b] - sd.vy[a]);
                edir[s][k] = d;
                int cv = a, cd = d;
                if (d >= 4)
                {
                    cv = b;
                    cd = 7 - d;
                }
                by_dir[cd].push_back({s, -voff[s][cv], k});
            }
            for (int c = 0; c < sd.nchord; ++c)
                chord_dir[s][c] = dir_from_delta(sd.vx[sd.cb[c]] - sd.vx[sd.ca[c]], sd.vy[sd.cb[c]] - sd.vy[sd.ca[c]]);
        }
    }
};

inline const ShapeTables &shape_tables()
{
    static const ShapeTables t;
    return t;
}

// All legal, not-yet-placed edges between two connected vertices with the raw area (x2) of the
// smallest polygon each would close. Depends on the sticks only, not on rocks.
RS_HOT inline void compute_closure_edges(const State &s, std::vector<Closure> &out)
{
    out.clear();
    State &ms = const_cast<State &>(s);
    const bool claims = s.may_claim();
    for (int v : s.conn)
    {
        std::uint8_t mask = s.sticks[v];
        for (int d = 0; d < 4; ++d)
        {
            if ((mask >> d) & 1)
                continue;
            int e = v + DOFF[d];
            if (!s.sticks[e])
                continue;
            if (IS_DIAG[d] && s.diag[State::diag_cell(v, d)])
                continue;
            if (claims && (s.claimed(v) || s.claimed(e)))
                continue;
            Closure c;
            c.v = v;
            c.d = d;
            c.can[0] = c.can[1] = false;
            c.pa[0] = c.pa[1] = 0;
            ms.sticks[v] |= (std::uint8_t)(1u << d);
            ms.sticks[e] |= (std::uint8_t)(1u << (7 - d));
            c.area2 = s.cycle_area2_placed(v, e);
            ms.sticks[v] &= (std::uint8_t) ~(1u << d);
            ms.sticks[e] &= (std::uint8_t) ~(1u << (7 - d));
            out.push_back(c);
        }
    }
}

// Fill in who may place each closure edge and drop the ones nobody can.
RS_HOT inline void finish_closures(const State &s, std::vector<Closure> &cl)
{
    size_t n = 0;
    for (size_t i = 0; i < cl.size(); ++i)
    {
        Closure c = cl[i];
        int e = c.v + DOFF[c.d];
        for (int p = 0; p < 2; ++p)
        {
            c.can[p] = s.can_place(c.v, p) || s.can_place(e, p);
            c.pa[p] = c.can[p] ? c.area2 : 0;
        }
        if (c.can[0] || c.can[1])
            cl[n++] = c;
    }
    cl.resize(n);
}

inline void compute_closures(const State &s, std::vector<Closure> &out)
{
    compute_closure_edges(s, out);
    finish_closures(s, out);
}

// ---- features ----
//
// Two families, both counted separately for the side to move ("mine") and the opponent:
//
//  * Rock pairs. A stick whose two ends are both rocked by one player can only ever be placed
//    by that player, so an adjacent pair of own rocks is the unit of attack: once the pair's
//    two vertices are joined through the stick graph (by a cycle of area >= 1) the owner
//    closes it and nobody can stop them. Pairs are bucketed by how close they are to that.
//  * Minimal shapes (area-1 polygons with at least one stick), bucketed by missing sticks and
//    rocked corners; the difference mine - theirs is used.
constexpr int N_ROCK_BUCKETS = 15;
constexpr int N_SHAPE_FEATS = 3 * N_ROCK_BUCKETS;

enum PairFeat
{
    PF_EXCL_SCORE = 0,   // both ends connected, closing scores: wins when on move
    PF_ARMED_GOOD,       // one end connected, a safe connecting stick exists, opponent cannot pre-spoil
    PF_ARMED_GOOD_KILL,  // same but the opponent can spoil with one stick first
    PF_ARMED_WEAK,       // one end connected, no safe connecting stick yet, cannot be pre-spoiled
    PF_ARMED_WEAK_KILL,
    PF_PAIR_FREE,        // neither end connected
    PF_POT_GOOD,         // own rock + free neighbour that would make an ARMED_GOOD pair
    PF_POT_GOOD_KILL,
    PF_POT_WEAK,
    N_PAIR_FEATS
};

constexpr int F_PAIR_MINE = N_SHAPE_FEATS;
constexpr int F_PAIR_THEIRS = F_PAIR_MINE + N_PAIR_FEATS;
constexpr int F_HAND_MINE = F_PAIR_THEIRS + N_PAIR_FEATS; // rocks I may still place before a stick
constexpr int F_HAND_THEIRS = F_HAND_MINE + 1;
constexpr int F_ROCK_CONN = F_HAND_THEIRS + 1;            // rocks on connected vertices (mine - theirs)
constexpr int F_ROCK_FREE = F_ROCK_CONN + 1;              // rocks on unconnected vertices (mine - theirs)
constexpr int F_TEMPO = F_ROCK_FREE + 1;                  // constant 1: side to move
constexpr int F_SECOND = F_TEMPO + 1;                     // +1 if side to move is player 1 else -1
constexpr int N_FEATS = F_SECOND + 1;

inline int rock_bucket(int mine, int theirs)
{
    // triangular index over mine + theirs <= 4
    static const int base[5] = {0, 5, 9, 12, 14};
    return base[mine] + theirs;
}

// A small fully connected network: N_FEATS inputs, ReLU hidden layers, one output (the win
// logit for the side to move), evaluated in float. Weights are stored [input][output] so that
// each layer is a sum of scaled rows, which vectorises, and zero inputs (most of them, on the
// small boards of real games) cost nothing.
struct Mlp
{
    static constexpr int MAX_WIDTH = 256;
    std::vector<int> sizes;                // N_FEATS, hidden..., 1
    std::vector<std::vector<float>> W, b;  // per layer
    float scale = 200.0f;                  // eval units per logit

    // out[0..n) += x * w[0..n): the inner loop of every layer (vectorised; no aliasing)
    static inline void axpy(float *__restrict out, const float *__restrict w, float x, int n)
    {
        for (int j = 0; j < n; ++j)
            out[j] += x * w[j];
    }

    RS_HOT int eval(const int *f) const
    {
        alignas(64) float a[MAX_WIDTH];
        alignas(64) float c[MAX_WIDTH];
        const int L = (int)sizes.size() - 1;
        int n_in = sizes[0], n_out = sizes[1];
        std::memcpy(a, b[0].data(), sizeof(float) * (size_t)n_out);
        const float *W0 = W[0].data();
        for (int i = 0; i < n_in; ++i)
            if (f[i])
                axpy(a, W0 + (size_t)i * (size_t)n_out, (float)f[i], n_out);
        for (int l = 1; l < L; ++l)
        {
            n_in = sizes[(size_t)l];
            n_out = sizes[(size_t)l + 1];
            std::memcpy(c, b[(size_t)l].data(), sizeof(float) * (size_t)n_out);
            const float *Wl = W[(size_t)l].data();
            for (int i = 0; i < n_in; ++i)
                if (a[i] > 0.0f) // ReLU
                    axpy(c, Wl + (size_t)i * (size_t)n_out, a[i], n_out);
            std::memcpy(a, c, sizeof(float) * (size_t)n_out);
        }
        const float v = a[0] * scale;
        return (int)std::lrint(std::max(-20000.0f, std::min(20000.0f, v)));
    }

    // File: "mlp" then the layer sizes, then per layer its weights ([input][output]) and biases.
    bool load(std::istream &in)
    {
        int n_layers = 0;
        if (!(in >> n_layers) || n_layers < 2 || n_layers > 8)
            return false;
        sizes.assign((size_t)n_layers, 0);
        for (int &x : sizes)
            if (!(in >> x) || x < 1 || x > MAX_WIDTH)
                return false;
        if (sizes.front() != N_FEATS || sizes.back() != 1)
            return false;
        W.assign((size_t)n_layers - 1, {});
        b.assign((size_t)n_layers - 1, {});
        for (int l = 0; l + 1 < n_layers; ++l)
        {
            W[(size_t)l].resize((size_t)sizes[(size_t)l] * (size_t)sizes[(size_t)l + 1]);
            b[(size_t)l].resize((size_t)sizes[(size_t)l + 1]);
            // read as double: a stream refuses floats below the normal range
            double x;
            for (float &dst : W[(size_t)l])
            {
                if (!(in >> x))
                    return false;
                dst = (float)x;
            }
            for (float &dst : b[(size_t)l])
            {
                if (!(in >> x))
                    return false;
                dst = (float)x;
            }
        }
        return true;
    }
};

struct Weights
{
    int w[N_FEATS];
    std::shared_ptr<const Mlp> mlp; // when set, the evaluation uses it instead of w
    Weights()
    {
        // Hand-set starting point; replaced by tuned values via load().
        for (int i = 0; i < N_FEATS; ++i)
            w[i] = 0;
        for (int m = 1; m <= 3; ++m)
            for (int a = 0; a <= 4; ++a)
                for (int b = 0; a + b <= 4; ++b)
                {
                    double base = (m == 1) ? 10.0 : (m == 2) ? 6.0 : 2.0;
                    double v = base * (1.0 + 0.8 * a) / (1.0 + 1.0 * b);
                    w[(m - 1) * N_ROCK_BUCKETS + rock_bucket(a, b)] = (int)v;
                }
        static const int pm[N_PAIR_FEATS] = {900, 500, 120, 90, 40, 20, 110, 40, 15};
        static const int pt[N_PAIR_FEATS] = {-700, -250, -60, -60, -25, -15, -70, -25, -10};
        for (int i = 0; i < N_PAIR_FEATS; ++i)
        {
            w[F_PAIR_MINE + i] = pm[i];
            w[F_PAIR_THEIRS + i] = pt[i];
        }
        w[F_HAND_MINE] = 12;
        w[F_HAND_THEIRS] = -10;
        w[F_ROCK_CONN] = 10;
        w[F_ROCK_FREE] = 3;
        w[F_TEMPO] = 10;
        w[F_SECOND] = 0;
    }
};

// The direction d with DOFF[d] == off, or -1; off may be anything within two steps.
inline int dir_of_offset(int off)
{
    struct Table
    {
        std::int8_t t[4 * GRID + 5];
        constexpr Table() : t()
        {
            for (int i = 0; i < 4 * GRID + 5; ++i)
                t[i] = -1;
            for (int d = 0; d < 8; ++d)
                t[DOFF[d] + 2 * GRID + 2] = (std::int8_t)d;
        }
    };
    static constexpr Table table{};
    const unsigned i = (unsigned)(off + 2 * GRID + 2);
    return i < (unsigned)(4 * GRID + 5) ? table.t[i] : -1;
}

// Classify the potential exclusive edge between `a` (connected) and `b` (not connected) for
// player x, assuming both are (or will be) rocked by x.
//   good: x has a connecting stick c->b after which closing a-b scores and cannot be spoiled
//   kill: the opponent can spoil right now with one stick (w->b where w-a is a stick)
RS_HOT inline void classify_pair(const State &s, int a, int b, int x, bool &good, bool &kill)
{
    good = false;
    kill = false;
    const int y = 1 - x;
    for (int d = 0; d < 8; ++d)
    {
        int c = b + DOFF[d];
        if (c == a || !s.sticks[c])
            continue;
        // can the edge c-b be placed at all?
        if (IS_DIAG[d] && s.diag[State::diag_cell(b, d)])
            continue;
        const int dir_ca = dir_of_offset(a - c);
        const bool c_adj_a = dir_ca >= 0;
        const bool c_stick_a = c_adj_a && s.has_stick(c, dir_ca);
        if (c_stick_a)
        {
            // c is a stick-neighbour of a: joining it to b makes a half triangle (worthless),
            // which is exactly how the opponent spoils the pair.
            if (s.rock[c] != x)
                kill = true;
            continue;
        }
        if (s.rock[c] == y)
            continue; // x cannot start a stick from c
        if (c_adj_a && s.rock[c] != x)
        {
            // after c->b the opponent answers c->a (half triangle) unless that edge is blocked
            bool blocked = IS_DIAG[dir_ca] && s.diag[State::diag_cell(c, dir_ca)];
            if (!blocked)
                continue;
        }
        good = true;
    }
}

// Features are counted from the point of view of player `me` (normally the side to move).
// `cl`, if given, is the position's closure list (saves a cycle search for rock pairs whose
// ends are both connected).
RS_HOT inline void shape_features(const State &s, int *f, int me)
{
    const ShapeTables &T = shape_tables();
    const int opp = 1 - me;

    for (int v : s.conn)
    {
        std::uint8_t mask = s.sticks[v];
        for (int d = 0; d < 4; ++d)
        {
            if (!((mask >> d) & 1))
                continue;
            for (const ShapeEdgeRef &ref : T.by_dir[d])
            {
                const int sh = ref.shape;
                const int o = v + ref.origin;
                const int *vo = T.voff[sh];
                // count each instance once: from its first present edge
                bool counted = false;
                for (int k = 0; k < ref.k; ++k)
                    if (s.has_stick(o + vo[k], T.edir[sh][k]))
                    {
                        counted = true;
                        break;
                    }
                if (counted)
                    continue;
                // dead if a chord is already there
                const ShapeDef &sd = SHAPES[sh];
                bool dead = false;
                for (int c = 0; c < sd.nchord; ++c)
                    if (s.has_stick(o + vo[sd.ca[c]], T.chord_dir[sh][c]))
                        dead = true;
                if (dead)
                    continue;
                int present = 0;
                bool alive[2] = {true, true};
                for (int k = 0; k < 4; ++k)
                {
                    int a = o + vo[k];
                    int ed = T.edir[sh][k];
                    if (s.has_stick(a, ed))
                    {
                        present++;
                        continue;
                    }
                    if (IS_DIAG[ed] && s.diag[State::diag_cell(a, ed)])
                    {
                        dead = true;
                        break;
                    }
                    int b = a + DOFF[ed];
                    // an edge with both ends rocked by one player can never be placed by the other
                    if (s.rock[a] >= 0 && s.rock[a] == s.rock[b])
                        alive[1 - s.rock[a]] = false;
                }
                if (dead || present >= 4)
                    continue;
                int r[2] = {0, 0};
                for (int k = 0; k < 4; ++k)
                {
                    int owner = s.rock[o + vo[k]];
                    if (owner >= 0)
                        r[owner]++;
                }
                int missing = 4 - present;
                if (alive[me])
                    f[(missing - 1) * N_ROCK_BUCKETS + rock_bucket(r[me], r[opp])]++;
                if (alive[opp])
                    f[(missing - 1) * N_ROCK_BUCKETS + rock_bucket(r[opp], r[me])]--;
            }
        }
    }

}

RS_HOT inline void pair_features(const State &s, int *f, const std::vector<Closure> *cl, int me)
{
    // ---- rock pairs ----
    for (int a : s.rocks)
    {
        const int x = s.rock[a];
        int *pf = f + (x == me ? F_PAIR_MINE : F_PAIR_THEIRS);
        const bool ca = s.sticks[a] != 0;
        int sign = (x == me) ? 1 : -1;
        if (ca)
            f[F_ROCK_CONN] += sign;
        else
            f[F_ROCK_FREE] += sign;
        for (int d = 0; d < 8; ++d)
        {
            int b = a + DOFF[d];
            if (s.has_stick(a, d))
                continue;
            if (IS_DIAG[d] && s.diag[State::diag_cell(a, d)])
                continue;
            const bool cb = s.sticks[b] != 0;
            if (s.rock[b] == x)
            {
                if (b < a)
                    continue; // each pair once
                if (ca && cb)
                {
                    int area = 0;
                    if (cl)
                    {
                        int cv = a, cd = d;
                        if (d >= 4)
                        {
                            cv = b;
                            cd = 7 - d;
                        }
                        for (const Closure &c : *cl)
                            if (c.v == cv && c.d == cd)
                            {
                                area = c.pa[x];
                                break;
                            }
                    }
                    else
                        area = s.closure_area2(a, d);
                    if (area >= 2)
                        pf[PF_EXCL_SCORE]++;
                }
                else if (ca || cb)
                {
                    bool good, kill;
                    classify_pair(s, ca ? a : b, ca ? b : a, x, good, kill);
                    pf[good ? (kill ? PF_ARMED_GOOD_KILL : PF_ARMED_GOOD) : (kill ? PF_ARMED_WEAK_KILL : PF_ARMED_WEAK)]++;
                }
                else
                    pf[PF_PAIR_FREE]++;
            }
            else if (s.rock[b] < 0 && (ca != cb))
            {
                // one more rock (on b) would arm the pair
                bool good, kill;
                classify_pair(s, ca ? a : b, ca ? b : a, x, good, kill);
                pf[good ? (kill ? PF_POT_GOOD_KILL : PF_POT_GOOD) : PF_POT_WEAK]++;
            }
        }
    }

}

inline void compute_features(const State &s, int *f, const std::vector<Closure> *cl = nullptr)
{
    std::memset(f, 0, sizeof(int) * N_FEATS);
    const int me = s.cp, opp = 1 - me;
    shape_features(s, f, me);
    pair_features(s, f, cl, me);
    f[F_HAND_MINE] = s.num_rocks[me];
    f[F_HAND_THEIRS] = s.num_rocks[opp];
    f[F_TEMPO] = 1;
    f[F_SECOND] = (me == 1) ? 1 : -1;
}

inline int evaluate(const State &s, const Weights &w, const std::vector<Closure> *cl = nullptr)
{
    int f[N_FEATS];
    compute_features(s, f, cl);
    if (w.mlp)
        return w.mlp->eval(f);
    int v = 0;
    for (int i = 0; i < N_FEATS; ++i)
        v += f[i] * w.w[i];
    return v;
}

// The evaluation splits into a part that depends only on the sticks and rocks on the board
// (everything but rocks in hand, tempo and side) and a cheap remainder. board_eval gives the
// board part for both players from one feature count: seen from the other player, shape and
// rock counts change sign and the mine/theirs pair counts swap. The search caches it by the
// board hash, so the same board with the other side to move (after a pass) or with different
// rocks in hand needs no recount. board_eval + rest_eval == evaluate, exactly.
inline void board_eval(const State &s, const Weights &w, const std::vector<Closure> *cl, int out[2])
{
    int f[N_FEATS];
    std::memset(f, 0, sizeof(f));
    shape_features(s, f, 0);
    pair_features(s, f, cl, 0);
    int v0 = 0, v1 = 0;
    for (int i = 0; i < N_SHAPE_FEATS; ++i)
    {
        v0 += f[i] * w.w[i];
        v1 -= f[i] * w.w[i];
    }
    for (int i = 0; i < N_PAIR_FEATS; ++i)
    {
        v0 += f[F_PAIR_MINE + i] * w.w[F_PAIR_MINE + i] + f[F_PAIR_THEIRS + i] * w.w[F_PAIR_THEIRS + i];
        v1 += f[F_PAIR_THEIRS + i] * w.w[F_PAIR_MINE + i] + f[F_PAIR_MINE + i] * w.w[F_PAIR_THEIRS + i];
    }
    for (int i : {F_ROCK_CONN, F_ROCK_FREE})
    {
        v0 += f[i] * w.w[i];
        v1 -= f[i] * w.w[i];
    }
    out[0] = v0;
    out[1] = v1;
}

// Load evaluation weights: a network file (starting "mlp") or N_FEATS linear weights.
inline bool load_weights_file(const std::string &path, Weights &w)
{
    std::ifstream in(path);
    if (!in)
        return false;
    std::string first;
    if (!(in >> first))
        return false;
    if (first == "mlp")
    {
        auto m = std::make_shared<Mlp>();
        if (!m->load(in))
            return false;
        w.mlp = m;
        return true;
    }
    std::vector<int> v;
    try
    {
        v.push_back(std::stoi(first));
    }
    catch (...)
    {
        return false;
    }
    int x;
    while (in >> x)
        v.push_back(x);
    if ((int)v.size() != N_FEATS)
        return false;
    for (int i = 0; i < N_FEATS; ++i)
        w.w[i] = v[(size_t)i];
    w.mlp.reset();
    return true;
}

// Board features for player 0 (shapes, rock pairs, rocks on and off the structure: everything
// that depends only on the sticks and rocks), as stored by the search's feature cache, and how
// to turn them into the full feature vector for either side to move.
constexpr int N_BOARD_FEATS = F_PAIR_THEIRS + N_PAIR_FEATS + 2; // + rock conn, rock free

inline void board_features(const State &s, const std::vector<Closure> *cl, std::int16_t *out)
{
    int f[N_FEATS];
    std::memset(f, 0, sizeof(f));
    shape_features(s, f, 0);
    pair_features(s, f, cl, 0);
    for (int i = 0; i < F_PAIR_THEIRS + N_PAIR_FEATS; ++i)
        out[i] = (std::int16_t)std::max(-32000, std::min(32000, f[i]));
    out[F_PAIR_THEIRS + N_PAIR_FEATS] = (std::int16_t)f[F_ROCK_CONN];
    out[F_PAIR_THEIRS + N_PAIR_FEATS + 1] = (std::int16_t)f[F_ROCK_FREE];
}

inline void features_from_board(const State &s, const std::int16_t *b, int *f)
{
    const int me = s.cp;
    if (me == 0)
    {
        for (int i = 0; i < F_PAIR_THEIRS + N_PAIR_FEATS; ++i)
            f[i] = b[i];
        f[F_ROCK_CONN] = b[F_PAIR_THEIRS + N_PAIR_FEATS];
        f[F_ROCK_FREE] = b[F_PAIR_THEIRS + N_PAIR_FEATS + 1];
    }
    else
    {
        // seen by player 1: shape counts change sign, mine and theirs swap
        for (int i = 0; i < N_SHAPE_FEATS; ++i)
            f[i] = -b[i];
        for (int i = 0; i < N_PAIR_FEATS; ++i)
        {
            f[F_PAIR_MINE + i] = b[F_PAIR_THEIRS + i];
            f[F_PAIR_THEIRS + i] = b[F_PAIR_MINE + i];
        }
        f[F_ROCK_CONN] = -b[F_PAIR_THEIRS + N_PAIR_FEATS];
        f[F_ROCK_FREE] = -b[F_PAIR_THEIRS + N_PAIR_FEATS + 1];
    }
    f[F_HAND_MINE] = s.num_rocks[me];
    f[F_HAND_THEIRS] = s.num_rocks[1 - me];
    f[F_TEMPO] = 1;
    f[F_SECOND] = (me == 1) ? 1 : -1;
}

inline int rest_eval(const State &s, const Weights &w)
{
    const int me = s.cp;
    return s.num_rocks[me] * w.w[F_HAND_MINE] + s.num_rocks[1 - me] * w.w[F_HAND_THEIRS] + w.w[F_TEMPO] + (me == 1 ? 1 : -1) * w.w[F_SECOND];
}

// The move a player would use to place a closure edge (from whichever end they may start at).
inline Move closure_move(const State &s, const Closure &c, int player)
{
    if (s.can_place(c.v, player))
        return make_move(c.v, c.d);
    return make_move(c.v + DOFF[c.d], 7 - c.d);
}

} // namespace rs
