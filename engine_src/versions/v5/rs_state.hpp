// Fast game state for Rocks and Sticks.
//
// This mirrors the rules implemented in game.py exactly (including the flood-fill based
// claimed-region check). tests/test_rs_engine_parity.py fuzzes this against the Python
// reference; change the two together or not at all.
#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>

// Profiling builds (-DRS_PROFILE) keep the main functions out of line so that a sampling
// profiler can tell them apart; normal builds inline as usual.
#ifdef RS_PROFILE
#define RS_HOT __attribute__((noinline))
#else
#define RS_HOT
#endif

namespace rs
{

constexpr int GRID = 128;           // board is GRID x GRID vertices
constexpr int OFF = GRID / 2;       // coordinate (0,0) lives at (OFF, OFF)
constexpr int NCELL = GRID * GRID;
constexpr int SAFE = 3;             // moves are not generated within SAFE of the grid edge
constexpr int N_ROCKS = 2;

// Direction order matches models.Direction: 0:N 1:E 2:NE 3:SE 4:NW 5:SW 6:W 7:S (reverse = 7 - d)
constexpr int DX[8] = {0, 1, 1, 1, -1, -1, -1, 0};
constexpr int DY[8] = {1, 0, 1, -1, 1, -1, 0, -1};
constexpr int DOFF[8] = {GRID, 1, GRID + 1, -GRID + 1, GRID - 1, -GRID - 1, -1, -GRID};
constexpr bool IS_DIAG[8] = {false, false, true, true, true, true, false, false};
// Directions in anticlockwise order (E, NE, N, NW, W, SW, S, SE) and each direction's place in it.
constexpr int ACW[8] = {1, 2, 0, 4, 6, 5, 7, 3};
constexpr int ACW_POS[8] = {2, 0, 1, 7, 3, 5, 4, 6};

constexpr int T_ROCK = 8;
constexpr int T_PASS = 9;

using Move = std::uint32_t; // (idx << 4) | type, type: 0..7 stick direction, 8 rock, 9 pass
constexpr Move MOVE_PASS = T_PASS;
constexpr Move MOVE_NONE = 0xFFFFFFFFu;

inline Move make_move(int idx, int type) { return ((Move)idx << 4) | (Move)type; }
inline int move_idx(Move m) { return (int)(m >> 4); }
inline int move_type(Move m) { return (int)(m & 15); }
inline int idx_of(int x, int y) { return (y + OFF) * GRID + (x + OFF); }
inline int x_of(int idx) { return idx % GRID - OFF; }
inline int y_of(int idx) { return idx / GRID - OFF; }

inline std::uint64_t mix64(std::uint64_t x)
{
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

struct Undo
{
    Move m;
    int cp, turn, winner;
    int scores[2];
    int num_rocks[2];
    int conn_size;
    int cnt_lo, cnt_hi;
    std::uint64_t hash, shash;
};

struct State
{
    std::uint8_t sticks[NCELL];   // bitmask of directions that carry a stick
    std::int8_t rock[NCELL];      // -1 none, else owner
    std::uint8_t diag[NCELL];     // per cell (lower-left vertex): non-zero if a diagonal stick occupies it
    std::vector<int> conn;        // vertices with at least one stick (stack: undo pops)
    std::vector<int> rocks;       // rocked vertices (stack)
    std::vector<Undo> hist;

    int cp = 0, turn = 0, winner = -1;
    int scores[2] = {0, 0};
    int num_rocks[2] = {N_ROCKS, N_ROCKS};
    // A claimed vertex can only exist if some vertex has sticks to all four lexicographically
    // smaller neighbours (W, SW, NW, S) and some vertex has sticks to all four larger ones.
    int cnt_lo = 0, cnt_hi = 0;
    std::uint64_t hash = 0;       // sticks + rocks only
    std::uint64_t shash = 0;      // sticks only (closure areas depend on nothing else)

    // claimed-region flood fill cache
    mutable bool flood_valid = false;
    mutable int fl_x0 = 0, fl_x1 = -1, fl_y0 = 0, fl_y1 = -1;
    mutable std::vector<std::uint8_t> fl_reach;
    mutable std::vector<int> fl_queue;

    // scratch for cycle DFS / move generation
    mutable std::uint32_t stamp[NCELL];
    mutable std::uint32_t stamp_gen = 0;
    mutable int walk_pos[NCELL];  // position of a vertex in the current face walk, or -1
    mutable std::vector<int> walk;

    State() { reset(); }

    void reset()
    {
        std::memset(sticks, 0, sizeof(sticks));
        std::memset(rock, -1, sizeof(rock));
        std::memset(diag, 0, sizeof(diag));
        std::memset(stamp, 0, sizeof(stamp));
        std::memset(walk_pos, -1, sizeof(walk_pos));
        conn.clear();
        rocks.clear();
        hist.clear();
        cp = 0;
        turn = 0;
        winner = -1;
        scores[0] = scores[1] = 0;
        num_rocks[0] = num_rocks[1] = N_ROCKS;
        cnt_lo = cnt_hi = 0;
        hash = 0;
        shash = 0;
        flood_valid = false;
        stamp_gen = 0;
    }

    // Standard start: one vertical stick (0,0)-(0,1).
    void set_start()
    {
        reset();
        add_stick_raw(idx_of(0, 0), 0);
    }

    // ---- raw board edits (used for setup from an external position) ----

    static std::uint64_t edge_hash(int v, int d)
    {
        if (d >= 4)
        {
            v += DOFF[d];
            d = 7 - d;
        }
        return mix64((std::uint64_t)v * 16 + (std::uint64_t)d);
    }
    static std::uint64_t rock_hash(int v, int owner) { return mix64((std::uint64_t)v * 16 + 8 + (std::uint64_t)owner); }
    static int diag_cell(int v, int d) { return idx_of(std::min(x_of(v), x_of(v) + DX[d]), std::min(y_of(v), y_of(v) + DY[d])); }

    bool has_stick(int v, int d) const { return (sticks[v] >> d) & 1; }
    bool connected(int v) const { return sticks[v] != 0; }
    static int popcount8(std::uint8_t b) { return __builtin_popcount((unsigned)b); }

    void add_stick_raw(int v, int d)
    {
        int e = v + DOFF[d];
        if (!sticks[v])
            conn.push_back(v);
        if (!sticks[e])
            conn.push_back(e);
        set_bit(v, d);
        set_bit(e, 7 - d);
        if (IS_DIAG[d])
            diag[diag_cell(v, d)] = 1;
        std::uint64_t h = edge_hash(v, d);
        hash ^= h;
        shash ^= h;
        flood_valid = false;
    }

    void set_bit(int v, int d)
    {
        std::uint8_t before = sticks[v];
        std::uint8_t after = (std::uint8_t)(before | (1u << d));
        sticks[v] = after;
        if ((before & 0xF0) != 0xF0 && (after & 0xF0) == 0xF0)
            cnt_lo++;
        if ((before & 0x0F) != 0x0F && (after & 0x0F) == 0x0F)
            cnt_hi++;
    }

    bool may_claim() const { return cnt_lo > 0 && cnt_hi > 0; }

    void add_rock_raw(int v, int owner)
    {
        rock[v] = (std::int8_t)owner;
        rocks.push_back(v);
        hash ^= rock_hash(v, owner);
    }

    // Full key including side to move etc. Turn only matters while it is 0 or 1.
    std::uint64_t key() const
    {
        std::uint64_t t = (std::uint64_t)std::min(turn, 2);
        std::uint64_t s = t | ((std::uint64_t)cp << 2) | ((std::uint64_t)num_rocks[0] << 3) | ((std::uint64_t)num_rocks[1] << 5) |
                          ((std::uint64_t)(scores[0] & 0xFFF) << 8) | ((std::uint64_t)(scores[1] & 0xFFF) << 20);
        return hash ^ mix64(s ^ 0xA5A5A5A5F00DULL);
    }

    // ---- rules ----

    bool can_place(int v, int player) const { return rock[v] < 0 || rock[v] == player; }

    // game.py coord_in_claimed_region: a vertex inside the bounding box (+1 margin) of the
    // connected vertices that a king-move flood fill from the box border cannot reach, where
    // a king move is blocked only if it runs along an existing stick. A claimed vertex needs
    // a vertex with >= 4 sticks to exist, so this is nearly always false.
    bool claimed(int v) const
    {
        if (!may_claim())
            return false;
        if (!flood_valid)
            flood();
        int x = x_of(v), y = y_of(v);
        if (x < fl_x0 || x > fl_x1 || y < fl_y0 || y > fl_y1)
            return false;
        return !fl_reach[(size_t)(y - fl_y0) * (size_t)(fl_x1 - fl_x0 + 1) + (size_t)(x - fl_x0)];
    }

    RS_HOT void flood() const
    {
        int minx = 1 << 30, maxx = -(1 << 30), miny = 1 << 30, maxy = -(1 << 30);
        for (int v : conn)
        {
            int x = x_of(v), y = y_of(v);
            minx = std::min(minx, x);
            maxx = std::max(maxx, x);
            miny = std::min(miny, y);
            maxy = std::max(maxy, y);
        }
        fl_x0 = minx - 1;
        fl_x1 = maxx + 1;
        fl_y0 = miny - 1;
        fl_y1 = maxy + 1;
        int w = fl_x1 - fl_x0 + 1, h = fl_y1 - fl_y0 + 1;
        fl_reach.assign((size_t)w * (size_t)h, 0);
        fl_queue.clear();
        auto push = [&](int x, int y)
        {
            size_t k = (size_t)(y - fl_y0) * (size_t)w + (size_t)(x - fl_x0);
            if (!fl_reach[k])
            {
                fl_reach[k] = 1;
                fl_queue.push_back(idx_of(x, y));
            }
        };
        for (int x = fl_x0; x <= fl_x1; ++x)
        {
            push(x, fl_y0);
            push(x, fl_y1);
        }
        for (int y = fl_y0; y <= fl_y1; ++y)
        {
            push(fl_x0, y);
            push(fl_x1, y);
        }
        for (size_t qi = 0; qi < fl_queue.size(); ++qi)
        {
            int v = fl_queue[qi];
            int x = x_of(v), y = y_of(v);
            for (int d = 0; d < 8; ++d)
            {
                int nx = x + DX[d], ny = y + DY[d];
                if (nx < fl_x0 || nx > fl_x1 || ny < fl_y0 || ny > fl_y1)
                    continue;
                if (has_stick(v, d))
                    continue;
                push(nx, ny);
            }
        }
        flood_valid = true;
    }

    static bool in_safe_bounds(int v)
    {
        int gx = v % GRID, gy = v / GRID;
        return gx >= SAFE && gx < GRID - SAFE && gy >= SAFE && gy < GRID - SAFE;
    }

    // Is stick (v, d) legal for `player`, started from v? Mirrors game.py valid_move.
    bool stick_legal_from(int v, int d, int player) const
    {
        if (!sticks[v] || has_stick(v, d) || !can_place(v, player))
            return false;
        if (IS_DIAG[d] && diag[diag_cell(v, d)])
            return false;
        int e = v + DOFF[d];
        if (!in_safe_bounds(e))
            return false;
        if (may_claim() && (claimed(v) || claimed(e)))
            return false;
        return true;
    }

    // Can `player` place the edge between v and v+d at all (from either end)?
    bool edge_placeable(int v, int d, int player) const
    {
        return stick_legal_from(v, d, player) || stick_legal_from(v + DOFF[d], 7 - d, player);
    }

    bool rock_legal(int v, int player) const
    {
        if (turn == 0 || num_rocks[player] <= 0 || rock[v] >= 0 || !in_safe_bounds(v))
            return false;
        if (may_claim() && claimed(v))
            return false;
        for (int d = 0; d < 8; ++d)
        {
            int n = v + DOFF[d];
            if (sticks[n] || rock[n] >= 0)
                return true;
        }
        return sticks[v] != 0;
    }

    bool move_legal(Move m, int player) const
    {
        int t = move_type(m);
        if (t == T_PASS)
            return true;
        int v = move_idx(m);
        if (v < 0 || v >= NCELL || !in_safe_bounds(v))
            return false;
        if (t == T_ROCK)
            return rock_legal(v, player);
        return stick_legal_from(v, t, player);
    }

    // Twice the area of the smallest polygon through the stick start-end, which must already be
    // on the board; 0 if there is none. Same method as game.py _path_of_smallest_area: sticks
    // never cross, so the smallest polygon through a stick is the outline of the smaller of
    // the two faces beside it, found by walking round each face.
    RS_HOT int cycle_area2_placed(int start, int end) const
    {
        if (popcount8(sticks[start]) < 2 || popcount8(sticks[end]) < 2)
            return 0;
        int a = face_area2(start, end);
        int b = face_area2(end, start);
        if (a == 0)
            return b;
        if (b == 0)
            return a;
        return std::min(a, b);
    }

    // Walk the face on one side of u->v, always turning to the next stick anticlockwise from
    // the one just walked, cutting out the loops made by dead ends and by anything that hangs
    // off the boundary at one vertex. Returns twice the area of the remaining polygon.
    RS_HOT int face_area2(int u, int v) const
    {
        walk.clear();
        walk.push_back(v);
        walk_pos[v] = 0;
        int cur = v;
        int back = 7 - dir_between(u, v);
        int result = 0;
        const int limit = 4 * (int)conn.size() + 64;
        for (int step = 0; step < limit; ++step)
        {
            int d = back;
            const std::uint8_t mask = sticks[cur];
            for (int k = 1; k <= 8; ++k)
            {
                d = ACW[(ACW_POS[back] + k) & 7];
                if ((mask >> d) & 1)
                    break;
            }
            int nxt = cur + DOFF[d];
            if (cur == u && nxt == v)
            {
                if (walk.size() >= 3)
                {
                    long cross = 0;
                    int px = x_of(walk.back()), py = y_of(walk.back());
                    for (int w : walk)
                    {
                        int wx = x_of(w), wy = y_of(w);
                        cross += (long)px * wy - (long)wx * py;
                        px = wx;
                        py = wy;
                    }
                    result = (int)std::labs(cross);
                }
                break;
            }
            int j = walk_pos[nxt];
            if (j < 0)
            {
                walk_pos[nxt] = (int)walk.size();
                walk.push_back(nxt);
            }
            else
            {
                for (size_t i = (size_t)j + 1; i < walk.size(); ++i)
                    walk_pos[walk[i]] = -1;
                walk.resize((size_t)j + 1);
            }
            back = 7 - d;
            cur = nxt;
        }
        for (int w : walk)
            walk_pos[w] = -1;
        return result;
    }

    static int dir_between(int from, int to)
    {
        int off = to - from;
        for (int d = 0; d < 8; ++d)
            if (DOFF[d] == off)
                return d;
        return 0;
    }

    // Area (x2) that placing stick (v, d) would score for the mover (0 if none / half triangle).
    // Does not check legality.
    int closure_area2(int v, int d) const
    {
        int e = v + DOFF[d];
        if (!sticks[e] || !sticks[v])
            return 0;
        State *self = const_cast<State *>(this);
        self->sticks[v] |= (std::uint8_t)(1u << d);
        self->sticks[e] |= (std::uint8_t)(1u << (7 - d));
        int a = cycle_area2_placed(v, e);
        self->sticks[v] &= (std::uint8_t) ~(1u << d);
        self->sticks[e] &= (std::uint8_t) ~(1u << (7 - d));
        return a == 1 ? 0 : a;
    }

    // known_area: the raw closure area (x2) for a stick joining two connected vertices, if the
    // caller already has it (saves the cycle search); -1 to compute.
    RS_HOT void do_move(Move m, int known_area = -1)
    {
        Undo u;
        u.m = m;
        u.cp = cp;
        u.turn = turn;
        u.winner = winner;
        u.scores[0] = scores[0];
        u.scores[1] = scores[1];
        u.num_rocks[0] = num_rocks[0];
        u.num_rocks[1] = num_rocks[1];
        u.conn_size = (int)conn.size();
        u.cnt_lo = cnt_lo;
        u.cnt_hi = cnt_hi;
        u.hash = hash;
        u.shash = shash;
        hist.push_back(u);

        int t = move_type(m);
        int player = cp;
        if (t == T_ROCK)
        {
            add_rock_raw(move_idx(m), player);
            num_rocks[player]--;
        }
        else if (t != T_PASS)
        {
            int v = move_idx(m);
            int e = v + DOFF[t];
            bool end_connected = sticks[e] != 0;
            num_rocks[player] = N_ROCKS;
            add_stick_raw(v, t);
            if (end_connected)
            {
                int a = known_area >= 0 ? known_area : cycle_area2_placed(v, e);
                if (a >= 2)
                    scores[player] += a;
            }
        }

        cp++;
        if (cp == 2)
        {
            cp = 0;
            turn++;
            int a0 = scores[0], a1 = scores[1];
            if (a0 != a1)
            {
                int leader = a0 > a1 ? 0 : 1;
                if (!(leader == 0 && turn == 1))
                    winner = leader;
            }
        }
    }

    RS_HOT void undo_move()
    {
        Undo u = hist.back();
        hist.pop_back();
        int t = move_type(u.m);
        if (t == T_ROCK)
        {
            int v = move_idx(u.m);
            rock[v] = -1;
            rocks.pop_back();
        }
        else if (t != T_PASS)
        {
            int v = move_idx(u.m);
            int e = v + DOFF[t];
            sticks[v] &= (std::uint8_t) ~(1u << t);
            sticks[e] &= (std::uint8_t) ~(1u << (7 - t));
            if (IS_DIAG[t])
                diag[diag_cell(v, t)] = 0;
            conn.resize((size_t)u.conn_size);
            flood_valid = false;
        }
        cp = u.cp;
        turn = u.turn;
        winner = u.winner;
        scores[0] = u.scores[0];
        scores[1] = u.scores[1];
        num_rocks[0] = u.num_rocks[0];
        num_rocks[1] = u.num_rocks[1];
        cnt_lo = u.cnt_lo;
        cnt_hi = u.cnt_hi;
        hash = u.hash;
        shash = u.shash;
    }

    bool last_was_pass() const { return !hist.empty() && move_type(hist.back().m) == T_PASS; }

    // All legal moves for the side to move. With dedupe, an edge that can be started from
    // either end is emitted once; without it the list matches game.py get_possible_moves.
    RS_HOT void gen_moves(std::vector<Move> &out, bool dedupe) const
    {
        out.clear();
        int player = cp;
        const bool claims = may_claim();
        for (int v : conn)
        {
            if (!can_place(v, player))
                continue;
            if (claims && claimed(v))
                continue;
            std::uint8_t mask = sticks[v];
            for (int d = 0; d < 8; ++d)
            {
                if ((mask >> d) & 1)
                    continue;
                if (IS_DIAG[d] && diag[diag_cell(v, d)])
                    continue;
                int e = v + DOFF[d];
                if (!in_safe_bounds(e))
                    continue;
                if (claims && claimed(e))
                    continue;
                if (dedupe && e < v && sticks[e] && can_place(e, player))
                    continue;
                out.push_back(make_move(v, d));
            }
        }
        if (turn != 0 && num_rocks[player] > 0)
        {
            ++stamp_gen;
            auto consider = [&](int a)
            {
                for (int dy = -1; dy <= 1; ++dy)
                    for (int dx = -1; dx <= 1; ++dx)
                    {
                        int c = a + dy * GRID + dx;
                        if (stamp[c] == stamp_gen)
                            continue;
                        stamp[c] = stamp_gen;
                        if (rock[c] >= 0 || !in_safe_bounds(c))
                            continue;
                        if (claims && claimed(c))
                            continue;
                        out.push_back(make_move(c, T_ROCK));
                    }
            };
            for (int a : conn)
                consider(a);
            for (int a : rocks)
                consider(a);
        }
        out.push_back(MOVE_PASS);
    }

};

} // namespace rs
