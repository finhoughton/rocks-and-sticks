#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <random>
#include <iostream>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <queue>
#include <algorithm>
#include <climits>
#include <memory>
#include <cmath>
#include <sstream>
#include <set>
#include <tuple>
#include <chrono>
#include <iomanip>
#include <cstdint>
#include <array>
#include <cstdlib>

namespace py = pybind11;
using namespace py::literals;

// Direction deltas: 0:N,1:E,2:NE,3:SE,4:NW,5:SW,6:W,7:S
static const int DIR_DELTAS[8][2] = {
    {0, 1}, {1, 0}, {1, 1}, {1, -1}, {-1, 1}, {-1, -1}, {-1, 0}, {0, -1}};

struct Move
{
    int x = 0;
    int y = 0;
    char t = 'P';
};

struct MoveKey
{
    int x;
    int y;
    char t;

    bool operator==(const MoveKey &o) const
    {
        return x == o.x && y == o.y && t == o.t;
    }
};

constexpr std::uint64_t splitmix64(std::uint64_t x);

// hash collisions are rare but catastrophic
struct TTKey
{
    std::uint64_t board_hash;
    std::uint64_t sticks_hash;
    int turn_number;
    int current_player;
    int winner;
    std::array<int, 2> players_scores;
    std::array<int, 2> num_rocks;

    bool operator==(const TTKey &o) const
    {
        return board_hash == o.board_hash && sticks_hash == o.sticks_hash && turn_number == o.turn_number && current_player == o.current_player &&
               winner == o.winner && players_scores == o.players_scores && num_rocks == o.num_rocks;
    }
};

struct TTKeyHash
{
    std::size_t operator()(const TTKey &k) const noexcept
    {
        std::uint64_t h = splitmix64(k.board_hash ^ 0xA4093822299F31D0ULL);
        h ^= splitmix64(k.sticks_hash ^ 0x243F6A8885A308D3ULL);
        h ^= splitmix64((std::uint64_t)(std::uint32_t)k.turn_number ^ 0xB492B66FBE98F273ULL);
        h ^= splitmix64((std::uint64_t)(std::uint32_t)k.current_player ^ 0x6A09E667F3BCC909ULL);
        h ^= splitmix64((std::uint64_t)(std::uint32_t)k.winner ^ 0x3C6EF372FE94F82BULL);
        for (int i = 0; i < 2; ++i)
        {
            h ^= splitmix64((std::uint64_t)(std::uint32_t)k.players_scores[i] ^ (0x9E3779B97F4A7C15ULL * (i + 1)) ^ 0xBB67AE8584CAA73BULL);
            h ^= splitmix64((std::uint64_t)(std::uint32_t)k.num_rocks[i] ^ (0xBF58476D1CE4E5B9ULL * (i + 1)) ^ 0x510E527FADE682D1ULL);
        }
        return (std::size_t)splitmix64(h);
    }
};

static inline std::uint64_t ttkey_digest(const TTKey &k)
{
    return (std::uint64_t)TTKeyHash{}(k);
}

struct EdgeKey
{
    TTKey s;
    MoveKey m;

    bool operator==(const EdgeKey &o) const
    {
        return s == o.s && m == o.m;
    }
};

constexpr std::uint64_t splitmix64(std::uint64_t x)
{
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

constexpr std::uint64_t pack_i32_pair(int x, int y)
{
    return (std::uint64_t)(std::uint32_t)x << 32 | (std::uint64_t)(std::uint32_t)y;
}

constexpr std::uint64_t move_key_u64(const MoveKey &m)
{
    std::uint64_t a = pack_i32_pair(m.x, m.y);
    std::uint64_t b = (std::uint64_t)(std::uint8_t)m.t;
    return splitmix64(a ^ (b * 0x9e3779b97f4a7c15ULL) ^ 0xD1B54A32D192ED03ULL);
}

struct MoveKeyHash
{
    std::size_t operator()(const MoveKey &m) const noexcept
    {
        return (std::size_t)move_key_u64(m);
    }
};

struct EdgeKeyHash
{
    std::size_t operator()(const EdgeKey &e) const noexcept
    {
        std::uint64_t h = splitmix64((std::uint64_t)TTKeyHash{}(e.s) ^ 0xA0761D6478BD642FULL);
        h ^= move_key_u64(e.m);
        return (std::size_t)splitmix64(h);
    }
};

using Coord = std::pair<int, int>;

struct Node
{
    int x, y;
    int rocked_by = -1;
    bool in_connected_points = false;
    int connected_points_index = -1;
    Node *neighbours[8] = {nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};
    uint8_t empty_mask = 0xFF;
    Node(int x_, int y_) : x(x_), y(y_), empty_mask(0xFF) {}
    Coord c() const { return {x, y}; }
};

constexpr Coord calc_end(Coord p, int d)
{
    return {p.first + DIR_DELTAS[d][0], p.second + DIR_DELTAS[d][1]};
}

class GameState
{
public:
    GameState()
    {
        turn_number = 0;
        current_player = 0;
        players_scores = {0, 0};
        num_rocks = {2, 2};
        winner = -1;
        add_node({0, 0});
        // starting stick
        {
            Node *start = get_node({0, 0});
            int d = 0; // North
            Node *end = get_node({0, 1});
            start->neighbours[d] = end;
            end->neighbours[reverse_dir(d)] = start;
            start->empty_mask &= static_cast<uint8_t>(~(1u << d));
            end->empty_mask &= static_cast<uint8_t>(~(1u << reverse_dir(d)));
            connected_points_push_unique(start);
            connected_points_push_unique(end);

            {
                std::uint64_t k1 = key_from_coord({start->x, start->y});
                std::uint64_t k2 = key_from_coord({end->x, end->y});
                std::uint64_t lo = std::min(k1, k2);
                std::uint64_t hi = std::max(k1, k2);
                std::uint64_t edge_feat = splitmix64(lo ^ (hi * 0x9e3779b97f4a7c15ULL) ^ 0x243F6A8885A308D3ULL);
                board_hash ^= edge_feat;
                sticks_hash ^= edge_feat;
            }
        }

        if (std::getenv("PYTEST_CURRENT_TEST") == nullptr)
        {
            rng.seed(std::random_device{}());
        }
    }

    int turn_number;
    int current_player;
    std::array<int, 2> players_scores;
    std::array<int, 2> num_rocks;
    static constexpr int num_players = 2;
    int winner;

    std::unordered_map<std::uint64_t, std::unique_ptr<Node>> points;
    std::vector<Node *> connected_points;
    std::vector<Node *> rocks;
    std::vector<Move> moves;

    std::uint64_t board_hash = 0;

    std::uint64_t sticks_hash = 0;

    mutable std::vector<Coord> scratch_q;
    mutable std::unordered_set<std::uint64_t> scratch_seen;
    mutable std::unordered_set<Node *> scratch_node_set;
    mutable std::vector<Node *> scratch_nodes;
    mutable std::unordered_map<Node *, int> scratch_idx_map;

    mutable int cached_minx = 0;
    mutable int cached_maxx = 0;
    mutable int cached_miny = 0;
    mutable int cached_maxy = 0;
    mutable bool bbox_valid = false;

    mutable std::uint64_t reachable_cache_key = 0;
    mutable bool reachable_cache_valid = false;
    mutable int reachable_lowx = 0;
    mutable int reachable_highx = 0;
    mutable int reachable_lowy = 0;
    mutable int reachable_highy = 0;
    mutable size_t reachable_width = 0;
    mutable size_t reachable_height = 0;
    mutable bool reachable_dense = true;

    mutable std::shared_ptr<std::vector<char>> reachable_grid_ptr;
    mutable std::unordered_set<std::uint64_t> reachable_seen_set;

    mutable std::vector<std::uint8_t> reachable_blocked_grid;
    mutable std::unordered_map<std::uint64_t, std::uint8_t> reachable_blocked_map;

    struct ReachEntry
    {
        std::uint64_t key = 0;
        int lowx = 0;
        int highx = 0;
        int lowy = 0;
        int highy = 0;
        size_t width = 0;
        size_t height = 0;
        bool dense = true;
        std::shared_ptr<std::vector<char>> grid;
        std::uint64_t stamp = 0;
    };
    static constexpr int REACH_CACHE_MAX = 32;
    mutable std::array<ReachEntry, REACH_CACHE_MAX> reach_cache;
    mutable int reach_cache_size = 0;
    mutable std::uint64_t reach_cache_clock = 0;

    std::vector<std::uint64_t> claimed_cycle_stack;
    std::unordered_set<std::uint64_t> claimed_cycle_keys;

    struct HistRec
    {
        Move m;
        int prev_player;
        std::array<int, 2> prev_scores;
        std::array<int, 2> prev_rocks;
        int prev_turn_number;
        int prev_winner;
        size_t claimed_cycle_stack_size;
    };
    std::vector<HistRec> history;

    // keys must be well-defined for negative coords.
    // Never left-shift negative signed integers (UB)
    static constexpr std::uint64_t key_from_coord(Coord c)
    {
        return pack_i32_pair(c.first, c.second);
    }

    static constexpr std::uint64_t intersect_key(int x, int y, int d1, int d2)
    {
        return pack_i32_pair(2 * x + d1, 2 * y + d2);
    }

    TTKey tt_key() const
    {
        TTKey k;
        k.board_hash = board_hash;
        k.sticks_hash = sticks_hash;
        k.turn_number = turn_number;
        k.current_player = current_player;
        k.winner = winner;
        k.players_scores = players_scores;
        k.num_rocks = num_rocks;
        return k;
    }

    std::uint64_t state_key() const
    {
        std::uint64_t h = board_hash;
        h ^= splitmix64((std::uint64_t)(std::uint32_t)turn_number ^ 0xB492B66FBE98F273ULL);
        h ^= splitmix64((std::uint64_t)(std::uint32_t)current_player ^ 0x6A09E667F3BCC909ULL);
        h ^= splitmix64((std::uint64_t)(std::uint32_t)winner ^ 0x3C6EF372FE94F82BULL);

        for (int i = 0; i < num_players; ++i)
        {
            std::uint64_t v = (std::uint64_t)(std::uint32_t)players_scores[i];
            h ^= splitmix64(v ^ (0x9E3779B97F4A7C15ULL * (i + 1)) ^ 0xBB67AE8584CAA73BULL);
        }
        for (int i = 0; i < num_players; ++i)
        {
            std::uint64_t v = (std::uint64_t)(std::uint32_t)num_rocks[i];
            h ^= splitmix64(v ^ (0xBF58476D1CE4E5B9ULL * (i + 1)) ^ 0x510E527FADE682D1ULL);
        }
        return splitmix64(h);
    }

    int debug_empty_mask_drift_count() const
    {
        int drift = 0;
        for (const auto &kv : points)
        {
            const Node *n = kv.second.get();
            for (int d = 0; d < 8; ++d)
            {
                const bool has_nbr = (n->neighbours[d] != nullptr);
                const bool bit_empty = (n->empty_mask & static_cast<std::uint8_t>(1u << d)) != 0;
                if (has_nbr == bit_empty)
                    ++drift;
            }
        }
        return drift;
    }

    std::mt19937 rng_snapshot() const { return rng; }
    void rng_restore(const std::mt19937 &snapshot) { rng = snapshot; }

    void set_current_player0()
    {
        current_player = 0;
    }

    bool intersects_stick(Coord start, int d) const
    {
        if (d < 2 || d > 5)
            return false;

        int dx = DIR_DELTAS[d][0];
        int dy = DIR_DELTAS[d][1];

        Coord a{start.first + dx, start.second};
        Coord b{start.first, start.second + dy};

        auto it_a = points.find(key_from_coord(a));
        if (it_a == points.end())
            return false;
        Node *na = it_a->second.get();

        int cross_dir = -1;
        if (-dx == 1 && dy == 1)
            cross_dir = 2; // NE
        else if (-dx == 1 && dy == -1)
            cross_dir = 3; // SE
        else if (-dx == -1 && dy == 1)
            cross_dir = 4; // NW
        else if (-dx == -1 && dy == -1)
            cross_dir = 5; // SW
        else
            return false;

        Node *nb = na->neighbours[cross_dir];
        if (!nb)
            return false;
        return (nb->x == b.first && nb->y == b.second);
    }

    Node *get_node(Coord c)
    {
        auto k = key_from_coord(c);
        auto it = points.find(k);
        if (it != points.end())
            return it->second.get();
        auto n = std::make_unique<Node>(c.first, c.second);
        Node *p = n.get();
        points[k] = std::move(n);
        return p;
    }

    void add_node(Coord c)
    {
        get_node(c);
        for (int d = 0; d < 8; ++d)
        {
            Coord e = calc_end(c, d);
            auto k = key_from_coord(e);
            if (points.find(k) == points.end())
                points[k] = std::make_unique<Node>(e.first, e.second);
        }
    }

    std::vector<Move> get_possible_moves_for_player(int player_number)
    {
        std::vector<Move> out;
        out.reserve(connected_points.size() * 3 + points.size() / 4);
        for (Node *p : connected_points)
        {
            if (!node_is_connected(p))
                continue;
            if (p->rocked_by != -1 && p->rocked_by != player_number)
                continue;
            Coord pc = p->c();
            if (coord_in_claimed_region_cached(pc))
                continue;
            for (int d = 0; d < 8; ++d)
            {
                if ((p->empty_mask & static_cast<std::uint8_t>(1u << d)) == 0)
                    continue;
                if (intersects_stick(pc, d))
                    continue;
                Coord endc = calc_end(pc, d);
                if (coord_in_claimed_region_cached(endc))
                    continue;

                {
                    auto it_end = points.find(key_from_coord(endc));
                    if (it_end != points.end())
                    {
                        Node *end = it_end->second.get();
                        int rd = reverse_dir(d);
                        if (end->neighbours[rd])
                            continue;
                    }
                }
                Move m;
                m.x = pc.first;
                m.y = pc.second;
                m.t = dir_name_char(d);
                out.push_back(m);
            }
        }
        if (turn_number > 0 && num_rocks[player_number] > 0)
        {
            std::unordered_set<std::uint64_t> seen;
            seen.reserve((connected_points.size() + rocks.size()) * 8);

            auto try_add_candidate = [&](int x, int y)
            {
                std::uint64_t k = key_from_coord({x, y});
                if (!seen.insert(k).second)
                    return;
                auto it = points.find(k);
                if (it != points.end())
                {
                    Node *p = it->second.get();
                    if (p->rocked_by != -1)
                        return;
                    if (coord_in_claimed_region_cached(p->c()))
                        return;
                    out.push_back(Move{p->x, p->y, 'R'});
                    return;
                }
                Coord c{x, y};
                if (coord_in_claimed_region_cached(c))
                    return;
                out.push_back(Move{x, y, 'R'});
            };

            auto add_anchor_neighborhood = [&](Node *a)
            {
                int ax = a->x;
                int ay = a->y;
                for (int dx = -1; dx <= 1; ++dx)
                {
                    for (int dy = -1; dy <= 1; ++dy)
                    {
                        try_add_candidate(ax + dx, ay + dy);
                    }
                }
            };

            for (Node *p : connected_points)
                add_anchor_neighborhood(p);
            for (Node *p : rocks)
                add_anchor_neighborhood(p);
        }
        out.push_back(Move{0, 0, 'P'});
        return out;
    }

    static constexpr bool HALF_AREA_COUNTS = false;
    static constexpr int MAX_CYCLE_PATHS = 100;

    static inline bool coord_lt(Coord a, Coord b)
    {
        if (a.first != b.first)
            return a.first < b.first;
        return a.second < b.second;
    }

    static inline bool coord_leq(Coord a, Coord b)
    {
        return coord_lt(a, b) || (a.first == b.first && a.second == b.second);
    }

    static int polygon_area2_from_path(const std::vector<Node *> &path)
    {
        if (path.size() < 2)
            return 0;
        long long area = 0;
        Node *first = path.front();
        Node *prev = first;
        for (size_t i = 1; i < path.size(); ++i)
        {
            Node *cur = path[i];
            area += (long long)prev->x * (long long)cur->y - (long long)cur->x * (long long)prev->y;
            prev = cur;
        }
        area += (long long)prev->x * (long long)first->y - (long long)first->x * (long long)prev->y;
        if (area < 0)
            area = -area;
        if (area > (long long)INT_MAX)
            return INT_MAX;
        return (int)area;
    }

    static std::uint64_t region_edge_key_digest_from_path(const std::vector<Node *> &path)
    {
        if (path.size() < 3)
            return 0;

        struct Edge4
        {
            int ax, ay, bx, by;
            bool operator<(const Edge4 &o) const
            {
                if (ax != o.ax)
                    return ax < o.ax;
                if (ay != o.ay)
                    return ay < o.ay;
                if (bx != o.bx)
                    return bx < o.bx;
                return by < o.by;
            }
        };

        std::vector<Edge4> edges;
        edges.reserve(path.size());

        Coord prev = path.back()->c();
        for (Node *curNode : path)
        {
            Coord cur = curNode->c();
            Coord a = prev;
            Coord b = cur;
            if (!coord_leq(a, b))
                std::swap(a, b);
            edges.push_back({a.first, a.second, b.first, b.second});
            prev = cur;
        }

        std::sort(edges.begin(), edges.end());

        std::uint64_t h = splitmix64(0xC2B2AE3D27D4EB4FULL ^ (std::uint64_t)edges.size());
        for (const auto &e : edges)
        {
            std::uint64_t ea = pack_i32_pair(e.ax, e.ay);
            std::uint64_t eb = pack_i32_pair(e.bx, e.by);
            h ^= splitmix64(ea ^ (eb * 0x9e3779b97f4a7c15ULL) ^ 0x243F6A8885A308D3ULL);
            h = splitmix64(h);
        }
        if (h == 0)
            h = 1;
        return h;
    }

    void fill_sorted_neighbours(Node *node, std::array<Node *, 8> &out, int &n_out) const
    {
        n_out = 0;
        for (int d = 0; d < 8; ++d)
        {
            Node *nbr = node->neighbours[d];
            if (nbr)
                out[n_out++] = nbr;
        }
        std::sort(out.begin(), out.begin() + n_out, [](Node *a, Node *b)
                  {
                      if (a->x != b->x)
                          return a->x < b->x;
                      return a->y < b->y; });
    }

    int best_new_cycle_area2(Node *start, Node *end, std::uint64_t &out_edge_key)
    {
        out_edge_key = 0;
        if (!start || !end || start == end)
            return 0;

        scratch_node_set.clear();
        std::vector<Node *> path;
        path.reserve(64);

        scratch_node_set.insert(start);
        path.push_back(start);

        struct Frame
        {
            Node *node;
            std::array<Node *, 8> nbrs;
            int n = 0;
            int next = 0;
        };

        int found = 0;
        int best_area2 = 0;
        std::uint64_t best_key = 0;
        bool have_best = false;

        std::vector<Frame> stack;
        stack.reserve(64);
        Frame root;
        root.node = start;
        fill_sorted_neighbours(start, root.nbrs, root.n);
        root.next = 0;
        stack.push_back(root);

        while (!stack.empty() && found < MAX_CYCLE_PATHS)
        {
            Frame &f = stack.back();
            if (f.next >= f.n)
            {
                Node *popped = f.node;
                stack.pop_back();
                if (!path.empty() && path.back() == popped)
                    path.pop_back();
                scratch_node_set.erase(popped);
                continue;
            }

            Node *nbr = f.nbrs[f.next++];
            if (scratch_node_set.find(nbr) != scratch_node_set.end())
                continue;

            scratch_node_set.insert(nbr);
            path.push_back(nbr);

            if (nbr == end)
            {
                ++found;
                int area2 = polygon_area2_from_path(path);
                if (area2 != 0)
                {
                    std::uint64_t ek = region_edge_key_digest_from_path(path);
                    if (ek == 0 || claimed_cycle_keys.find(ek) == claimed_cycle_keys.end())
                    {
                        if (!have_best || area2 < best_area2)
                        {
                            have_best = true;
                            best_area2 = area2;
                            best_key = ek;
                        }
                    }
                }

                scratch_node_set.erase(nbr);
                path.pop_back();
                continue;
            }

            Frame nf;
            nf.node = nbr;
            fill_sorted_neighbours(nbr, nf.nbrs, nf.n);
            nf.next = 0;
            stack.push_back(nf);
        }

        scratch_node_set.clear();

        if (!have_best)
            return 0;
        out_edge_key = best_key;
        return best_area2;
    }

    bool is_move_legal(const Move &m, int player_number) const
    {
        if (m.t == 'P')
            return true;
        if (m.t == 'R')
        {
            if (turn_number == 0)
                return false;
            if (num_rocks[player_number] <= 0)
                return false;
            Coord rc{m.x, m.y};
            if (const_cast<GameState *>(this)->coord_in_claimed_region_cached(rc))
                return false;

            bool near_anchor = false;
            for (Node *a : connected_points)
            {
                if (std::abs(a->x - m.x) <= 1 && std::abs(a->y - m.y) <= 1)
                {
                    near_anchor = true;
                    break;
                }
            }
            if (!near_anchor)
            {
                for (Node *a : rocks)
                {
                    if (std::abs(a->x - m.x) <= 1 && std::abs(a->y - m.y) <= 1)
                    {
                        near_anchor = true;
                        break;
                    }
                }
            }
            if (!near_anchor)
                return false;

            auto it = points.find(key_from_coord({m.x, m.y}));
            if (it != points.end() && it->second->rocked_by != -1)
                return false;
            return true;
        }

        Coord sc{m.x, m.y};
        if (const_cast<GameState *>(this)->coord_in_claimed_region_cached(sc))
            return false;

        auto it_start = points.find(key_from_coord(sc));
        if (it_start == points.end())
            return false;
        Node *start = it_start->second.get();
        if (!node_is_connected(start))
            return false;
        if (start->rocked_by != -1 && start->rocked_by != player_number)
            return false;

        int d = GameState::dir_from_name(m.t);
        if (d < 0 || d > 7)
            return false;
        if (start->neighbours[d])
            return false;
        if (const_cast<GameState *>(this)->intersects_stick(sc, d))
            return false;

        Coord endc = calc_end(sc, d);
        if (const_cast<GameState *>(this)->coord_in_claimed_region_cached(endc))
            return false;

        auto it_end = points.find(key_from_coord(endc));
        if (it_end != points.end())
        {
            Node *end = it_end->second.get();
            int rd = reverse_dir(d);
            if (end->neighbours[rd])
                return false;
        }
        return true;
    }

    void do_move(const Move &m, int player_number)
    {
        if (m.t == 'P')
        {
            history.push_back({m, current_player, players_scores, num_rocks, turn_number, winner, claimed_cycle_stack.size()});
            num_rocks[player_number] = 2;
        }
        else if (m.t == 'R')
        {
            if (turn_number == 0)
                throw std::runtime_error("Illegal rock move: rocks unavailable on turn 0");
            if (num_rocks[player_number] <= 0)
                throw std::runtime_error("Illegal rock move: no rocks left");

            Coord rc{m.x, m.y};
            if (coord_in_claimed_region_cached(rc))
                throw std::runtime_error("Illegal rock move: coord in claimed region");

            bool near_anchor = false;
            for (Node *a : connected_points)
            {
                if (std::abs(a->x - m.x) <= 1 && std::abs(a->y - m.y) <= 1)
                {
                    near_anchor = true;
                    break;
                }
            }
            if (!near_anchor)
            {
                for (Node *a : rocks)
                {
                    if (std::abs(a->x - m.x) <= 1 && std::abs(a->y - m.y) <= 1)
                    {
                        near_anchor = true;
                        break;
                    }
                }
            }
            if (!near_anchor)
                throw std::runtime_error("Illegal rock move: must be within Chebyshev distance 1 of a connected point or rock");

            Node *p = get_node({m.x, m.y});
            if (p->rocked_by != -1)
                throw std::runtime_error("Illegal rock move: cell already has a rock");

            history.push_back({m, current_player, players_scores, num_rocks, turn_number, winner, claimed_cycle_stack.size()});
            p->rocked_by = player_number;
            rocks.push_back(p);
            num_rocks[player_number] -= 1;

            MoveKey mk{m.x, m.y, 'R'};
            board_hash ^= splitmix64(move_key_u64(mk) ^ (std::uint64_t)(player_number + 1) * 0x94d049bb133111ebULL);
        }
        else
        {
            if (!is_move_legal(m, player_number))
                throw std::runtime_error("Illegal stick move");

            history.push_back({m, current_player, players_scores, num_rocks, turn_number, winner, claimed_cycle_stack.size()});
            num_rocks[player_number] = 2;
            Node *start = get_node({m.x, m.y});
            int d = dir_from_name(m.t);
            Node *end = get_node(calc_end(start->c(), d));
            start->neighbours[d] = end;
            end->neighbours[reverse_dir(d)] = start;
            start->empty_mask &= static_cast<uint8_t>(~(1 << d));
            end->empty_mask &= static_cast<uint8_t>(~(1 << reverse_dir(d)));

            {
                std::uint64_t k1 = key_from_coord({start->x, start->y});
                std::uint64_t k2 = key_from_coord({end->x, end->y});
                std::uint64_t lo = std::min(k1, k2);
                std::uint64_t hi = std::max(k1, k2);
                std::uint64_t edge_feat = splitmix64(lo ^ (hi * 0x9e3779b97f4a7c15ULL) ^ 0x243F6A8885A308D3ULL);
                board_hash ^= edge_feat;
                sticks_hash ^= edge_feat;
            }

            connected_points_push_unique(start);
            connected_points_push_unique(end);

            reachable_cache_valid = false;
            {
                std::uint64_t cycle_key = 0;
                int area2 = best_new_cycle_area2(start, end, cycle_key);
                if (area2 > 0 && (HALF_AREA_COUNTS || area2 != 1))
                {
                    players_scores[player_number] += area2;
                    if (cycle_key != 0)
                    {
                        claimed_cycle_stack.push_back(cycle_key);
                        claimed_cycle_keys.insert(cycle_key);
                    }
                }
            }
        }

        current_player += 1;
        if (current_player == num_players)
        {
            current_player = 0;
            turn_number++;

            if (turn_number > 0)
            {
                {
                    int a0 = players_scores[0];
                    int a1 = players_scores[1];
                    int max_a = std::max(a0, a1);
                    if (max_a > 0 && ((a0 == max_a) + (a1 == max_a) == 1))
                    {
                        int leader = (a0 > a1) ? 0 : 1;
                        if (leader == 0 && turn_number == 1)
                        {
                            // Player 1 scored first, player 2 gets a turn
                        }
                        else if (leader == 1 || (leader == 0 && turn_number > 1))
                        {
                            winner = leader;
                        }
                    }
                }
            }
        }
        moves.push_back(m);
    }

    void undo_move()
    {
        if (moves.empty() || history.empty())
            return;
        Move last = moves.back();
        moves.pop_back();
        auto rec = history.back();
        history.pop_back();
        players_scores = rec.prev_scores;
        num_rocks = rec.prev_rocks;
        current_player = rec.prev_player;
        turn_number = rec.prev_turn_number;
        winner = rec.prev_winner;

        while (claimed_cycle_stack.size() > rec.claimed_cycle_stack_size)
        {
            std::uint64_t k = claimed_cycle_stack.back();
            claimed_cycle_stack.pop_back();
            claimed_cycle_keys.erase(k);
        }
        if (last.t == 'R')
        {
            Node *p = get_node({last.x, last.y});
            p->rocked_by = -1;
            if (!rocks.empty())
                rocks.pop_back();

            int mover = rec.prev_player;
            MoveKey mk{last.x, last.y, 'R'};
            board_hash ^= splitmix64(move_key_u64(mk) ^ (std::uint64_t)(mover + 1) * 0x94d049bb133111ebULL);
        }
        else if (last.t != 'P')
        {
            Node *start = get_node({last.x, last.y});
            int d = dir_from_name(last.t);
            Coord endc = calc_end(start->c(), d);
            Node *end = get_node(endc);

            {
                std::uint64_t k1 = key_from_coord({start->x, start->y});
                std::uint64_t k2 = key_from_coord({end->x, end->y});
                std::uint64_t lo = std::min(k1, k2);
                std::uint64_t hi = std::max(k1, k2);
                std::uint64_t edge_feat = splitmix64(lo ^ (hi * 0x9e3779b97f4a7c15ULL) ^ 0x243F6A8885A308D3ULL);
                board_hash ^= edge_feat;
                sticks_hash ^= edge_feat;
            }

            // Undo must restore a canonical, symmetric adjacency state.
            // Using pointer-equality guards here can leave a stale occupied edge
            // if state ever gets slightly inconsistent during deep rollout/undo cycles.
            start->neighbours[d] = nullptr;
            int rd = reverse_dir(d);
            end->neighbours[rd] = nullptr;
            start->empty_mask |= static_cast<uint8_t>(1 << d);
            end->empty_mask |= static_cast<uint8_t>(1 << rd);
            if (!node_is_connected(start) && !(start->x == 0 && start->y == 0))
                connected_points_remove(start);
            if (!node_is_connected(end) && !(end->x == 0 && end->y == 0))
                connected_points_remove(end);

            reachable_cache_valid = false;
        }
    }

    bool coord_in_claimed_region_cached(Coord c)
    {
        if (connected_points.empty())
            return false;

        ensure_reachable_cache();
        if (!reachable_cache_valid)
            return false;

        if (c.first < reachable_lowx || c.first > reachable_highx || c.second < reachable_lowy || c.second > reachable_highy)
            return false;

        if (reachable_dense)
        {
            int ix = c.first - reachable_lowx;
            int iy = c.second - reachable_lowy;
            if (ix < 0 || iy < 0)
                return false;
            size_t idx = (size_t)ix + (size_t)iy * reachable_width;
            if (!reachable_grid_ptr)
                return false;
            if (idx >= reachable_grid_ptr->size())
                return false;
            return (*reachable_grid_ptr)[idx] == 0;
        }
        else
        {
            std::uint64_t k = key_from_coord(c);
            return reachable_seen_set.find(k) == reachable_seen_set.end();
        }
    }

    bool coord_in_claimed_region(Coord start)
    {
        if (connected_points.empty())
            return false;
        int minx, maxx, miny, maxy;
        if (bbox_valid)
        {
            minx = cached_minx;
            maxx = cached_maxx;
            miny = cached_miny;
            maxy = cached_maxy;
        }
        else
        {
            minx = INT_MAX;
            maxx = INT_MIN;
            miny = INT_MAX;
            maxy = INT_MIN;
            for (Node *p : connected_points)
            {
                minx = std::min(minx, p->x);
                maxx = std::max(maxx, p->x);
                miny = std::min(miny, p->y);
                maxy = std::max(maxy, p->y);
            }
            cached_minx = minx;
            cached_maxx = maxx;
            cached_miny = miny;
            cached_maxy = maxy;
            bbox_valid = true;
        }
        int margin = 1;
        int lowx = minx - margin, highx = maxx + margin, lowy = miny - margin, highy = maxy + margin;
        scratch_q.clear();
        size_t width = (size_t)(highx - lowx + 1);
        size_t height = (size_t)(highy - lowy + 1);
        size_t area = width * height;
        const size_t GRID_THRESHOLD = 20000;

        if (area > 0 && area <= GRID_THRESHOLD)
        {
            std::vector<char> visited(area, 0);
            auto in_bounds = [&](Coord c) -> bool
            {
                int ix = c.first - lowx;
                int iy = c.second - lowy;
                return !(ix < 0 || (size_t)ix >= width || iy < 0 || (size_t)iy >= height);
            };
            auto push_if_unseen_grid = [&](Coord c)
            {
                if (!in_bounds(c))
                    return false;
                int ix = c.first - lowx;
                int iy = c.second - lowy;
                size_t idx = (size_t)ix + (size_t)iy * width;
                if (!visited[idx])
                {
                    visited[idx] = 1;
                    scratch_q.push_back(c);
                    return true;
                }
                return false;
            };

            if (!in_bounds(start))
                return false;

            push_if_unseen_grid(start);
            size_t qi = 0;
            while (qi < scratch_q.size())
            {
                Coord cur = scratch_q[qi++];
                std::uint64_t kcur = key_from_coord(cur);
                Node *curNode = nullptr;
                auto it = points.find(kcur);
                if (it != points.end())
                    curNode = it->second.get();
                for (int d = 0; d < 8; ++d)
                {
                    if (curNode && curNode->neighbours[d])
                        continue;
                    Coord nc = calc_end(cur, d);
                    if (!in_bounds(nc))
                        return false;
                    push_if_unseen_grid(nc);
                }
            }
        }
        else
        {
            scratch_seen.clear();
            if (points.size() > 0)
                scratch_seen.reserve(points.size() * 2 + 16);
            auto push_if_unseen = [&](Coord c)
            {
                std::uint64_t k = key_from_coord(c);
                if (scratch_seen.insert(k).second)
                    scratch_q.push_back(c);
            };
            push_if_unseen(start);
            size_t qi = 0;
            while (qi < scratch_q.size())
            {
                Coord cur = scratch_q[qi++];
                if (cur.first < lowx || cur.first > highx || cur.second < lowy || cur.second > highy)
                    return false;
                std::uint64_t kcur = key_from_coord(cur);
                Node *curNode = nullptr;
                auto it = points.find(kcur);
                if (it != points.end())
                    curNode = it->second.get();
                for (int d = 0; d < 8; ++d)
                {
                    if (curNode && curNode->neighbours[d])
                        continue;
                    Coord nc = calc_end(cur, d);
                    push_if_unseen(nc);
                }
            }
        }
        return true;
    }

private:
    void ensure_reachable_cache() const
    {
        std::uint64_t key = sticks_hash;

        if (reachable_cache_valid && reachable_cache_key == key)
            return;

        for (int i = 0; i < reach_cache_size; ++i)
        {
            const ReachEntry &e = reach_cache[i];
            if (e.key != key)
                continue;
            reachable_cache_key = key;
            reachable_cache_valid = true;
            reachable_lowx = e.lowx;
            reachable_highx = e.highx;
            reachable_lowy = e.lowy;
            reachable_highy = e.highy;
            reachable_width = e.width;
            reachable_height = e.height;
            reachable_dense = e.dense;
            if (reachable_dense)
            {
                reachable_grid_ptr = e.grid;
                reachable_seen_set.clear();
            }
            else
            {
                reachable_grid_ptr.reset();
                reachable_seen_set.clear();
                reachable_cache_valid = false;
            }
            reach_cache[i].stamp = ++reach_cache_clock;
            return;
        }

        int minx = INT_MAX, maxx = INT_MIN, miny = INT_MAX, maxy = INT_MIN;
        for (Node *p : connected_points)
        {
            minx = std::min(minx, p->x);
            maxx = std::max(maxx, p->x);
            miny = std::min(miny, p->y);
            maxy = std::max(maxy, p->y);
        }
        cached_minx = minx;
        cached_maxx = maxx;
        cached_miny = miny;
        cached_maxy = maxy;
        bbox_valid = true;

        int margin = 1;
        int lowx = minx - margin, highx = maxx + margin, lowy = miny - margin, highy = maxy + margin;
        size_t width = (size_t)(highx - lowx + 1);
        size_t height = (size_t)(highy - lowy + 1);
        size_t area = width * height;

        reachable_cache_key = key;
        reachable_cache_valid = true;
        reachable_lowx = lowx;
        reachable_highx = highx;
        reachable_lowy = lowy;
        reachable_highy = highy;
        reachable_width = width;
        reachable_height = height;

        const size_t GRID_THRESHOLD = 60000;
        reachable_dense = (area > 0 && area <= GRID_THRESHOLD);

        scratch_q.clear();

        auto in_bounds = [&](Coord c) -> bool
        {
            int ix = c.first - lowx;
            int iy = c.second - lowy;
            return !(ix < 0 || (size_t)ix >= width || iy < 0 || (size_t)iy >= height);
        };

        if (reachable_dense)
        {
            if (reachable_blocked_grid.size() != area)
                reachable_blocked_grid.assign(area, 0);
            else
                std::fill(reachable_blocked_grid.begin(), reachable_blocked_grid.end(), 0);

            for (Node *p : connected_points)
            {
                int ix = p->x - lowx;
                int iy = p->y - lowy;
                if (ix < 0 || iy < 0 || (size_t)ix >= width || (size_t)iy >= height)
                    continue;
                // reachable_blocked_grid stores occupied directions (bit=1 means blocked).
                std::uint8_t mask = static_cast<std::uint8_t>(~p->empty_mask);
                size_t idx = (size_t)ix + (size_t)iy * width;
                reachable_blocked_grid[idx] |= mask;
            }
            reachable_blocked_map.clear();
        }
        else
        {
            reachable_blocked_map.clear();
            reachable_blocked_map.reserve(connected_points.size() * 2 + 16);
            for (Node *p : connected_points)
            {
                std::uint8_t mask = 0;
                for (int d = 0; d < 8; ++d)
                {
                    if (p->neighbours[d])
                        mask |= static_cast<std::uint8_t>(1u << d);
                }
                if (mask)
                    reachable_blocked_map[key_from_coord(p->c())] = mask;
            }
            reachable_blocked_grid.clear();
        }

        auto blocked_mask_at = [&](Coord c) -> std::uint8_t
        {
            if (reachable_dense)
            {
                int ix = c.first - lowx;
                int iy = c.second - lowy;
                if (ix < 0 || iy < 0 || (size_t)ix >= width || (size_t)iy >= height)
                    return 0;
                size_t idx = (size_t)ix + (size_t)iy * width;
                if (idx >= reachable_blocked_grid.size())
                    return 0;
                return reachable_blocked_grid[idx];
            }
            auto it = reachable_blocked_map.find(key_from_coord(c));
            if (it == reachable_blocked_map.end())
                return 0;
            return it->second;
        };

        if (reachable_dense)
        {
            auto grid = std::make_shared<std::vector<char>>(area, 0);
            auto push_if_unseen = [&](Coord c)
            {
                int ix = c.first - lowx;
                int iy = c.second - lowy;
                if (ix < 0 || (size_t)ix >= width || iy < 0 || (size_t)iy >= height)
                    return;
                size_t idx = (size_t)ix + (size_t)iy * width;
                if (!(*grid)[idx])
                {
                    (*grid)[idx] = 1;
                    scratch_q.push_back(c);
                }
            };

            auto seed_if_open_to_outside = [&](Coord c)
            {
                std::uint8_t mask = blocked_mask_at(c);
                for (int d = 0; d < 8; ++d)
                {
                    if (mask & static_cast<std::uint8_t>(1u << d))
                        continue;
                    Coord nc = calc_end(c, d);
                    if (!in_bounds(nc))
                    {
                        push_if_unseen(c);
                        return;
                    }
                }
            };

            for (int x = lowx; x <= highx; ++x)
            {
                seed_if_open_to_outside({x, lowy});
                seed_if_open_to_outside({x, highy});
            }
            for (int y = lowy; y <= highy; ++y)
            {
                seed_if_open_to_outside({lowx, y});
                seed_if_open_to_outside({highx, y});
            }

            size_t qi = 0;
            while (qi < scratch_q.size())
            {
                Coord cur = scratch_q[qi++];
                std::uint8_t mask = blocked_mask_at(cur);
                for (int d = 0; d < 8; ++d)
                {
                    if (mask & static_cast<std::uint8_t>(1u << d))
                        continue;
                    Coord nc = calc_end(cur, d);
                    if (!in_bounds(nc))
                        continue;
                    push_if_unseen(nc);
                }
            }
            reachable_grid_ptr = std::move(grid);
            reachable_seen_set.clear();
        }
        else
        {
            reachable_seen_set.clear();
            if (area > 0)
                reachable_seen_set.reserve(std::min(area, (size_t)200000));

            auto push_if_unseen = [&](Coord c)
            {
                std::uint64_t k = key_from_coord(c);
                if (reachable_seen_set.insert(k).second)
                    scratch_q.push_back(c);
            };

            auto seed_if_open_to_outside = [&](Coord c)
            {
                std::uint8_t mask = blocked_mask_at(c);
                for (int d = 0; d < 8; ++d)
                {
                    if (mask & static_cast<std::uint8_t>(1u << d))
                        continue;
                    Coord nc = calc_end(c, d);
                    if (nc.first < lowx || nc.first > highx || nc.second < lowy || nc.second > highy)
                    {
                        push_if_unseen(c);
                        return;
                    }
                }
            };

            for (int x = lowx; x <= highx; ++x)
            {
                seed_if_open_to_outside({x, lowy});
                seed_if_open_to_outside({x, highy});
            }
            for (int y = lowy; y <= highy; ++y)
            {
                seed_if_open_to_outside({lowx, y});
                seed_if_open_to_outside({highx, y});
            }

            size_t qi = 0;
            while (qi < scratch_q.size())
            {
                Coord cur = scratch_q[qi++];
                std::uint8_t mask = blocked_mask_at(cur);
                for (int d = 0; d < 8; ++d)
                {
                    if (mask & static_cast<std::uint8_t>(1u << d))
                        continue;
                    Coord nc = calc_end(cur, d);
                    if (nc.first < lowx || nc.first > highx || nc.second < lowy || nc.second > highy)
                        continue;
                    push_if_unseen(nc);
                }
            }
            reachable_grid_ptr.reset();
        }

        if (reachable_dense)
        {
            ReachEntry ne;
            ne.key = key;
            ne.lowx = reachable_lowx;
            ne.highx = reachable_highx;
            ne.lowy = reachable_lowy;
            ne.highy = reachable_highy;
            ne.width = reachable_width;
            ne.height = reachable_height;
            ne.dense = true;
            ne.grid = reachable_grid_ptr;
            ne.stamp = ++reach_cache_clock;

            if (reach_cache_size < REACH_CACHE_MAX)
            {
                reach_cache[reach_cache_size++] = std::move(ne);
            }
            else
            {
                int victim = 0;
                std::uint64_t best = reach_cache[0].stamp;
                for (int i = 1; i < REACH_CACHE_MAX; ++i)
                {
                    if (reach_cache[i].stamp < best)
                    {
                        best = reach_cache[i].stamp;
                        victim = i;
                    }
                }
                reach_cache[victim] = std::move(ne);
            }
        }
    }

    int tactical_branch_limit = 8;
    double rock_rollout_bonus_connected = 1.0;
    double rock_rollout_bonus_disconnected = 0.02;
    double stick_between_opp_rocks_bonus = 0.4;

    std::mt19937 rng{0};

    static std::string move_key_str_static(const Move &m)
    {
        return std::to_string(m.x) + ":" + std::to_string(m.y) + ":" + m.t;
    }
    double rock_bonus_for_cell(const GameState &g, Coord c, double connected_bonus, double disconnected_bonus) const
    {
        auto k = GameState::key_from_coord(c);
        auto it = g.points.find(k);
        if (it == g.points.end())
            return disconnected_bonus;
        Node *p = it->second.get();
        return p->in_connected_points ? connected_bonus : disconnected_bonus;
    }

    bool stick_between_opp_rocks(const GameState &g, int player_number, const Move &mv) const
    {
        if (mv.t == 'P' || mv.t == 'R')
            return false;
        int d = dir_from_name(mv.t);
        auto start_it = g.points.find(GameState::key_from_coord({mv.x, mv.y}));
        if (start_it == g.points.end())
            return false;
        Node *start = start_it->second.get();
        if (start->rocked_by == -1)
            return false;
        int opp = 1 - player_number;
        if (start->rocked_by != opp)
            return false;
        Coord endc = calc_end(start->c(), d);
        auto end_it = g.points.find(GameState::key_from_coord(endc));
        if (end_it == g.points.end())
            return false;
        Node *end = end_it->second.get();
        if (end->rocked_by != opp)
            return false;
        return true;
    }

    double eval_probability_simple(GameState &g, int player_number) const
    {
        if (g.winner != -1)
            return (g.winner == player_number) ? 1.0 : 0.0;
        return 0.5;
    }

    double score_after(GameState &g, int player_number, const Move &m) const
    {
        if (!g.is_move_legal(m, player_number))
            return -1e300;
        g.do_move(m, player_number);
        double s = eval_probability_simple(g, player_number);
        g.undo_move();
        return s;
    }

public:
    Move rollout_pick_move(GameState &game)
    {
        auto move_less = [](const Move &a, const Move &b) -> bool
        {
            if (a.x != b.x)
                return a.x < b.x;
            if (a.y != b.y)
                return a.y < b.y;
            return a.t < b.t;
        };

        int mover = game.current_player;
        auto moves = game.get_possible_moves_for_player(mover);
        if (moves.empty())
            return Move{0, 0, 'P'};
        if (moves.size() == 1)
            return moves[0];

        std::uniform_real_distribution<double> urd(0.0, 1.0);
        if (urd(rng) < 0.20)
        {
            std::uniform_int_distribution<size_t> uid(0, moves.size() - 1);
            return moves[uid(rng)];
        }

        if (moves.size() > 9)
        {
            std::shuffle(moves.begin(), moves.end(), rng);
            moves.resize(9);
        }

        double best_score = -1e300;
        Move best_move = moves[0];
        for (auto &m : moves)
        {
            double s = score_after(game, mover, m);
            if (m.t == 'R')
                s += rock_bonus_for_cell(game, {m.x, m.y}, rock_rollout_bonus_connected, rock_rollout_bonus_disconnected);
            if (stick_between_opp_rocks(game, mover, m))
            {
                s += stick_between_opp_rocks_bonus;
            }
            if (s > best_score || (s == best_score && move_less(m, best_move)))
            {
                best_score = s;
                best_move = m;
            }
        }
        return best_move;
    }

    bool allows_forced_loss_next_round(const Move &my_move, GameState &working_game, int player_number) const
    {
        std::unordered_map<std::uint64_t, std::vector<Move>> move_order_cache;

        auto top_k_by_heuristic = [&](int pnum) -> const std::vector<Move> &
        {
            std::uint64_t state_key = ttkey_digest(working_game.tt_key());
            std::uint64_t cache_key = state_key ^ ((std::uint64_t)pnum << 56);
            auto it = move_order_cache.find(cache_key);
            if (it != move_order_cache.end())
                return it->second;

            auto moves = working_game.get_possible_moves_for_player(pnum);
            std::vector<std::pair<double, Move>> scored;
            scored.reserve(moves.size());
            for (auto &mv : moves)
            {
                double sc = score_after(working_game, pnum, mv);
                scored.emplace_back(sc, mv);
            }
            std::sort(scored.begin(), scored.end(), [](auto &a, auto &b)
                      { return a.first > b.first; });

            int limit = std::min((int)scored.size(), tactical_branch_limit);
            std::vector<Move> top;
            top.reserve(limit);
            for (int i = 0; i < limit; ++i)
                top.push_back(scored[i].second);
            auto ins = move_order_cache.emplace(cache_key, std::move(top));
            return ins.first->second;
        };

        {
            if (!working_game.is_move_legal(my_move, player_number))
                return false;
            working_game.do_move(my_move, player_number);
            if (working_game.winner != -1 && working_game.winner != player_number)
            {
                working_game.undo_move();
                return true;
            }
            int opp = working_game.current_player;
            const auto &opp_moves = top_k_by_heuristic(opp);
            for (const auto &opp_move : opp_moves)
            {
                if (!working_game.is_move_legal(opp_move, opp))
                    continue;
                working_game.do_move(opp_move, opp);
                if (working_game.winner == opp)
                {
                    working_game.undo_move();
                    working_game.undo_move();
                    return true;
                }
                int me = working_game.current_player;
                const auto &responses = top_k_by_heuristic(me);
                bool defended = false;
                for (const auto &resp : responses)
                {
                    if (!working_game.is_move_legal(resp, me))
                        continue;
                    working_game.do_move(resp, me);
                    if (working_game.winner != opp)
                    {
                        defended = true;
                        working_game.undo_move();
                        break;
                    }
                    working_game.undo_move();
                }
                working_game.undo_move();
                if (!defended)
                {
                    working_game.undo_move();
                    return true;
                }
            }
            working_game.undo_move();
        }
        return false;
    }

private:
    void connected_points_push_unique(Node *n)
    {
        if (n->in_connected_points)
            return;
        n->in_connected_points = true;
        n->connected_points_index = (int)connected_points.size();
        connected_points.push_back(n);
        if (bbox_valid)
        {
            if (n->x < cached_minx)
                cached_minx = n->x;
            if (n->x > cached_maxx)
                cached_maxx = n->x;
            if (n->y < cached_miny)
                cached_miny = n->y;
            if (n->y > cached_maxy)
                cached_maxy = n->y;
        }
    }
    void connected_points_remove(Node *n)
    {
        if (!n->in_connected_points)
            return;
        int idx = n->connected_points_index;
        if (idx < 0 || idx >= (int)connected_points.size())
            return;

        Node *last = connected_points.back();
        connected_points[idx] = last;
        last->connected_points_index = idx;
        connected_points.pop_back();

        n->in_connected_points = false;
        n->connected_points_index = -1;

        if (bbox_valid && (n->x == cached_minx || n->x == cached_maxx || n->y == cached_miny || n->y == cached_maxy))
            bbox_valid = false;
    }
    inline bool node_is_connected(const Node *n) const
    {
        // empty_mask bit=1 means empty; if all 8 are empty, node is disconnected.
        return n->empty_mask != 0xFF;
    }

public:
    constexpr static int reverse_dir(int d) { return 7 - d; }
    constexpr static int dir_from_name(char name)
    {
        switch (name)
        {
        case 'N':
            return 0;
        case 'E':
            return 1;
        case 'A':
            return 2;
        case 'B':
            return 3;
        case 'C':
            return 4;
        case 'D':
            return 5;
        case 'W':
            return 6;
        case 'S':
            return 7;
        default:
            return -1;
        }
    }
    constexpr static char dir_name_char(int d)
    {
        switch (d)
        {
        case 0:
            return 'N';
        case 1:
            return 'E';
        case 2:
            return 'A';
        case 3:
            return 'B';
        case 4:
            return 'C';
        case 5:
            return 'D';
        case 6:
            return 'W';
        case 7:
            return 'S';
        default:
            return 'N';
        }
    }
};

class MCTSEngine
{
public:
    MCTSEngine(int seed = 0, double c_puct_ = 1.41421356)
        : rng(seed),
          c_puct(c_puct_),
          progressive_widening_c(1.6),
          progressive_widening_alpha(0.55),
          rave_k(250.0),
          prior_eval_cap(48),
          max_sim_depth(50),
          check_forced_losses(true),
          tactical_root_limit(20)
    {
        // silence when running as test
        verbose = (std::getenv("PYTEST_CURRENT_TEST") == nullptr);
    }
    Move choose_move(const GameState &root, int n_rollouts)
    {
        auto &game = const_cast<GameState &>(root);

        struct RolloutRngGuard
        {
            GameState &g;
            std::mt19937 snapshot;
            explicit RolloutRngGuard(GameState &gs) : g(gs), snapshot(gs.rng_snapshot()) {}
            ~RolloutRngGuard() { g.rng_restore(snapshot); }
        } rollout_rng_guard(game);

        TTKey root_key = game.tt_key();
        std::uint64_t root_key_digest = ttkey_digest(root_key);

        // Debug stuff (currently no-op)
        auto log_stage = [](const char *) {};
        auto log_value = [](const char *, int, int, int) {};
        auto log_rollout = [](const char *, int, int, int, int, size_t) {};
        log_stage("stage1_start");

        auto explain_illegal = [&](const Move &m, int player_number) -> std::string
        {
            std::ostringstream os;
            os << "m=(" << m.x << "," << m.y << "," << m.t << ")";
            os << " turn=" << game.turn_number << " cur=" << game.current_player;
            if (m.t == 'P')
                return os.str() + " pass";
            if (m.t == 'R')
            {
                if (game.turn_number == 0)
                    return os.str() + " illegal: turn0";
                if (game.num_rocks[player_number] <= 0)
                    return os.str() + " illegal: no_rocks";
                if (game.coord_in_claimed_region_cached({m.x, m.y}))
                    return os.str() + " illegal: claimed";
                bool near_anchor = false;
                for (Node *a : game.connected_points)
                    if (std::abs(a->x - m.x) <= 1 && std::abs(a->y - m.y) <= 1)
                        near_anchor = true;
                if (!near_anchor)
                    for (Node *a : game.rocks)
                        if (std::abs(a->x - m.x) <= 1 && std::abs(a->y - m.y) <= 1)
                            near_anchor = true;
                if (!near_anchor)
                    return os.str() + " illegal: not_adjacent";
                auto it = game.points.find(GameState::key_from_coord({m.x, m.y}));
                if (it != game.points.end() && it->second->rocked_by != -1)
                    return os.str() + " illegal: occupied";
                return os.str() + " legal?";
            }

            Coord sc{m.x, m.y};
            if (game.coord_in_claimed_region_cached(sc))
                return os.str() + " illegal: start_claimed";
            auto it_start = game.points.find(GameState::key_from_coord(sc));
            if (it_start == game.points.end())
                return os.str() + " illegal: start_missing";
            Node *start = it_start->second.get();
            {
                bool connected = false;
                for (int dd = 0; dd < 8; ++dd)
                    if (start->neighbours[dd])
                        connected = true;
                if (!connected)
                    return os.str() + " illegal: start_disconnected";
            }
            if (start->rocked_by != -1 && start->rocked_by != player_number)
                return os.str() + " illegal: cannot_place";
            int d = GameState::dir_from_name(m.t);
            if (d < 0 || d > 7)
                return os.str() + " illegal: bad_dir";
            if (start->neighbours[d])
                return os.str() + " illegal: edge_occupied";
            if (game.intersects_stick(sc, d))
                return os.str() + " illegal: intersects";
            Coord endc = calc_end(sc, d);
            if (game.coord_in_claimed_region_cached(endc))
                return os.str() + " illegal: end_claimed";
            auto it_end = game.points.find(GameState::key_from_coord(endc));
            if (it_end != game.points.end())
            {
                Node *end = it_end->second.get();
                int rd = 7 - d;
                if (end->neighbours[rd])
                    return os.str() + " illegal: reverse_occupied";
            }
            return os.str() + " legal?";
        };

        auto mk_of = [](const Move &m) -> MoveKey
        { return MoveKey{m.x, m.y, m.t}; };

        auto move_less = [](const Move &a, const Move &b) -> bool
        {
            if (a.x != b.x)
                return a.x < b.x;
            if (a.y != b.y)
                return a.y < b.y;
            return a.t < b.t;
        };

        auto ensure_state_initialized = [&](const TTKey &skey, GameState &g)
        {
            auto it_existing = legal_moves.find(skey);
            if (it_existing == legal_moves.end())
            {
                log_stage("stage2_gen_moves");
                auto gen_moves = g.get_possible_moves_for_player(g.current_player);
                auto it_ins = legal_moves.emplace(skey, std::move(gen_moves)).first;
                it_existing = it_ins;
                expanded_count[skey] = 0;
                Ns.emplace(skey, 0);
            }

            auto &moves = it_existing->second;

            // The move list is cached by TTKey (transposition table). In practice, we've observed
            // rare cases where the cached list can become inconsistent with the current state's
            // legality (e.g. after deep rollout/undo cycles or other subtle state interactions).
            // If that happens, MCTS can select an illegal move.
            //
            // Root-cause: cached move lists must be treated as derived data and validated.
            // If any cached move is illegal, regenerate the full list from the GameState.
            bool any_illegal = false;
            for (const auto &m : moves)
            {
                if (!g.is_move_legal(m, g.current_player))
                {
                    any_illegal = true;
                    break;
                }
            }
            if (any_illegal)
            {
                log_stage("stage2_regen_moves");
                moves = g.get_possible_moves_for_player(g.current_player);
                expanded_count[skey] = 0;
            }
            if (skey == root_key)
            {
                if (root_priors.empty())
                {
                    log_stage("stage3_build_priors");
                    const bool bypass_python_priors = false;
                    if (bypass_python_priors)
                    {
                        double uniform = 1.0 / std::max<size_t>(1, moves.size());
                        for (const auto &m : moves)
                            root_priors[mk_of(m)] = uniform;
                    }
                    else
                    {
                        try
                        {
                            py::gil_scoped_acquire gil;
                            if (!torch_module)
                            {
                                torch_module = py::module::import("torch");
                                pyg_data_module = py::module::import("torch_geometric.data");
                                pyg_data_Data = pyg_data_module.attr("Data");
                                pyg_data_Batch = pyg_data_module.attr("Batch");
                                gnn_module = py::module::import("gnn.model");
                                types_module = py::module::import("types");
                            }

                            auto eval_probs = [&](py::list encs) -> py::list
                            {
                                if (!model_override.is_none())
                                {
                                    py::list datas;
                                    for (auto item : encs)
                                    {
                                        py::object enc_obj = py::cast<py::object>(item);
                                        datas.append(enc_obj.attr("data"));
                                    }
                                    py::object batch = pyg_data_Batch.attr("from_data_list")(datas);
                                    if (!model_device.empty())
                                        batch = batch.attr("to")(py::cast(model_device));

                                    py::object no_grad = torch_module.attr("no_grad")();
                                    no_grad.attr("__enter__")();
                                    py::object logits = model_override(batch);
                                    no_grad.attr("__exit__")(py::none(), py::none(), py::none());
                                    py::object probs = torch_module.attr("sigmoid")(logits).attr("detach")().attr("cpu")();
                                    return py::cast<py::list>(probs.attr("tolist")());
                                }

                                py::object py_probs = gnn_module.attr("evaluate_encodings")(encs);
                                return py::cast<py::list>(py_probs);
                            };

                            for (const auto &m : moves)
                                root_priors[mk_of(m)] = 0.0;

                            py::list encs;
                            std::vector<size_t> valid_indices;
                            valid_indices.reserve(moves.size());

                            struct UndoGuard
                            {
                                GameState *g = nullptr;
                                bool active = false;
                                explicit UndoGuard(GameState &gs) : g(&gs), active(false) {}
                                void arm() { active = true; }
                                void disarm() { active = false; }
                                ~UndoGuard()
                                {
                                    if (active && g)
                                        g->undo_move();
                                }
                            };

                            for (size_t mi = 0; mi < moves.size(); ++mi)
                            {
                                const auto &m = moves[mi];
                                if (!g.is_move_legal(m, g.current_player))
                                    continue;

                                UndoGuard guard(g);
                                g.do_move(m, g.current_player);
                                guard.arm();

                                std::uint64_t enc_key = g.state_key();
                                auto enc_it = enc_cache.find(enc_key);
                                if (enc_it != enc_cache.end())
                                {
                                    encs.append(enc_it->second);
                                    valid_indices.push_back(mi);
                                    guard.disarm();
                                    g.undo_move();
                                    continue;
                                }

                                g.scratch_node_set.clear();
                                if (g.connected_points.size() + g.rocks.size() > 0)
                                    g.scratch_node_set.reserve(g.connected_points.size() + g.rocks.size());
                                for (auto *p : g.connected_points)
                                    g.scratch_node_set.insert(p);
                                for (auto *p : g.rocks)
                                    g.scratch_node_set.insert(p);

                                g.scratch_nodes.clear();
                                g.scratch_nodes.reserve(g.scratch_node_set.size());
                                for (auto *p : g.scratch_node_set)
                                    g.scratch_nodes.push_back(p);
                                std::sort(g.scratch_nodes.begin(), g.scratch_nodes.end(), [](Node *a, Node *b)
                                          {
                                if (a->x != b->x)
                                    return a->x < b->x;
                                return a->y < b->y; });

                                g.scratch_idx_map.clear();
                                if (g.scratch_nodes.size() > 0)
                                    g.scratch_idx_map.reserve(g.scratch_nodes.size() * 2);
                                for (size_t i = 0; i < g.scratch_nodes.size(); ++i)
                                    g.scratch_idx_map[g.scratch_nodes[i]] = (int)i;
                                std::vector<Node *> &nodes = g.scratch_nodes;
                                std::unordered_map<Node *, int> &idx_map = g.scratch_idx_map;

                                std::vector<std::vector<double>> node_feats;
                                node_feats.reserve(nodes.size());
                                std::vector<std::array<long long, 2>> coords;
                                coords.reserve(nodes.size());
                                for (auto *n : nodes)
                                {
                                    int num_players = GameState::num_players;
                                    std::vector<double> owner_one_hot(num_players + 1, 0.0);
                                    int owner_idx = (n->rocked_by >= 0) ? (n->rocked_by + 1) : 0;
                                    if (owner_idx >= 0 && owner_idx < (int)owner_one_hot.size())
                                        owner_one_hot[owner_idx] = 1.0;
                                    int neighbour_count = 0;
                                    for (int d = 0; d < 8; ++d)
                                        if (n->neighbours[d])
                                            neighbour_count++;
                                    double deg = (double)neighbour_count / 8.0;
                                    double is_leaf = (neighbour_count == 1) ? 1.0 : 0.0;
                                    double x = (double)n->x;
                                    double y = (double)n->y;
                                    double r2 = x * x + y * y;
                                    std::vector<double> feats = owner_one_hot;
                                    feats.push_back(deg);
                                    feats.push_back(is_leaf);
                                    feats.push_back(x);
                                    feats.push_back(y);
                                    feats.push_back(r2);
                                    node_feats.push_back(std::move(feats));
                                    coords.push_back({n->x, n->y});
                                }

                                std::vector<long long> srcs;
                                std::vector<long long> dsts;
                                std::vector<std::vector<double>> edge_attrs;
                                for (size_t i = 0; i < nodes.size(); ++i)
                                {
                                    Node *p = nodes[i];
                                    for (int d = 0; d < 8; ++d)
                                    {
                                        Node *q = p->neighbours[d];
                                        if (!q)
                                            continue;
                                        auto it = idx_map.find(q);
                                        if (it == idx_map.end())
                                            continue;
                                        int j = it->second;
                                        double dx = double(q->x - p->x);
                                        double dy = double(q->y - p->y);
                                        double is_diag = (std::abs(dx) == 1.0 && std::abs(dy) == 1.0) ? 1.0 : 0.0;
                                        double orth = 1.0 - is_diag;
                                        srcs.push_back((long long)i);
                                        dsts.push_back((long long)j);
                                        edge_attrs.push_back({orth, is_diag});
                                    }
                                }

                                py::object x_tensor = torch_module.attr("tensor")(py::cast(node_feats));
                                py::object edge_index;
                                if (!srcs.empty())
                                {
                                    std::vector<std::vector<long long>> ei = {srcs, dsts};
                                    edge_index = torch_module.attr("tensor")(py::cast(ei));
                                }
                                else
                                {
                                    edge_index = torch_module.attr("empty")(py::make_tuple(2, 0));
                                }
                                py::object edge_attr = edge_attrs.empty() ? torch_module.attr("empty")(py::make_tuple(0, 2)) : torch_module.attr("tensor")(py::cast(edge_attrs));
                                py::object batch = torch_module.attr("zeros")(py::make_tuple((py::int_)nodes.size())).attr("to")(torch_module.attr("long"));

                                double turn = (double)g.turn_number;
                                std::vector<double> cur_one_hot(GameState::num_players, 0.0);
                                if (g.current_player >= 0 && g.current_player < GameState::num_players)
                                    cur_one_hot[g.current_player] = 1.0;
                                std::vector<double> scores;
                                for (auto s : g.players_scores)
                                    scores.push_back((double)s);
                                std::vector<double> rocks_left;
                                for (auto r : g.num_rocks)
                                    rocks_left.push_back((double)r);
                                std::vector<double> rocks_placed(GameState::num_players, 0.0);
                                for (auto *n : nodes)
                                {
                                    if (n->rocked_by != -1)
                                    {
                                        rocks_placed[n->rocked_by] += 1.0;
                                    }
                                }
                                double max_r2 = 0.0;
                                for (auto &pr : coords)
                                {
                                    double r2 = double(pr[0] * pr[0] + pr[1] * pr[1]);
                                    if (r2 > max_r2)
                                        max_r2 = r2;
                                }
                                std::vector<double> global_feats;
                                global_feats.push_back(turn);
                                for (double v : cur_one_hot)
                                    global_feats.push_back(v);
                                for (double v : scores)
                                    global_feats.push_back(v);
                                for (double v : rocks_left)
                                    global_feats.push_back(v);
                                for (double v : rocks_placed)
                                    global_feats.push_back(v);
                                global_feats.push_back(max_r2);

                                py::object global_tensor = torch_module.attr("tensor")(py::cast(global_feats)).attr("unsqueeze")(0);

                                auto enc_start = std::chrono::high_resolution_clock::now();
                                py::object data = pyg_data_Data("x"_a = x_tensor, "edge_index"_a = edge_index, "edge_attr"_a = edge_attr, "batch"_a = batch, "global_feats"_a = global_tensor);
                                data.attr("node_coords") = torch_module.attr("tensor")(py::cast(coords));
                                auto enc_end = std::chrono::high_resolution_clock::now();
                                total_encode_time += std::chrono::duration<double>(enc_end - enc_start).count();

                                py::object enc_obj = types_module.attr("SimpleNamespace")("data"_a = data);
                                encs.append(enc_obj);
                                valid_indices.push_back(mi);

                                if (enc_cache.size() > ENC_CACHE_MAX)
                                    enc_cache.clear();
                                enc_cache[enc_key] = enc_obj;

                                guard.disarm();
                                g.undo_move();
                            }

                            if (!valid_indices.empty())
                            {
                                log_stage("stage4_eval_model");
                                auto model_start = std::chrono::high_resolution_clock::now();
                                py::list probs_list = eval_probs(encs);
                                auto model_end = std::chrono::high_resolution_clock::now();
                                total_model_time += std::chrono::duration<double>(model_end - model_start).count();
                                for (size_t j = 0; j < valid_indices.size(); ++j)
                                {
                                    size_t mi = valid_indices[j];
                                    double p = py::cast<double>(probs_list[j]);
                                    root_priors[mk_of(moves[mi])] = p;
                                }
                            }
                        }
                        catch (const py::error_already_set &e)
                        {
                            log_stage("stage_err_pycall");
                            throw std::runtime_error(
                                std::string("GNN evaluation is mandatory and failed. ") +
                                "Ensure a GNN evaluator is loaded in Python (call gnn.model.load_model(...)) before running MCTSPlayerCPP. " +
                                std::string("Python error: ") + e.what());
                        }
                    }
                }
            }

            // AlphaZero-style root exploration: mix Dirichlet noise into root priors.
            if (skey == root_key && dirichlet_epsilon > 0.0 && dirichlet_alpha > 0.0 && moves.size() > 1)
            {
                std::gamma_distribution<double> gamma(dirichlet_alpha, 1.0);
                std::vector<double> noise;
                noise.reserve(moves.size());
                double sum = 0.0;
                for (size_t i = 0; i < moves.size(); ++i)
                {
                    double v = std::max(0.0, gamma(rng));
                    noise.push_back(v);
                    sum += v;
                }
                if (sum > 0.0)
                {
                    for (size_t i = 0; i < moves.size(); ++i)
                    {
                        MoveKey mk = mk_of(moves[i]);
                        double p = 0.0;
                        auto itp = root_priors.find(mk);
                        if (itp != root_priors.end())
                            p = itp->second;
                        double n = noise[i] / sum;
                        root_priors[mk] = (1.0 - dirichlet_epsilon) * p + dirichlet_epsilon * n;
                    }
                }
            }

            double total = 0.0;
            for (auto &m : moves)
            {
                MoveKey mk = mk_of(m);
                EdgeKey ek{skey, mk};
                double p = 1.0;
                if (skey == root_key)
                {
                    auto it = root_priors.find(mk);
                    if (it != root_priors.end())
                        p = it->second;
                }
                Psa[ek] = p;
                total += p;
            }
            if (total <= 0.0)
            {
                for (auto &m : moves)
                {
                    MoveKey mk = mk_of(m);
                    EdgeKey ek{skey, mk};
                    Psa[ek] = 1.0 / std::max(1, (int)moves.size());
                }
            }
            else
            {
                for (auto &m : moves)
                {
                    MoveKey mk = mk_of(m);
                    EdgeKey ek{skey, mk};
                    Psa[ek] /= total;
                }
            }
        };

        auto ensure_progressive_widening = [&](const TTKey &skey)
        {
            auto it = Ns.find(skey);
            int ns = (it != Ns.end()) ? it->second : 0;
            int target = (int)std::floor(progressive_widening_c * std::pow((double)(ns + 1), progressive_widening_alpha));
            int min_k = 1;
            if (skey == root_key && legal_moves[skey].size() > 1)
                min_k = 6;
            target = std::max(min_k, std::min(target, (int)legal_moves[skey].size()));
            if (expanded_count[skey] < target)
                expanded_count[skey] = target;
        };

        auto ensure_py_modules = [&]
        {
            if (torch_module)
                return;
            py::gil_scoped_acquire gil;
            torch_module = py::module::import("torch");
            pyg_data_module = py::module::import("torch_geometric.data");
            pyg_data_Data = pyg_data_module.attr("Data");
            gnn_module = py::module::import("gnn.model");
            types_module = py::module::import("types");
        };

        auto encode_state = [&](GameState &g) -> py::object
        {
            ensure_py_modules();
            std::uint64_t enc_key = g.state_key();
            auto it_cache = enc_cache.find(enc_key);
            if (it_cache != enc_cache.end())
                return it_cache->second;

            g.scratch_node_set.clear();
            if (g.connected_points.size() + g.rocks.size() > 0)
                g.scratch_node_set.reserve(g.connected_points.size() + g.rocks.size());
            for (auto *p : g.connected_points)
                g.scratch_node_set.insert(p);
            for (auto *p : g.rocks)
                g.scratch_node_set.insert(p);

            g.scratch_nodes.clear();
            g.scratch_nodes.reserve(g.scratch_node_set.size());
            for (auto *p : g.scratch_node_set)
                g.scratch_nodes.push_back(p);
            std::sort(g.scratch_nodes.begin(), g.scratch_nodes.end(), [](Node *a, Node *b)
                      {
                if (a->x != b->x)
                    return a->x < b->x;
                return a->y < b->y; });

            g.scratch_idx_map.clear();
            if (!g.scratch_nodes.empty())
                g.scratch_idx_map.reserve(g.scratch_nodes.size() * 2);
            for (size_t i = 0; i < g.scratch_nodes.size(); ++i)
                g.scratch_idx_map[g.scratch_nodes[i]] = (int)i;

            std::vector<std::vector<double>> node_feats;
            node_feats.reserve(g.scratch_nodes.size());
            std::vector<std::array<long long, 2>> coords;
            coords.reserve(g.scratch_nodes.size());
            for (auto *n : g.scratch_nodes)
            {
                int num_players = GameState::num_players;
                std::vector<double> owner_one_hot(num_players + 1, 0.0);
                int owner_idx = (n->rocked_by >= 0) ? (n->rocked_by + 1) : 0;
                if (owner_idx >= 0 && owner_idx < (int)owner_one_hot.size())
                    owner_one_hot[owner_idx] = 1.0;
                int neighbour_count = 0;
                for (int d = 0; d < 8; ++d)
                    if (n->neighbours[d])
                        neighbour_count++;
                double deg = (double)neighbour_count / 8.0;
                double is_leaf = (neighbour_count == 1) ? 1.0 : 0.0;
                double x = (double)n->x;
                double y = (double)n->y;
                double r2 = x * x + y * y;
                std::vector<double> feats = owner_one_hot;
                feats.push_back(deg);
                feats.push_back(is_leaf);
                feats.push_back(x);
                feats.push_back(y);
                feats.push_back(r2);
                node_feats.push_back(std::move(feats));
                coords.push_back({n->x, n->y});
            }

            std::vector<long long> srcs;
            std::vector<long long> dsts;
            std::vector<std::vector<double>> edge_attrs;
            for (size_t i = 0; i < g.scratch_nodes.size(); ++i)
            {
                Node *p = g.scratch_nodes[i];
                for (int d = 0; d < 8; ++d)
                {
                    Node *q = p->neighbours[d];
                    if (!q)
                        continue;
                    auto it_idx = g.scratch_idx_map.find(q);
                    if (it_idx == g.scratch_idx_map.end())
                        continue;
                    int j = it_idx->second;
                    double dx = double(q->x - p->x);
                    double dy = double(q->y - p->y);
                    double is_diag = (std::abs(dx) == 1.0 && std::abs(dy) == 1.0) ? 1.0 : 0.0;
                    double orth = 1.0 - is_diag;
                    srcs.push_back((long long)i);
                    dsts.push_back((long long)j);
                    edge_attrs.push_back({orth, is_diag});
                }
            }

            py::gil_scoped_acquire gil;
            py::object x_tensor = torch_module.attr("tensor")(py::cast(node_feats));
            py::object edge_index;
            if (!srcs.empty())
            {
                std::vector<std::vector<long long>> ei = {srcs, dsts};
                edge_index = torch_module.attr("tensor")(py::cast(ei));
            }
            else
            {
                edge_index = torch_module.attr("empty")(py::make_tuple(2, 0));
            }
            py::object edge_attr = edge_attrs.empty() ? torch_module.attr("empty")(py::make_tuple(0, 2)) : torch_module.attr("tensor")(py::cast(edge_attrs));
            py::object batch = torch_module.attr("zeros")(py::make_tuple((py::int_)g.scratch_nodes.size())).attr("to")(torch_module.attr("long"));

            double turn = (double)g.turn_number;
            std::vector<double> cur_one_hot(GameState::num_players, 0.0);
            if (g.current_player >= 0 && g.current_player < GameState::num_players)
                cur_one_hot[g.current_player] = 1.0;
            std::vector<double> scores;
            for (auto s : g.players_scores)
                scores.push_back((double)s);
            std::vector<double> rocks_left;
            for (auto r : g.num_rocks)
                rocks_left.push_back((double)r);
            std::vector<double> rocks_placed(GameState::num_players, 0.0);
            for (auto *n : g.scratch_nodes)
            {
                if (n->rocked_by != -1)
                    rocks_placed[n->rocked_by] += 1.0;
            }
            double max_r2 = 0.0;
            for (auto &pr : coords)
            {
                double r2 = double(pr[0] * pr[0] + pr[1] * pr[1]);
                if (r2 > max_r2)
                    max_r2 = r2;
            }
            std::vector<double> global_feats;
            global_feats.push_back(turn);
            for (double v : cur_one_hot)
                global_feats.push_back(v);
            for (double v : scores)
                global_feats.push_back(v);
            for (double v : rocks_left)
                global_feats.push_back(v);
            for (double v : rocks_placed)
                global_feats.push_back(v);
            global_feats.push_back(max_r2);

            py::object global_tensor = torch_module.attr("tensor")(py::cast(global_feats)).attr("unsqueeze")(0);

            py::object data = pyg_data_Data("x"_a = x_tensor, "edge_index"_a = edge_index, "edge_attr"_a = edge_attr, "batch"_a = batch, "global_feats"_a = global_tensor);
            data.attr("node_coords") = torch_module.attr("tensor")(py::cast(coords));
            py::object enc_obj = types_module.attr("SimpleNamespace")("data"_a = data);

            if (enc_cache.size() > ENC_CACHE_MAX)
                enc_cache.clear();
            enc_cache[enc_key] = enc_obj;
            return enc_obj;
        };

        auto gnn_value = [&](GameState &g, int perspective) -> double
        {
            if (g.winner != -1)
                return (g.winner == perspective) ? 1.0 : 0.0;
            // Avoid asking the model to score an empty graph; return neutral instead.
            if (g.connected_points.empty() && g.rocks.empty())
                return 0.5;
            ensure_py_modules();
            py::gil_scoped_acquire gil;
            py::list encs;
            encs.append(encode_state(g));

            if (!model_override.is_none())
            {
                py::list datas;
                for (auto item : encs)
                {
                    py::object enc_obj = py::cast<py::object>(item);
                    datas.append(enc_obj.attr("data"));
                }
                py::object batch = pyg_data_Batch.attr("from_data_list")(datas);
                if (!model_device.empty())
                    batch = batch.attr("to")(py::cast(model_device));

                py::object no_grad = torch_module.attr("no_grad")();
                no_grad.attr("__enter__")();
                py::object logits = model_override(batch);
                no_grad.attr("__exit__")(py::none(), py::none(), py::none());
                py::object probs = torch_module.attr("sigmoid")(logits).attr("detach")().attr("cpu")();
                py::list probs_list = py::cast<py::list>(probs.attr("tolist")());
                if (py::len(probs_list) == 0)
                    return 0.5;
                double p = py::cast<double>(probs_list[0]);
                if (g.current_player != perspective)
                    p = 1.0 - p;
                if (p < 1e-6)
                    p = 1e-6;
                if (p > 1.0 - 1e-6)
                    p = 1.0 - 1e-6;
                return p;
            }

            py::object py_probs = gnn_module.attr("evaluate_encodings")(encs);
            py::list probs_list = py::cast<py::list>(py_probs);
            if (py::len(probs_list) == 0)
                return 0.5;
            double p = py::cast<double>(probs_list[0]);
            if (g.current_player != perspective)
                p = 1.0 - p;
            if (p < 1e-6)
                p = 1e-6;
            if (p > 1.0 - 1e-6)
                p = 1.0 - 1e-6;
            return p;
        };

        auto puct_score = [&](const TTKey &skey, const Move &m)
        {
            MoveKey mk = mk_of(m);
            EdgeKey ek{skey, mk};
            auto it_nsa = Nsa.find(ek);
            int nsa = (it_nsa != Nsa.end()) ? it_nsa->second : 0;
            auto it_wsa = Wsa.find(ek);
            double wsa = (it_wsa != Wsa.end()) ? it_wsa->second : 0.0;
            double q_ucb = (nsa > 0) ? (wsa / (double)nsa) : 0.5;
            double q = q_ucb;
            if (skey == root_key && rave_k > 0)
            {
                EdgeKey rek{root_key, mk};
                auto it_na = N_amaf.find(rek);
                int n_amaf = (it_na != N_amaf.end()) ? it_na->second : 0;
                if (n_amaf > 0)
                {
                    auto it_wa = W_amaf.find(rek);
                    double w_amaf = (it_wa != W_amaf.end()) ? it_wa->second : 0.0;
                    double q_amaf = w_amaf / (double)n_amaf;
                    double beta = rave_k / (rave_k + (double)nsa);
                    q = (1.0 - beta) * q_ucb + beta * q_amaf;
                }
            }
            auto it_p = Psa.find(ek);
            double p = (it_p != Psa.end()) ? it_p->second : 1.0;
            auto it_ns = Ns.find(skey);
            double ns = (it_ns != Ns.end()) ? (double)it_ns->second : 0.0;
            double u = c_puct * p * (std::sqrt(ns + 1e-9) / (1.0 + nsa));
            return q + u;
        };

        _root_key = root_key_digest;
        int root_player = game.current_player;
        ensure_state_initialized(root_key, game);
        auto &root_moves = legal_moves[root_key];
        std::vector<Move> safe_root_moves;
        safe_root_moves.reserve(root_moves.size());
        for (auto &m : root_moves)
        {
            if (!game.is_move_legal(m, game.current_player))
                continue;
            game.do_move(m, game.current_player);
            if (game.winner == root_player)
            {
                game.undo_move();
                return m;
            }
            if (game.winner == -1)
                safe_root_moves.push_back(m);
            game.undo_move();
        }
        std::vector<Move> allowed_root_moves = safe_root_moves;
        if (check_forced_losses && !safe_root_moves.empty())
        {
            int limit = std::min((int)safe_root_moves.size(), tactical_root_limit);
            std::vector<Move> to_check;
            to_check.reserve((size_t)limit);
            to_check.insert(to_check.end(), safe_root_moves.begin(), safe_root_moves.begin() + limit);
            std::vector<Move> non_forced;
            non_forced.reserve(to_check.size());
            for (auto &m : to_check)
            {
                if (!game.allows_forced_loss_next_round(m, const_cast<GameState &>(game), root_player))
                    non_forced.push_back(m);
            }
            if (!non_forced.empty())
                allowed_root_moves = non_forced;
        }

        auto __mcts_start_time = std::chrono::high_resolution_clock::now();
        int __mcts_rollouts_done = 0;

        std::vector<std::tuple<TTKey, Move, int>> path;
        path.reserve((size_t)max_sim_depth + 16);
        std::vector<std::pair<int, MoveKey>> sim_moves;
        sim_moves.reserve((size_t)max_sim_depth + 16);
        std::unordered_set<MoveKey, MoveKeyHash> root_played;
        root_played.reserve((size_t)max_sim_depth * 2 + 64);

        for (int it = 0; it < n_rollouts; ++it)
        {
            int base_moves = (int)game.moves.size();
            log_rollout("rollout_start", it, 0, game.winner, game.current_player, game.moves.size());
            path.clear();
            TTKey skey = root_key;

            while (true)
            {
                ensure_state_initialized(skey, game);
                ensure_progressive_widening(skey);
                auto &legal = legal_moves[skey];
                int k = expanded_count[skey];
                if (k <= 0 || legal.empty())
                    break;
                int n_expanded = std::min((int)legal.size(), k);
                if (n_expanded <= 0)
                    break;

                double best_score = -1e300;
                Move best_move = legal[0];
                for (int mi = 0; mi < n_expanded; ++mi)
                {
                    const auto &m = legal[mi];
                    double sc = puct_score(skey, m);
                    if (sc > best_score || (sc == best_score && move_less(m, best_move)))
                    {
                        best_score = sc;
                        best_move = m;
                    }
                }

                int mover_idx = game.current_player;
                if (!game.is_move_legal(best_move, game.current_player))
                {
                    std::ostringstream dbg;
                    dbg << "Internal error: MCTS selected an illegal move (legal_moves should be filtered). "
                        << explain_illegal(best_move, game.current_player);
                    throw std::runtime_error(dbg.str());
                }
                game.do_move(best_move, game.current_player);
                path.emplace_back(skey, best_move, mover_idx);

                EdgeKey ek{skey, mk_of(best_move)};
                auto it_edge = Nsa.find(ek);
                int nsa = (it_edge != Nsa.end()) ? it_edge->second : 0;
                if (nsa == 0)
                    break;
                skey = game.tt_key();
            }

            log_rollout("after_selection", it, (int)path.size(), game.winner, game.current_player, game.moves.size());

            int simd = 0;
            sim_moves.clear();
            int sim_start_player = game.current_player;
            while (simd < max_sim_depth)
            {
                if (game.winner != -1)
                    break;
                Move sm = game.rollout_pick_move(game);
                log_rollout("sim_step", it, simd, game.winner, game.current_player, game.moves.size());
                sim_moves.emplace_back(game.current_player, mk_of(sm));
                if (!game.is_move_legal(sm, game.current_player))
                    throw std::runtime_error("Internal error: rollout produced an illegal move");
                game.do_move(sm, game.current_player);
                ++simd;
            }

            log_rollout("sim_end", it, simd, game.winner, game.current_player, game.moves.size());

            double reward = 0.5;
            if (game.winner != -1)
                reward = (game.winner == sim_start_player) ? 1.0 : 0.0;
            else
            {
                const bool bypass_python_value = false;
                if (bypass_python_value)
                {
                    reward = 0.5;
                }
                else
                {
                    int nodes = (int)game.connected_points.size() + (int)game.rocks.size();
                    int edges = 0;
                    for (auto *n : game.connected_points)
                        for (int d = 0; d < 8; ++d)
                            if (n->neighbours[d])
                                edges++;
                    for (auto *n : game.rocks)
                        for (int d = 0; d < 8; ++d)
                            if (n->neighbours[d])
                                edges++;
                    log_value("value_start", simd, nodes, edges);
                    try
                    {
                        reward = gnn_value(game, sim_start_player);
                    }
                    catch (const std::exception &e)
                    {
                        fprintf(stderr, "[players_ext choose_move] value_exception: %s\n", e.what());
                        fflush(stderr);
                        throw;
                    }
                    log_value("value_end", simd, nodes, edges);
                }
            }

            root_played.clear();
            for (auto &pr : sim_moves)
                if (pr.first == sim_start_player)
                    root_played.insert(pr.second);
            for (auto &t : path)
            {
                int mover = std::get<2>(t);
                if (mover == sim_start_player)
                    root_played.insert(mk_of(std::get<1>(t)));
            }
            for (auto &mk : root_played)
            {
                EdgeKey ek{root_key, mk};
                N_amaf[ek] += 1;
                W_amaf[ek] += reward;
            }

            for (int i = (int)path.size() - 1; i >= 0; --i)
            {
                TTKey sk = std::get<0>(path[i]);
                const Move &mv = std::get<1>(path[i]);
                EdgeKey ek{sk, mk_of(mv)};
                Ns[sk] += 1;
                Nsa[ek] += 1;
                Wsa[ek] += reward;
                reward = 1.0 - reward;
            }

            while ((int)game.moves.size() > base_moves)
                game.undo_move();
            log_rollout("rollout_end", it, simd, game.winner, game.current_player, game.moves.size());
            __mcts_rollouts_done++;
        }

        auto __mcts_end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> __mcts_elapsed = __mcts_end_time - __mcts_start_time;
        if (verbose)
        {
            std::cout << "rollouts = " << __mcts_rollouts_done << " for player " << root_player
                      << ". time taken: " << std::fixed << std::setprecision(2) << __mcts_elapsed.count() << "s\n";
        }

        ensure_state_initialized(root_key, game);
        auto &legal_root = legal_moves[root_key];
        int kroot = expanded_count[root_key];
        std::vector<Move> expanded_root(legal_root.begin(), legal_root.begin() + std::min((int)legal_root.size(), kroot));
        if (expanded_root.empty())
        {
            auto moves = game.get_possible_moves_for_player(game.current_player);
            if (moves.empty())
                return Move{0, 0, 'P'};
            return moves[0];
        }

        auto candidates = (!allowed_root_moves.empty()) ? allowed_root_moves : expanded_root;

        auto visits = [&](const Move &m) -> int
        {
            EdgeKey ek{root_key, mk_of(m)};
            auto it = Nsa.find(ek);
            return (it != Nsa.end()) ? it->second : 0;
        };

        int max_vis = -1;
        for (auto &m : candidates)
            max_vis = std::max(max_vis, visits(m));
        std::vector<Move> ranked;
        ranked.reserve(candidates.size());
        for (auto &m : candidates)
            ranked.push_back(m);
        std::sort(ranked.begin(), ranked.end(), [&](const Move &a, const Move &b)
                  {
            int va = visits(a), vb = visits(b);
            if (va != vb) return va > vb;
            if (a.x != b.x) return a.x < b.x;
            if (a.y != b.y) return a.y < b.y;
            return a.t < b.t; });

        int safety_limit = std::min((int)ranked.size(), std::max(tactical_root_limit, 40));
        std::vector<Move> safe_moves;
        safe_moves.reserve((size_t)safety_limit);
        std::vector<double> safe_visits;
        safe_visits.reserve((size_t)safety_limit);

        for (int i = 0; i < safety_limit; ++i)
        {
            Move m = ranked[i];
            if (!game.is_move_legal(m, game.current_player))
                continue;
            game.do_move(m, game.current_player);
            if (game.winner == root_player)
            {
                game.undo_move();
                return m;
            }
            if (game.winner != -1 && game.winner != root_player)
            {
                game.undo_move();
                continue;
            }
            game.undo_move();
            if (game.allows_forced_loss_next_round(m, const_cast<GameState &>(game), root_player))
                continue;

            safe_moves.push_back(m);
            safe_visits.push_back((double)std::max(0, visits(m)));
        }

        if (!safe_moves.empty())
        {
            const bool explore = (temperature > 0.0 && temperature_moves > 0 && game.turn_number < temperature_moves);
            if (explore)
            {
                double inv_temp = 1.0 / std::max(1e-9, temperature);
                std::vector<double> weights;
                weights.reserve(safe_visits.size());
                for (double v : safe_visits)
                    weights.push_back(std::pow(std::max(1e-12, v), inv_temp));
                std::discrete_distribution<size_t> dd(weights.begin(), weights.end());
                return safe_moves[dd(rng)];
            }
            return safe_moves[0];
        }

        for (const auto &m : ranked)
        {
            if (game.is_move_legal(m, game.current_player))
                return m;
        }

        return Move{0, 0, 'P'};
    }

    void set_c_puct(double v) { c_puct = v; }
    void set_verbose(bool v) { verbose = v; }
    void set_progressive_widening(double c, double alpha)
    {
        progressive_widening_c = c;
        progressive_widening_alpha = alpha;
    }
    void set_rave_k(double v) { rave_k = v; }
    void set_prior_eval_cap(int v) { prior_eval_cap = v; }
    void set_max_sim_depth(int v) { max_sim_depth = v; }
    void clear_root_priors() { root_priors.clear(); }

    void set_exploration(double alpha, double epsilon, double temp, int temp_moves)
    {
        dirichlet_alpha = std::max(0.0, alpha);
        dirichlet_epsilon = std::max(0.0, std::min(1.0, epsilon));
        temperature = std::max(0.0, temp);
        temperature_moves = std::max(0, temp_moves);
    }

    void set_model_checkpoint(const std::string &path, const std::string &device)
    {
        py::gil_scoped_acquire gil;

        if (!torch_module)
        {
            torch_module = py::module::import("torch");
            pyg_data_module = py::module::import("torch_geometric.data");
            pyg_data_Data = pyg_data_module.attr("Data");
            pyg_data_Batch = pyg_data_module.attr("Batch");
            gnn_module = py::module::import("gnn.model");
            types_module = py::module::import("types");
        }

        py::object sample_enc = py::module::import("gnn.encode").attr("SAMPLE_ENC");
        int node_dim = sample_enc.attr("data").attr("x").attr("size")(1).cast<int>();
        int global_dim = sample_enc.attr("data").attr("global_feats").attr("size")(1).cast<int>();

        py::object GNNEval = gnn_module.attr("GNNEval");
        py::object model = GNNEval("node_feat_dim"_a = node_dim, "global_feat_dim"_a = global_dim);
        py::object state = torch_module.attr("load")(py::cast(path), "map_location"_a = py::cast(device));
        model.attr("load_state_dict")(state);
        model.attr("to")(py::cast(device));
        model.attr("eval")();

        model_override = model;
        model_device = device;

        // Search caches are model-dependent.
        clear_stats();
    }

    void reset_search()
    {
        clear_stats();
    }

    std::uint64_t get_current_root_key() const { return _root_key; }

    // Return per-move visit counts at the provided root.
    // Each element is a dict: {"x": int, "y": int, "t": str, "visits": int}
    py::list get_root_visit_stats_py(const GameState &root)
    {
        auto &game = const_cast<GameState &>(root);
        TTKey root_key = game.tt_key();

        const std::vector<Move> *moves_ptr = nullptr;
        std::vector<Move> tmp;
        auto it = legal_moves.find(root_key);
        if (it != legal_moves.end())
        {
            moves_ptr = &it->second;
        }
        else
        {
            tmp = game.get_possible_moves_for_player(game.current_player);
            moves_ptr = &tmp;
        }

        py::list out;
        for (const auto &m : *moves_ptr)
        {
            EdgeKey ek{root_key, MoveKey{m.x, m.y, m.t}};
            int v = 0;
            auto itv = Nsa.find(ek);
            if (itv != Nsa.end())
                v = itv->second;

            py::dict d;
            d["x"] = m.x;
            d["y"] = m.y;
            d["t"] = py::cast(std::string(1, m.t));
            d["visits"] = v;
            out.append(d);
        }
        return out;
    }

    // takes iterable of (x,y,t,prior)
    void set_root_priors_py(py::iterable priors)
    {
        root_priors.clear();
        for (auto item : priors)
        {
            py::sequence seq = py::cast<py::sequence>(item);
            if (seq.size() != 4)
                continue;
            int x = seq[0].cast<int>();
            int y = seq[1].cast<int>();
            std::string t = seq[2].cast<std::string>();
            double p = seq[3].cast<double>();
            char tc = t.empty() ? 'P' : t[0];
            root_priors[MoveKey{x, y, tc}] = p;
        }
    }

    void clear_stats()
    {
        Ns.clear();
        Nsa.clear();
        Wsa.clear();
        Psa.clear();
        N_amaf.clear();
        W_amaf.clear();
        legal_moves.clear();
        expanded_count.clear();
        enc_cache.clear();
        _root_key = 0;

        total_encode_time = 0.0;
        total_model_time = 0.0;
    }

    py::dict get_profile_stats()
    {
        py::dict d;
        d["total_encode_time"] = total_encode_time;
        d["total_model_time"] = total_model_time;
        return d;
    }

    void advance_root(const GameState &game)
    {
        _root_key = ttkey_digest(game.tt_key());
    }

    void prune_tables(int max_states)
    {
        if (max_states <= 0)
            return;
        int cur = (int)Ns.size();
        if (cur <= max_states)
            return;

        std::vector<std::pair<TTKey, int>> items;
        items.reserve(Ns.size());
        for (auto &kv : Ns)
            items.emplace_back(kv.first, kv.second);
        std::sort(items.begin(), items.end(), [](auto &a, auto &b)
                  { return a.second < b.second; });
        int remove_count = cur - max_states;
        std::unordered_set<TTKey, TTKeyHash> to_remove;
        for (int i = 0; i < remove_count; ++i)
            to_remove.insert(items[i].first);

        for (auto &k : to_remove)
        {
            Ns.erase(k);
            legal_moves.erase(k);
            expanded_count.erase(k);
        }

        auto erase_edges_for_removed = [&](auto &map_like)
        {
            std::vector<typename std::remove_reference_t<decltype(map_like)>::key_type> keys;
            keys.reserve(map_like.size());
            for (auto &kv : map_like)
                keys.push_back(kv.first);
            for (auto &key : keys)
            {
                if (to_remove.find(key.s) != to_remove.end())
                    map_like.erase(key);
            }
        };

        erase_edges_for_removed(Nsa);
        erase_edges_for_removed(Wsa);
        erase_edges_for_removed(Psa);
        erase_edges_for_removed(N_amaf);
        erase_edges_for_removed(W_amaf);
    }

private:
    std::mt19937 rng;
    double c_puct;
    bool verbose = true;
    double progressive_widening_c;
    double progressive_widening_alpha;
    double rave_k;
    int prior_eval_cap;
    int max_sim_depth;
    bool check_forced_losses;
    int tactical_root_limit;

    std::unordered_map<TTKey, int, TTKeyHash> Ns;
    std::unordered_map<EdgeKey, int, EdgeKeyHash> Nsa;
    std::unordered_map<EdgeKey, double, EdgeKeyHash> Wsa;
    std::unordered_map<EdgeKey, double, EdgeKeyHash> Psa;
    std::unordered_map<EdgeKey, int, EdgeKeyHash> N_amaf;
    std::unordered_map<EdgeKey, double, EdgeKeyHash> W_amaf;
    std::unordered_map<TTKey, std::vector<Move>, TTKeyHash> legal_moves;
    std::unordered_map<TTKey, int, TTKeyHash> expanded_count;
    std::uint64_t _root_key = 0;
    std::unordered_map<MoveKey, double, MoveKeyHash> root_priors;

    py::object torch_module;
    py::object pyg_data_module;
    py::object pyg_data_Data;
    py::object pyg_data_Batch;
    py::object gnn_module;
    py::object types_module;

    py::object model_override = py::none();
    std::string model_device = "cpu";

    double dirichlet_alpha = 0.0;
    double dirichlet_epsilon = 0.0;
    double temperature = 0.0;
    int temperature_moves = 0;

    std::unordered_map<std::uint64_t, py::object> enc_cache;
    static constexpr size_t ENC_CACHE_MAX = 4096;

    double total_encode_time = 0.0;
    double total_model_time = 0.0;
};

class AlphaBetaEngine
{
public:
    AlphaBetaEngine(int seed = 0, double pass_penalty = 1.2)
        : rng((seed == 0) ? std::mt19937(std::random_device{}()) : std::mt19937((std::uint32_t)seed)), pass_penalty(pass_penalty)
    {
    }

    Move choose_move(const GameState &root, int depth = 3, int move_cap = 48)
    {
        auto &game = const_cast<GameState &>(root);

        struct RngGuard
        {
            GameState &g;
            std::mt19937 snapshot;
            explicit RngGuard(GameState &gs) : g(gs), snapshot(gs.rng_snapshot()) {}
            ~RngGuard() { g.rng_restore(snapshot); }
        } rng_guard(game);

        if (depth <= 0)
        {
            auto moves = game.get_possible_moves_for_player(game.current_player);
            if (moves.empty())
                return Move{0, 0, 'P'};
            return moves[0];
        }

        root_player = game.current_player;
        if (last_root_player != -1 && root_player != last_root_player)
        {
            // TT / eval caches depend on root_player perspective.
            tt.clear();
            eval_cache.clear();
        }
        last_root_player = root_player;
        this->move_cap = std::max(1, move_cap);

        {
            auto moves = game.get_possible_moves_for_player(game.current_player);
            order_moves_inplace(moves);
            for (auto &m : moves)
            {
                game.do_move(m, game.current_player);
                if (game.winner == root_player)
                {
                    game.undo_move();
                    return m;
                }
                game.undo_move();
            }
        }

        auto moves = game.get_possible_moves_for_player(game.current_player);
        if (!moves.empty())
        {
            moves = filter_search_moves(moves, game, game.current_player);
        }
        if (moves.empty())
            return Move{0, 0, 'P'};

        order_moves_inplace(moves);
        if ((int)moves.size() > this->move_cap)
            moves.resize((size_t)this->move_cap);

        Move best_move = moves[0];
        double best_value = -1e300;
        double alpha = -1e300;
        double beta = 1e300;

        for (auto &m : moves)
        {
            if (!game.is_move_legal(m, game.current_player))
                continue;
            int mover = game.current_player;
            game.do_move(m, mover);
            double v = alpha_beta(game, depth - 1, alpha, beta);
            game.undo_move();

            if (m.t == 'P')
                v -= pass_penalty;

            if (v > best_value || (v == best_value && move_less(m, best_move)))
            {
                best_value = v;
                best_move = m;
            }
            alpha = std::max(alpha, best_value);
            if (alpha >= beta)
                break;
        }

        return best_move;
    }

    void set_model_checkpoint(const std::string &path, const std::string &device)
    {
        py::gil_scoped_acquire gil;

        if (!torch_module)
        {
            torch_module = py::module::import("torch");
            pyg_data_module = py::module::import("torch_geometric.data");
            pyg_data_Data = pyg_data_module.attr("Data");
            pyg_data_Batch = pyg_data_module.attr("Batch");
            gnn_module = py::module::import("gnn.model");
            types_module = py::module::import("types");
        }

        py::object sample_enc = py::module::import("gnn.encode").attr("SAMPLE_ENC");
        int node_dim = sample_enc.attr("data").attr("x").attr("size")(1).cast<int>();
        int global_dim = sample_enc.attr("data").attr("global_feats").attr("size")(1).cast<int>();

        py::object GNNEval = gnn_module.attr("GNNEval");
        py::object model = GNNEval("node_feat_dim"_a = node_dim, "global_feat_dim"_a = global_dim);
        py::object state = torch_module.attr("load")(py::cast(path), "map_location"_a = py::cast(device));
        model.attr("load_state_dict")(state);
        model.attr("to")(py::cast(device));
        model.attr("eval")();

        model_override = model;
        model_device = device;

        // Caches depend on the model.
        clear_stats();
    }

    void clear_stats()
    {
        tt.clear();
        eval_cache.clear();
        enc_cache.clear();
        total_encode_time = 0.0;
        total_model_time = 0.0;
    }

    py::dict get_profile_stats()
    {
        py::dict d;
        d["total_encode_time"] = total_encode_time;
        d["total_model_time"] = total_model_time;
        d["model_calls"] = (int)model_calls;
        d["model_batch_items"] = (int)model_batch_items;
        d["tt_entries"] = (int)tt.size();
        d["eval_cache_entries"] = (int)eval_cache.size();
        d["enc_cache_entries"] = (int)enc_cache.size();
        return d;
    }

private:
    struct TTEntry
    {
        int depth = 0;
        double value = 0.0;
        // 0 exact, 1 lower bound, 2 upper bound
        int flag = 0;
        Move best{0, 0, 'P'};
    };

    static inline bool move_less(const Move &a, const Move &b)
    {
        if (a.x != b.x)
            return a.x < b.x;
        if (a.y != b.y)
            return a.y < b.y;
        return a.t < b.t;
    }

    static inline int move_type_rank(const Move &m)
    {
        if (m.t == 'P')
            return 3;
        if (m.t == 'R')
            return 2;
        return 1;
    }

    static void order_moves_inplace(std::vector<Move> &moves)
    {
        std::sort(moves.begin(), moves.end(), [](const Move &a, const Move &b)
                  {
            int ra = move_type_rank(a);
            int rb = move_type_rank(b);
            if (ra != rb) return ra < rb;
            return move_less(a,b); });
    }

    static bool rock_is_search_worthy(GameState &g, const Move &m)
    {
        if (m.t != 'R')
            return true;
        auto it = g.points.find(GameState::key_from_coord({m.x, m.y}));
        if (it != g.points.end())
        {
            Node *p = it->second.get();
            if (p->in_connected_points)
                return true;
            for (int d = 0; d < 8; ++d)
                if (p->neighbours[d])
                    return true;
        }

        std::unordered_set<std::uint64_t> rock_coords;
        rock_coords.reserve(g.rocks.size() * 2 + 8);
        for (Node *r : g.rocks)
            rock_coords.insert(GameState::key_from_coord({r->x, r->y}));
        int adjacent = 0;
        for (int dx = -1; dx <= 1; ++dx)
        {
            for (int dy = -1; dy <= 1; ++dy)
            {
                if (dx == 0 && dy == 0)
                    continue;
                if (rock_coords.find(GameState::key_from_coord({m.x + dx, m.y + dy})) != rock_coords.end())
                {
                    adjacent++;
                    if (adjacent >= 2)
                        return true;
                }
            }
        }
        return false;
    }

    static std::vector<Move> filter_search_moves(const std::vector<Move> &moves, GameState &g, int player)
    {
        std::vector<Move> out;
        out.reserve(moves.size());
        for (const auto &m : moves)
        {
            if (m.t == 'R' && !rock_is_search_worthy(g, m))
                continue;
            out.push_back(m);
        }
        return out;
    }

    static inline double clamp_prob(double p)
    {
        if (p < 1e-4)
            p = 1e-4;
        if (p > 1.0 - 1e-4)
            p = 1.0 - 1e-4;
        return p;
    }

    static inline double prob_to_value(double prob)
    {
        // Match players/ai.py::_evaluate_with_gnn scaling.
        prob = clamp_prob(prob);
        double logit = std::log(prob / (1.0 - prob));
        logit /= 2.0;
        double p2 = 1.0 / (1.0 + std::exp(-logit));
        double x = (p2 - 0.5) * 2.0;
        x = std::max(-0.999999, std::min(0.999999, x));
        return 6.0 * std::atanh(x);
    }

    void ensure_py_modules()
    {
        if (torch_module)
            return;
        py::gil_scoped_acquire gil;
        torch_module = py::module::import("torch");
        pyg_data_module = py::module::import("torch_geometric.data");
        pyg_data_Data = pyg_data_module.attr("Data");
        pyg_data_Batch = pyg_data_module.attr("Batch");
        gnn_module = py::module::import("gnn.model");
        types_module = py::module::import("types");
    }

    py::object encode_state(GameState &g)
    {
        ensure_py_modules();

        std::uint64_t enc_key = g.state_key();
        auto it_cache = enc_cache.find(enc_key);
        if (it_cache != enc_cache.end())
            return it_cache->second;

        g.scratch_node_set.clear();
        if (g.connected_points.size() + g.rocks.size() > 0)
            g.scratch_node_set.reserve(g.connected_points.size() + g.rocks.size());
        for (auto *p : g.connected_points)
            g.scratch_node_set.insert(p);
        for (auto *p : g.rocks)
            g.scratch_node_set.insert(p);

        g.scratch_nodes.clear();
        g.scratch_nodes.reserve(g.scratch_node_set.size());
        for (auto *p : g.scratch_node_set)
            g.scratch_nodes.push_back(p);
        std::sort(g.scratch_nodes.begin(), g.scratch_nodes.end(), [](Node *a, Node *b)
                  {
            if (a->x != b->x) return a->x < b->x;
            return a->y < b->y; });

        g.scratch_idx_map.clear();
        if (!g.scratch_nodes.empty())
            g.scratch_idx_map.reserve(g.scratch_nodes.size() * 2);
        for (size_t i = 0; i < g.scratch_nodes.size(); ++i)
            g.scratch_idx_map[g.scratch_nodes[i]] = (int)i;

        std::vector<std::vector<double>> node_feats;
        node_feats.reserve(g.scratch_nodes.size());
        std::vector<std::array<long long, 2>> coords;
        coords.reserve(g.scratch_nodes.size());
        for (auto *n : g.scratch_nodes)
        {
            int num_players = GameState::num_players;
            std::vector<double> owner_one_hot(num_players + 1, 0.0);
            int owner_idx = (n->rocked_by >= 0) ? (n->rocked_by + 1) : 0;
            if (owner_idx >= 0 && owner_idx < (int)owner_one_hot.size())
                owner_one_hot[owner_idx] = 1.0;
            int neighbour_count = 0;
            for (int d = 0; d < 8; ++d)
                if (n->neighbours[d])
                    neighbour_count++;
            double deg = (double)neighbour_count / 8.0;
            double is_leaf = (neighbour_count == 1) ? 1.0 : 0.0;
            double x = (double)n->x;
            double y = (double)n->y;
            double r2 = x * x + y * y;
            std::vector<double> feats = owner_one_hot;
            feats.push_back(deg);
            feats.push_back(is_leaf);
            feats.push_back(x);
            feats.push_back(y);
            feats.push_back(r2);
            node_feats.push_back(std::move(feats));
            coords.push_back({n->x, n->y});
        }

        std::vector<long long> srcs;
        std::vector<long long> dsts;
        std::vector<std::vector<double>> edge_attrs;
        for (size_t i = 0; i < g.scratch_nodes.size(); ++i)
        {
            Node *p = g.scratch_nodes[i];
            for (int d = 0; d < 8; ++d)
            {
                Node *q = p->neighbours[d];
                if (!q)
                    continue;
                auto it_idx = g.scratch_idx_map.find(q);
                if (it_idx == g.scratch_idx_map.end())
                    continue;
                int j = it_idx->second;
                double dx = double(q->x - p->x);
                double dy = double(q->y - p->y);
                double is_diag = (std::abs(dx) == 1.0 && std::abs(dy) == 1.0) ? 1.0 : 0.0;
                double orth = 1.0 - is_diag;
                srcs.push_back((long long)i);
                dsts.push_back((long long)j);
                edge_attrs.push_back({orth, is_diag});
            }
        }

        py::gil_scoped_acquire gil;
        auto enc_start = std::chrono::high_resolution_clock::now();

        py::object x_tensor = torch_module.attr("tensor")(py::cast(node_feats));
        py::object edge_index;
        if (!srcs.empty())
        {
            std::vector<std::vector<long long>> ei = {srcs, dsts};
            edge_index = torch_module.attr("tensor")(py::cast(ei));
        }
        else
        {
            edge_index = torch_module.attr("empty")(py::make_tuple(2, 0));
        }
        py::object edge_attr = edge_attrs.empty() ? torch_module.attr("empty")(py::make_tuple(0, 2)) : torch_module.attr("tensor")(py::cast(edge_attrs));
        py::object batch = torch_module.attr("zeros")(py::make_tuple((py::int_)g.scratch_nodes.size())).attr("to")(torch_module.attr("long"));

        double turn = (double)g.turn_number;
        std::vector<double> cur_one_hot(GameState::num_players, 0.0);
        if (g.current_player >= 0 && g.current_player < GameState::num_players)
            cur_one_hot[g.current_player] = 1.0;
        std::vector<double> scores;
        for (auto s : g.players_scores)
            scores.push_back((double)s);
        std::vector<double> rocks_left;
        for (auto r : g.num_rocks)
            rocks_left.push_back((double)r);
        std::vector<double> rocks_placed(GameState::num_players, 0.0);
        for (auto *n : g.scratch_nodes)
            if (n->rocked_by != -1)
                rocks_placed[n->rocked_by] += 1.0;
        double max_r2 = 0.0;
        for (auto &pr : coords)
        {
            double r2 = double(pr[0] * pr[0] + pr[1] * pr[1]);
            if (r2 > max_r2)
                max_r2 = r2;
        }
        std::vector<double> global_feats;
        global_feats.push_back(turn);
        for (double v : cur_one_hot)
            global_feats.push_back(v);
        for (double v : scores)
            global_feats.push_back(v);
        for (double v : rocks_left)
            global_feats.push_back(v);
        for (double v : rocks_placed)
            global_feats.push_back(v);
        global_feats.push_back(max_r2);
        py::object global_tensor = torch_module.attr("tensor")(py::cast(global_feats)).attr("unsqueeze")(0);

        py::object data = pyg_data_Data("x"_a = x_tensor, "edge_index"_a = edge_index, "edge_attr"_a = edge_attr, "batch"_a = batch, "global_feats"_a = global_tensor);
        data.attr("node_coords") = torch_module.attr("tensor")(py::cast(coords));
        py::object enc_obj = types_module.attr("SimpleNamespace")("data"_a = data);

        auto enc_end = std::chrono::high_resolution_clock::now();
        total_encode_time += std::chrono::duration<double>(enc_end - enc_start).count();

        if (enc_cache.size() > ENC_CACHE_MAX)
            enc_cache.clear();
        enc_cache[enc_key] = enc_obj;
        return enc_obj;
    }

    double gnn_prob_root(GameState &g)
    {
        // Returns P(root_player wins) from current state.
        if (g.winner != -1)
            return (g.winner == root_player) ? 1.0 : 0.0;
        if (g.connected_points.empty() && g.rocks.empty())
            return 0.5;

        ensure_py_modules();
        py::gil_scoped_acquire gil;
        py::list encs;
        encs.append(encode_state(g));

        double p = 0.5;
        try
        {
            auto model_start = std::chrono::high_resolution_clock::now();
            model_calls += 1;
            model_batch_items += 1;
            if (!model_override.is_none())
            {
                py::list datas;
                for (auto item : encs)
                {
                    py::object enc_obj = py::cast<py::object>(item);
                    datas.append(enc_obj.attr("data"));
                }
                py::object batch = pyg_data_Batch.attr("from_data_list")(datas);
                if (!model_device.empty())
                    batch = batch.attr("to")(py::cast(model_device));

                py::object no_grad = torch_module.attr("no_grad")();
                no_grad.attr("__enter__")();
                py::object logits = model_override(batch);
                no_grad.attr("__exit__")(py::none(), py::none(), py::none());
                py::object probs = torch_module.attr("sigmoid")(logits).attr("detach")().attr("cpu")();
                py::list probs_list = py::cast<py::list>(probs.attr("tolist")());
                p = (py::len(probs_list) > 0) ? py::cast<double>(probs_list[0]) : 0.5;
            }
            else
            {
                py::object py_probs = gnn_module.attr("evaluate_encodings")(encs);
                py::list probs_list = py::cast<py::list>(py_probs);
                p = (py::len(probs_list) > 0) ? py::cast<double>(probs_list[0]) : 0.5;
            }
            auto model_end = std::chrono::high_resolution_clock::now();
            total_model_time += std::chrono::duration<double>(model_end - model_start).count();
        }
        catch (const py::error_already_set &e)
        {
            throw std::runtime_error(
                std::string("GNN evaluation is mandatory for AlphaBetaEngine and failed. ") +
                "Ensure a GNN evaluator is loaded in Python (call gnn.model.load_model(...)) or use set_model_checkpoint(...). " +
                std::string("Python error: ") + e.what());
        }

        if (g.current_player != root_player)
            p = 1.0 - p;
        return clamp_prob(p);
    }

    std::vector<double> gnn_probs_root_for_encodings(const py::list &encs)
    {
        // Returns P(root_player wins) for each encoding.
        ensure_py_modules();
        py::gil_scoped_acquire gil;

        std::vector<double> probs;
        probs.reserve((size_t)py::len(encs));

        try
        {
            auto model_start = std::chrono::high_resolution_clock::now();
            model_calls += 1;
            model_batch_items += (size_t)py::len(encs);
            if (!model_override.is_none())
            {
                py::list datas;
                for (auto item : encs)
                {
                    py::object enc_obj = py::cast<py::object>(item);
                    datas.append(enc_obj.attr("data"));
                }
                py::object batch = pyg_data_Batch.attr("from_data_list")(datas);
                if (!model_device.empty())
                    batch = batch.attr("to")(py::cast(model_device));

                py::object no_grad = torch_module.attr("no_grad")();
                no_grad.attr("__enter__")();
                py::object logits = model_override(batch);
                no_grad.attr("__exit__")(py::none(), py::none(), py::none());
                py::object out = torch_module.attr("sigmoid")(logits).attr("detach")().attr("cpu")();
                py::list out_list = py::cast<py::list>(out.attr("tolist")());
                for (auto v : out_list)
                    probs.push_back(py::cast<double>(v));
            }
            else
            {
                py::object py_probs = gnn_module.attr("evaluate_encodings")(encs);
                py::list probs_list = py::cast<py::list>(py_probs);
                for (auto v : probs_list)
                    probs.push_back(py::cast<double>(v));
            }
            auto model_end = std::chrono::high_resolution_clock::now();
            total_model_time += std::chrono::duration<double>(model_end - model_start).count();
        }
        catch (const py::error_already_set &e)
        {
            throw std::runtime_error(
                std::string("GNN evaluation is mandatory for AlphaBetaEngine and failed. ") +
                "Ensure a GNN evaluator is loaded in Python (call gnn.model.load_model(...)) or use set_model_checkpoint(...). " +
                std::string("Python error: ") + e.what());
        }

        if (probs.size() != (size_t)py::len(encs))
        {
            // Defensive: fall back to 0.5 for missing outputs.
            probs.resize((size_t)py::len(encs), 0.5);
        }
        for (auto &p : probs)
            p = clamp_prob(p);
        return probs;
    }

    double evaluate(GameState &g)
    {
        TTKey key = g.tt_key();
        auto it = eval_cache.find(key);
        if (it != eval_cache.end())
            return it->second;

        double p = gnn_prob_root(g);
        double v = prob_to_value(p);
        eval_cache[key] = v;
        return v;
    }

    std::vector<double> evaluate_children_depth1_batched(GameState &g, const std::vector<Move> &moves, bool parent_maximising)
    {
        // For depth==1, children are evaluated at depth 0.
        // We batch all uncached leaf evals into a single model call.
        std::vector<double> values;
        values.resize(moves.size(), 0.0);

        std::vector<TTKey> keys;
        keys.resize(moves.size());

        std::vector<size_t> need_eval_indices;
        need_eval_indices.reserve(moves.size());

        std::vector<char> flip_flags;
        flip_flags.reserve(moves.size());

        py::list encs;

        for (size_t i = 0; i < moves.size(); ++i)
        {
            const auto &m = moves[i];
            int mover = g.current_player;
            g.do_move(m, mover);

            TTKey key = g.tt_key();
            keys[i] = key;
            auto it = eval_cache.find(key);
            if (it != eval_cache.end())
            {
                values[i] = it->second;
                g.undo_move();
                continue;
            }

            // Terminal/empty fast paths without model.
            if (g.winner != -1)
            {
                double p = (g.winner == root_player) ? 1.0 : 0.0;
                if (g.current_player != root_player)
                    p = 1.0 - p;
                double v = prob_to_value(p);
                eval_cache[key] = v;
                values[i] = v;
                g.undo_move();
                continue;
            }
            if (g.connected_points.empty() && g.rocks.empty())
            {
                double v = prob_to_value(0.5);
                eval_cache[key] = v;
                values[i] = v;
                g.undo_move();
                continue;
            }

            // Need model: build encoding for this child.
            encs.append(encode_state(g));
            need_eval_indices.push_back(i);
            flip_flags.push_back((g.current_player != root_player) ? 1 : 0);
            g.undo_move();
        }

        if (!need_eval_indices.empty())
        {
            auto probs = gnn_probs_root_for_encodings(encs);
            for (size_t j = 0; j < need_eval_indices.size(); ++j)
            {
                size_t i = need_eval_indices[j];
                double p = probs[j];
                if (flip_flags[j])
                    p = 1.0 - p;

                double v = prob_to_value(p);
                eval_cache[keys[i]] = v;
                values[i] = v;
            }
        }

        // Apply pass penalty at the parent node (matches alpha_beta recursion).
        for (size_t i = 0; i < moves.size(); ++i)
        {
            if (moves[i].t == 'P')
            {
                if (parent_maximising)
                    values[i] -= pass_penalty;
                else
                    values[i] += pass_penalty;
            }
        }
        return values;
    }

    void order_moves_by_child_eval_inplace(std::vector<Move> &moves, GameState &g, bool parent_maximising)
    {
        if (moves.size() <= 1)
            return;

        std::vector<double> scores;
        scores.resize(moves.size(), 0.0);

        std::vector<TTKey> keys;
        keys.resize(moves.size());

        std::vector<size_t> need_eval_indices;
        need_eval_indices.reserve(moves.size());
        std::vector<char> flip_flags;
        flip_flags.reserve(moves.size());

        py::list encs;

        for (size_t i = 0; i < moves.size(); ++i)
        {
            const auto &m = moves[i];
            int mover = g.current_player;
            g.do_move(m, mover);

            TTKey key = g.tt_key();
            keys[i] = key;
            auto it = eval_cache.find(key);
            if (it != eval_cache.end())
            {
                scores[i] = it->second;
                g.undo_move();
                continue;
            }

            if (g.winner != -1)
            {
                double p = (g.winner == root_player) ? 1.0 : 0.0;
                if (g.current_player != root_player)
                    p = 1.0 - p;
                double v = prob_to_value(p);
                eval_cache[key] = v;
                scores[i] = v;
                g.undo_move();
                continue;
            }
            if (g.connected_points.empty() && g.rocks.empty())
            {
                double v = prob_to_value(0.5);
                eval_cache[key] = v;
                scores[i] = v;
                g.undo_move();
                continue;
            }

            encs.append(encode_state(g));
            need_eval_indices.push_back(i);
            flip_flags.push_back((g.current_player != root_player) ? 1 : 0);
            g.undo_move();
        }

        if (!need_eval_indices.empty())
        {
            auto probs = gnn_probs_root_for_encodings(encs);
            for (size_t j = 0; j < need_eval_indices.size(); ++j)
            {
                size_t i = need_eval_indices[j];
                double p = probs[j];
                if (flip_flags[j])
                    p = 1.0 - p;
                double v = prob_to_value(p);
                eval_cache[keys[i]] = v;
                scores[i] = v;
            }
        }

        for (size_t i = 0; i < moves.size(); ++i)
        {
            if (moves[i].t == 'P')
            {
                if (parent_maximising)
                    scores[i] -= pass_penalty;
                else
                    scores[i] += pass_penalty;
            }
        }

        std::vector<size_t> idx(moves.size());
        for (size_t i = 0; i < idx.size(); ++i)
            idx[i] = i;

        std::stable_sort(idx.begin(), idx.end(), [&](size_t ia, size_t ib)
                         {
            double a = scores[ia];
            double b = scores[ib];
            if (a != b) return parent_maximising ? (a > b) : (a < b);
            int ra = move_type_rank(moves[ia]);
            int rb = move_type_rank(moves[ib]);
            if (ra != rb) return ra < rb;
            return move_less(moves[ia], moves[ib]); });

        std::vector<Move> reordered;
        reordered.reserve(moves.size());
        for (size_t i : idx)
            reordered.push_back(moves[i]);
        moves.swap(reordered);
    }

    double alpha_beta(GameState &g, int depth, double alpha, double beta)
    {
        TTKey key = g.tt_key();
        auto it = tt.find(key);
        if (it != tt.end() && it->second.depth >= depth)
        {
            const TTEntry &e = it->second;
            if (e.flag == 0)
                return e.value;
            if (e.flag == 1)
                alpha = std::max(alpha, e.value);
            else if (e.flag == 2)
                beta = std::min(beta, e.value);
            if (alpha >= beta)
                return e.value;
        }

        if (depth <= 0 || g.winner != -1)
            return evaluate(g);

        bool maximising = (g.current_player == root_player);
        double best = maximising ? -1e300 : 1e300;
        Move best_move{0, 0, 'P'};
        double a0 = alpha;
        double b0 = beta;

        auto moves = g.get_possible_moves_for_player(g.current_player);
        moves = filter_search_moves(moves, g, g.current_player);
        order_moves_inplace(moves);
        if ((int)moves.size() > move_cap)
            moves.resize((size_t)move_cap);

        // Use a one-shot batched child evaluation for move ordering at depth==2.
        // This adds one model call but can improve pruning significantly.
        if (depth == 2)
        {
            order_moves_by_child_eval_inplace(moves, g, maximising);
        }

        if (moves.empty())
            return evaluate(g);

        if (depth == 1)
        {
            auto vals = evaluate_children_depth1_batched(g, moves, maximising);
            for (size_t i = 0; i < moves.size(); ++i)
            {
                const auto &m = moves[i];
                double v = vals[i];

                if (maximising)
                {
                    if (v > best || (v == best && move_less(m, best_move)))
                    {
                        best = v;
                        best_move = m;
                    }
                    alpha = std::max(alpha, best);
                    if (alpha >= beta)
                        break;
                }
                else
                {
                    if (v < best || (v == best && move_less(m, best_move)))
                    {
                        best = v;
                        best_move = m;
                    }
                    beta = std::min(beta, best);
                    if (alpha >= beta)
                        break;
                }
            }
        }
        else
        {
            for (auto &m : moves)
            {
                int mover = g.current_player;
                g.do_move(m, mover);
                double v = alpha_beta(g, depth - 1, alpha, beta);
                g.undo_move();

                if (m.t == 'P')
                {
                    if (maximising)
                        v -= pass_penalty;
                    else
                        v += pass_penalty;
                }

                if (maximising)
                {
                    if (v > best || (v == best && move_less(m, best_move)))
                    {
                        best = v;
                        best_move = m;
                    }
                    alpha = std::max(alpha, best);
                    if (alpha >= beta)
                        break;
                }
                else
                {
                    if (v < best || (v == best && move_less(m, best_move)))
                    {
                        best = v;
                        best_move = m;
                    }
                    beta = std::min(beta, best);
                    if (alpha >= beta)
                        break;
                }
            }
        }

        TTEntry e;
        e.depth = depth;
        e.value = best;
        e.best = best_move;
        if (best <= a0)
            e.flag = 2; // upper
        else if (best >= b0)
            e.flag = 1; // lower
        else
            e.flag = 0; // exact
        tt[key] = e;
        return best;
    }

    std::mt19937 rng;
    double pass_penalty = 1.2;
    int move_cap = 48;
    int root_player = 0;
    int last_root_player = -1;

    std::unordered_map<TTKey, TTEntry, TTKeyHash> tt;
    std::unordered_map<TTKey, double, TTKeyHash> eval_cache;

    py::object torch_module;
    py::object pyg_data_module;
    py::object pyg_data_Data;
    py::object pyg_data_Batch;
    py::object gnn_module;
    py::object types_module;

    py::object model_override = py::none();
    std::string model_device = "cpu";

    std::unordered_map<std::uint64_t, py::object> enc_cache;
    static constexpr size_t ENC_CACHE_MAX = 4096;

    double total_encode_time = 0.0;
    double total_model_time = 0.0;

    size_t model_calls = 0;
    size_t model_batch_items = 0;
};

PYBIND11_MODULE(players_ext, m)
{
    py::class_<Move>(m, "Move")
        .def(py::init<>())
        .def_readwrite("x", &Move::x)
        .def_readwrite("y", &Move::y)
        .def_readwrite("t", &Move::t)
        .def("__repr__", [](const Move &mv)
             {
            std::ostringstream os;
            os << "players_ext.Move(" << mv.x << ", " << mv.y << ", '" << mv.t << "')";
            return os.str(); });

    py::class_<GameState>(m, "GameState")
        .def(py::init<>())
        .def("get_possible_moves", &GameState::get_possible_moves_for_player, py::arg("player_number"))
        .def("do_move", &GameState::do_move, py::arg("move"), py::arg("player_number"))
        .def("undo_move", &GameState::undo_move)
        .def("state_key", &GameState::state_key)
        .def("debug_empty_mask_drift_count", &GameState::debug_empty_mask_drift_count)
        .def("set_current_player0", &GameState::set_current_player0)
        .def_readwrite("current_player", &GameState::current_player);

    py::class_<MCTSEngine>(m, "MCTSEngine")
        .def(py::init<int, double>(), py::arg("seed") = 0, py::arg("c_puct") = 1.41421356)
        .def("choose_move", &MCTSEngine::choose_move, py::arg("root"), py::arg("n_rollouts") = 1000)
        .def("set_c_puct", &MCTSEngine::set_c_puct)
        .def("set_verbose", &MCTSEngine::set_verbose, py::arg("verbose"))
        .def("set_progressive_widening", &MCTSEngine::set_progressive_widening)
        .def("set_rave_k", &MCTSEngine::set_rave_k)
        .def("set_prior_eval_cap", &MCTSEngine::set_prior_eval_cap)
        .def("set_max_sim_depth", &MCTSEngine::set_max_sim_depth)
        .def("clear_root_priors", &MCTSEngine::clear_root_priors)
        .def("set_exploration", &MCTSEngine::set_exploration, py::arg("dirichlet_alpha"), py::arg("dirichlet_epsilon"), py::arg("temperature"), py::arg("temperature_moves"))
        .def("set_model_checkpoint", &MCTSEngine::set_model_checkpoint, py::arg("path"), py::arg("device") = "cpu")
        .def("reset_search", &MCTSEngine::reset_search)
        .def("set_root_priors", &MCTSEngine::set_root_priors_py)
        .def("get_current_root_key", &MCTSEngine::get_current_root_key)
        .def("get_root_visit_stats", &MCTSEngine::get_root_visit_stats_py, py::arg("root"))
        .def("clear_stats", &MCTSEngine::clear_stats)
        .def("get_profile_stats", &MCTSEngine::get_profile_stats)
        .def("advance_root", &MCTSEngine::advance_root, py::arg("game"))
        .def("prune_tables", &MCTSEngine::prune_tables, py::arg("max_states"));

    py::class_<AlphaBetaEngine>(m, "AlphaBetaEngine")
        .def(py::init<int, double>(), py::arg("seed") = 0, py::arg("pass_penalty") = 1.2)
        .def("choose_move", &AlphaBetaEngine::choose_move, py::arg("root"), py::arg("depth") = 3, py::arg("move_cap") = 48)
        .def("set_model_checkpoint", &AlphaBetaEngine::set_model_checkpoint, py::arg("path"), py::arg("device") = "cpu")
        .def("clear_stats", &AlphaBetaEngine::clear_stats)
        .def("get_profile_stats", &AlphaBetaEngine::get_profile_stats);
}