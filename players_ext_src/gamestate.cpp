#include "gamestate.hpp"

#include <cmath>
#include <stdexcept>

GameState::GameState()
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

GameState::GameState(const GameState &other)
    : GameState()
{
    // Copy config knobs / RNG first (do_move doesn't use RNG, but keep consistent).
    tactical_branch_limit = other.tactical_branch_limit;
    rock_rollout_bonus_connected = other.rock_rollout_bonus_connected;
    rock_rollout_bonus_disconnected = other.rock_rollout_bonus_disconnected;
    stick_between_opp_rocks_bonus = other.stick_between_opp_rocks_bonus;
    rng = other.rng;

    // Reconstruct by replaying the exact movers from the initial state.
    // Important: some callers (e.g. training randomization) legally apply moves out-of-turn,
    // so the mover is not always equal to the state's current_player.
    for (const auto &rec : other.history)
    {
        do_move(rec.m, rec.mover);
    }
}

GameState &GameState::operator=(const GameState &other)
{
    if (this == &other)
        return *this;
    GameState tmp(other);
    *this = std::move(tmp);
    return *this;
}

TTKey GameState::tt_key() const
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

std::uint64_t GameState::state_key() const
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

std::mt19937 GameState::rng_snapshot() const { return rng; }
void GameState::rng_restore(const std::mt19937 &snapshot) { rng = snapshot; }

void GameState::set_current_player0()
{
    current_player = 0;
}

bool GameState::intersects_stick(Coord start, int d) const
{
    if (d < 2 || d > 5)
        return false;
    int dx = DIR_DELTAS[d][0];
    int dy = DIR_DELTAS[d][1];
    std::uint64_t k = intersect_key(start.first, start.second, dx, dy);
    return intersects_cache.find(k) != intersects_cache.end();
}

Node *GameState::get_node(Coord c)
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

void GameState::add_node(Coord c)
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

std::vector<Move> GameState::get_possible_moves_for_player(int player_number)
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
            // Trust the actual adjacency pointers for legality.
            if (p->neighbours[d])
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

bool GameState::coord_lt(Coord a, Coord b)
{
    if (a.first != b.first)
        return a.first < b.first;
    return a.second < b.second;
}

bool GameState::coord_leq(Coord a, Coord b)
{
    return coord_lt(a, b) || (a.first == b.first && a.second == b.second);
}

int GameState::polygon_area2_from_path(const std::vector<Node *> &path)
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

std::uint64_t GameState::region_edge_key_digest_from_path(const std::vector<Node *> &path)
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

void GameState::fill_sorted_neighbours(Node *node, std::array<Node *, 8> &out, int &n_out) const
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

int GameState::best_new_cycle_area2(Node *start, Node *end, std::uint64_t &out_edge_key)
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

bool GameState::is_move_legal(const Move &m, int player_number) const
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

std::string GameState::explain_illegal_move(const Move &m, int player_number)
{
    std::ostringstream os;
    os << "m=(" << m.x << "," << m.y << "," << m.t << ")";
    os << " player=" << player_number;
    os << " cur=" << current_player;
    os << " turn=" << turn_number;
    os << " winner=" << winner;

    if (m.t == 'P')
        return os.str() + " pass";

    if (m.t == 'R')
    {
        if (turn_number == 0)
            return os.str() + " illegal: turn0";
        if (num_rocks[player_number] <= 0)
            return os.str() + " illegal: no_rocks";
        Coord rc{m.x, m.y};
        if (coord_in_claimed_region_cached(rc))
            return os.str() + " illegal: claimed";
        auto it = points.find(key_from_coord(rc));
        if (it != points.end() && it->second->rocked_by != -1)
            return os.str() + " illegal: occupied";

        bool near_anchor = false;
        for (Node *a : connected_points)
            if (std::abs(a->x - m.x) <= 1 && std::abs(a->y - m.y) <= 1)
                near_anchor = true;
        if (!near_anchor)
            for (Node *a : rocks)
                if (std::abs(a->x - m.x) <= 1 && std::abs(a->y - m.y) <= 1)
                    near_anchor = true;
        if (!near_anchor)
            return os.str() + " illegal: not_adjacent";
        return os.str() + " legal?";
    }

    Coord sc{m.x, m.y};
    if (coord_in_claimed_region_cached(sc))
        return os.str() + " illegal: start_claimed";
    auto it_start = points.find(key_from_coord(sc));
    if (it_start == points.end())
        return os.str() + " illegal: start_missing";
    Node *start = it_start->second.get();
    if (!node_is_connected(start))
        return os.str() + " illegal: start_disconnected";
    if (start->rocked_by != -1 && start->rocked_by != player_number)
        return os.str() + " illegal: cannot_place";

    int d = GameState::dir_from_name(m.t);
    if (d < 0 || d > 7)
        return os.str() + " illegal: bad_dir";
    if (start->neighbours[d])
        return os.str() + " illegal: edge_occupied";
    if (intersects_stick(sc, d))
        return os.str() + " illegal: intersects";
    Coord endc = calc_end(sc, d);
    if (coord_in_claimed_region_cached(endc))
        return os.str() + " illegal: end_claimed";
    auto it_end = points.find(key_from_coord(endc));
    if (it_end != points.end())
    {
        Node *end = it_end->second.get();
        int rd = reverse_dir(d);
        if (end->neighbours[rd])
            return os.str() + " illegal: reverse_occupied";
    }
    return os.str() + " legal?";
}

void GameState::do_move(const Move &m, int player_number)
{
    if (m.t == 'P')
    {
        history.push_back({m, player_number, current_player, players_scores, num_rocks, turn_number, winner, claimed_cycle_stack.size()});
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

        history.push_back({m, player_number, current_player, players_scores, num_rocks, turn_number, winner, claimed_cycle_stack.size()});
        p->rocked_by = player_number;
        rocks.push_back(p);
        num_rocks[player_number] -= 1;

        MoveKey mk{m.x, m.y, 'R'};
        board_hash ^= splitmix64(move_key_u64(mk) ^ (std::uint64_t)(player_number + 1) * 0x94d049bb133111ebULL);
    }
    else
    {
        if (!is_move_legal(m, player_number))
        {
            std::ostringstream os;
            os << "Illegal stick move: " << explain_illegal_move(m, player_number);
            throw std::runtime_error(os.str());
        }

        history.push_back({m, player_number, current_player, players_scores, num_rocks, turn_number, winner, claimed_cycle_stack.size()});
        num_rocks[player_number] = 2;
        Node *start = get_node({m.x, m.y});
        int d = dir_from_name(m.t);

        // Track diagonal midpoints for intersection detection (matches Python intersects cache).
        if (d >= 2 && d <= 5)
        {
            int dx = DIR_DELTAS[d][0];
            int dy = DIR_DELTAS[d][1];
            intersects_cache.insert(intersect_key(m.x, m.y, dx, dy));
        }

        Node *end = get_node(calc_end(start->c(), d));
        start->neighbours[d] = end;
        end->neighbours[reverse_dir(d)] = start;

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

void GameState::undo_move()
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

        int mover = rec.mover;
        MoveKey mk{last.x, last.y, 'R'};
        board_hash ^= splitmix64(move_key_u64(mk) ^ (std::uint64_t)(mover + 1) * 0x94d049bb133111ebULL);
    }
    else if (last.t != 'P')
    {
        Node *start = get_node({last.x, last.y});
        int d = dir_from_name(last.t);

        if (d >= 2 && d <= 5)
        {
            int dx = DIR_DELTAS[d][0];
            int dy = DIR_DELTAS[d][1];
            intersects_cache.erase(intersect_key(last.x, last.y, dx, dy));
        }

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
        if (!node_is_connected(start) && !(start->x == 0 && start->y == 0))
            connected_points_remove(start);
        if (!node_is_connected(end) && !(end->x == 0 && end->y == 0))
            connected_points_remove(end);

        reachable_cache_valid = false;
    }
}

bool GameState::coord_in_claimed_region_cached(Coord c)
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

bool GameState::coord_in_claimed_region(Coord start)
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

void GameState::ensure_reachable_cache() const
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
            std::uint8_t mask = 0;
            for (int d = 0; d < 8; ++d)
            {
                if (p->neighbours[d])
                    mask |= static_cast<std::uint8_t>(1u << d);
            }
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

std::string GameState::move_key_str_static(const Move &m)
{
    return std::to_string(m.x) + ":" + std::to_string(m.y) + ":" + m.t;
}

double GameState::rock_bonus_for_cell(const GameState &g, Coord c, double connected_bonus, double disconnected_bonus) const
{
    auto k = GameState::key_from_coord(c);
    auto it = g.points.find(k);
    if (it == g.points.end())
        return disconnected_bonus;
    Node *p = it->second.get();
    return p->in_connected_points ? connected_bonus : disconnected_bonus;
}

bool GameState::stick_between_opp_rocks(const GameState &g, int player_number, const Move &mv) const
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

double GameState::eval_probability_simple(GameState &g, int player_number) const
{
    if (g.winner != -1)
        return (g.winner == player_number) ? 1.0 : 0.0;
    return 0.5;
}

double GameState::score_after(GameState &g, int player_number, const Move &m) const
{
    if (!g.is_move_legal(m, player_number))
        return -1e300;
    g.do_move(m, player_number);
    double s = eval_probability_simple(g, player_number);
    g.undo_move();
    return s;
}

Move GameState::rollout_pick_move(GameState &game)
{
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
    int tie_count = 0;
    constexpr double SCORE_TIE_EPS = 1e-12;
    for (auto &m : moves)
    {
        double s = score_after(game, mover, m);
        if (m.t == 'R')
            s += rock_bonus_for_cell(game, {m.x, m.y}, rock_rollout_bonus_connected, rock_rollout_bonus_disconnected);
        if (stick_between_opp_rocks(game, mover, m))
        {
            s += stick_between_opp_rocks_bonus;
        }
        if (s > best_score + SCORE_TIE_EPS)
        {
            best_score = s;
            best_move = m;
            tie_count = 1;
        }
        else if (std::fabs(s - best_score) <= SCORE_TIE_EPS)
        {
            tie_count += 1;
            std::uniform_int_distribution<int> uid(0, tie_count - 1);
            if (uid(rng) == 0)
                best_move = m;
        }
    }
    return best_move;
}

bool GameState::allows_forced_loss_next_round(const Move &my_move, GameState &working_game, int player_number) const
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

void GameState::connected_points_push_unique(Node *n)
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

void GameState::connected_points_remove(Node *n)
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

bool GameState::node_is_connected(const Node *n) const
{
    // Use neighbour pointers as the source of truth.
    for (int d = 0; d < 8; ++d)
    {
        if (n->neighbours[d])
            return true;
    }
    return false;
}
