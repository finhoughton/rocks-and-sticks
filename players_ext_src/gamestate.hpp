#pragma once

#include "common.hpp"

class GameState
{
public:
    GameState();
    GameState(const GameState &other);
    GameState &operator=(const GameState &other);

    GameState(GameState &&) noexcept = default;
    GameState &operator=(GameState &&) noexcept = default;

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

    // Midpoint cache for diagonal stick intersections.
    // Matches Python Game._intersects_cache key: (2*x + dx, 2*y + dy) where dx,dy in {-1,+1}.
    std::unordered_set<std::uint64_t> intersects_cache;

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
        int mover;
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

    TTKey tt_key() const;
    std::uint64_t state_key() const;

    std::mt19937 rng_snapshot() const;
    void rng_restore(const std::mt19937 &snapshot);

    void set_current_player0();

    bool intersects_stick(Coord start, int d) const;

    Node *get_node(Coord c);
    void add_node(Coord c);

    std::vector<Move> get_possible_moves_for_player(int player_number);

    static constexpr bool HALF_AREA_COUNTS = false;
    static constexpr int MAX_CYCLE_PATHS = 100;

    static bool coord_lt(Coord a, Coord b);
    static bool coord_leq(Coord a, Coord b);
    static int polygon_area2_from_path(const std::vector<Node *> &path);
    static std::uint64_t region_edge_key_digest_from_path(const std::vector<Node *> &path);

    void fill_sorted_neighbours(Node *node, std::array<Node *, 8> &out, int &n_out) const;
    int best_new_cycle_area2(Node *start, Node *end, std::uint64_t &out_edge_key);

    bool is_move_legal(const Move &m, int player_number) const;
    std::string explain_illegal_move(const Move &m, int player_number);
    void do_move(const Move &m, int player_number);
    void undo_move();

    bool coord_in_claimed_region_cached(Coord c);
    bool coord_in_claimed_region(Coord start);

private:
    void ensure_reachable_cache() const;

    int tactical_branch_limit = 8;
    double rock_rollout_bonus_connected = 1.0;
    double rock_rollout_bonus_disconnected = 0.02;
    double stick_between_opp_rocks_bonus = 0.4;

    std::mt19937 rng{0};

    static std::string move_key_str_static(const Move &m);

    double rock_bonus_for_cell(const GameState &g, Coord c, double connected_bonus, double disconnected_bonus) const;
    bool stick_between_opp_rocks(const GameState &g, int player_number, const Move &mv) const;
    double eval_probability_simple(GameState &g, int player_number) const;
    double score_after(GameState &g, int player_number, const Move &m) const;

public:
    Move rollout_pick_move(GameState &game);
    bool allows_forced_loss_next_round(const Move &my_move, GameState &working_game, int player_number) const;

private:
    void connected_points_push_unique(Node *n);
    void connected_points_remove(Node *n);
    bool node_is_connected(const Node *n) const;

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
