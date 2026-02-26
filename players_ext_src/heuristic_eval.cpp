#include "heuristic_eval.hpp"
#include <algorithm>
#include <vector>

int heval_closure_area2_for_stick(GameState &g, const Move &m)
{
    auto it = g.points.find(GameState::key_from_coord({m.x, m.y}));
    if (it == g.points.end())
        return 0;
    Node *start = it->second.get();
    int d = GameState::dir_from_name(m.t);
    if (d < 0)
        return 0;
    Node *end = start->neighbours[d];
    if (end != nullptr)
        return 0;
    Coord end_c = calc_end({m.x, m.y}, d);
    auto it2 = g.points.find(GameState::key_from_coord(end_c));
    if (it2 == g.points.end())
        return 0;
    end = it2->second.get();
    if (!end->in_connected_points)
        return 0;

    std::uint64_t edge_key = 0;
    int area2 = g.best_new_cycle_area2(start, end, edge_key);
    return area2;
}

TacticalInfo heval_compute_tactical(GameState &g, int player_number)
{
    TacticalInfo info;

    // Gather stick moves for this player
    std::vector<Move> stick_moves;
    stick_moves.reserve(48);
    for (Node *cp : g.connected_points)
    {
        if (!heval_can_place(cp, player_number))
            continue;
        if (g.coord_in_claimed_region_cached(cp->c()))
            continue;
        for (int d = 0; d < 8; ++d)
        {
            if (cp->neighbours[d] != nullptr)
                continue;
            if (g.intersects_stick(cp->c(), d))
                continue;
            Coord end_c = calc_end(cp->c(), d);
            if (g.coord_in_claimed_region_cached(end_c))
                continue;
            stick_moves.push_back(Move{cp->x, cp->y, GameState::dir_name_char(d)});
        }
    }

    info.stick_move_count = (int)stick_moves.size();
    if (stick_moves.empty())
    {
        // potential_area, blocking, rock_value still computed below
    }
    else
    {
        int before_score = g.players_scores[player_number];
        std::vector<double> gains;
        gains.reserve(stick_moves.size());

        int cap = std::min((int)stick_moves.size(), 32);
        std::sort(stick_moves.begin(), stick_moves.end(), [](const Move &a, const Move &b)
                  {
                      if (a.x != b.x) return a.x < b.x;
                      if (a.y != b.y) return a.y < b.y;
                      return a.t < b.t;
                  });

        for (int i = 0; i < cap; ++i)
        {
            const Move &mv = stick_moves[i];
            if (!g.is_move_legal(mv, player_number))
                continue;
            g.do_move(mv, player_number);
            if (g.winner == player_number)
            {
                info.max_immediate_gain = std::max(info.max_immediate_gain, 999.0);
                info.scoring_move_count++;
                gains.push_back(999.0);
                g.undo_move();
                continue;
            }
            double gain = (double)(g.players_scores[player_number] - before_score);
            g.undo_move();

            if (gain > 0)
            {
                info.scoring_move_count++;
                info.max_immediate_gain = std::max(info.max_immediate_gain, gain);
                gains.push_back(gain);
            }
            else
            {
                int a2 = heval_closure_area2_for_stick(g, mv);
                if (a2 == 2)
                    info.bad_closure_count++;
            }
        }

        std::sort(gains.begin(), gains.end(), std::greater<double>());
        for (int i = 0; i < std::min((int)gains.size(), 3); ++i)
            info.top3_gain_sum += gains[i];

        // best_reply_gain
        if (info.scoring_move_count > 0 && info.max_immediate_gain < 999.0)
        {
            Move best_scoring_move{0, 0, 'P'};
            double best_gain = 0.0;
            int before = g.players_scores[player_number];
            int reply_cap = std::min((int)stick_moves.size(), 4);
            for (int i = 0; i < reply_cap; ++i)
            {
                const Move &mv = stick_moves[i];
                if (!g.is_move_legal(mv, player_number)) continue;
                g.do_move(mv, player_number);
                double gain_val = (double)(g.players_scores[player_number] - before);
                if (gain_val > best_gain)
                {
                    best_gain = gain_val;
                    best_scoring_move = mv;
                }
                g.undo_move();
            }
            if (best_gain > 0.0)
            {
                g.do_move(best_scoring_move, player_number);
                int opp = 1 - player_number;
                int opp_before = g.players_scores[opp];
                double opp_best_reply = 0.0;
                for (Node *cp : g.connected_points)
                {
                    if (!heval_can_place(cp, opp)) continue;
                    if (g.coord_in_claimed_region_cached(cp->c())) continue;
                    for (int d = 0; d < 8; ++d)
                    {
                        if (cp->neighbours[d] != nullptr) continue;
                        if (g.intersects_stick(cp->c(), d)) continue;
                        Coord end_c = calc_end(cp->c(), d);
                        if (g.coord_in_claimed_region_cached(end_c)) continue;
                        Move rmv{cp->x, cp->y, GameState::dir_name_char(d)};
                        if (!g.is_move_legal(rmv, opp)) continue;
                        g.do_move(rmv, opp);
                        double rgain = (double)(g.players_scores[opp] - opp_before);
                        opp_best_reply = std::max(opp_best_reply, rgain);
                        g.undo_move();
                        if (opp_best_reply > 0.0) break;
                    }
                    if (opp_best_reply > 0.0) break;
                }
                g.undo_move();
                info.best_reply_gain = opp_best_reply;
            }
        }
    }

    // potential_area
    {
        double pa = 0.0;
        for (Node *cp : g.connected_points)
        {
            if (!heval_can_place(cp, player_number))
                continue;
            for (int d = 0; d < 8; ++d)
            {
                if (cp->neighbours[d] != nullptr)
                    continue;
                Coord end_c = calc_end(cp->c(), d);
                auto it = g.points.find(GameState::key_from_coord(end_c));
                if (it != g.points.end() && heval_can_place(it->second.get(), player_number))
                    pa += 1.0;
            }
        }
        info.potential_area = pa;
    }

    // blocking_power
    {
        double blocked = 0.0;
        for (Node *rock : g.rocks)
        {
            if (rock->rocked_by != player_number)
                continue;
            if (!rock->in_connected_points)
                continue;
            if (g.coord_in_claimed_region_cached(rock->c()))
                continue;
            for (int d = 0; d < 8; ++d)
            {
                if (rock->neighbours[d] != nullptr)
                    continue;
                if (g.intersects_stick(rock->c(), d))
                    continue;
                Coord end_c = calc_end(rock->c(), d);
                if (g.coord_in_claimed_region_cached(end_c))
                    continue;
                blocked += 1.0;
                int a2 = heval_closure_area2_for_stick(g, Move{rock->x, rock->y, GameState::dir_name_char(d)});
                if (a2 > 0)
                    blocked += 0.25 * heval_scored_gain_from_area(a2);
            }
        }
        info.blocking_power = blocked;
    }

    // rock_value
    {
        int rocks_remaining = g.num_rocks[player_number];
        if (rocks_remaining <= 0)
        {
            info.rock_value = 0.0;
        }
        else
        {
            int opp = 1 - player_number;
            double best_impact = 0.0;
            int rock_count = 0;
            auto all_moves = g.get_possible_moves_for_player(player_number);
            for (const auto &m : all_moves)
            {
                if (m.t != 'R')
                    continue;
                auto it = g.points.find(GameState::key_from_coord({m.x, m.y}));
                if (it == g.points.end())
                    continue;
                Node *p = it->second.get();
                if (!p->in_connected_points)
                    continue;
                if (!heval_can_place(p, opp))
                    continue;

                double impact = 0.0;
                for (int d = 0; d < 8; ++d)
                {
                    if (p->neighbours[d] != nullptr)
                        continue;
                    if (g.intersects_stick(p->c(), d))
                        continue;
                    impact += 1.0;
                    int a2 = heval_closure_area2_for_stick(g, Move{p->x, p->y, GameState::dir_name_char(d)});
                    if (a2 > 0)
                        impact += 0.30 * heval_scored_gain_from_area(a2);
                }
                best_impact = std::max(best_impact, impact);
                if (++rock_count >= 24)
                    break;
            }
            double opportunity = std::min(3.0, best_impact / 4.0);
            info.rock_value = (double)rocks_remaining * opportunity;
        }
    }

    return info;
}

double heval_evaluate(GameState &g, int perspective_player)
{
    if (g.winner != -1)
        return (g.winner == perspective_player) ? 1000.0 : -1000.0;

    if (g.connected_points.empty() && g.rocks.empty())
        return 0.0;

    int me = perspective_player;
    int opp = 1 - me;

    TacticalInfo my_ts = heval_compute_tactical(g, me);
    TacticalInfo opp_ts = heval_compute_tactical(g, opp);

    bool my_turn = (g.current_player == me);
    double w_me  = my_turn ? 1.0 : 0.4;
    double w_opp = my_turn ? 0.4 : 1.0;

    double v =
        1.2 * my_ts.blocking_power +
        0.7 * (double)my_ts.stick_move_count +
        1.3 * my_ts.potential_area -
        2.0 * opp_ts.potential_area +
        1.8 * (my_ts.rock_value - opp_ts.rock_value) +
        1.2 * (my_ts.max_immediate_gain - opp_ts.max_immediate_gain) +
        0.3 * (my_ts.top3_gain_sum - opp_ts.top3_gain_sum) +
        0.3 * (double)(my_ts.scoring_move_count - opp_ts.scoring_move_count) +
        0.1 * (double)(my_ts.stick_move_count - opp_ts.stick_move_count) -
        0.1 * (double)(my_ts.bad_closure_count - opp_ts.bad_closure_count);

    // Best-reply gain
    v -= 0.5 * w_me  * my_ts.best_reply_gain;
    v += 0.5 * w_opp * opp_ts.best_reply_gain;

    // Score difference (very important in endgame)
    v += 10.0 * (double)(g.players_scores[me] - g.players_scores[opp]);

    return v;
}
