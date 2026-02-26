#pragma once

#include "gamestate.hpp"

// Standalone heuristic evaluation functions, shared between AlphaBeta and MCTS.

struct TacticalInfo {
    double max_immediate_gain = 0.0;
    int scoring_move_count = 0;
    int stick_move_count = 0;
    double top3_gain_sum = 0.0;
    int bad_closure_count = 0;
    double potential_area = 0.0;
    double blocking_power = 0.0;
    double rock_value = 0.0;
    double best_reply_gain = 0.0;
};

// Helper: can player_number place a stick starting at node n?
inline bool heval_can_place(const Node *n, int player_number)
{
    return n->rocked_by == -1 || n->rocked_by == player_number;
}

// Convert 2x-area to scored gain
inline double heval_scored_gain_from_area(int area2)
{
    int area = area2 / 2;
    if (area <= 0) return 0.0;
    if (area == 1) return 0.0;  // HALF_AREA_COUNTS = false
    return (double)area;
}

// Check if a stick placement creates a cycle; return 2*area.
int heval_closure_area2_for_stick(GameState &g, const Move &m);

// Compute tactical features for a given player.
TacticalInfo heval_compute_tactical(GameState &g, int player_number);

// Evaluate position from perspective_player's perspective.
// Returns a heuristic score (positive = good for perspective_player).
double heval_evaluate(GameState &g, int perspective_player);

// Convert heuristic score to probability via sigmoid.
// Returns P(perspective_player wins) in [0, 1].
inline double heval_score_to_prob(double score, double temperature = 6.0)
{
    return 1.0 / (1.0 + std::exp(-score / temperature));
}
