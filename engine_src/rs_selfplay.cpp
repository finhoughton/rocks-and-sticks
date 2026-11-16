// Self-play driver for the rs engine: generates evaluation-tuning data and plays matches
// between two weight files.
//
//   rs_selfplay gen   --games N --nodes K --threads T --seed S --weights W --out data.bin
//                     [--leaves L --leaf-nodes K2 --leaf-walk W]
//   either mode:      [--openings FILE]  one move sequence per line ("0,0S 0,1N 0,1R P ..."):
//                     game (pair) k starts after line k % n, e.g. the opening book's lines
//   rs_selfplay match --games N --nodes K --threads T --seed S --weights A --weights2 B
//                     [--nodes2 K2]
//
// Data records are int16: [result*2 (0 loss, 1 draw, 2 win for side to move), search score
// (clamped), tactical flag, ply, game id low bits, N_FEATS features].
//
// With --leaves L, each game position is also the start of L random walks of 1..W moves, and
// the position each walk reaches is recorded with the score of a K2-node search from it and
// result -2 (none). Those positions look like the ones deep in a search tree, where the
// evaluation is used most, rather than like positions from good play.
#include <atomic>
#include <cstdio>
#include <fstream>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

#include "rs_search.hpp"

using namespace rs;

static bool load_weights(const std::string &path, Weights &w)
{
    return path.empty() || load_weights_file(path, w);
}

struct Args
{
    std::string mode;
    int games = 100;
    std::int64_t nodes = 20000, nodes2 = 0;
    int threads = 4;
    std::uint64_t seed = 1;
    std::string weights, weights2, out = "selfplay.bin";
    int max_plies = 120;
    int open_min = 2, open_max = 6;
    int margin = 12;
    // opening variety: for the first open_noise_plies moves, choose among moves within
    // open_noise of the best (instead of within margin)
    int open_noise = 0, open_noise_plies = 0;
    std::vector<Move> first_moves; // if set, game pair k opens with first_moves[k % n]
    std::vector<std::vector<Move>> openings; // if set, game pair k starts after openings[k % n]
    int leaves = 0, leaf_walk = 8;
    std::int64_t leaf_nodes = 20000;
    std::vector<std::string> opts, opts2; // name=value search options for engine A / B
};

static bool any_scoring(const State &s)
{
    std::vector<Closure> cl;
    compute_closures(s, cl);
    for (auto &c : cl)
        if (c.pa[0] >= 2 || c.pa[1] >= 2)
            return true;
    return false;
}

static void random_opening(State &s, std::mt19937_64 &rng, int plies)
{
    std::vector<Move> mv, ok;
    for (int i = 0; i < plies && s.winner < 0; ++i)
    {
        s.gen_moves(mv, true);
        ok.clear();
        for (Move m : mv)
        {
            int t = move_type(m);
            if (t == T_PASS)
                continue;
            if (t == T_ROCK)
            {
                // keep rocks near the structure
                bool near = s.sticks[move_idx(m)] != 0;
                for (int d = 0; d < 8 && !near; ++d)
                    near = s.sticks[move_idx(m) + DOFF[d]] != 0;
                if (!near)
                    continue;
            }
            else if (s.sticks[move_idx(m) + DOFF[t]] && s.closure_area2(move_idx(m), t) >= 2)
                continue; // do not hand out free points in the opening
            ok.push_back(m);
        }
        if (ok.empty())
            break;
        s.do_move(ok[rng() % ok.size()]);
    }
}

// "x,yD" (D a direction name, R for a rock) or "P"; returns false if it is not a move.
static bool parse_move(const std::string &tok, Move &m)
{
    static const char *names[10] = {"N", "E", "NE", "SE", "NW", "SW", "W", "S", "R", "P"};
    if (tok == "P")
    {
        m = MOVE_PASS;
        return true;
    }
    int x = 0, y = 0;
    char t[4] = {0};
    if (std::sscanf(tok.c_str(), "%d,%d%3s", &x, &y, t) != 3)
        return false;
    for (int i = 0; i < 9; ++i)
        if (std::string(t) == names[i])
        {
            m = make_move(idx_of(x, y), i);
            return true;
        }
    return false;
}

struct Record
{
    std::vector<std::int16_t> v; // per position, without result
    int stm;
    bool leaf = false;           // a random-walk position: no game result
};

// A random move for a walk: any stick, or a rock next to the structure (as the search uses).
static Move random_walk_move(const State &s, std::mt19937_64 &rng, std::vector<Move> &mv)
{
    s.gen_moves(mv, true);
    std::vector<Move> ok;
    for (Move m : mv)
    {
        int t = move_type(m);
        if (t == T_PASS)
            continue;
        if (t == T_ROCK)
        {
            int v = move_idx(m);
            bool near = s.sticks[v] != 0;
            for (int d = 0; d < 8 && !near; ++d)
                near = s.sticks[v + DOFF[d]] != 0;
            if (!near)
                continue;
        }
        ok.push_back(m);
    }
    if (ok.empty())
        return MOVE_PASS;
    return ok[(size_t)(rng() % ok.size())];
}

int main(int argc, char **argv)
{
    Args a;
    if (argc < 2)
    {
        std::fprintf(stderr, "usage: rs_selfplay gen|match [options]\n");
        return 2;
    }
    a.mode = argv[1];
    for (int i = 2; i + 1 < argc; i += 2)
    {
        std::string k = argv[i], v = argv[i + 1];
        if (k == "--games") a.games = std::stoi(v);
        else if (k == "--nodes") a.nodes = std::stoll(v);
        else if (k == "--nodes2") a.nodes2 = std::stoll(v);
        else if (k == "--threads") a.threads = std::stoi(v);
        else if (k == "--seed") a.seed = std::stoull(v);
        else if (k == "--weights") a.weights = v;
        else if (k == "--weights2") a.weights2 = v;
        else if (k == "--out") a.out = v;
        else if (k == "--max-plies") a.max_plies = std::stoi(v);
        else if (k == "--open-min") a.open_min = std::stoi(v);
        else if (k == "--open-max") a.open_max = std::stoi(v);
        else if (k == "--margin") a.margin = std::stoi(v);
        else if (k == "--open-noise") a.open_noise = std::stoi(v);
        else if (k == "--open-noise-plies") a.open_noise_plies = std::stoi(v);
        else if (k == "--leaves") a.leaves = std::stoi(v);
        else if (k == "--leaf-nodes") a.leaf_nodes = std::stoll(v);
        else if (k == "--leaf-walk") a.leaf_walk = std::stoi(v);
        else if (k == "--openings")
        {
            std::ifstream in(v);
            if (!in)
            {
                std::fprintf(stderr, "cannot read %s\n", v.c_str());
                return 2;
            }
            std::string line;
            while (std::getline(in, line))
            {
                std::vector<Move> seq;
                size_t pos = 0;
                while (pos < line.size())
                {
                    size_t end = line.find(' ', pos);
                    std::string tok = line.substr(pos, end == std::string::npos ? std::string::npos : end - pos);
                    pos = (end == std::string::npos) ? line.size() : end + 1;
                    if (tok.empty())
                        continue;
                    Move m;
                    if (!parse_move(tok, m))
                    {
                        std::fprintf(stderr, "bad move %s in %s\n", tok.c_str(), v.c_str());
                        return 2;
                    }
                    seq.push_back(m);
                }
                if (!seq.empty())
                    a.openings.push_back(seq);
            }
        }
        else if (k == "--first-moves")
        {
            // e.g. "0,0S 0,0E 0,1N": x,y then a direction name
            static const char *names[8] = {"N", "E", "NE", "SE", "NW", "SW", "W", "S"};
            size_t pos = 0;
            while (pos < v.size())
            {
                size_t end = v.find(' ', pos);
                std::string tok = v.substr(pos, end == std::string::npos ? std::string::npos : end - pos);
                pos = (end == std::string::npos) ? v.size() : end + 1;
                if (tok.empty())
                    continue;
                int x = 0, y = 0;
                char dir[4] = {0};
                if (std::sscanf(tok.c_str(), "%d,%d%3s", &x, &y, dir) != 3)
                {
                    std::fprintf(stderr, "bad first move %s\n", tok.c_str());
                    return 2;
                }
                int d = -1;
                for (int i = 0; i < 8; ++i)
                    if (std::string(dir) == names[i])
                        d = i;
                if (d < 0)
                {
                    std::fprintf(stderr, "bad direction in %s\n", tok.c_str());
                    return 2;
                }
                a.first_moves.push_back(make_move(idx_of(x, y), d));
            }
        }
        else if (k == "--opt") a.opts.push_back(v);
        else if (k == "--opt2") a.opts2.push_back(v);
        else
        {
            std::fprintf(stderr, "unknown option %s\n", k.c_str());
            return 2;
        }
    }
    if (a.nodes2 == 0)
        a.nodes2 = a.nodes;
    Weights w1, w2;
    if (!load_weights(a.weights, w1) || !load_weights(a.weights2.empty() ? a.weights : a.weights2, w2))
    {
        std::fprintf(stderr, "could not load weights\n");
        return 2;
    }
    const bool match = a.mode == "match";

    std::atomic<int> next_game{0};
    std::mutex mu;
    std::ofstream out;
    if (!match)
        out.open(a.out, std::ios::binary);
    long wins = 0, draws = 0, losses = 0, total_plies = 0, n_records = 0;
    long p0_wins = 0, p1_wins = 0, n_claims = 0, n_broken = 0, n_retries = 0, n_lost = 0;
    // how games were decided (counted by side, player 0 = moves first)
    long first_by[2] = {0, 0};        // who completed the first polygon
    long p0_first_won = 0;            // ... player 0 did and player 1 could not answer
    long p0_first_exceeded = 0;       // ... player 0 did and player 1 answered with a bigger one
    long p0_first_matched = 0;        // ... player 0 did and player 1 matched it: game went on
    long p0_first_matched_p0won = 0;  // ... of those, player 0 still won

    auto worker = [&](int tid)
    {
        (void)tid;
        // State and Searcher are far too big for a thread stack.
        auto s_ptr = std::make_unique<State>();
        auto sa_ptr = std::make_unique<Searcher>(18);
        auto sb_ptr = std::make_unique<Searcher>(18);
        State &s = *s_ptr;
        Searcher &sa = *sa_ptr, &sb = *sb_ptr;
        sa.weights = w1;
        sb.weights = w2;
        sa.root_margin = sb.root_margin = a.margin;
        auto apply = [](Searcher &se, const std::vector<std::string> &opts)
        {
            for (const std::string &o : opts)
            {
                size_t eq = o.find('=');
                if (eq == std::string::npos || !se.set_opt(o.substr(0, eq), std::stoi(o.substr(eq + 1))))
                {
                    std::fprintf(stderr, "bad option %s\n", o.c_str());
                    std::exit(2);
                }
            }
        };
        apply(sa, a.opts);
        apply(sb, a.opts2);
        std::vector<Record> recs;
        int f[N_FEATS];
        while (true)
        {
            int g = next_game.fetch_add(1);
            if (g >= a.games)
                break;
            std::mt19937_64 rng(a.seed * 1000003ULL + (std::uint64_t)(match ? g / 2 : g));
            s.set_start();
            int open = a.open_min + (int)(rng() % (std::uint64_t)(a.open_max - a.open_min + 1));
            random_opening(s, rng, open);
            sa.clear();
            sb.clear();
            sa.rng.seed(rng());
            sb.rng.seed(rng());
            // in match mode engine A plays player (g % 2); each opening is played from both sides
            int a_side = match ? (g % 2) : 0;
            recs.clear();
            bool claimed_win[2] = {false, false};
            bool had_win[2] = {false, false};
            int plies = 0;
            int first_scorer = -1, first_score_ply = -1, p0_first_outcome = -1;
            if (!a.openings.empty())
            {
                for (Move om : a.openings[(size_t)((match ? g / 2 : g) % (int)a.openings.size())])
                {
                    if (s.winner >= 0 || !s.move_legal(om, s.cp))
                    {
                        std::fprintf(stderr, "opening move is not legal (game %d)\n", g);
                        std::exit(2);
                    }
                    s.do_move(om);
                    plies++;
                }
            }
            if (!a.first_moves.empty())
            {
                Move fm = a.first_moves[(size_t)((match ? g / 2 : g) % (int)a.first_moves.size())];
                if (!s.move_legal(fm, s.cp))
                {
                    std::fprintf(stderr, "first move is not legal here\n");
                    std::exit(2);
                }
                s.do_move(fm);
                plies++;
            }
            while (s.winner < 0 && plies < a.max_plies)
            {
                bool a_to_move = !match || s.cp == a_side;
                Searcher &se = a_to_move ? sa : sb;
                SearchLimits lim;
                lim.max_nodes = a_to_move ? a.nodes : a.nodes2;
                se.root_margin = (plies < a.open_noise_plies) ? a.open_noise : a.margin;
                SearchResult r = se.search(s, lim);
                // A forced win announced on the previous move is still there; if this search
                // did not find it again it only ran out of budget, so give it more.
                if (had_win[s.cp] && r.score < MATE_BOUND)
                {
                    for (int retry = 0; retry < 3 && r.score < MATE_BOUND; ++retry)
                    {
                        lim.max_nodes *= 4;
                        r = se.search(s, lim);
                    }
                    std::lock_guard<std::mutex> lk(mu);
                    n_retries++;
                    if (r.score < MATE_BOUND)
                    {
                        n_lost++;
                        std::string line;
                        for (const Undo &u : s.hist)
                        {
                            int t = move_type(u.m);
                            static const char *names[10] = {"N", "E", "NE", "SE", "NW", "SW", "W", "S", "R", "P"};
                            if (t == T_PASS)
                                line += "P ";
                            else
                                line += std::to_string(x_of(move_idx(u.m))) + "," + std::to_string(y_of(move_idx(u.m))) + names[t] + " ";
                        }
                        std::fprintf(stderr, "LOST WIN (game %d, player %d, score now %d): %s\n", g, s.cp + 1, r.score, line.c_str());
                    }
                }
                had_win[s.cp] = r.score >= MATE_BOUND;
                if (r.score >= MATE_BOUND)
                    claimed_win[s.cp] = true;
                if (!match)
                {
                    Record rec;
                    rec.stm = s.cp;
                    compute_features(s, f);
                    rec.v.push_back((std::int16_t)std::max(-32000, std::min(32000, r.score)));
                    rec.v.push_back((std::int16_t)(any_scoring(s) ? 1 : 0));
                    rec.v.push_back((std::int16_t)plies);
                    rec.v.push_back((std::int16_t)(g & 0x7FFF));
                    for (int i = 0; i < N_FEATS; ++i)
                        rec.v.push_back((std::int16_t)f[i]);
                    recs.push_back(std::move(rec));
                    // random walks from here, labelled by a short search (engine B's searcher)
                    std::vector<Move> wmv;
                    for (int w = 0; w < a.leaves; ++w)
                    {
                        int len = 1 + (int)(rng() % (std::uint64_t)a.leaf_walk), made = 0;
                        while (made < len && s.winner < 0)
                        {
                            s.do_move(random_walk_move(s, rng, wmv));
                            made++;
                        }
                        if (s.winner < 0)
                        {
                            SearchLimits ll;
                            ll.max_nodes = a.leaf_nodes;
                            sb.root_margin = 0;
                            SearchResult lr = sb.search(s, ll);
                            if (std::abs(lr.score) < MATE_BOUND)
                            {
                                Record leaf;
                                leaf.stm = s.cp;
                                leaf.leaf = true;
                                compute_features(s, f);
                                leaf.v.push_back((std::int16_t)lr.score);
                                leaf.v.push_back((std::int16_t)(any_scoring(s) ? 1 : 0));
                                leaf.v.push_back((std::int16_t)(plies + made));
                                leaf.v.push_back((std::int16_t)(g & 0x7FFF));
                                for (int i = 0; i < N_FEATS; ++i)
                                    leaf.v.push_back((std::int16_t)f[i]);
                                recs.push_back(std::move(leaf));
                            }
                        }
                        while (made-- > 0)
                            s.undo_move();
                    }
                }
                const int mover = s.cp, before = s.scores[mover];
                s.do_move(r.best);
                plies++;
                if (first_scorer < 0 && s.scores[mover] > before)
                {
                    first_scorer = mover;
                    first_score_ply = plies;
                }
                if (first_scorer == 0 && p0_first_outcome < 0 && plies == first_score_ply + 1)
                    p0_first_outcome = (s.winner == 0) ? 0 : (s.winner == 1) ? 1 : 2;
            }
            std::lock_guard<std::mutex> lk(mu);
            total_plies += plies;
            for (int p = 0; p < 2; ++p)
                if (claimed_win[p])
                {
                    n_claims++;
                    if (s.winner != p)
                        n_broken++;
                }
            if (s.winner == 0)
                p0_wins++;
            else if (s.winner == 1)
                p1_wins++;
            if (first_scorer >= 0)
                first_by[first_scorer]++;
            if (p0_first_outcome == 0)
                p0_first_won++;
            else if (p0_first_outcome == 1)
                p0_first_exceeded++;
            else if (p0_first_outcome == 2)
            {
                p0_first_matched++;
                if (s.winner == 0)
                    p0_first_matched_p0won++;
            }
            if (match)
            {
                if (s.winner < 0)
                    draws++;
                else if (s.winner == a_side)
                    wins++;
                else
                    losses++;
            }
            else
            {
                if (s.winner < 0)
                    draws++;
                for (auto &rec : recs)
                {
                    std::int16_t res = rec.leaf ? -2 : (s.winner < 0) ? 1 : (s.winner == rec.stm ? 2 : 0);
                    out.write((const char *)&res, sizeof(res));
                    out.write((const char *)rec.v.data(), (std::streamsize)(rec.v.size() * sizeof(std::int16_t)));
                    n_records++;
                }
            }
            int done = (int)(wins + draws + losses + (match ? 0 : p0_wins + p1_wins));
            if (done % 200 == 0)
                std::fprintf(stderr, "  %d/%d games\n", done, a.games);
        }
    };

    std::vector<std::thread> th;
    for (int t = 0; t < a.threads; ++t)
        th.emplace_back(worker, t);
    for (auto &t : th)
        t.join();

    if (match)
    {
        double n = (double)(wins + draws + losses);
        double score = (wins + 0.5 * draws) / std::max(1.0, n);
        std::printf("match A vs B: +%ld =%ld -%ld  score %.3f  (P0 wins %ld, P1 wins %ld, avg plies %.1f)\n", wins, draws, losses, score,
                    p0_wins, p1_wins, (double)total_plies / std::max(1.0, n));
    }
    else
    {
        std::printf("gen: %d games, %ld records, record width %d, P0 wins %ld, P1 wins %ld, draws %ld, avg plies %.1f\n", a.games, n_records,
                    N_FEATS + 5, p0_wins, p1_wins, draws, (double)total_plies / std::max(1, a.games));
    }
        std::printf("first polygon by player 1: %ld (unanswered, player 1 won: %ld; answered with a bigger one: %ld; matched, game went on: %ld, "
                "of which player 1 still won %ld); first polygon by player 2: %ld\n",
                first_by[0], p0_first_won, p0_first_exceeded, p0_first_matched, p0_first_matched_p0won, first_by[1]);
    std::printf("forced-win announcements: %ld, not converted: %ld (needed a longer search to keep the win: %ld, lost it anyway: %ld)\n", n_claims,
                n_broken, n_retries, n_lost);
    return 0;
}
