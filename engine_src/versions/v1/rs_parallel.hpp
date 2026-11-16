// Lazy-SMP driver: several searchers run the same iterative deepening on copies of the
// position and share one transposition table. The main searcher decides the move.
#pragma once

#include <thread>

#include "rs_search.hpp"

namespace rs
{

class ParallelSearcher
{
  public:
    ParallelSearcher(int tt_bits, int threads) : tt(std::make_shared<TransTable>(tt_bits))
    {
        set_threads(threads);
    }

    void set_threads(int threads)
    {
        threads = std::max(1, threads);
        while ((int)workers.size() < threads)
        {
            workers.push_back(std::make_unique<Searcher>(tt));
            states.push_back(std::make_unique<State>());
            workers.back()->helper_id = (int)workers.size() - 1;
            if (workers.size() > 1)
            {
                workers.back()->weights = workers[0]->weights;
                std::memcpy(workers.back()->opt, workers[0]->opt, sizeof(workers[0]->opt));
            }
        }
        workers.resize((size_t)threads);
        states.resize((size_t)threads);
    }

    Searcher &main() { return *workers[0]; }

    void set_weights(const Weights &w)
    {
        for (auto &wk : workers)
            wk->weights = w;
        clear();
    }

    void clear()
    {
        tt->clear();
        for (auto &wk : workers)
            wk->clear_local();
    }

    SearchResult search(State &root, const SearchLimits &lim)
    {
        if (workers.size() == 1)
        {
            workers[0]->shared_stop = nullptr;
            return workers[0]->search(root, lim);
        }
        stop.store(false);
        std::vector<std::thread> th;
        SearchLimits hl; // helpers run until told to stop
        hl.max_depth = lim.max_depth;
        for (size_t i = 1; i < workers.size(); ++i)
        {
            *states[i] = root;
            workers[i]->shared_stop = &stop;
            workers[i]->rng.seed(workers[0]->rng() + i);
            th.emplace_back([this, i, hl]() { workers[i]->search(*states[i], hl); });
        }
        workers[0]->shared_stop = nullptr;
        SearchResult res = workers[0]->search(root, lim);
        stop.store(true);
        for (auto &t : th)
            t.join();
        for (size_t i = 1; i < workers.size(); ++i)
            res.nodes += workers[i]->nodes;
        return res;
    }

  private:
    std::shared_ptr<TransTable> tt;
    std::vector<std::unique_ptr<Searcher>> workers;
    std::vector<std::unique_ptr<State>> states;
    std::atomic<bool> stop{false};
};

} // namespace rs
