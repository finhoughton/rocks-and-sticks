#pragma once

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
    Node(int x_, int y_) : x(x_), y(y_) {}
    Coord c() const { return {x, y}; }
};

constexpr Coord calc_end(Coord p, int d)
{
    return {p.first + DIR_DELTAS[d][0], p.second + DIR_DELTAS[d][1]};
}

