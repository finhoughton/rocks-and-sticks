#include "nn_eval.hpp"

#include <fstream>
#include <cstring>
#include <algorithm>
#include <numeric>
#include <iostream>

#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#define HAS_BLAS 1
#else
#define HAS_BLAS 0
#endif

// ---- BLAS / fallback helpers ----

void NativeNNEval::matmul(const float *A, const float *B, float *C,
                          int M, int N, int K)
{
    // C[M,N] = A[M,K] @ B[N,K]^T  (B stored row-major as [N,K])
#if HAS_BLAS
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                M, N, K, 1.0f, A, K, B, K, 0.0f, C, N);
#else
    // Naive fallback
    for (int i = 0; i < M; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k)
                sum += A[i * K + k] * B[j * K + k];
            C[i * N + j] = sum;
        }
    }
#endif
}

void NativeNNEval::add_bias(float *data, const float *bias, int rows, int cols)
{
    for (int i = 0; i < rows; ++i)
        for (int c = 0; c < cols; ++c)
            data[i * cols + c] += bias[c];
}

void NativeNNEval::relu_inplace(float *data, int n)
{
    for (int i = 0; i < n; ++i)
        if (data[i] < 0.0f)
            data[i] = 0.0f;
}

// ---- Weight loading ----

static bool read_int32(std::ifstream &f, int32_t &v)
{
    f.read(reinterpret_cast<char *>(&v), 4);
    return f.good();
}

static bool read_float32(std::ifstream &f, float &v)
{
    f.read(reinterpret_cast<char *>(&v), 4);
    return f.good();
}

static bool read_floats(std::ifstream &f, std::vector<float> &v, size_t n)
{
    v.resize(n);
    f.read(reinterpret_cast<char *>(v.data()), n * 4);
    return f.good();
}

bool NativeNNEval::load_weights(const std::string &path)
{
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open())
    {
        std::cerr << "NativeNNEval: cannot open " << path << std::endl;
        return false;
    }

    // Magic: "NNWT"
    char magic[4];
    f.read(magic, 4);
    if (!f.good() || std::strncmp(magic, "NNWT", 4) != 0)
    {
        std::cerr << "NativeNNEval: bad magic in " << path << std::endl;
        return false;
    }

    int32_t nfd, gfd, hidden, nlayers, edge_dim;
    if (!read_int32(f, nfd) || !read_int32(f, gfd) || !read_int32(f, hidden) ||
        !read_int32(f, nlayers) || !read_int32(f, edge_dim))
    {
        std::cerr << "NativeNNEval: truncated header" << std::endl;
        return false;
    }

    node_feat_dim_ = nfd;
    global_feat_dim_ = gfd;
    hidden_ = hidden;
    num_layers_ = nlayers;
    edge_dim_ = edge_dim;

    conv_weights_.resize(nlayers);
    norm_weights_.resize(nlayers);

    for (int i = 0; i < nlayers; ++i)
    {
        auto &cw = conv_weights_[i];
        int32_t in_ch, out_ch, ed;
        if (!read_int32(f, in_ch) || !read_int32(f, out_ch) || !read_int32(f, ed))
            return false;
        if (!read_float32(f, cw.eps))
            return false;
        cw.in_ch = in_ch;
        cw.out_ch = out_ch;

        if (!read_floats(f, cw.lin_w, in_ch * ed))
            return false;
        if (!read_floats(f, cw.lin_b, in_ch))
            return false;
        if (!read_floats(f, cw.nn0_w, out_ch * in_ch))
            return false;
        if (!read_floats(f, cw.nn0_b, out_ch))
            return false;
        if (!read_floats(f, cw.nn2_w, out_ch * out_ch))
            return false;
        if (!read_floats(f, cw.nn2_b, out_ch))
            return false;
    }

    for (int i = 0; i < nlayers; ++i)
    {
        auto &nw = norm_weights_[i];
        int32_t ch;
        if (!read_int32(f, ch))
            return false;
        nw.ch = ch;
        // mean_scale is per-channel: [ch] floats
        if (!read_floats(f, nw.mean_scale, ch))
            return false;
        if (!read_floats(f, nw.weight, ch))
            return false;
        if (!read_floats(f, nw.bias, ch))
            return false;
    }

    // Head
    int head_in = hidden + gfd;
    if (!read_floats(f, head0_w_, hidden * head_in))
        return false;
    if (!read_floats(f, head0_b_, hidden))
        return false;
    if (!read_floats(f, head3_w_, hidden))
        return false; // [1, hidden] flattened
    if (!read_float32(f, head3_b_))
        return false;

    loaded_ = true;
    return true;
}

// ---- Forward pass (single position) ----

float NativeNNEval::evaluate(const NNInput &input) const
{
    if (!loaded_)
        return 0.0f;

    int N = input.num_nodes;
    int E = input.num_edges;

    if (N == 0)
        return 0.0f; // empty graph -> 0.5 prob (logit=0)

    // Working buffers
    std::vector<float> h(N * hidden_, 0.0f);
    std::vector<float> h_in(N * hidden_, 0.0f);

    for (int layer = 0; layer < num_layers_; ++layer)
    {
        const auto &cw = conv_weights_[layer];
        const auto &nw = norm_weights_[layer];
        int in_ch = cw.in_ch;
        int out_ch = cw.out_ch;

        // Pointer to current node features
        const float *x = (layer == 0) ? input.node_feats.data() : h.data();

        // Save h for residual connection
        if (in_ch == out_ch)
            std::copy(x, x + N * in_ch, h_in.begin());

        // 1. Project edge attributes: edge_proj[E, in_ch] = edge_attr[E, edge_dim] @ lin_w^T + lin_b
        std::vector<float> edge_proj(E * in_ch, 0.0f);
        if (E > 0)
        {
            matmul(input.edge_attr.data(), cw.lin_w.data(), edge_proj.data(),
                   E, in_ch, edge_dim_);
            add_bias(edge_proj.data(), cw.lin_b.data(), E, in_ch);
        }

        // 2. Messages: msg[e] = relu(x_j[e] + edge_proj[e])
        //    x_j = x[edge_src[e]]
        std::vector<float> msg(E * in_ch, 0.0f);
        for (int e = 0; e < E; ++e)
        {
            int src = input.edge_src[e];
            for (int c = 0; c < in_ch; ++c)
            {
                float val = x[src * in_ch + c] + edge_proj[e * in_ch + c];
                msg[e * in_ch + c] = val > 0.0f ? val : 0.0f;
            }
        }

        // 3. Scatter add: agg[dst] += msg
        std::vector<float> agg(N * in_ch, 0.0f);
        for (int e = 0; e < E; ++e)
        {
            int dst = input.edge_dst[e];
            for (int c = 0; c < in_ch; ++c)
                agg[dst * in_ch + c] += msg[e * in_ch + c];
        }

        // 4. Combined = agg + (1 + eps) * x
        std::vector<float> combined(N * in_ch, 0.0f);
        float eps_factor = 1.0f + cw.eps;
        for (int i = 0; i < N * in_ch; ++i)
            combined[i] = agg[i] + eps_factor * x[i];

        // 5. MLP: nn.0 (Linear + ReLU) + nn.2 (Linear)
        //    temp = combined @ nn0_w^T + nn0_b → [N, out_ch]
        std::vector<float> temp(N * out_ch, 0.0f);
        matmul(combined.data(), cw.nn0_w.data(), temp.data(),
               N, out_ch, in_ch);
        add_bias(temp.data(), cw.nn0_b.data(), N, out_ch);
        relu_inplace(temp.data(), N * out_ch);

        //    h = temp @ nn2_w^T + nn2_b → [N, out_ch]
        std::fill(h.begin(), h.begin() + N * out_ch, 0.0f);
        matmul(temp.data(), cw.nn2_w.data(), h.data(),
               N, out_ch, out_ch);
        add_bias(h.data(), cw.nn2_b.data(), N, out_ch);

        // 6. GraphNorm (single graph, batch_size=1)
        //    mean = mean(h, dim=0) → [out_ch]
        std::vector<float> mean(out_ch, 0.0f);
        float inv_N = 1.0f / static_cast<float>(N);
        for (int i = 0; i < N; ++i)
            for (int c = 0; c < out_ch; ++c)
                mean[c] += h[i * out_ch + c];
        for (int c = 0; c < out_ch; ++c)
            mean[c] *= inv_N;

        //    h = h - mean_scale * mean (per-channel mean_scale)
        for (int i = 0; i < N; ++i)
            for (int c = 0; c < out_ch; ++c)
                h[i * out_ch + c] -= nw.mean_scale[c] * mean[c];

        //    var = mean(h^2, dim=0) → [out_ch]
        std::vector<float> var(out_ch, 0.0f);
        for (int i = 0; i < N; ++i)
            for (int c = 0; c < out_ch; ++c)
            {
                float v = h[i * out_ch + c];
                var[c] += v * v;
            }
        for (int c = 0; c < out_ch; ++c)
            var[c] *= inv_N;

        //    h = weight * h / sqrt(var + eps) + bias
        for (int c = 0; c < out_ch; ++c)
        {
            float std_inv = 1.0f / std::sqrt(var[c] + 1e-5f);
            for (int i = 0; i < N; ++i)
                h[i * out_ch + c] = nw.weight[c] * h[i * out_ch + c] * std_inv + nw.bias[c];
        }

        // 7. ReLU
        relu_inplace(h.data(), N * out_ch);

        // 8. Residual connection (skip for layer 0 where dims differ)
        if (in_ch == out_ch)
        {
            for (int i = 0; i < N * out_ch; ++i)
                h[i] += h_in[i];
        }
    }

    // 9. Global mean pool → [hidden_]
    std::vector<float> pooled(hidden_, 0.0f);
    float inv_N2 = 1.0f / static_cast<float>(N);
    for (int i = 0; i < N; ++i)
        for (int c = 0; c < hidden_; ++c)
            pooled[c] += h[i * hidden_ + c];
    for (int c = 0; c < hidden_; ++c)
        pooled[c] *= inv_N2;

    // 10. Concatenate [pooled, global_feats] → [hidden_ + global_feat_dim_]
    int cat_dim = hidden_ + global_feat_dim_;
    std::vector<float> cat(cat_dim, 0.0f);
    std::copy(pooled.begin(), pooled.end(), cat.begin());
    std::copy(input.global_feats.begin(), input.global_feats.end(), cat.begin() + hidden_);

    // 11. Head: Linear(hidden+gfd, hidden) → ReLU → Linear(hidden, 1)
    std::vector<float> head_out(hidden_, 0.0f);
    matmul(cat.data(), head0_w_.data(), head_out.data(),
           1, hidden_, cat_dim);
    add_bias(head_out.data(), head0_b_.data(), 1, hidden_);
    relu_inplace(head_out.data(), hidden_);

    // Dropout is disabled during eval (no-op)

    // Final linear: logit = head_out @ head3_w^T + head3_b (scalar output)
    float logit = head3_b_;
    for (int c = 0; c < hidden_; ++c)
        logit += head_out[c] * head3_w_[c];

    return logit;
}

// ---- Batch evaluation ----

std::vector<float> NativeNNEval::evaluate_batch(const std::vector<NNInput> &inputs) const
{
    // For now, evaluate one by one. The BLAS operations at N=20-40 are fast
    // and the overhead of merging graphs isn't worth it for typical batch sizes.
    // Can be optimized to merged-graph batch later if profiling shows need.
    std::vector<float> results;
    results.reserve(inputs.size());
    for (auto &inp : inputs)
        results.push_back(evaluate(inp));
    return results;
}
