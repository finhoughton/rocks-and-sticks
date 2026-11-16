#pragma once

#include <string>
#include <vector>
#include <cstdint>
#include <cmath>

// Lightweight input for native NN evaluation (no Python/PyTorch dependency)
struct NNInput
{
    std::vector<float> node_feats;   // flat [num_nodes * node_feat_dim]
    int num_nodes = 0;
    std::vector<int32_t> edge_src;   // [num_edges]
    std::vector<int32_t> edge_dst;   // [num_edges]
    std::vector<float> edge_attr;    // flat [num_edges * edge_dim]
    int num_edges = 0;
    std::vector<float> global_feats; // flat [global_feat_dim]
};

// Native C++ implementation of GNNEval forward pass.
// Uses Accelerate BLAS on macOS for matrix multiplies, no Python/GIL needed.
class NativeNNEval
{
public:
    NativeNNEval() = default;

    // Load weights from binary file exported by scripts/export_nn_weights.py
    bool load_weights(const std::string &path);
    bool is_loaded() const { return loaded_; }

    // Evaluate a single position, returns logit (apply sigmoid for probability)
    float evaluate(const NNInput &input) const;

    // Evaluate multiple positions, returns logits
    std::vector<float> evaluate_batch(const std::vector<NNInput> &inputs) const;

    // Policy head: run backbone once, then score each move cheaply.
    // backbone_forward returns cat(pooled, global_feats) vector [hidden + gfd].
    std::vector<float> backbone_forward(const NNInput &input) const;
    // policy_score: given backbone output and a move feature vector, return raw logit.
    float policy_score(const std::vector<float> &backbone_out,
                       const float *move_feat, int mfd) const;
    bool has_policy() const { return has_policy_; }
    int move_feat_dim() const { return move_feat_dim_; }

    int node_feat_dim() const { return node_feat_dim_; }
    int global_feat_dim() const { return global_feat_dim_; }
    int hidden_dim() const { return hidden_; }

private:
    bool loaded_ = false;
    int node_feat_dim_ = 0;
    int global_feat_dim_ = 0;
    int hidden_ = 0;
    int num_layers_ = 0;
    int edge_dim_ = 2;

    struct ConvWeights
    {
        int in_ch = 0, out_ch = 0;
        float eps = 0.0f;
        std::vector<float> lin_w; // [in_ch, edge_dim] row-major
        std::vector<float> lin_b; // [in_ch]
        std::vector<float> nn0_w; // [out_ch, in_ch] (first Linear in MLP)
        std::vector<float> nn0_b; // [out_ch]
        std::vector<float> nn2_w; // [out_ch, out_ch] (second Linear in MLP)
        std::vector<float> nn2_b; // [out_ch]
    };

    struct NormWeights
    {
        int ch = 0;
        std::vector<float> mean_scale; // [ch] per-channel mean scale
        std::vector<float> weight;     // [ch]
        std::vector<float> bias;       // [ch]
    };

    std::vector<ConvWeights> conv_weights_;
    std::vector<NormWeights> norm_weights_;

    // Head: Sequential(Linear(hidden+gfd, hidden), ReLU, Dropout(skip), Linear(hidden, 1))
    std::vector<float> head0_w_; // [hidden, hidden + global_feat_dim]
    std::vector<float> head0_b_; // [hidden]
    std::vector<float> head3_w_; // [1, hidden] stored as [hidden]
    float head3_b_ = 0.0f;

    // Policy head (optional): Sequential(Linear(policy_in, hidden), ReLU, Linear(hidden, 1))
    // policy_in = hidden + global_feat_dim + move_feat_dim
    bool has_policy_ = false;
    int move_feat_dim_ = 0;
    std::vector<float> pol0_w_; // [hidden, policy_in]
    std::vector<float> pol0_b_; // [hidden]
    std::vector<float> pol2_w_; // [1, hidden] stored as [hidden]
    float pol2_b_ = 0.0f;
    mutable std::vector<float> buf_pol_in_, buf_pol_hidden_;

    // Matrix multiply: C = A @ B^T, A:[M,K], B:[N,K], C:[M,N]
    static void matmul(const float *A, const float *B, float *C,
                       int M, int N, int K);

    // Pre-allocated working buffers — reused each evaluate() call to avoid
    // repeated heap allocation. Mutable so evaluate() can stay logically const.
    mutable std::vector<float> buf_h_, buf_h_in_;
    mutable std::vector<float> buf_edge_proj_, buf_msg_, buf_agg_;
    mutable std::vector<float> buf_combined_, buf_temp_;
    mutable std::vector<float> buf_mean_, buf_var_;
    mutable std::vector<float> buf_pooled_, buf_cat_, buf_head_out_;
    // C += bias broadcast over rows
    static void add_bias(float *data, const float *bias, int rows, int cols);
    // In-place ReLU
    static void relu_inplace(float *data, int n);
};
