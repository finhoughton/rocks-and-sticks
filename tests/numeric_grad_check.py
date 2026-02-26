import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import inspect
import math
from pathlib import Path

import rl.train as train_mod


def load_batch():
    p = Path('tests/fixtures/tiny_dataset.pt')
    import inspect

    import torch_geometric.data.data as tgd
    classes = [getattr(tgd, n) for n in dir(tgd) if inspect.isclass(getattr(tgd, n))]
    with torch.serialization.safe_globals(classes):
        obj = torch.load(p)
    # use first two samples to form a batch
    samples = obj[:2]
    batch = train_mod._collate_grouped(samples)
    return batch


def numeric_check(eps=1e-4, tol=1e-3):
    batch = load_batch()
    device = torch.device('cpu')
    # infer dims
    g = batch['graph']
    node_feat_dim = g.x.size(1)
    global_feat_dim = g.global_feats.size(1)
    model = train_mod.PolicyValueNet(node_feat_dim=node_feat_dim, global_feat_dim=global_feat_dim, move_feat_dim=int(batch['move_feat'].size(1)))
    # Use eval mode for deterministic behaviour of normalization layers
    model.eval()
    # single forward/backward to get analytic grads
    graph = batch['graph']
    move_feat = batch['move_feat']
    move_owner = batch['move_owner']
    p_targets = batch['policy']
    v_target = batch['value']

    pooled, g_flat = model.encode_graph(graph)
    v_in = torch.cat([pooled, g_flat], dim=-1)
    v = model.value_mlp(v_in).squeeze(-1)
    # grouped log-softmax (match training implementation)
    p_logit = model.policy_logits_grouped(pooled, move_feat, move_owner)
    num_groups = int(move_owner.max().item()) + 1 if move_owner.numel() else 0
    logp = torch.empty_like(p_logit)
    for g in range(num_groups):
        mask = move_owner == g
        if not mask.any():
            continue
        lg = p_logit[mask]
        logp[mask] = torch.nn.functional.log_softmax(lg, dim=0)
    p = logp.exp()
    logp = torch.log(p.clamp(min=1e-12))
    bsz = int(pooled.size(0))
    loss_p = -(p_targets * logp).sum() / max(1, bsz)
    loss_v = torch.nn.MSELoss()(torch.sigmoid(v), v_target.view(-1))
    loss = loss_p + loss_v
    loss.backward()

    # pick one parameter from value_mlp and one from policy_mlp
    pv = next(model.value_mlp.parameters())
    pp = next(model.policy_mlp.parameters())
    gv = pv.grad.clone()
    gp = pp.grad.clone()

    def finite_grad(param, idx):
        orig = param.data.clone()
        flat = param.data.view(-1)
        orig_val = float(flat[idx].item())
        flat[idx] = orig_val + eps
        plus = compute_loss(model, batch)
        flat[idx] = orig_val - eps
        minus = compute_loss(model, batch)
        flat[idx] = orig_val
        # Debug prints to help diagnose zero finite-diff
        print(f'finite_grad idx={idx} orig={orig_val} plus={plus} minus={minus}')
        return (plus - minus) / (2 * eps)

    def compute_loss(model, batch):
        # Compute loss without modifying model running stats or tracking gradients
        was_training = model.training
        model.eval()
        with torch.no_grad():
            pooled, g_flat = model.encode_graph(batch['graph'])
            v_in = torch.cat([pooled, g_flat], dim=-1)
            v = model.value_mlp(v_in).squeeze(-1)
            # grouped log-softmax (match training implementation)
            p_logit = model.policy_logits_grouped(pooled, batch['move_feat'], batch['move_owner'])
            num_groups = int(batch['move_owner'].max().item()) + 1 if batch['move_owner'].numel() else 0
            logp = torch.empty_like(p_logit)
            for g in range(num_groups):
                mask = batch['move_owner'] == g
                if not mask.any():
                    continue
                lg = p_logit[mask]
                logp[mask] = torch.nn.functional.log_softmax(lg, dim=0)
            bsz = int(pooled.size(0))
            loss_p = -(batch['policy'] * logp).sum() / max(1, bsz)
            loss_v = torch.nn.MSELoss()(torch.sigmoid(v), batch['value'].view(-1))
            out = float((loss_p + loss_v).item())
        if was_training:
            model.train()
        return out

    # check a couple indices
    idx_v = 0
    idx_p = 0
    fg_v = finite_grad(pv, idx_v)
    fg_p = finite_grad(pp, idx_p)
    ag_v = float(gv.view(-1)[idx_v].item())
    ag_p = float(gp.view(-1)[idx_p].item())
    rel_v = abs(ag_v - fg_v) / max(1e-8, abs(fg_v))
    rel_p = abs(ag_p - fg_p) / max(1e-8, abs(fg_p))
    abs_v = abs(ag_v - fg_v)
    abs_p = abs(ag_p - fg_p)
    print('value ag',ag_v,'fg',fg_v,'rel',rel_v,'abs_diff',abs_v)
    print('policy ag',ag_p,'fg',fg_p,'rel',rel_p,'abs_diff',abs_p)
    # Accept if relative error small OR absolute difference is tiny
    ok_v = (rel_v < tol) or (abs_v < 1e-3)
    ok_p = (rel_p < tol) or (abs_p < 1e-4)
    return ok_v and ok_p

if __name__ == '__main__':
    ok = numeric_check()
    print('NUMERIC CHECK', 'PASS' if ok else 'FAIL')
