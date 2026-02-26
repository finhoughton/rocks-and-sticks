import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import inspect

import rl.train as train_mod


def safe_load_batch():
    p = Path('tests/fixtures/tiny_dataset.pt')
    import torch_geometric.data.data as tgd
    classes = []
    for n in dir(tgd):
        try:
            o = getattr(tgd, n)
        except Exception:
            continue
        if inspect.isclass(o):
            classes.append(o)
    with torch.serialization.safe_globals(classes):
        obj = torch.load(p)
    samples = obj[:2]
    batch = train_mod._collate_grouped(samples)
    return batch


def inspect_flow():
    batch = safe_load_batch()
    graph = batch['graph']
    move_feat = batch['move_feat']
    move_owner = batch['move_owner']
    p_targets = batch['policy']
    v_target = batch['value']

    print('graph.x requires_grad', getattr(graph.x, 'requires_grad', None))
    print('move_feat requires_grad', move_feat.requires_grad)
    print('move_owner requires_grad', move_owner.requires_grad)
    print('policy targets requires_grad', p_targets.requires_grad)
    print('value targets requires_grad', v_target.requires_grad)

    node_feat_dim = graph.x.size(1)
    global_feat_dim = graph.global_feats.size(1)
    model = train_mod.PolicyValueNet(node_feat_dim=node_feat_dim, global_feat_dim=global_feat_dim, move_feat_dim=int(batch['move_feat'].size(1)))
    model.train()

    # forward
    pooled, g_flat = model.encode_graph(graph)
    print('pooled.requires_grad', pooled.requires_grad, 'g_flat.requires_grad', g_flat.requires_grad)
    # policy logits
    p_logit = model.policy_logits_grouped(pooled, move_feat, move_owner)
    print('p_logit.requires_grad', p_logit.requires_grad, 'shape', p_logit.shape)

    # compute grouped log-softmax as in training
    num_groups = int(move_owner.max().item()) + 1 if move_owner.numel() else 0
    logp = torch.empty_like(p_logit)
    for g in range(num_groups):
        mask = move_owner == g
        if not mask.any():
            continue
        lg = p_logit[mask]
        lp = torch.nn.functional.log_softmax(lg, dim=0)
        logp[mask] = lp
    print('logp.requires_grad', logp.requires_grad)

    bsz = int(pooled.size(0))
    loss_p = -(p_targets * logp).sum() / max(1, bsz)
    v_in = torch.cat([pooled, g_flat], dim=-1)
    v = model.value_mlp(v_in).squeeze(-1)
    loss_v = torch.nn.MSELoss()(torch.sigmoid(v), v_target.view(-1))
    loss = loss_p + loss_v

    print('loss requires_grad', loss.requires_grad)

    params = [p for p in model.parameters() if p.requires_grad]
    names = [n for n, _ in model.named_parameters() if _.requires_grad]
    print('param count', len(params))

    # compute autograd.grad for all params
    grads_via_autograd = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
    print('Computed autograd.grad for params')

    # now backward
    for p in params:
        if p.grad is not None:
            p.grad.zero_()
    loss.backward()
    print('After backward, .grad presence and norms:')
    for name, p, ag in zip(names, params, grads_via_autograd):
        g = p.grad
        gn = None
        if g is not None:
            try:
                gn = float(g.norm().item())
            except Exception:
                gn = 'err'
        agn = None
        if ag is not None:
            try:
                agn = float(ag.norm().item())
            except Exception:
                agn = 'err'
        print(f'{name}: requires_grad={p.requires_grad} grad_present={g is not None} grad_norm={gn} autograd_grad_norm={agn}')


if __name__ == '__main__':
    inspect_flow()
