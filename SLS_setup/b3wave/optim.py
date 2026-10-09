"""
Optimizer param groups for the B3 wave: layer-wise LR decay (--llrd) and bottom-k freezing
(--freeze_bottom_k) over the backbone's transformer layers.

Group layout (AdamW: each split into decay / no-decay like the original build_optimizer):

    backbone        the top transformer layer, at --lr          <- param_groups[0], so the
                                                                   epoch log's LR= stays "the backbone LR"
    backbone_lXX    layer XX at --lr * decay ** (L - 1 - XX)
    backbone_stem   conv stem, positional embedding, everything not in a layer: --lr * decay ** L
    backbone_top    the final norm (ln_post / layer_norm): --lr
    head            SLS + BN + fc1 + fc2 at --lr * --head_lr_mult

With --llrd 1.0 and --freeze_bottom_k 0 the trainer keeps the original two-group optimizer, so
the reproduction run is untouched. Every trainable parameter must land in exactly one group
(asserted).
"""

import torch


def backbone_layers(model):
    return list(model.ssl_model.encoder.layers)


def top_norm_module(model):
    enc = model.ssl_model.encoder
    return getattr(enc, "ln_post", None) or getattr(enc, "layer_norm", None)


def freeze_bottom(model, k):
    """requires_grad=False on the stem and the first k transformer layers; returns #frozen params."""
    layers = backbone_layers(model)
    if k <= 0:
        return 0
    if k > len(layers):
        raise ValueError(f"--freeze_bottom_k {k} > {len(layers)} layers")
    layer_ids = {id(p) for layer in layers for p in layer.parameters()}
    top_ids = {id(p) for p in (top_norm_module(model).parameters() if top_norm_module(model) else [])}
    frozen = 0
    for p in model.ssl_model.parameters():          # stem = backbone params outside every layer / top norm
        if id(p) not in layer_ids and id(p) not in top_ids and p.requires_grad:
            p.requires_grad_(False)
            frozen += p.numel()
    for layer in layers[:k]:
        for p in layer.parameters():
            if p.requires_grad:
                p.requires_grad_(False)
                frozen += p.numel()
    return frozen


def _split(params, lr, wd, name, adamw):
    params = [p for p in params if p.requires_grad]
    if not params:
        return []
    if adamw:
        decay = [p for p in params if p.ndim > 1]
        no_decay = [p for p in params if p.ndim <= 1]
        groups = []
        if decay:
            groups.append({"params": decay, "lr": lr, "weight_decay": wd, "name": name})
        if no_decay:
            groups.append({"params": no_decay, "lr": lr, "weight_decay": 0.0, "name": name})
        return groups
    return [{"params": params, "lr": lr, "weight_decay": wd, "name": name}]


def build_optimizer_b3(model, trainable, args):
    """LLRD param groups (see module doc). `trainable` is the list the trainer already computed."""
    decay = float(args.llrd)
    if not 0.0 < decay <= 1.0:
        raise ValueError(f"--llrd must be in (0, 1], got {decay}")
    adamw = args.optim == "adamw"
    wd = args.weight_decay
    layers = backbone_layers(model)
    L = len(layers)
    top_norm = top_norm_module(model)

    layer_ids = {id(p) for layer in layers for p in layer.parameters()}
    top_ids = {id(p) for p in (top_norm.parameters() if top_norm is not None else [])}
    backbone_ids = {id(p) for p in model.ssl_model.parameters()}
    stem = [p for p in model.ssl_model.parameters() if id(p) not in layer_ids and id(p) not in top_ids]
    head = [p for p in trainable if id(p) not in backbone_ids]

    groups = []
    groups += _split(layers[L - 1].parameters(), args.lr, wd, "backbone", adamw)           # top layer first
    for i in range(L - 2, -1, -1):
        groups += _split(layers[i].parameters(), args.lr * decay ** (L - 1 - i), wd, f"backbone_l{i:02d}", adamw)
    groups += _split(stem, args.lr * decay ** L, wd, "backbone_stem", adamw)
    if top_norm is not None:
        groups += _split(top_norm.parameters(), args.lr, wd, "backbone_top", adamw)
    groups += _split(head, args.lr * args.head_lr_mult, wd, "head", adamw)

    grouped = [id(p) for g in groups for p in g["params"]]
    assert len(grouped) == len(set(grouped)), "a parameter landed in two groups"
    assert set(grouped) == {id(p) for p in trainable}, \
        f"grouped {len(grouped)} params but {len(trainable)} are trainable"
    opt_cls = torch.optim.AdamW if adamw else torch.optim.Adam
    return opt_cls(groups, lr=args.lr, weight_decay=wd)


def describe_groups(optimizer):
    """One line: the LR ladder (first group per name) and the number of groups."""
    lrs = {}
    for g in optimizer.param_groups:
        lrs.setdefault(g.get("name", "lr"), g["lr"])
    names = list(lrs)
    layer_names = sorted(n for n in names if n.startswith("backbone_l"))
    picks = ["backbone_stem"] + layer_names[:1] + layer_names[len(layer_names) // 2:len(layer_names) // 2 + 1] \
        + layer_names[-1:] + ["backbone", "backbone_top", "head"]
    parts = [f"{n}={lrs[n]:.2e}" for n in picks if n in lrs]
    return f"{len(optimizer.param_groups)} groups: " + ", ".join(parts)
