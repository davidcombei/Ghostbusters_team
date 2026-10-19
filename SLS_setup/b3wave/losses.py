"""
Consistency losses for the two-view training step of the B3 wave (--consistency).

    js_divergence           Jensen-Shannon divergence between the two views' class posteriors
                            (AugMix-style; symmetric, bounded by ln 2)
    embedding_consistency   1 - cosine similarity of the two views' pooled embeddings (the 1024-d
                            fc1 output of ModelSLSB3), optionally with the first view as a
                            stop-gradient "clean teacher"
    consistency_loss        weighted sum of the two, with the components for logging
"""

import math

import torch
import torch.nn.functional as F

LN2 = math.log(2.0)


def js_divergence(logits_a, logits_b, detach_a=False):
    """Mean over the batch of JS(p_a || p_b), computed in log space; in [0, ln 2]."""
    log_pa = F.log_softmax(logits_a.float(), dim=-1)
    log_pb = F.log_softmax(logits_b.float(), dim=-1)
    if detach_a:
        log_pa = log_pa.detach()
    # log m = log((p_a + p_b) / 2)
    log_m = torch.logsumexp(torch.stack([log_pa, log_pb], dim=0), dim=0) - LN2
    kl_a = (log_pa.exp() * (log_pa - log_m)).sum(dim=-1)
    kl_b = (log_pb.exp() * (log_pb - log_m)).sum(dim=-1)
    return (0.5 * (kl_a + kl_b)).mean()


def embedding_consistency(h_a, h_b, stop_grad_a=True):
    """1 - cos(h_a, h_b), batch mean; the first view is detached when stop_grad_a."""
    if stop_grad_a:
        h_a = h_a.detach()
    return (1.0 - F.cosine_similarity(h_a.float(), h_b.float(), dim=-1)).mean()


def consistency_loss(logits_a, logits_b, h_a=None, h_b=None, mode="js_emb",
                     lam_js=0.3, lam_emb=0.3, clean_view=False):
    """
    Returns (total, components). mode "js": JS only; "js_emb": JS + embedding term.
    clean_view=True treats view a as the teacher: both terms detach it.
    """
    if mode not in ("js", "js_emb"):
        raise ValueError(f"unknown consistency mode {mode}")
    js = js_divergence(logits_a, logits_b, detach_a=clean_view)
    total = lam_js * js
    comps = {"js": float(js.detach())}
    if mode == "js_emb":
        if h_a is None or h_b is None:
            raise ValueError("js_emb needs the embeddings of both views")
        emb = embedding_consistency(h_a, h_b, stop_grad_a=clean_view)
        total = total + lam_emb * emb
        comps["emb"] = float(emb.detach())
    return total, comps
