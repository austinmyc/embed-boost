#!/usr/bin/env python3
"""
WP1 --- oracle headroom for input-space steering, measured in retrieval units.

What this answers
-----------------
WP0 established that s -> Delta(I, s) is nonlinear. It could not say whether
that curvature is *useful*. This script optimizes s directly against the
retrieval objective with gold labels visible -- deliberate cheating, because
the number wanted is a CEILING, not an achievable score. Nothing downstream
(amortization, a learned generator, distillation) can exceed the oracle.

The unit under test is the ENSEMBLE, so everything is a curve over K stages
rather than a pair of arms. Two curves, plus a floor:

    arm0        no steering                            floor
    greedy@k    k slots, stages 1..k-1 FROZEN,         the boosting curve --
                stage k optimized against them         what the method actually
                                                       produces at depth k
    joint@k     k slots, all optimized together        the ceiling at depth k

greedy@k is real boosting: stage k is fitted to what the current ensemble
already gets wrong, and earlier stages are never revisited (they were also
optimized without knowing later stages were coming, which is faithful -- a
boosted ensemble does not get to retune tree 1 after tree 40).

Three separate questions fall out of the two curves, and they have different
answers:

    headroom      greedy@1 - arm0        is there ANY usable room
    depth         greedy@K - greedy@1    does the ensemble keep paying as it
                                         grows, or saturate after one stage
    capture       (greedy@K - greedy@1)  can GREEDY staging reach the capacity
                  / (joint@K - greedy@1) that K slots jointly can

`capture` is the actual boosting question and the one WP0's rho_cat could only
proxy. Boosting's premise is that greedy sequential fitting recovers most of
what joint optimization would. If joint@K climbs while greedy@K flattens,
the capacity is real but staging cannot reach it, and the stagewise framing is
dead no matter how nonlinear the encoder turned out to be.

Note greedy@1 and joint@1 are the same computation (one active slot), so the
two curves share their first point and every gain is measured from it.

One confound is deliberate rather than controlled: slot count grows with k, and
WP0 measured that even an EMPTY extra slot moves the embedding by ~8% of its
norm. Each greedy@k is nonetheless a real deployable configuration, which is
what the decision is about. The clean, slot-matched comparison is greedy@K vs
joint@K -- identical slot counts, differing only in how they were optimized --
and that is the comparison `capture` is built from.

Decision rule --- commit before looking
---------------------------------------
    headroom ~ 0                   mechanism has no usable room. Stop.
    headroom >> 0, depth ~ 0       single-stage steering is the paper.
    depth > 0, capture < 0.5       capacity exists, greedy cannot reach it.
    depth > 0, capture > 0.5       boosting is alive. Proceed to amortization.

Two traps this script is built around
-------------------------------------
**The oracle must not be allowed to cheat by magnitude.** With 1024 free
parameters per query, an unconstrained s would push the gold document to rank 1
for almost any query and the ceiling would read ~100% while meaning nothing.
s is therefore reparameterized as scale * v/||v||, so ||s|| is pinned exactly
at the operating point WP0 identified (0.5x the mean token-embedding norm).
`--free-norm` lifts the constraint and should be run once as a control: if it
saturates, that confirms the constraint is what makes the number meaningful.

Equal-norm slots are the wrong boosting analogue. A later stage with the same
||s|| as stage 1 can dominate attention rather than correct a residual.
`--scale-decay` / `--slot-scales` shrink later slots (stage 1 stays at `--scale`)
so each new slot is a small step -- the same role ν plays in F ← F + ν h_k.

**Only queries are steered.** Passages are encoded once, frozen, and cached.
Steering the corpus would be a different method and would also make the index
query-dependent, which defeats the point of dense retrieval.

Usage
-----
    python wp1_oracle.py --dataset scifact --device cuda
    python wp1_oracle.py --dataset nfcorpus --device cuda --steps 300
    python wp1_oracle.py --dataset fiqa --device cuda --corpus-dtype float16
    python wp1_oracle.py --dataset nfcorpus --device cuda --n-stages 8 \
        --scale 0.5 --scale-decay 0.1
    python wp1_oracle.py --dataset nfcorpus --device cuda --n-stages 8 \
        --slot-scales 0.5 0.1 0.005
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


class _Tee:
    """Mirror a stream into a log file, flushing every write so the log is
    tailable while a multi-hour run is still going."""

    def __init__(self, path_or_file, stream):
        self.f = open(path_or_file, "w") if isinstance(path_or_file, str) else path_or_file
        self.stream = stream

    def write(self, s):
        self.stream.write(s)
        self.f.write(s)
        self.flush()
        return len(s)

    def flush(self):
        self.stream.flush()
        self.f.flush()


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #


def load_beir(name: str, split: str) -> tuple[list[str], list[str], list[str], dict]:
    """BEIR from the HuggingFace mirror.

    Returns (doc_ids, doc_texts, query_ids, query_texts, qrels) where qrels maps
    query_id -> {doc_id: relevance}. Relevance is graded on some sets
    (nfcorpus) and binary on others (scifact); nDCG below handles both.
    """
    from datasets import load_dataset

    corpus = load_dataset(f"BeIR/{name}", "corpus")["corpus"]
    queries = load_dataset(f"BeIR/{name}", "queries")["queries"]
    qrels_ds = load_dataset(f"BeIR/{name}-qrels")[split]

    qrels: dict[str, dict[str, int]] = {}
    for row in qrels_ds:
        qid, did, rel = str(row["query-id"]), str(row["corpus-id"]), int(row["score"])
        if rel > 0:
            qrels.setdefault(qid, {})[did] = rel

    doc_ids, doc_texts = [], []
    for row in corpus:
        title, text = (row.get("title") or "").strip(), (row.get("text") or "").strip()
        doc_ids.append(str(row["_id"]))
        doc_texts.append(f"{title} {text}".strip() if title else text)

    # keep only queries that actually have judgements in this split
    q_ids, q_texts = [], []
    for row in queries:
        qid = str(row["_id"])
        if qid in qrels:
            q_ids.append(qid)
            q_texts.append(row["text"])

    return doc_ids, doc_texts, q_ids, q_texts, qrels


# --------------------------------------------------------------------------- #
# steerable encoder (gradient-enabled)
# --------------------------------------------------------------------------- #


class SteerableEncoder:
    """Frozen encoder with steering vectors appended in embedding space.

    The injection path is intentionally identical to probe_nonlinearity's:
    steering slots go immediately after the last real token and before padding,
    and the steering positions are excluded from the mean pool so s reaches the
    output only through attention from the real tokens. `self_check` asserts
    numerical agreement with the WP0 implementation -- if that fails, WP1 is
    measuring a different mechanism than WP0 gated on.

    Unlike WP0 this class does NOT wrap forwards in no_grad: gradients must
    flow back to the steering tensor. Model parameters stay frozen via
    requires_grad_(False).
    """

    def __init__(self, model_name: str, device: str = "cuda",
                 dtype: torch.dtype = torch.float32, max_len: int = 128):
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name, dtype=dtype)
        self.model.to(device).eval()
        self.model.requires_grad_(False)
        self.device = device
        self.dtype = dtype
        self.max_len = max_len
        self.wte = self.model.get_input_embeddings()
        self.d = self.model.config.hidden_size

    def token_norm(self) -> float:
        return self.wte.weight.detach().float().norm(dim=-1).mean().item()

    def tokenize(self, texts: list[str]) -> list[list[int]]:
        return self.tok(texts, truncation=True, max_length=self.max_len)["input_ids"]

    def forward_ids(self, ids: list[list[int]], steer: torch.Tensor | None = None,
                    pool_include_steer: bool = False) -> torch.Tensor:
        """Mean-pooled PRE-normalization embeddings, shape (B, d).

        steer: None or (B, S, d), differentiable.
        """
        B = len(ids)
        S = 0 if steer is None else steer.shape[1]
        L = max(len(x) for x in ids) + S

        embs = torch.zeros(B, L, self.d, device=self.device, dtype=self.dtype)
        attn = torch.zeros(B, L, dtype=torch.long, device=self.device)
        real = torch.zeros(B, L, dtype=torch.long, device=self.device)

        for b, seq in enumerate(ids):
            n = len(seq)
            embs[b, :n] = self.wte(torch.tensor(seq, device=self.device)).to(self.dtype)
            real[b, :n] = 1
            attn[b, :n] = 1
            if S:
                embs[b, n:n + S] = steer[b].to(self.dtype)
                attn[b, n:n + S] = 1

        h = self.model(inputs_embeds=embs, attention_mask=attn).last_hidden_state.float()
        m = (attn if pool_include_steer else real).unsqueeze(-1).float()
        return (h * m).sum(1) / m.sum(1).clamp(min=1e-9)

    @torch.no_grad()
    def encode(self, texts: list[str], batch_size: int = 64,
               show_every: int = 0) -> torch.Tensor:
        """Unsteered encoding for the corpus. Returns L2-normalized (N, d) on CPU."""
        outs = []
        t0 = time.time()
        for i in range(0, len(texts), batch_size):
            ids = self.tokenize(texts[i:i + batch_size])
            e = self.forward_ids(ids)
            outs.append(F.normalize(e, dim=-1).cpu())
            if show_every and (i // batch_size) % show_every == 0:
                done = min(i + batch_size, len(texts))
                rate = done / max(time.time() - t0, 1e-9)
                print(f"    encoded {done}/{len(texts)}  ({rate:.0f}/s)")
        return torch.cat(outs, dim=0)

    @torch.no_grad()
    def self_check(self, texts: list[str]) -> dict:
        """Injection path must be an exact no-op with nothing injected, and must
        not depend on batch composition once something is injected."""
        enc = self.tok(texts, padding=True, truncation=True,
                       max_length=self.max_len, return_tensors="pt").to(self.device)
        h = self.model(**enc).last_hidden_state.float()
        m = enc["attention_mask"].unsqueeze(-1).float()
        ref = (h * m).sum(1) / m.sum(1).clamp(min=1e-9)
        ids = self.tokenize(texts)
        ours = self.forward_ids(ids)
        err_identity = (ref - ours).abs().max().item()

        single = torch.cat([self.forward_ids(self.tokenize([t])) for t in texts], dim=0)
        err_batch = (single - ours).abs().max().item()

        g = torch.Generator().manual_seed(0)
        s = torch.randn(len(texts), 1, self.d, generator=g)
        s = F.normalize(s, dim=-1) * self.token_norm()
        s = s.to(self.device, self.dtype)
        big = self.forward_ids(ids, steer=s)
        one = torch.cat([self.forward_ids(self.tokenize([t]), steer=s[i:i + 1])
                         for i, t in enumerate(texts)], dim=0)
        err_steer_batch = (big - one).abs().max().item()

        return {"identity_vs_hf": err_identity,
                "batch_invariance": err_batch,
                "batch_invariance_steered": err_steer_batch}


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #


def evaluate(scores: torch.Tensor, doc_ids: list[str], q_ids: list[str],
             qrels: dict, ks=(10, 20, 100)) -> dict:
    """scores: (Q, N) similarity. Returns nDCG@10, R@k, MRR@10."""
    maxk = max(max(ks), 10)
    top = scores.topk(maxk, dim=-1).indices.cpu().numpy()

    ndcg, mrr = [], []
    recall = {k: [] for k in ks}
    for qi, qid in enumerate(q_ids):
        rel = qrels[qid]
        ranked = [doc_ids[j] for j in top[qi]]
        gains = np.array([rel.get(d, 0) for d in ranked], dtype=float)

        disc = 1.0 / np.log2(np.arange(2, 12))
        dcg = float((gains[:10] * disc).sum())
        ideal = np.sort(np.array(list(rel.values()), dtype=float))[::-1][:10]
        idcg = float((ideal * disc[:len(ideal)]).sum())
        ndcg.append(dcg / idcg if idcg > 0 else 0.0)

        hit = np.nonzero(gains[:10] > 0)[0]
        mrr.append(1.0 / (hit[0] + 1) if len(hit) else 0.0)

        n_rel = len(rel)
        for k in ks:
            recall[k].append(float((gains[:k] > 0).sum()) / n_rel)

    out = {"ndcg@10": float(np.mean(ndcg)), "mrr@10": float(np.mean(mrr))}
    out.update({f"recall@{k}": float(np.mean(v)) for k, v in recall.items()})
    return out


# --------------------------------------------------------------------------- #
# oracle optimization
# --------------------------------------------------------------------------- #


def expand_slot_scales(K: int, base: float, decay: float | None,
                       explicit: list[float] | None) -> list[float]:
    """Per-stage ||s|| as a multiple of the mean token-embedding norm.

    Stage 1 stays at `base` unless `--slot-scales` overrides it. Later stages
    shrink so they can only make residual corrections -- boosting shrinkage,
    not a mixing weight on the output. Unset decay keeps the old equal-norm
    schedule (every slot at `base`).
    """
    if explicit:
        scales = [float(x) for x in explicit]
        if len(scales) >= K:
            return scales[:K]
        if len(scales) == 1:
            return scales * K
        ratio = scales[-1] / scales[-2] if scales[-2] != 0 else 0.0
        while len(scales) < K:
            scales.append(scales[-1] * ratio)
        return scales
    if decay is None:
        return [base] * K
    if not (0.0 < decay < 1.0):
        raise SystemExit(f"--scale-decay must be in (0, 1), got {decay}")
    return [base * (decay ** k) for k in range(K)]


def _steer_from_raw(raw: torch.Tensor, scale: float | torch.Tensor,
                    free_norm: bool) -> torch.Tensor:
    """Reparameterize so ||s|| is pinned at `scale` unless free_norm is set.

    `scale` is a scalar, or a tensor broadcastable to `raw` (used for per-slot
    decaying norms). This is the guard that keeps the ceiling meaningful: an
    unconstrained s can drag almost any gold document to rank 1, which would
    report a ~100% ceiling that says nothing about whether the mechanism is
    usable.
    """
    if free_norm:
        return raw
    s = F.normalize(raw, dim=-1)
    if torch.is_tensor(scale):
        return s * scale.to(device=s.device, dtype=s.dtype)
    return s * scale


def _nll(q: torch.Tensor, C: torch.Tensor, pos: list[list[int]], temp: float) -> torch.Tensor:
    """Multi-positive InfoNCE over the FULL corpus.

    Full-corpus softmax rather than mined negatives: the oracle is supposed to
    measure the ceiling against the same corpus the metrics are computed on. A
    fixed negative pool would let s overfit that pool and inflate the loss
    improvement without moving nDCG.
    """
    # matmul in the corpus dtype (fp16 keeps a large index resident and hits
    # tensor cores), softmax upcast to fp32 -- logsumexp over a corpus-sized
    # axis in fp16 loses too much.
    logits = (F.normalize(q, dim=-1).to(C.dtype) @ C.T).float() / temp
    denom = torch.logsumexp(logits, dim=-1)
    loss = 0.0
    for i, p in enumerate(pos):
        num = torch.logsumexp(logits[i, p], dim=-1)
        loss = loss + (denom[i] - num)
    return loss / len(pos)


def optimize_arm(enc, q_ids_tok, C, positives, n_slots, trainable, scale,
                 steps, lr, temp, batch_size, free_norm, init_raw=None,
                 frozen=None, seed=0, label=""):
    """Optimize the steering slots listed in `trainable` for every query.

    Queries are independent, so a batch of them is optimized simultaneously --
    the loss is separable and this is what makes full-scale runs tractable.

    `scale` is either a scalar ||s|| applied to every slot, or a sequence of
    per-slot norms of length >= n_slots (absolute, already multiplied by the
    token-embedding norm). Decaying later entries is boosting shrinkage.

    frozen: dict {slot_index: (Q, d) tensor} held fixed (used by the greedy arm).
    Returns (steer_raw (Q, n_slots, d) on CPU, final query embeddings (Q, d)).
    """
    Q, d, dev = len(q_ids_tok), enc.d, enc.device
    g = torch.Generator().manual_seed(seed)
    raw = torch.randn(Q, n_slots, d, generator=g) if init_raw is None else init_raw.clone()
    raw = raw.to(dev)
    if isinstance(scale, (int, float)):
        scale_t = float(scale)
    else:
        scale_t = torch.tensor([float(x) for x in list(scale)[:n_slots]],
                               device=dev).view(1, n_slots, 1)

    out_emb = torch.zeros(Q, d)
    t0 = time.time()
    for i in range(0, Q, batch_size):
        sl = slice(i, min(i + batch_size, Q))
        ids = q_ids_tok[sl]
        pos = positives[sl]
        blk = raw[sl].clone().requires_grad_(True)
        opt = torch.optim.Adam([blk], lr=lr)

        for step in range(steps):
            opt.zero_grad(set_to_none=True)
            s = _steer_from_raw(blk, scale_t, free_norm)
            parts = []
            for k in range(n_slots):
                if frozen is not None and k in frozen:
                    parts.append(frozen[k][sl].to(dev).unsqueeze(1))
                elif k in trainable:
                    parts.append(s[:, k:k + 1])
                else:
                    parts.append(torch.zeros_like(s[:, k:k + 1]))
            steer = torch.cat(parts, dim=1)
            e = enc.forward_ids(ids, steer=steer)
            loss = _nll(e, C, pos, temp)
            loss.backward()
            opt.step()

        with torch.no_grad():
            s = _steer_from_raw(blk, scale_t, free_norm)
            parts = []
            for k in range(n_slots):
                if frozen is not None and k in frozen:
                    parts.append(frozen[k][sl].to(dev).unsqueeze(1))
                elif k in trainable:
                    parts.append(s[:, k:k + 1])
                else:
                    parts.append(torch.zeros_like(s[:, k:k + 1]))
            steer = torch.cat(parts, dim=1)
            out_emb[sl] = F.normalize(enc.forward_ids(ids, steer=steer), dim=-1).cpu()
            raw[sl] = blk.detach()

        done = min(i + batch_size, Q)
        el = time.time() - t0
        print(f"    [{label}] {done}/{Q} queries   loss={loss.item():.4f}   "
              f"{el:.0f}s elapsed, ~{el / done * (Q - done):.0f}s left")

    return raw.cpu(), out_emb


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="intfloat/e5-large-v2")
    ap.add_argument("--dataset", default="scifact",
                    help="BEIR name: scifact, nfcorpus, fiqa, scidocs, arguana, trec-covid")
    ap.add_argument("--split", default="test")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    ap.add_argument("--corpus-dtype", default="float32", choices=["float32", "float16"],
                    help="dtype of the resident corpus matrix; float16 halves GPU memory")
    ap.add_argument("--query-prefix", default="query: ")
    ap.add_argument("--doc-prefix", default="passage: ")
    ap.add_argument("--scale", type=float, default=0.5,
                    help="||s|| of stage 1 as a multiple of the mean token-embedding "
                         "norm; 0.5 is the operating point WP0 identified")
    ap.add_argument("--scale-decay", type=float, default=None,
                    help="boosting shrinkage: ||s_k|| = scale * decay^(k-1) * token_norm. "
                         "Unset = every slot at --scale (old equal-norm schedule). "
                         "e.g. --scale 0.5 --scale-decay 0.1 -> 0.5, 0.05, 0.005, ...")
    ap.add_argument("--slot-scales", type=float, nargs="+", default=None,
                    help="explicit ||s||/token_norm per stage, overriding --scale-decay. "
                         "If shorter than K, continue with the last ratio "
                         "(e.g. 0.5 0.1 0.005 -> ... 0.00025)")
    ap.add_argument("--free-norm", action="store_true",
                    help="control run: lift the norm constraint (expect saturation)")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--temp", type=float, default=0.02)
    ap.add_argument("--batch-size", type=int, default=32, help="queries optimized at once")
    ap.add_argument("--encode-batch-size", type=int, default=128)
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--n-queries", type=int, default=0, help="0 = all")
    ap.add_argument("--n-stages", type=int, default=4,
                    help="K: ensemble depth. The greedy curve is evaluated at every "
                         "k=1..K, so this is the boosting curve's length")
    ap.add_argument("--joint-at", type=int, nargs="+", default=None,
                    help="depths at which to also run the joint ceiling "
                         "(default: just K; k=1 is shared with the greedy curve)")
    ap.add_argument("--skip-baseline", action="store_true")
    ap.add_argument("--zero-curve", action="store_true",
                    help="null control: evaluate k EMPTY slots for k=1..K with no "
                         "optimization at all. Any decline here is pure slot "
                         "perturbation, and must be subtracted from the greedy "
                         "curve before reading it as a staging result")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cache-dir", default="results/cache")
    ap.add_argument("--out", default=None)
    ap.add_argument("--log", default=None)
    args = ap.parse_args()

    args.out = args.out or f"results/wp1_{args.dataset}.json"
    log_path = args.log or os.path.splitext(args.out)[0] + ".log"
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    log_file = open(log_path, "w")
    sys.stdout = _Tee(log_file, sys.stdout)
    sys.stderr = _Tee(log_file, sys.stderr)
    t_start = time.time()
    print(f"logging to {log_path}")
    print(f"started {datetime.now():%Y-%m-%d %H:%M:%S}")

    torch.manual_seed(args.seed)
    dtype = dict(float32=torch.float32, float16=torch.float16,
                 bfloat16=torch.bfloat16)[args.dtype]

    print(f"loading {args.model} on {args.device} ({args.dtype})")
    enc = SteerableEncoder(args.model, args.device, dtype, args.max_len)
    tnorm = enc.token_norm()
    K = args.n_stages
    rel_scales = expand_slot_scales(K, args.scale, args.scale_decay, args.slot_scales)
    abs_scales = [s * tnorm for s in rel_scales]
    sched = ", ".join(f"{s:.4g}" for s in rel_scales)
    print(f"  d={enc.d}   token_norm={tnorm:.4f}   |s|/tnorm = [{sched}]"
          f"{'  [FREE NORM]' if args.free_norm else ''}")

    print(f"\nloading BEIR/{args.dataset} ({args.split})")
    doc_ids, doc_texts, q_ids, q_texts, qrels = load_beir(args.dataset, args.split)
    if args.n_queries:
        q_ids, q_texts = q_ids[:args.n_queries], q_texts[:args.n_queries]
    print(f"  {len(doc_texts)} docs   {len(q_ids)} queries")

    chk = enc.self_check([args.query_prefix + t for t in q_texts[:4]])
    print("\nself-check (max abs err, want < 1e-4):")
    for k, v in chk.items():
        print(f"  [{'ok  ' if v < 1e-4 else 'FAIL'}] {k:26s} {v:.2e}")
    if max(chk.values()) > 1e-4:
        print("  -> injection path differs from WP0. Everything below is unreliable.")

    os.makedirs(args.cache_dir, exist_ok=True)
    cache = os.path.join(args.cache_dir,
                         f"{args.dataset}_{args.model.replace('/', '_')}_{args.dtype}.pt")
    if os.path.exists(cache):
        print(f"\nloading cached corpus embeddings from {cache}")
        C_cpu = torch.load(cache)
    else:
        print(f"\nencoding {len(doc_texts)} passages (frozen, once)")
        C_cpu = enc.encode([args.doc_prefix + t for t in doc_texts],
                           batch_size=args.encode_batch_size, show_every=20)
        torch.save(C_cpu, cache)
        print(f"  cached to {cache}")

    cdt = torch.float16 if args.corpus_dtype == "float16" else torch.float32
    C = C_cpu.to(enc.device, cdt)
    print(f"  corpus matrix {tuple(C.shape)} {args.corpus_dtype} "
          f"= {C.numel() * C.element_size() / 1e9:.2f} GB on {enc.device}")

    did_to_row = {d: i for i, d in enumerate(doc_ids)}
    positives = [[did_to_row[d] for d in qrels[q] if d in did_to_row] for q in q_ids]
    keep = [i for i, p in enumerate(positives) if p]
    if len(keep) != len(q_ids):
        print(f"  dropped {len(q_ids) - len(keep)} queries with no in-corpus gold")
        q_ids = [q_ids[i] for i in keep]
        q_texts = [q_texts[i] for i in keep]
        positives = [positives[i] for i in keep]

    q_tok = enc.tokenize([args.query_prefix + t for t in q_texts])
    results = {"config": vars(args) | {"d": enc.d, "token_norm": tnorm,
                                       "n_docs": len(doc_ids), "n_queries": len(q_ids),
                                       "slot_scales": rel_scales},
               "self_check": chk, "arms": {}}

    def run_eval(name, emb):
        with torch.no_grad():
            sc = emb.to(enc.device, cdt) @ C.T
            m = evaluate(sc.float(), doc_ids, q_ids, qrels)
        results["arms"][name] = m
        print(f"  {name:<6} nDCG@10={m['ndcg@10']:.4f}  R@20={m['recall@20']:.4f}  "
              f"R@100={m['recall@100']:.4f}  MRR@10={m['mrr@10']:.4f}")
        return m

    common = dict(scale=abs_scales, steps=args.steps, lr=args.lr, temp=args.temp,
                  batch_size=args.batch_size, free_norm=args.free_norm, seed=args.seed)

    joint_at = sorted(set(args.joint_at or [K]) - {1})   # k=1 shared with greedy
    print("\n" + "=" * 92)
    print(f"ensemble depth K={K}  (steps={args.steps} lr={args.lr} temp={args.temp} "
          f"|s|/tnorm=[{sched}], {args.batch_size} queries per optimization batch)")
    print("=" * 92)

    if not args.skip_baseline:
        t0 = time.time()
        with torch.no_grad():
            e0 = torch.cat([F.normalize(enc.forward_ids(q_tok[i:i + args.batch_size]),
                                        dim=-1).cpu()
                            for i in range(0, len(q_tok), args.batch_size)], dim=0)
        run_eval("arm0", e0)
        print(f"         [{time.time() - t0:.0f}s]")

    # ---- null curve: k empty slots, nothing optimized --------------------- #
    # WP0 measured slot_bias at ~8% of the embedding norm for a SINGLE empty
    # attended position. Stacking k of them moves the embedding on its own, so a
    # declining greedy curve is only evidence about staging to the extent it
    # declines faster than this.
    if args.zero_curve:
        for k in range(1, K + 1):
            with torch.no_grad():
                embs = []
                for i in range(0, len(q_tok), args.batch_size):
                    ids = q_tok[i:i + args.batch_size]
                    z = torch.zeros(len(ids), k, enc.d, device=enc.device)
                    embs.append(F.normalize(enc.forward_ids(ids, steer=z), dim=-1).cpu())
                run_eval(f"zero@{k}", torch.cat(embs, dim=0))

    # ---- greedy curve: the boosting ensemble itself ----------------------- #
    # Stage k is fitted against stages 1..k-1 held frozen. Earlier stages are
    # never revisited, which is what makes this boosting rather than joint
    # optimization with extra steps.
    frozen: dict[int, torch.Tensor] = {}
    for k in range(1, K + 1):
        t0 = time.time()
        raw, emb = optimize_arm(enc, q_tok, C, positives, k, {k - 1},
                                frozen=dict(frozen), label=f"greedy@{k}", **common)
        run_eval(f"greedy@{k}", emb)
        frozen[k - 1] = _steer_from_raw(raw[:, k - 1], abs_scales[k - 1],
                                        args.free_norm)
        print(f"         [{time.time() - t0:.0f}s]")

    # ---- joint ceiling at selected depths --------------------------------- #
    for k in joint_at:
        t0 = time.time()
        _, emb = optimize_arm(enc, q_tok, C, positives, k, set(range(k)),
                              label=f"joint@{k}", **common)
        run_eval(f"joint@{k}", emb)
        print(f"         [{time.time() - t0:.0f}s]")

    # ---- verdict --------------------------------------------------------- #
    a = results["arms"]
    m = "ndcg@10"
    print("\n" + "=" * 92)
    print(f"{'depth':<8}{'greedy':>10}{'joint':>10}{'zero':>10}"
          f"{'greedy gain':>14}{'net of zero':>13}")
    for k in range(1, K + 1):
        gk = a.get(f"greedy@{k}", {}).get(m)
        jk = a.get(f"joint@{k}", {}).get(m)
        zk = a.get(f"zero@{k}", {}).get(m)
        if gk is None:
            continue
        gain = gk - a["greedy@1"][m]
        # the greedy curve minus the null curve: what staging did beyond simply
        # occupying k positions
        net = (f"{gain - (zk - a['zero@1'][m]):+.4f}"
               if zk is not None and "zero@1" in a else "--")
        print(f"k={k:<6}{gk:>10.4f}{(f'{jk:.4f}' if jk else '--'):>10}"
              f"{(f'{zk:.4f}' if zk else '--'):>10}{gain:>+14.4f}{net:>13}")

    if "arm0" in a and f"joint@{K}" in a and K > 1:
        headroom = a["greedy@1"][m] - a["arm0"][m]
        depth = a[f"greedy@{K}"][m] - a["greedy@1"][m]
        ceiling = a[f"joint@{K}"][m] - a["greedy@1"][m]
        capture = depth / ceiling if abs(ceiling) > 1e-9 else float("nan")
        results["verdict"] = {"headroom_ndcg": headroom, "depth_gain": depth,
                              "joint_ceiling_gain": ceiling,
                              "greedy_capture_frac": capture, "K": K}
        print("-" * 92)
        print(f"headroom  greedy@1 - arm0            {headroom:+.4f} nDCG@10")
        print(f"depth     greedy@{K} - greedy@1        {depth:+.4f}")
        print(f"ceiling   joint@{K}  - greedy@1        {ceiling:+.4f}")
        print(f"capture   depth / ceiling            {capture * 100:.1f}%")
        print("-" * 92)
        if headroom < 0.01:
            print("STOP    oracle steering barely beats the frozen baseline. The "
                  "mechanism has no usable room; no generator can do better.")
        elif depth < 0.01 and ceiling < 0.01:
            print("PIVOT   the ensemble adds nothing beyond stage 1, even jointly. "
                  "Single-stage steering is the paper; drop the stagewise framing.")
        elif capture < 0.5:
            print("PIVOT   K slots hold real capacity but greedy staging cannot "
                  "reach it. Joint optimization is the method; boosting is dead.")
        else:
            print("GO      greedy staging recovers most of the joint ceiling. "
                  "Boosting is alive -- next question is the amortization gap.")
        print("        Check the per-k column too: a curve that flattens after "
              "k=2 caps useful ensemble depth regardless of the verdict above.")
        print("        Reminder: every number here uses gold labels. These are "
              "ceilings, not achievable scores.")
    print("=" * 92)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nwrote {args.out}")
    print(f"finished {datetime.now():%Y-%m-%d %H:%M:%S} "
          f"({(time.time() - t_start) / 60:.1f} min total)")


if __name__ == "__main__":
    main()
