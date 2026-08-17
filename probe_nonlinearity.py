#!/usr/bin/env python3
"""
WP0 --- nonlinearity probe for input-space steering of a frozen text encoder.

Core mechanism under test
-------------------------
A steering vector s in R^d is appended in the encoder's *embedding* space
(no discretization, no token). The induced delta is

    Delta(I, s) := M([emb(I) ; s]) - M(I)

The GO/NO-GO question for the whole project: is the map s -> Delta(I, s)
nonlinear enough that stacking stages can add expressivity? Two failure modes
are measured separately.

    rho_sum  --- superposition inside ONE slot: D(s1 + s2) vs D(s1) + D(s2).
                 If ~0, the single-slot map is affine and the *additive*
                 variant collapses to a single stage.

    rho_cat  --- composition across TWO slots: D(s1, s2) vs D(s1, 0) + D(0, s2).
                 If ~0, the *concatenative* variant collapses to a single stage.

    rank     --- rank of the reachable set {Delta(I, s) : s} for fixed I.
                 Upper-bounds how many stages can ever be useful.

Baselines matter more than they look
------------------------------------
Delta is NOT zero at s = 0: merely appending an extra attended position
perturbs the output. So an encoder that was perfectly affine in s would still
score rho > 0 if deltas were taken against M(I) -- the residual would pick up
that constant offset and fake a GO. Every rho here is therefore measured
against a *slot-matched* baseline: M([emb(I); 0]) for the one-slot test and
M([emb(I); 0; 0]) for the two-slot test, with the unused slot zeroed rather
than removed. Under any affine model both rho's are then exactly 0, so a
nonzero value is real curvature. The size of that offset is reported
separately as `slot_bias`.

Both rho's are reported on the PRE-normalization pooled vector (isolates the
encoder's own nonlinearity) and after L2 normalization (what retrieval sees).
The pre-norm number is the honest gate -- L2 projection is itself nonlinear
and would inflate rho even for an affine encoder.

Usage
-----
    python probe_nonlinearity.py --quick                       # ~2 min smoke test
    python probe_nonlinearity.py --model intfloat/e5-large-v2 --device cuda
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
from transformers import AutoModel, AutoTokenizer


class _Tee:
    """Mirror a stream into a log file.

    Flushes on every write: the whole point is that the log is tailable while
    a multi-hour sweep is still running, and CPU runs here are long enough
    that a buffered log would be empty exactly when it is wanted.
    """

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

DEFAULT_TEXTS = [
    "what is the capital of australia",
    "how long does it take to boil an egg",
    "symptoms of vitamin d deficiency in adults",
    "average cost of a wedding in the united states",
    "difference between weather and climate",
    "who wrote the book the great gatsby",
    "how do noise cancelling headphones work",
    "best time of year to visit iceland",
    "what causes the northern lights",
    "how many calories in a banana",
    "define monetary policy",
    "why is the sky blue",
    "how to remove a red wine stain from carpet",
    "history of the roman aqueduct system",
    "what does a product manager actually do",
    "side effects of long term ibuprofen use",
    "how does a heat pump work in cold weather",
    "population of tokyo metropolitan area",
    "what is the difference between tcp and udp",
    "recipe for sourdough starter from scratch",
    "when was the eiffel tower built",
    "how do vaccines produce immunity",
    "explain compound interest with an example",
    "signs of a failing car alternator",
    "what is the halting problem",
    "how deep is the mariana trench",
    "rules of offside in soccer",
    "how to calculate standard deviation by hand",
    "what is quantitative easing",
    "life cycle of a monarch butterfly",
    "why do cats purr",
    "how does gps determine your location",
    "what is the boiling point of water at altitude",
    "difference between raid 0 and raid 1",
    "how long do lithium ion batteries last",
    "who discovered penicillin and when",
    "what is a black hole event horizon",
    "how to read a nutrition label",
    "what does the spleen do",
    "origins of the silk road trade routes",
    "how are tsunamis formed",
    "what is the difference between llc and s corp",
    "how does anesthesia work on the brain",
    "best practices for password security",
    "what is the greenhouse effect",
    "how do plants perform photosynthesis",
    "what caused the 2008 financial crisis",
    "how does a nuclear reactor generate power",
    "what is machine learning in simple terms",
    "how far away is the nearest star",
    "why does bread rise when baking",
    "what is the role of the federal reserve",
    "how do birds navigate during migration",
    "what is the difference between ram and ssd",
    "how is glass made from sand",
    "what are the stages of sleep",
    "how does inflation affect savings",
    "what is the tallest mountain in europe",
    "how do solar panels convert light to electricity",
    "what is the function of mitochondria",
    "when did the berlin wall fall",
    "how do earthquakes get measured",
    "what is the difference between virus and bacteria",
    "how does a refrigerator keep food cold",
]


# --------------------------------------------------------------------------- #
# steerable encoder
# --------------------------------------------------------------------------- #


class SteerableEncoder:
    """Frozen encoder with steering vectors appended in embedding space.

    Steering vectors are inserted immediately after the last *real* token of
    each sequence and before any padding, so a given (text, s) pair produces
    an identical delta regardless of what else is in the batch. That
    batch-invariance is not free: naive right-padding puts s after the pads,
    which makes its position embedding depend on the batch's longest sequence
    and turns every number in this file into noise. `self_check` asserts it.
    """

    def __init__(self, model_name: str, device: str = "cpu",
                 dtype: torch.dtype = torch.float32, max_len: int = 128):
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name, torch_dtype=dtype)
        self.model.to(device).eval()
        self.model.requires_grad_(False)
        self.device = device
        self.dtype = dtype
        self.max_len = max_len
        self.wte = self.model.get_input_embeddings()
        self.d = self.model.config.hidden_size

    # -- helpers ----------------------------------------------------------- #

    def token_norm(self) -> float:
        """Mean L2 norm of a real input embedding: the natural scale for s."""
        return self.wte.weight.detach().float().norm(dim=-1).mean().item()

    def sample_s(self, n: int, kind: str, scale: float, g: torch.Generator) -> torch.Tensor:
        """(n, d) steering vectors on CPU, each with L2 norm == `scale`.

        gauss: isotropic random direction, almost surely off the token manifold.
        token: direction of a real vocabulary embedding, i.e. in-distribution.
        """
        if kind == "gauss":
            s = torch.randn(n, self.d, generator=g)
        elif kind == "token":
            V = self.wte.weight.shape[0]
            idx = torch.randint(0, V, (n,), generator=g)
            s = self.wte.weight.detach()[idx.to(self.wte.weight.device)].float().cpu().clone()
        else:
            raise ValueError(f"unknown s distribution: {kind}")
        return s / s.norm(dim=-1, keepdim=True).clamp(min=1e-9) * scale

    # -- forward ----------------------------------------------------------- #

    @torch.no_grad()
    def encode(self, texts: list[str], steer: torch.Tensor | None = None,
               pool_include_steer: bool = False, batch_size: int = 16) -> torch.Tensor:
        """Mean-pooled, PRE-normalization embeddings, shape (N, d).

        steer: None, or (N, S, d) -- one steering block of S slots per text.
        """
        if steer is not None:
            assert steer.shape[0] == len(texts), "need one steering block per text"
            assert steer.shape[2] == self.d, "steering vector has wrong width"
        outs = []
        for i in range(0, len(texts), batch_size):
            blk = None if steer is None else steer[i:i + batch_size]
            outs.append(self._encode_batch(texts[i:i + batch_size], blk, pool_include_steer))
        return torch.cat(outs, dim=0)

    @torch.no_grad()
    def _encode_batch(self, texts, steer, pool_include_steer):
        ids = self.tok(texts, truncation=True, max_length=self.max_len)["input_ids"]
        S = 0 if steer is None else steer.shape[1]
        B = len(texts)
        L = max(len(x) for x in ids) + S

        embs = torch.zeros(B, L, self.d, device=self.device, dtype=self.dtype)
        attn = torch.zeros(B, L, dtype=torch.long, device=self.device)   # real + steer
        real = torch.zeros(B, L, dtype=torch.long, device=self.device)   # real only

        for b, seq in enumerate(ids):
            n = len(seq)
            embs[b, :n] = self.wte(torch.tensor(seq, device=self.device)).to(self.dtype)
            real[b, :n] = 1
            attn[b, :n] = 1
            if S:
                embs[b, n:n + S] = steer[b].to(self.device, self.dtype)
                attn[b, n:n + S] = 1

        h = self.model(inputs_embeds=embs, attention_mask=attn).last_hidden_state.float()
        m = (attn if pool_include_steer else real).unsqueeze(-1).float()
        return (h * m).sum(1) / m.sum(1).clamp(min=1e-9)

    # -- correctness ------------------------------------------------------- #

    @torch.no_grad()
    def self_check(self, texts: list[str]) -> dict:
        """The injection path must be an exact no-op with nothing injected,
        and must not depend on batch composition once something is."""
        enc = self.tok(texts, padding=True, truncation=True,
                       max_length=self.max_len, return_tensors="pt").to(self.device)
        h = self.model(**enc).last_hidden_state.float()
        m = enc["attention_mask"].unsqueeze(-1).float()
        ref = (h * m).sum(1) / m.sum(1).clamp(min=1e-9)
        ours = self.encode(texts, batch_size=len(texts))
        err_identity = (ref - ours).abs().max().item()

        single = torch.cat([self.encode([t]) for t in texts], dim=0)
        err_batch = (single - ours).abs().max().item()

        g = torch.Generator().manual_seed(0)
        s = self.sample_s(len(texts), "gauss", self.token_norm(), g).unsqueeze(1)
        big = self.encode(texts, steer=s, batch_size=len(texts))
        one = torch.cat([self.encode([t], steer=s[i:i + 1], batch_size=1)
                         for i, t in enumerate(texts)], dim=0)
        err_steer_batch = (big - one).abs().max().item()

        return {"identity_vs_hf": err_identity,
                "batch_invariance": err_batch,
                "batch_invariance_steered": err_steer_batch}


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #


def _norm(x: torch.Tensor) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True).clamp(min=1e-9)


def _rho(d12: torch.Tensor, d1: torch.Tensor, d2: torch.Tensor) -> dict:
    """Per-row rho, plus cosine and norm ratio of composed vs summed delta."""
    lin = d1 + d2
    rho = (d12 - lin).norm(dim=-1) / (d1.norm(dim=-1) + d2.norm(dim=-1)).clamp(min=1e-9)
    cos = torch.nn.functional.cosine_similarity(d12, lin, dim=-1)
    ratio = d12.norm(dim=-1) / lin.norm(dim=-1).clamp(min=1e-9)
    return {"rho": rho, "cos": cos, "ratio": ratio}


def _stat(x: torch.Tensor) -> dict:
    x = x.float()
    return {"mean": x.mean().item(),
            "std": x.std().item() if x.numel() > 1 else 0.0,
            "p10": x.quantile(0.10).item(),
            "p90": x.quantile(0.90).item()}


def _agg(rows: list[dict]) -> dict:
    return {m: _stat(torch.cat([r[m] for r in rows])) for m in ("rho", "cos", "ratio")}


def probe_rho(enc, texts, kind, scale, n_pairs, seed, pool_include_steer, batch_size) -> dict:
    g = torch.Generator().manual_seed(seed)
    N = len(texts)
    Z = torch.zeros(N, 1, enc.d)

    def E(s):
        return enc.encode(texts, steer=s, pool_include_steer=pool_include_steer,
                          batch_size=batch_size)

    e_plain = E(None)                              # M(I), no slot at all
    b1 = E(Z)                                      # M([I; 0])
    b2 = E(torch.cat([Z, Z], dim=1))               # M([I; 0; 0])
    b1n, b2n = _norm(b1), _norm(b2)

    slot_bias = (b1 - e_plain).norm(dim=-1) / e_plain.norm(dim=-1).clamp(min=1e-9)

    acc = {k: [] for k in ("sum_raw", "sum_nrm", "cat_raw", "cat_nrm")}
    dmag = []

    for _ in range(n_pairs):
        s1 = enc.sample_s(N, kind, scale, g).unsqueeze(1)      # (N,1,d)
        s2 = enc.sample_s(N, kind, scale, g).unsqueeze(1)

        # --- one slot: is s -> Delta affine within a single slot? ---------- #
        def D1(s):
            e = E(s)
            return e - b1, _norm(e) - b1n

        a1_r, a1_n = D1(s1)
        a2_r, a2_n = D1(s2)
        a12_r, a12_n = D1(s1 + s2)
        acc["sum_raw"].append(_rho(a12_r, a1_r, a2_r))
        acc["sum_nrm"].append(_rho(a12_n, a1_n, a2_n))

        # --- two slots: do the slots interact? ----------------------------- #
        # slot count held fixed at 2 throughout, unused slot zeroed, so an
        # encoder linear in each slot scores exactly 0.
        def D2(s):
            e = E(s)
            return e - b2, _norm(e) - b2n

        c1_r, c1_n = D2(torch.cat([s1, Z], dim=1))
        c2_r, c2_n = D2(torch.cat([Z, s2], dim=1))
        c12_r, c12_n = D2(torch.cat([s1, s2], dim=1))
        acc["cat_raw"].append(_rho(c12_r, c1_r, c2_r))
        acc["cat_nrm"].append(_rho(c12_n, c1_n, c2_n))

        dmag.append(a1_r.norm(dim=-1) / e_plain.norm(dim=-1).clamp(min=1e-9))

    return {k: _agg(v) for k, v in acc.items()} | {
        "delta_norm_rel_e0": _stat(torch.cat(dmag)),
        "slot_bias_rel_e0": _stat(slot_bias),
    }


def probe_rank(enc, texts, kind, scale, n_s, seed, pool_include_steer, batch_size) -> dict:
    """Rank of the reachable delta set {Delta(I, s)} for each fixed input I.

    Deltas are taken against M([I; 0]) so the constant slot offset does not
    contribute a spurious dominant direction.
    """
    g = torch.Generator().manual_seed(seed + 1)
    per_input, spectra = [], []

    for t in texts:
        b1 = enc.encode([t], steer=torch.zeros(1, 1, enc.d),
                        pool_include_steer=pool_include_steer)
        s = enc.sample_s(n_s, kind, scale, g).unsqueeze(1)
        e = enc.encode([t] * n_s, steer=s, pool_include_steer=pool_include_steer,
                       batch_size=batch_size)
        D = (e - b1).double()
        sv = torch.linalg.svdvals(D)
        en = sv ** 2
        cum = torch.cumsum(en, 0) / en.sum()
        per_input.append({
            "participation_ratio": (en.sum() ** 2 / (en ** 2).sum()).item(),
            "k90": int((cum < 0.90).sum().item()) + 1,
            "k95": int((cum < 0.95).sum().item()) + 1,
            "k99": int((cum < 0.99).sum().item()) + 1,
            "top1_energy": (en[0] / en.sum()).item(),
        })
        spectra.append(sv[:16].tolist())

    keys = list(per_input[0].keys())
    return {"n_s": n_s,
            "mean": {k: float(np.mean([p[k] for p in per_input])) for k in keys},
            "min": {k: float(np.min([p[k] for p in per_input])) for k in keys},
            "per_input": per_input,
            "top_singular_values": spectra}


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="intfloat/e5-large-v2")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    ap.add_argument("--prefix", default="query: ", help="e5 wants a query:/passage: prefix")
    ap.add_argument("--texts-file", default=None, help="one text per line; overrides defaults")
    ap.add_argument("--n-texts", type=int, default=32)
    ap.add_argument("--n-pairs", type=int, default=4, help="(s1,s2) draws per text")
    ap.add_argument("--n-rank-texts", type=int, default=4)
    ap.add_argument("--n-rank-s", type=int, default=128)
    ap.add_argument("--scales", type=float, nargs="+", default=[0.25, 0.5, 1.0, 2.0, 4.0],
                    help="|s| as a multiple of the mean real-token embedding norm")
    ap.add_argument("--dists", nargs="+", default=["gauss", "token"])
    ap.add_argument("--pool-include-steer", action="store_true",
                    help="include steer positions in the mean pool (leaky: s then "
                         "reaches the output without passing through the stack)")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quick", action="store_true", help="tiny smoke-test config")
    ap.add_argument("--skip-self-check", action="store_true")
    ap.add_argument("--out", default="results/wp0_nonlinearity.json")
    ap.add_argument("--log", default=None,
                    help="mirror stdout+stderr here (default: <out> with .log)")
    args = ap.parse_args()

    if args.quick:
        args.n_texts, args.n_pairs = 8, 1
        args.n_rank_texts, args.n_rank_s = 2, 32
        args.scales, args.dists = [0.5, 2.0], ["gauss"]

    # stderr shares the file so a crash or a transformers warning lands in the
    # log too -- an unattended sweep that dies should say why.
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
    print(f"  d={enc.d}   mean token-embedding norm={tnorm:.4f}")

    if args.texts_file:
        texts = [l.strip() for l in open(args.texts_file) if l.strip()]
    else:
        texts = DEFAULT_TEXTS
    texts = [args.prefix + t for t in texts[:args.n_texts]]
    rank_texts = texts[:args.n_rank_texts]
    print(f"  {len(texts)} texts   pool_include_steer={args.pool_include_steer}")

    results = {"config": vars(args) | {"d": enc.d, "token_norm": tnorm}, "probes": {}}

    if not args.skip_self_check:
        chk = enc.self_check(texts[:4])
        print("\nself-check (max abs err, want < 1e-4):")
        for k, v in chk.items():
            print(f"  [{'ok  ' if v < 1e-4 else 'FAIL'}] {k:26s} {v:.2e}")
        results["self_check"] = chk
        if max(chk.values()) > 1e-4:
            print("  -> injection path is not identity / not batch-invariant. "
                  "Everything below is unreliable; fix this first.")

    print("\n" + "=" * 96)
    print("pre-normalization deltas, slot-matched baselines (affine encoder => rho == 0)")
    print(f"{'dist':<7}{'|s|/tok':>9}{'rho_sum':>10}{'rho_cat':>10}{'cos_sum':>10}"
          f"{'cos_cat':>10}{'|D|/|e0|':>11}{'slotbias':>10}")
    print("=" * 96)

    for kind in args.dists:
        for a in args.scales:
            t0 = time.time()
            r = probe_rho(enc, texts, kind, a * tnorm, args.n_pairs, args.seed,
                          args.pool_include_steer, args.batch_size)
            results["probes"][f"rho/{kind}/{a}"] = r
            print(f"{kind:<7}{a:>9.2f}"
                  f"{r['sum_raw']['rho']['mean']:>10.4f}"
                  f"{r['cat_raw']['rho']['mean']:>10.4f}"
                  f"{r['sum_raw']['cos']['mean']:>10.4f}"
                  f"{r['cat_raw']['cos']['mean']:>10.4f}"
                  f"{r['delta_norm_rel_e0']['mean']:>11.4f}"
                  f"{r['slot_bias_rel_e0']['mean']:>10.4f}   [{time.time() - t0:.0f}s]")

    print("-" * 96)
    print("same draws, measured after L2 normalization (what retrieval sees; "
          "inflated by the projection itself)")
    for kind in args.dists:
        for a in args.scales:
            r = results["probes"][f"rho/{kind}/{a}"]
            print(f"{kind:<7}{a:>9.2f}"
                  f"{r['sum_nrm']['rho']['mean']:>10.4f}"
                  f"{r['cat_nrm']['rho']['mean']:>10.4f}"
                  f"{r['sum_nrm']['cos']['mean']:>10.4f}"
                  f"{r['cat_nrm']['cos']['mean']:>10.4f}")

    print(f"\nreachable delta-space rank per input ({args.n_rank_s} random s, "
          f"ceiling = {args.n_rank_s}):")
    for kind in args.dists:
        for a in args.scales:
            t0 = time.time()
            rk = probe_rank(enc, rank_texts, kind, a * tnorm, args.n_rank_s,
                            args.seed, args.pool_include_steer, args.batch_size)
            results["probes"][f"rank/{kind}/{a}"] = rk
            m = rk["mean"]
            print(f"  {kind:<7} |s|={a:>5.2f}x   PR={m['participation_ratio']:>7.2f}   "
                  f"k90={m['k90']:>5.1f}  k95={m['k95']:>5.1f}  k99={m['k99']:>5.1f}   "
                  f"top1_energy={m['top1_energy']:.3f}   [{time.time() - t0:.0f}s]")

    # ---- verdict --------------------------------------------------------- #
    cfg = [(k, a) for k in args.dists for a in args.scales]
    rho_best = max(results["probes"][f"rho/{k}/{a}"]["sum_raw"]["rho"]["mean"] for k, a in cfg)
    cat_best = max(results["probes"][f"rho/{k}/{a}"]["cat_raw"]["rho"]["mean"] for k, a in cfg)
    pr_best = max(results["probes"][f"rank/{k}/{a}"]["mean"]["participation_ratio"] for k, a in cfg)
    results["verdict"] = {"rho_sum_max": rho_best, "rho_cat_max": cat_best,
                          "participation_ratio_max": pr_best,
                          "go": bool(rho_best > 0.15 and pr_best > 8)}

    print("\n" + "=" * 96)
    print(f"GATE   max rho_sum = {rho_best:.4f}    max rho_cat = {cat_best:.4f}    "
          f"max PR = {pr_best:.2f}")
    if results["verdict"]["go"]:
        print("GO     single-slot map is nonlinear and the reachable set is high-rank: "
              "stages can add expressivity. Proceed to WP1.")
    else:
        print("NO-GO  deltas compose near-linearly and/or the reachable set is low-rank: "
              "K stages will collapse toward one.")
        print("       Which one failed matters. rho_cat >> rho_sum still supports the "
              "concatenative variant; a low PR at every scale is the more damaging "
              "result and is the negative-note pivot.")
    print("=" * 96)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nwrote {args.out}")
    print(f"finished {datetime.now():%Y-%m-%d %H:%M:%S} "
          f"({(time.time() - t_start) / 60:.1f} min total)")


if __name__ == "__main__":
    main()
