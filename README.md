# embed-boost — WP0

Nonlinearity probe for input-space steering of a frozen text embedding model.
This is the GO/NO-GO gate for the stagewise steering project: it answers only
whether the mechanism *could* work, not whether it helps retrieval.

## Run

```bash
pip install -r requirements.txt
python probe_nonlinearity.py --quick                              # ~2 min, sanity
python probe_nonlinearity.py --model intfloat/e5-large-v2 --device cpu
python probe_nonlinearity.py --device cuda --n-texts 64 --n-pairs 8 --n-rank-s 256
```

Results land in `results/wp0_nonlinearity.json`.

## What it measures

Steering vector `s ∈ R^d` is appended in the encoder's embedding space, right
after the last real token and before padding. `Δ(I, s) = M([emb(I); s]) − M(I)`.

| metric | question | collapse if |
|---|---|---|
| `rho_sum` | is one slot affine in `s`? | `≈ 0` → additive variant collapses to one stage |
| `rho_cat` | do two slots interact? | `≈ 0` → concatenative variant collapses to one stage |
| `PR`, `k90/95/99` | rank of the reachable set `{Δ(I,s)}` | low → few useful stages exist regardless |
| `slot_bias` | how much does an *empty* slot move the embedding? | large → the intervention is mostly a constant offset |

Gate: `rho_sum > 0.15` **and** `PR > 8`. The script prints the verdict.

## Three things that will bite you if changed

**Baselines are slot-matched.** `Δ` is not zero at `s = 0` — appending an
extra attended position moves the embedding on its own. If deltas were taken
against `M(I)`, that constant would land in the residual and a perfectly
*affine* encoder would score `rho > 0`, faking a GO. So the one-slot test
baselines on `M([I; 0])` and the two-slot test on `M([I; 0; 0])` with the
unused slot zeroed rather than removed. Under any affine model both `rho`s are
then exactly 0. Same reason `probe_rank` baselines on `M([I; 0])`: otherwise
the shared offset shows up as a spurious dominant singular direction and makes
the reachable set look rank-1.

**`rho` is reported pre-normalization first.** L2 projection to the sphere is
itself nonlinear, so the post-normalization `rho` is inflated even for an
affine encoder. Gate on the pre-norm number; the normalized rows are there
because that's the geometry retrieval actually operates in.

**Steering goes before the padding, not after it.** Right-padding then
appending would place `s` at position `max_len_in_batch`, so the same
`(text, s)` pair would give different deltas in different batches. `self_check`
asserts batch invariance with steering on — if that line says FAIL, no other
number in the output means anything.

## Pooling choice

By default the steering positions are **excluded** from the mean pool, so `s`
can only influence the output through attention from the real tokens — the
manifold-constrained path the project is actually about. `--pool-include-steer`
lets `s` write to the output directly; expect much lower `rho` and a mostly
linear map, since that's a near pass-through. Worth running once as a control.

## Interpreting a NO-GO

Which sub-metric failed decides the pivot:

- `rho_cat >> rho_sum` — additive boosting is dead, concatenative is alive.
  Reframe WP2 around the concatenative variant.
- both `rho` low but `PR` high — the map is affine but the reachable set is
  rich: a single stage with a stronger `θ` is the paper, not boosting.
- `PR` low at every scale — the damaging case. The encoder barely moves in
  response to appended vectors; write the geometry note and skip to WP3.

## Not covered here

Decoder backbones (`gte-Qwen2`, `Qwen3-Embedding`) need last-token pooling and
left padding, which changes where `s` must be inserted. Add after the encoder
result is in. Nothing here is differentiable-by-design yet — WP1's oracle needs
gradients through `inputs_embeds`, which the current `@torch.no_grad()`
wrappers block deliberately.
