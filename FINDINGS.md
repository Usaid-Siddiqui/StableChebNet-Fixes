# Stable-ChebNet damping: findings

Investigation of the damping term in Stable-ChebNet (arXiv:2506.07624), branch `fejer-damping`.
Status: **experimental phase closed** pending one replication run (§5).

The layer is `X^{l+1} = X^l + ε·Σ_{k<K} T_k(L̃)·X·(W_k − W_kᵀ − γI)`. We added a
`damping_kernel` flag controlling how the fixed scalar γ is distributed over the K
Chebyshev orders: `dirichlet` (γ on every order — the original), `uniform` (γ pulled
out, applied once to the signal), `fejer` (γ·(1−k/K) on order k).

---

## 1. Established (high confidence)

**The diagnosis is correct.** The applied damping is `s(λ̃) = Σ_{k<K} w_k T_k(λ̃)` — a
Dirichlet kernel for the original weights. It is sign-changing for **K ≥ 3**, with the
negative fraction → ¼ as K grows (measured 0.2435 on real Peptides graphs at K=10,
min s = −1.58; exactly 0 at K=2). The exact per-mode gain

    ‖G_i‖² = (1 − εγ·s_i)² + ε²·σ_max(B_i)²

matches the true autograd Jacobian to **<1e-14** (independently reproduced at 9e-15).
Because any γ>0 amplifies the negative-s modes, worst-case-gain tuning collapses γ→0 —
matching the paper's own grid (Table 7: γ*=0.00 SSSP, 0.00 Ecc, 0.01 Diameter).

**γ magnitude is decisive; the kernel is not.** Peptides, 16 layers, matched γ=2.2, seed 0:

| arm | test AP | diverged |
|---|---|---|
| dirichlet @ γ=0.1 (paper-like) | 0.3794 | **yes** (3/4 seeds) |
| **dirichlet @ γ=2.2** | **0.6487** | no |
| **fejer @ γ=2.2** | **0.6486** | no |
| uniform @ γ=2.2 | 0.3715 | **yes** (ep. 39) |

Raising γ rescues Dirichlet completely, and it then matches Fejér to four decimals.
On ring 32 at matched γ, Dirichlet is **better** (solves at depth 24 vs Fejér's 28).

**Flat damping is harmful.** `uniform` is the only arm that diverges at adequate γ, on
Peptides *and* fails on every ring. Its damping has zero spread across frequencies
(std(s)=0 vs 2.12 dirichlet, 1.19 fejér), while all three share mean(s)=1.

**The linear-norm diagnostic does not predict success.** `uniform` has the *lowest*
initial ‖J‖ (2.76) and fails; `dirichlet` has the *highest* (9.30) and succeeds.
Measured epoch-0 norms reproduce theory exactly (9.296/2.764/5.229 vs 9.31/2.82/5.23).

## 2. Falsified — including our own hypotheses

- **"Fejér is the fix / uniquely enables depth."** Falsified by the matched-γ controls.
  The original headline compared `dirichlet@0.1` against `fejer@2.2` — kernel and γ
  varied together. At matched γ the advantage disappears or reverses.
- **"Pull γ out of the sum" (Amendment 1's proposed fix).** Falsified — `uniform` is the
  *worst* arm, not the best.
- **Phase 0's worst-case-gain-at-initialization metric does not predict trained
  behaviour.** It asks whether contraction is *achievable*; trained models run at
  ‖J‖≈25 and never contract. The math was right; its relevance was not.

## 3. Mechanism (low–moderate confidence)

For K=2 the damping term contributes a **fixed neighbour-transport path**:

    (1 − εγ)·X  +  εγ·w₁·P·X       w₁ = 1 (dirichlet), ½ (fejér), 0 (uniform)

Dirichlet supplies twice Fejér's fixed transport and Uniform supplies none — which
predicts the observed ring ordering (solves at depth 24 / 28 / never) quantitatively.
This is a plausible mechanism, not a demonstrated one: it is untested against
saturation- and optimization-based alternatives, and requires activation/gradient
measurements to confirm.

## 4. Known flaws and confounds

1. **The ring benchmark used K=2, where the Dirichlet kernel is nonnegative** (`s=1+λ̃≥0`).
   It therefore cannot test the sign-change diagnosis at all. Design error.
2. **The ring has only 5 unique labelled inputs** per ring size (5 classes, fixed graph);
   train/val/test resample the same patterns. It measures transport and optimization,
   **not generalization**.
3. **ε=2.5 was chosen to separate the arms.** Rankings change with ε (at ε=0.5 `uniform`
   wins; at ε=2.5 it loses).
4. **Matched-γ results are n=1** (seed 0). Not yet replicated.
5. **No deep γ=0 control.** At γ=0 all three kernels are bit-identical — the missing
   control that separates "damping" from "depth".
6. `summarize_depth.py` silently collapsed cells sharing (layers, kernel) but differing
   in γ/seed — **fixed** (now warns). Conclusions were drawn from collision-free files.
7. `summarize_seeds.py` reports survivor-only means grouped by kernel alone; Dirichlet's
   "mean 0.3723" is a single survivor.
8. **Inherited from the original code** (not ours): the multi-label Peptides task is
   trained with `CrossEntropyLoss` (log-softmax over non-exclusive labels) rather than
   BCE, and the LR scheduler is fed only the final validation *batch's* loss.
9. **Run-to-run nondeterminism at 3 layers is 0.0122** (measured from six γ=0 runs, which
   are provably the same model). The floor at 16 layers is unmeasured.

## 5. Remaining for closure — one run

Replicate the matched-γ control across the seeds we already have, and add the missing
deep γ=0 control:

```bash
cd Peptides/Stable
for s in 1 2 3; do for k in dirichlet fejer; do
  python ChebStable_peptide.py --num_layers 16 --damping_kernel $k \
    --dissipative_force 2.2 --seed $s \
    --results_csv ../../results/peptides_L16_matched_seeds.csv \
    --run_name "L16_${k}_g2.2_s${s}"; done; done
# deep gamma=0 control (all kernels identical here by construction)
for s in 0 1; do
  python ChebStable_peptide.py --num_layers 16 --damping_kernel dirichlet \
    --dissipative_force 0.0 --seed $s \
    --results_csv ../../results/peptides_L16_matched_seeds.csv \
    --run_name "L16_g0_s${s}"; done
```

If Dirichlet@2.2 survives across seeds, §1 is settled and the experiment closes.

## 6. What this supports as a claim

> Stable-ChebNet's damping coefficient is untunable by construction: the sign-changing
> Dirichlet kernel makes the worst-case stability signal monotonically worse in γ, so
> tuning collapses it to γ≈0 — precisely the regime that cannot be trained at depth. The
> flaw is invisible at the shallow depths the paper reports and silently caps usable
> depth. The obvious repair — pulling γ out to a flat profile — is actively harmful.
> Reshaping the kernel (Fejér) is **not** required; adequate γ suffices.

This is a **correctness and tuning** result, not a benchmark result. Every benchmark the
paper leads on is shallow or small-diameter, where all damping choices tie (Phase 1: all
12 kernel×γ cells within 0.68–0.70, reproducing the paper's 70.32±0.26). We do not beat
it there, and the fix should not be presented as if we do.
