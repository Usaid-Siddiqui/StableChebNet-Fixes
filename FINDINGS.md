# Stable-ChebNet damping: findings

Investigation of the damping term in Stable-ChebNet (arXiv:2506.07624), branch `fejer-damping`.
Status: **experimental phase CLOSED** (2026-09-10). All claims in §1 rest on n=4 seeds.

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

**γ magnitude is decisive; the kernel is not.** Peptides, 16 layers, 4 seeds each:

| γ | arm | diverged | mean ± std (survivors) |
|---|---|---|---|
| 0.0 | all kernels (identical at γ=0) | **3/3** | — |
| 0.1 (paper-tuned) | dirichlet | **3/4** | — |
| 2.2 | **dirichlet** | **0/4** | **0.6330 ± 0.0099** |
| 2.2 | **fejer** | **0/4** | **0.6397 ± 0.0067** |
| 2.2 | uniform (flat) | **4/4** | — |

Raising γ rescues Dirichlet across every seed. The dirichlet–fejér difference (0.0067)
is **below the measured L16 noise floor (0.0118)** — statistically indistinguishable.
There is a real γ threshold between 0.1 and 2.2: with no damping at all (γ=0) depth-16
diverges, so damping *is* necessary — it is simply not the *kernel* that matters.
On ring 32 at matched γ, Dirichlet is **better** (solves at depth 24 vs Fejér's 28).

**The L16 noise floor is 0.0118.** The three γ=0 runs are bit-identical models, so their
spread (test_AP 0.3714–0.3832, std 0.0048) is pure run-to-run nondeterminism — matching
the L3 floor of 0.0122 measured the same way.

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
4. ~~Matched-γ results are n=1.~~ **RESOLVED** — replicated across 4 seeds; dirichlet@2.2 survives 4/4.
5. ~~No deep γ=0 control.~~ **RESOLVED** — γ=0 at L16 diverges 3/3, so damping is
   genuinely necessary at depth; the failure is not depth alone.
6. `summarize_depth.py` silently collapsed cells sharing (layers, kernel) but differing
   in γ/seed — **fixed** (now warns). Conclusions were drawn from collision-free files.
7. `summarize_seeds.py` reports survivor-only means grouped by kernel alone; Dirichlet's
   "mean 0.3723" is a single survivor.
8. **Inherited from the original code** (not ours): the multi-label Peptides task is
   trained with `CrossEntropyLoss` (log-softmax over non-exclusive labels) rather than
   BCE, and the LR scheduler is fed only the final validation *batch's* loss.
9. ~~The noise floor at 16 layers is unmeasured.~~ **RESOLVED** — 0.0118 at L16 (from the
   three bit-identical γ=0 runs), matching 0.0122 at L3. The kernel difference (0.0067)
   falls below it.

## 5. Closure runs — DONE

`experiments/run_closure.sh` ran Part A (dirichlet@2.2 seeds 1-3) and Part B (γ=0
control under all three kernel flags). Results merged into
`results/peptides_L16_matched_final.csv`. Outcome: dirichlet@2.2 survives 4/4 seeds and
is indistinguishable from fejér; γ=0 diverges 3/3. Nothing further is required to
support the claim in §6.

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

## 7. Reproducing

Setup (cluster):

```bash
conda create -n fejer python=3.11 -y && conda activate fejer
pip install torch --index-url https://download.pytorch.org/whl/cu121   # match cluster CUDA
pip install -r requirements.txt
python experiments/prefetch_data.py --root Peptides/Stable    # LRGB datasets
# GraphProp data ships in the repo: cd GraphProp/graph_prop_pred && tar xzf data.tar.gz
```

The `--damping_kernel {dirichlet,uniform,fejer}` flag is threaded through
`Peptides/Stable/EulerConv.py` (the layer), `model_euler*.py`, and the entry scripts;
`GraphProp/.../dgn_GraphProp.py` and `LongRange/run_ring.py` carry the same port.
`WANDB_MODE` defaults to `offline`; result CSVs are tracked in git, checkpoints are not.

| script | produces |
|---|---|
| `tools/check_damping.py --real` | Phase 0: sign profile, gain-vs-Jacobian gate, γ sweep |
| `tools/test_kernel_equiv.py` | guard: γ=0 must be bit-identical across all three arms |
| `experiments/run_phase1.sh` | shallow (3-layer) kernel×γ sweep — reproduces the paper |
| `experiments/run_depth.sh` | Peptides depth sweep 3→16 |
| `experiments/run_depth_seeds.sh` | L16 seed confirmation |
| `experiments/run_closure.sh` | matched-γ replication + γ=0 control (§5) |
| `experiments/run_ring.sh`, `run_ksweep.sh` | ring-transfer sweeps (see flaws 1–3) |
| `experiments/summarize_depth.py`, `summarize_seeds.py` | read the CSVs |

Note `summarize_seeds.py` groups by kernel alone — for files mixing γ values, group by
(kernel, γ) instead, or the γ=0 control rows will be pooled with the γ=2.2 ones.
