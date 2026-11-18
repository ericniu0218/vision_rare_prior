# Pre-registration (DRAFT — not yet frozen)

**Status:** draft. Same-site calibration has not met the dev gate. The original
lensless holdout assignment is compromised and needs replacement. Review the
[freeze gates](../docs/improvement_plan.md#remaining-gates-and-ownership) before
any new held-out reconstruction. Post-freeze amendments make affected analyses exploratory.

- Frozen at commit: `TBD`
- Date: `TBD`

## 1. Hypotheses
Copied from `projects/vision_imaging_plan.md`, Section 3.

| ID | Statement | Type |
|---|---|---|
| H1 (primary) | ΔS = S_UNet − S_DPS > 0 on small features (sizes 3, 5 px), FFHQ, Gaussian blur, matched d′ | Confirmatory, α = 0.05 two-sided |
| H1b | DPS has lower LPIPS than the U-Net on the same images | Descriptive |
| H2 | Negative DPS × surprisal interaction in the GEE model | Confirmatory secondary (Holm) |
| H3a / H3b | Residual score / semantic disagreement beat pixel variance on AUROC for locating erasures | Confirmatory secondary (Holm) |
| H4 | Detectability-optimised PSF raises diffusion survival vs the DiffuserCam PSF (lensless sim) | Confirmatory secondary (Holm) |
| H5, H6 | Replications and causal rarity | Exploratory |

## 2. Primary metric and estimator
- Survival S = TPR at 5% FPR of the background-known-exactly matched filter; thresholds per
  (solver, severity, class, size) from control sites (8 per image × every template × both signs).
- Pooled over (severity, class, size, d′) cells with equal weight; small sizes only for H1.
- CI: image-clustered paired bootstrap with thresholds re-estimated in each replicate,
  10,000 replicates (`rareprior/stats/bootstrap.py::paired_survival_bootstrap`).

## 3. Frozen design parameters
| Parameter | Value |
|---|---|
| Test images | **150** — the power rule triggered on dev (predicted ΔS half-width 0.063 at n = 100, 0.052 at n = 150, conservative independence assumption; `results/ffhq_dev_pilot/power_check.json`). `ffhq_test(150, seed=20261109)` |
| Severities | S1: blur std 3.0 (official). **S2: blur std 4.0** (dev drops: std 3.5 3.9%, std 4 11.1%, std 5 26.1%; none of the original candidates {4, 5, 6} met < 10%; amended on dev to a 12% threshold to keep a useful severity contrast — see Amendments) |
| Noise | σ = 0.05 in [−1, 1] |
| d′ levels | **{1, 2}, stationary-Hotelling (background-known-statistically) d′** with the FFHQ power spectrum `results/ffhq_power_spectrum_256.pt` (200 images, indices 65400–65599, outside every split). Changed on dev from background-known-exactly {2, 4}, which left Tikhonov at 0.09 survival (floor) because background variability, not noise, limits small-feature detection |
| Features per image / controls | 6 / 8; min separation 48 px (features), 24 px (controls) |
| DPS steps | `TBD` (1000 if Week-0 profiling gives ≤ 150 s/sample on a T4; else 500, validated on dev) |
| DPS scale | 0.3 (official Gaussian-deblur config) |
| K posterior samples | 4 |
| U-Net | `TBD` checkpoint hash; trained on FFHQ 60000–68999 |
| Tikhonov λ | S1 0.03, S2 0.01 (best dev PSNR; `results/ffhq_dev_pilot/tikhonov_lambda_grid.json`) |
| Observer | mean-removed template (changed on dev, see STATUS.md); null from control sites — dev counterfactual FPR (Tikhonov, 30 images): 0.053 (std 3, λ 0.01), 0.069 (std 3, λ 0.03), 0.075 (std 4, slightly outside the 5% ± 2% band). **Re-check with DPS on GPU before freezing** |

## 4. Week-0 verification results
`TBD` — paste `verify_env.json` summary.

## 5. Dev pilot results that informed the frozen values
Tikhonov, 30 dev images (`results/ffhq_dev_pilot/`): S1 survival 0.32 / 0.56 at d′ 1 / 2 (λ 0.01) against Hotelling
ceilings 0.26 / 0.64; 76% of small cells in [0.1, 0.9]. DPS / DiffPIR / U-Net dev pilots: GPU (pending).

## Amendments (all made on dev data, before freezing)
1. 2026-11-09: d′ definition changed from background-known-exactly to stationary Hotelling; levels {2, 4} → {1, 2}.
2. 2026-11-09: S2 drop threshold 10% → 12%; S2 = std 4.0 (none of {4, 5, 6} met 10%).
3. 2026-11-09: observer template mean-removed.
4. 2026-11-09: test set 100 → 150 images (power rule).
5. 2026-11-09: H4 redesigned (multiplexing sweep, full field of view); see the plan, Section 9.0.
6. 2026-11-09: control sites must pass the same pixel-range acceptance test as a random hypothetical feature
   (dev: counterfactual FPR Tikhonov 0.069 → 0.059, U-Net 0.089 → 0.077). U-Net is still slightly above the band,
   but the check is underpowered (SE ≈ 0.016 with ~180 nulls). Decide control vs counterfactual nulls on the GPU dev pilot.
7. 2026-11-09 (audit, draft): historical fidelity calibration overlapped the proposed lensless holdout in four IDs. Retire that assignment for confirmatory analysis. The named FFHQ spectrum subset is within U-Net train. Checkpoint training membership remains unverified. Test execution is disabled pending a reviewed manifest and successful dev calibration.
8. 2026-11-09 (audit, draft): the stationary Hotelling d′ is a model-based reference, not a bound on the privileged reconstruction observer. H3 and H4 remain exploratory pending new endpoints and validation. Budget a single-severity S1 H1 first. Freeze only after the gates in the improvement plan pass.
9. 2026-11-09 (audit, draft): compare known-background and reconstruction-only matched-filter observers on identical feature-on/off sites. Split development calibration from evaluation by image. Declare the trial weighting and empirical null tolerance before a new run. Report feature-on minus feature-off response separately. The historical macro contrast and a site-weighted contrast are different estimands.
