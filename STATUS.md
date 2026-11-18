# Status: what has been run, and what it showed

**Last updated:** 2026-11-09. The audit added CPU and Apple MPS development runs.

> **Current evidence.** The preregistration is still a draft. The historical lensless
> fidelity audit exposed four scenes in the proposed lensless holdout. New runs used
> development data. The historical bit-exact DPS result applies to spatial blur on
> a narrow check. Current FFT trajectories fail the five-step tolerance. See the
> [audit correction](#audit-correction-2026-11-09) and [final audit snapshot](#final-audit-snapshot).

## Historical status before independent audit

## Week-0 checklist (plan Section 20)
| Item | Result |
|---|---|
| FFHQ checkpoint from HF (`deepinv/diffunet`) | ✅ SHA-256 `81d53574…237faa`, strict load of all weights |
| Official DPS runs on modern torch | ✅ torch 2.14 / NumPy 2.5 on CPU (core sampling loop) |
| Own DPS = official DPS | ✅ **bit-exact** on the real checkpoint (3 steps, max diff 0.0; identical network outputs and blur operator); tiny-U-Net 20-step trajectory < 1e-4 (unit test) |
| DiffuserCam HF test split + PSF | ✅ 999 pairs at 270×480; captures use ~6 bits (max ≈ 68); PSF 1080×1920×3 = 4× measurement size |
| SigPy | ✅ 0.1.27 on Python 3.13 / NumPy 2.5: poisson, ESPIRiT, Sense, L1-wavelet all run; our SENSE = SigPy's to 1e-8 |
| FFHQ mirror indices | ✅ `merkol/ffhq-256` keeps original filenames; loader matches on them |
| ImageNet checkpoint identity | ⏳ not downloaded yet (2.2 GB; do it on Kaggle with `verify_env.py --full`) |
| mridata.org | ❌ blocked from this container (DNS); check from Kaggle or a laptop |
| Kaggle quota, T4 speed | ⏳ needs Kaggle (CPU here: 2.6 s per DPS step at 256 px on 4 threads) |

## Tests
`pytest`: **42 passed** (including the slow CI-coverage test), 15 minutes on one CPU core. CI runs on CPU (`.github/workflows/vision_ci.yml`).
Included: adjoint tests for every operator, official-DPS equivalence, a Gaussian toy with exact score,
insertion hits target d′ to 1e-4, the observer matches the analytic survival of a linear reconstruction,
and the planted-effect pipeline test (recovers planted survival, 5% FPR, CI coverage, null difference).

## Findings from executing the plan
1. **H4 as planned was degenerate** (`results/psf_design_128/`). With equal light, read noise and a scene no
   larger than the sensor, MSE, Gaussian-MI and Hotelling-detectability objectives are all maximised by a
   perfect lens. The MI optimiser converged to a single-pixel PSF, which also beats the "detectability-optimised"
   PSF on its own objective. H4 is now redesigned (plan Section 9.0).
2. **Multiplexing sweep (redesigned H4, linear reconstruction, 20 dev scenes, exploratory).**
   Small features (3–5 px), fixed amplitudes 0.2 and 0.4, Tikhonov with λ tuned per camera for PSNR.

   *Full field of view* (scene 2× the sensor; `results/multiplexing_full_fov/`):
   | Camera | Participation ratio | PSNR (dB) | Small-feature survival | inside / outside footprint |
   |---|---|---|---|---|
   | lens (k = 1) | 0.0015 | 9.30 | 0.31 | 0.96 / 0.02 |
   | k = 4 | 0.006 | 9.83 | 0.19 | 0.42 / 0.09 |
   | k = 16 | 0.023 | 10.42 | 0.15 | 0.38 / 0.06 |
   | k = 64 | 0.080 | 10.96 | 0.10 | 0.25 / 0.04 |
   | k = 256 | 0.259 | 11.39 | 0.10 | 0.17 / 0.07 |
   | DiffuserCam (real PSF) | 0.455 | 11.85 | 0.06 | 0.17 / 0.02 |

   Lens − DiffuserCam survival: +0.34 [0.17, 0.40]; k = 4 − DiffuserCam: +0.15 [0.03, 0.20].
   **PSNR rises with multiplexing while small-feature survival falls.** A camera designed for PSNR would be the
   wrong camera for rare features. Caveat: linear reconstruction only, 20 images, dev data, simulation (see 4).

   *Sensor-sized field of view* (`results/multiplexing_sensor_fov/`): the lens dominates as predicted
   (PSNR 34.0 dB, survival 1.00); survival falls monotonically to 0.14 for DiffuserCam.
3. **Observer revised.** The first observer used the raw template. Under the lens camera, smooth background
   errors outside the footprint dominated the null distribution, giving 0% survival even for perfectly imaged
   features. Templates are now mean-removed, and sweep thresholds are per region. The plan is updated.
4. **Lensless forward model vs real captures** (`results/lensless_fidelity/`, 20 real test captures):
   correlation 0.98 (wrong-scene control 0.79); high-pass correlation 0.32 (control 0.04). The simulator gets
   the coarse caustic right but only part of the fine detail, which is where small features live. This supports
   the plan's hybrid real-data test and calls for caveats on simulation-only lensless claims.
5. **Respaced DDPM under-disperses** (Gaussian toy, exact score): std 0.25 / 0.27 / 0.29 / 0.30 at
   50 / 100 / 250 / 1000 steps (truth 0.30). Reducing DPS steps changes sample diversity, which matters for the
   posterior-disagreement detector (H3).
6. **Official DPS quirks:** the residual norm couples samples within a batch; the model loader silently falls back
   to random weights on failure; the repository has no license file.

7. **MRI pipeline demo on a phantom** (`results/mri_phantom/`; mridata.org is blocked here). SigPy birdcage
   coils (8), Poisson-disc masks, lesion-like features inserted with the background phase, 12 phantom
   variants. Small-feature survival, L1-wavelet / Tikhonov / zero-filled: R2 1.00 / 1.00 / 0.75, R4 0.96 / 0.98 / 0.42,
   R6 0.90 / 0.81 / 0.25, R8 0.73 / 0.56 / 0.13. L1-wavelet PSNR 33.1 → 27.9 dB. **Pipeline check only:** Shepp-Logan
   is piecewise constant, which flatters L1-wavelet. While building it I found and fixed a bug: noise was being
   added to unsampled k-space, which made SigPy treat the data as fully sampled.
8. **Engineering:** the blur operator now uses FFT convolution (identical to conv2d to 1e-10, 10× faster on CPU),
   which matters for DiffPIR's CG data step.

9. **The matched-d′ premise needed fixing (dev pilot).** With background-known-exactly d′ ∈ {2, 4}, Tikhonov kept only
   9% of small features. On FFHQ, the unknown background texture at the feature's scale is a 6–12× larger
   "noise" than σ = 0.05 (3-px blob at blur std 3: BKE d′ 18.5 per unit amplitude vs Hotelling d′ 3.0). Features are
   now matched on the **stationary-Hotelling d′**. After the change, Tikhonov's survival (0.32 / 0.56 at d′ 1 / 2)
   tracks the Hotelling ceiling (0.26 / 0.64), as it should for a linear method. Severity calibration: drop rates
   std 3.5 / 4 / 5 = 3.9% / 11.1% / 26.1%, so **S2 = std 4** (amendment). Counterfactual FPR 0.053–0.075.
10. **PSNR-tuning erases features even within one linear method:** Tikhonov λ 0.01 → 0.03 gains +0.05 dB but loses
    0.10 small-feature survival (0.43 → 0.34).
11. **Power check:** predicted ΔS half-width 0.063 at n = 100 → the rule triggers → **n = 150** test images.

12. **Dev pilot, U-Net vs Tikhonov** (30 dev images, S1, Hotelling-matched features; the U-Net is a small CPU-trained
    1.9M-parameter model, 2,000 steps): U-Net PSNR 28.0 vs 26.1 dB, small-feature survival 0.47 vs 0.27,
    **U-Net − Tikhonov = +0.19 [0.08, 0.35]**. A learned MSE network keeps *more* small features than the linear
    method, so H1's comparator is a strong baseline (`results/ffhq_dev_pilot/unet_vs_tikhonov_matched_controls/`).
13. **Control-site location bias found and fixed.** Feature sites must pass a pixel-range check, which control sites
    did not, so the two came from different location distributions (U-Net counterfactual FPR 0.089). Controls now pass
    the same check: 0.077 for the U-Net, 0.059 for Tikhonov (SE ≈ 0.016 at this n).
14. **DiffPIR dev pilot** (4 images, K = 2, CPU): pipeline and detector run end to end; PSNR 26.3 dB; survival point
    estimates are not below Tikhonov's, but n = 4 is uninformative (CI ±0.3). Detector: only 8 erased sites, AUROC residual
    0.61 vs variance 0.48, CIs span zero. These are pipeline checks only.
15. **Figure:** `results/multiplexing_survival_vs_psnr.png`.

## FFHQ dev pilot (CPU, in progress)
Done on CPU: Tikhonov (30 images), U-Net (30), DiffPIR (4, K = 2), DPS (official config: 1000 steps, scale 0.3, K = 2;
2 images, about 1.8 h per image).

**DPS pilot, 2 dev images, 12 features shared by every method** (identical sites and noise;
`results/ffhq_dev_pilot/dps_pilot_2images.json`). Median amplitude-recovery ratio (threshold-free; 1 = fully kept):

| Method | All sizes | Small (3–5 px) | PSNR (dB) |
|---|---|---|---|
| DPS (own, official config) | 0.14 | 0.09 | 26.2 |
| U-Net (small, CPU-trained) | 0.36 | 0.28 | 29.8 |
| Tikhonov | 0.36 | 0.28 | 27.5 |
| DiffPIR | 0.44 | 0.39 | 26.1 |

DPS kept less than the U-Net on **12 of 12** features (mean difference −0.27). **This is not evidence yet:** 2 images,
and the features within an image are not independent. Two things make it worth the GPU run:
1. It points the same way as H1.
2. **DiffPIR, also a diffusion method, kept the most.** So the erasure may come from DPS's gradient-based measurement
   guidance (approximate, and known to be biased: Jensen gap, Feynman–Kac analysis) rather than from diffusion priors
   as such. DiffPIR enforces the measurement with an exact proximal step. The GPU analysis should report DPS vs DiffPIR
   as a prominent exploratory contrast (both run on the same grid already) and avoid claiming "diffusion priors erase"
   if only DPS does.

## Next steps (GPU, Kaggle)
1. `python scripts/verify_env.py --full` (records T4 speed; decides 1000 vs 500 DPS steps).
2. Train the U-Net (`scripts/train_unet.py`), run the dev pilot for DPS / U-Net / Tikhonov / DiffPIR, calibrate S2, run
   the power check, then **freeze `preregistration.md`**.
3. Week 3: the H1 test grid (see `KAGGLE.md`).
4. Redesigned H4 with DiffPIR + ImageNet prior: `scripts/multiplexing_sweep.py` extended to diffusion (GPU).

## Audit correction (2026-11-09)

The statements above describe historical artifacts. The following corrections supersede their interpretation.

- The historical lensless fidelity script exposed four scene IDs from the proposed lensless holdout, counting both target and wrong-scene controls. That holdout needs replacement. Current entry points use development IDs only.
- The FFHQ spectrum indices 65400–65599 are inside the U-Net training pool. They do not overlap the local FFHQ dev/holdout IDs. The checkpoint's own training membership is unverified.
- In the saved full-FOV optics data, lens − DiffuserCam is +.244 by site weighting and +.344 [.174, .399] by the historical macro estimator. Inside the footprint it is 23/24 vs 4/24. Outside it is 1/54 each. These are 20 development scenes and a single layout per camera.
- The CPU rerun of five real-checkpoint DPS steps matched network output but differed by up to .00249 on the cumulative trajectory with FFT blur. The stated five-step tolerance failed. The three saved bit-exact steps are historical.
- Phantom PSNR reproduced to rounding, but R8 detections were 38/48, 35/48 and 10/48 on current code, versus 35/48, 27/48 and 6/48 in the saved rows (L1-wavelet, CG, adjoint). The original lensless PSNR also reproduced, while site survival changed from .308/.064 to .269/.077 for lens/DiffuserCam. These are observer-sensitive dev diagnostics.
- The old FFHQ ratio is a known-background projection proxy, not a direct measurement of causal erasure. The stationary Hotelling curve is a Gaussian-model reference. No confirmatory diffusion result or clinically validated detector exists.

See [audit](../docs/pi_review.md) and [execution plan](../docs/improvement_plan.md).

### Run integrity

The editable package now installs. The run manifest records selected image IDs, code revision, backend and input hashes. Resuming checks for the full per-image artifact set and publishes the parquet last. Analysis checks arm, severity, operator settings, image/site pairing and completion. Same-site feature-off runs now log both privileged and reconstruction-only observer scores. Pixel-pair distance and LPIPS are separate detector channels. With 15 fit and 15 evaluation dev faces, the preliminary 200-step U-Net changes rank relative to CG when clean background access is removed. This comparison is an observer sensitivity check. A 50-step real-checkpoint DPS smoke and a 50-NFE DiffPIR smoke each used two dev faces. Neither is a validated reference setting.

### Additional development evidence

- Raw knee engineering case: one public legacy Stanford acquisition, three hybrid slices, two masks, 24×24 ACS. Explicit-mask SENSE adjoint agrees with SigPy to relative error below (10^{-7}). This validates an acquisition and software path. The acquisition is the independent unit.
- Lensless development: the pinned capture/PSF model gives mean high-pass Pearson .293 vs .046 on 30 dev pairs and wrong-scene controls. In the 20-scene repeat with separate calibration, the lens advantage remains concentrated inside the footprint, and several lenslet layouts miss the FPR band. A general monotonicity claim fails across layouts.
- Exact-posterior toy: 4,096 synthetic chains per setting with fixed likelihood and three prior masses. At 10% prior mass the exact posterior mass is .355 and 100-step DPS yields .105, .367, .671 at guidance scales 0, .3, 1. This illustrates mechanisms under a scalar model.
- Proximal check: the default FFHQ DiffPIR is a generalized variant with a Gaussian rather than measurement-based start and reflect blur. Current circular Gaussian blur uses an FFT proximal solve. The FFT and CG check concerns the data subproblem only.

### Final audit snapshot

The same-site calibration gate failed. With the seeded 200-step U-Net and 15/15 FFHQ development calibration/evaluation split, U-Net and CG detected 30/61 and 21/61 small sites with clean background access, and 17/61 and 23/61 without it. A reconstruction-free template observer at the same sites detected 7/61. In lensless simulation, same-site null FPR reached .189/.265 for the centered lens and DiffuserCam. The control-site layout results remain a geometry-sensitive development diagnostic.

The MPS dtype fix made a 50-step DPS dev smoke and a four-image, 25-NFE DiffPIR detector smoke runnable. The latter had 55 candidate sites and 8 positives. Residual AUROC .734 versus variance .484 had paired difference +.250 [−.038, .495]. This evaluates prespecified sites and does not validate a deployable detector. Raw MRI, optics, the scalar prior toy, the color control and COCO development artifacts are in [audit_dev](results/audit_dev/). The protocol stays draft.

The learned scalar control adds three training seeds per prior mass, nine small
denoisers total. At prior .10 and guidance .3, presence ranged .218–.366 across
the three runs, compared with .355 exact and .367 for analytic-score DPS.
See [learned toy results](results/audit_dev/prior_toy_learned.json).
