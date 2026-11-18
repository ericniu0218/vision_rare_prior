# rareprior — Small-feature recovery in image reconstruction

**Evidence status.** Development studies test whether small planted signals remain
detectable after reconstruction. Results support a spatially conditional linear
lensless trade-off and a raw-knee pipeline check. The core diffusion claim remains
untested on a sealed holdout. See [the evidence audit](../docs/pi_review.md),
[execution report](../docs/improvement_plan.md) and [current status](STATUS.md).

Code for the vision/imaging project planned in [`../projects/vision_imaging_plan.md`](../projects/vision_imaging_plan.md).
Literature review: [`../projects/vision_imaging_literature.md`](../projects/vision_imaging_literature.md).
Progress and results so far: [`STATUS.md`](STATUS.md).

## What it does
Paste small synthetic features into images at a fixed **measurement-domain detectability d′**, simulate a
measurement, and reconstruct with diffusion and non-diffusion methods. Observers
target a 5% false-alarm rate. Evaluation rates and the observer's inputs must be
reported alongside detection rates. The common runner covers blur and lensless
simulation. The MRI raw case uses a separate engineering script.

## Layout
```
rareprior/
  operators/   blur.py (matches official DPS), lensless.py (pad-conv-crop), mri.py (SENSE, masks)
  diffusion/   schedule.py (DDPM, respacing; matches official DPS), loader.py, guided/ (vendored OpenAI U-Net, MIT)
  solvers/     dps.py (own DPS), diffpir.py (own DiffPIR, any linear operator), cg.py, unet.py, registry.py
  features/    library.py (5 shape classes), insert.py (d'-matched amplitudes, placement)
  metrics/     observer.py (matched filter, per-solver thresholds, survival), image_quality.py
  detect/      scores.py (residual, semantic, variance)
  optics/      psf.py (free-form + phase-mask PSFs; MSE, Gaussian-MI, Hotelling-detectability objectives)
  stats/       bootstrap.py (clustered paired bootstrap, Holm), gee.py (H2)
  data/        splits.py (dev guards), loaders.py (pinned FFHQ and DiffuserCam mirrors)
scripts/       verify_env.py, run_survival.py, analyze_survival.py, train_unet.py, design_psf.py
configs/       experiment YAMLs (inherit base.yaml; override with key=value)
tests/         adjoints, official-DPS equivalence, Gaussian toy, observer vs analytic, planted-effect pipeline
```

## Quick start (CPU)
```bash
pip install -e ".[dev,mri,data,eval]"
git clone https://github.com/DPS2022/diffusion-posterior-sampling third_party/dps
git -C third_party/dps checkout effbde7325b22ce8dc3e2c06c160c021e743a12d
pytest -q
ruff check rareprior scripts tests
python scripts/run_survival.py configs/experiment/smoke_synthetic.yaml
python scripts/analyze_survival.py --runs artifacts/survival/smoke_synthetic/* --out artifacts/smoke_analysis
```
The draft guard rejects held-out runs before downloading inputs. GPU development runs: see [`KAGGLE.md`](KAGGLE.md).

## Notes
- The official DPS repository has **no license file**, so none of its code is copied here. It is cloned
  into `third_party/dps` (git-ignored) only so the tests can compare against it.
- `rareprior/diffusion/guided/` is vendored from `openai/guided-diffusion` (MIT, license included).
