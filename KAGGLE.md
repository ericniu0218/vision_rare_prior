# Running on Kaggle (free T4)

Everything heavy runs on Kaggle; this folder's code is the same on CPU and GPU.

## One-time setup
1. Kaggle → *Create* → *Notebook*. Settings: **Accelerator: GPU T4 x2**, **Internet: on**, Persistence: files.
2. First cell:
   ```bash
   !git clone https://github.com/<your-user>/bair-plan.git
   %cd bair-plan/vision_project
   !pip install -q sigpy lpips tifffile
   !git clone -q https://github.com/DPS2022/diffusion-posterior-sampling third_party/dps
   !git -C third_party/dps checkout -q effbde7325b22ce8dc3e2c06c160c021e743a12d
   ```
3. Week-0 check (about 10 minutes; downloads the 374 MB FFHQ checkpoint from Hugging Face):
   ```bash
   !python scripts/verify_env.py --full
   ```
   Paste the printed table into `preregistration.md` (the DPS speed line decides 1000 vs 500 steps).
4. Tests, including the GPU equivalence check:
   ```bash
   !pytest -q -m "not slow"
   ```

## Week 1–3 commands (primary arm)
```bash
# DPS dev pilot, two GPUs in parallel (shards 0 and 1)
!python scripts/run_survival.py configs/experiment/ffhq_blur_S1_dps.yaml split=dev shard=0 num_shards=2 device=cuda:0 &
!python scripts/run_survival.py configs/experiment/ffhq_blur_S1_dps.yaml split=dev shard=1 num_shards=2 device=cuda:1
# baselines
!python scripts/train_unet.py configs/experiment/ffhq_blur_S1_unet.yaml
!python scripts/run_survival.py configs/experiment/ffhq_blur_S1_unet.yaml split=dev solver.weights=<path printed above>
!python scripts/run_survival.py configs/experiment/ffhq_blur_S1_tikhonov.yaml split=dev
# analysis
!python scripts/analyze_survival.py --runs artifacts/survival/ffhq_blur_S1_*/* --contrast unet dps --small-only --out results/dev
```
Runs are resumable: if a session dies, re-run the same command and finished images are skipped.
Save `artifacts/` as a private Kaggle Dataset at the end of each session.

**Holdout status:** the preregistration is draft. The runner rejects held-out
runs before loading data. Use the [dev calibration and freeze gates](../docs/improvement_plan.md).
Confirm the account's available free compute. At 150 seconds per sample,
150 scenes × 4 samples × feature-on/off already costs 50 GPU-hours per severity.

## Updated after the CPU dev pilot (2026-11-09)
- Features are now matched on the stationary-Hotelling d′ (`results/ffhq_power_spectrum_256.pt` is committed;
  `base.yaml` points to it). S2 is blur std 4 (`ffhq_blur_S2_*.yaml`).
- Candidate holdout size is 150. Re-estimate paired precision on dev before freezing.
- U-Net (GPU config, 20k steps, base 48): `python scripts/train_unet.py configs/experiment/ffhq_blur_S1_unet.yaml`
  and the same for S2; put the printed path into `solver.weights`.
- Before freezing: rerun the counterfactual check with DPS on dev
  (`run_survival.py ... ffhq_blur_S1_dps.yaml split=dev counterfactual=true`), and decide control vs counterfactual nulls.
- Exploratory detector (H3): `python scripts/analyze_detector.py --runs artifacts/survival/ffhq_blur_S1_dps/* --out results/h3`
- Exploratory DiffPIR lensless development run. Benchmark runtime before budgeting:
  `python scripts/multiplexing_sweep.py --out results/h4_diffpir --sensor 128 --fov full --solver diffpir --K 2 --n-images 30`
