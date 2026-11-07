
# Directory structure

- `configs/`: preprocessing and experiment configuration files.
- `src/preprocessing/`: SH093 R3DPR, whole-face, and cheek/eye/lip ROI
  preprocessing programs.
- `src/datasets/`, `src/models/`, `src/losses/`, and `src/metrics/`: reusable
  dataset, model, loss, and metric implementations.
- `src/scripts/`: five-fold split construction, training, evaluation, fusion,
  and Grad-CAM entry points.
- `src/figures/`: source programs for the retained manuscript and
  supplementary figures.
- `vendor/`: retained upstream R3DPR and rendering source dependencies. The
  empty `checkpoints/`, `BFM/`, `pretrained/`, and `trained_model/` directories
  are intentional runtime placement directories; no model files are included.
- `environment.yml`: the single package environment specification.
- `THIRD_PARTY_LICENSES.md`: centralized notices for retained upstream code.

# Environment preparation

Create the single pinned environment and expose `src/` on `PYTHONPATH`:

```bash
conda env create -f environment.yml
conda activate face-media-review
export PYTHONPATH="$PWD/src"
```

In Windows PowerShell:

```powershell
conda env create -f environment.yml
conda activate face-media-review
$env:PYTHONPATH = "$PWD\src"
```

The R3DPR preprocessing and model-training programs require a compatible
PyTorch/CUDA runtime. The package does not download missing resources
automatically. Supply approved input data and the required runtime resources
at the paths selected in the relevant configuration files before execution.

# Reproduction order

1. Replace the `protected_inputs/` placeholders in the selected YAML files
   with approved, de-identified input paths.
2. Validate and run SH093 R3DPR preprocessing:

   ```bash
   python src/preprocessing/preprocess_r3dpr_sh093.py \
     --config configs/preprocess_r3dpr_sh093.yaml --validate-only
   python src/preprocessing/preprocess_r3dpr_sh093.py \
     --config configs/preprocess_r3dpr_sh093.yaml
   ```

3. Build the participant-grouped five-fold split:

   ```bash
   python src/scripts/5fold/build_nyha_2class_sex_stratified_group_5fold_split.py \
     --labels-csv protected_inputs/approved_labels.csv \
     --output-dir runs/split_5fold
   ```

4. Run the whole-face and ROI preprocessing programs with the matching
   `configs/preprocess_*.yaml` files. The ROI program publishes the retained
   `roi_masked/` output.
5. Train the current experiment configurations with the shared entry point:

   ```bash
   python src/scripts/train/train_r3dpr_resnet_binary_nested_direct_5fold.py \
     --config configs/wholeface_resnet18.yaml \
     --output-dir runs/wholeface_resnet18
   ```

   Repeat for the selected whole-face, cheek, eye, lip, and without-R3DPR
   configurations.
6. Summarize or evaluate the outer-fold predictions, then run the applicable
   fixed-fusion and Grad-CAM programs using the approved experiment outputs.
7. Run the retained programs under `src/figures/` with their approved result
   tables or workbooks to regenerate the corresponding figures.