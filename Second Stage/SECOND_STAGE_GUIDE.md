# Style4D Second-Stage Training Guide

The second stage fine-tunes a completed first-stage 4D Gaussian Splatting model against stylized frames. It keeps the learned geometry and motion field fixed and optimizes the appended per-Gaussian appearance MLP. The loader accepts both the original first-stage PLY layout and the expanded second-stage PLY layout, so the same command can start a new stylization or resume from an exported Style4D model.

## 1. Install the second-stage environment

Run the following commands from the repository root. Use a CUDA toolkit and PyTorch build that are compatible with your GPU.

```bash
cd "Second Stage"
pip install -r requirements.txt
pip install ./submodules/simple-knn
pip install ./submodules/depth-diff-gaussian-rasterization
```

Reinstall `depth-diff-gaussian-rasterization` whenever its CUDA source is changed. A stale compiled extension will not contain the second-stage MLP renderer.

## 2. Prepare the stylized dataset

The stylized dataset must use exactly the same cameras, camera order, and frame order as the scene used in stage one. For a DyNeRF scene, the minimum supported layout is:

```text
stylized_scene/
|-- poses_bounds.npy
|-- points3D_downsample2.ply
|-- cam00/images/0000.png ...
|-- cam01/images/0000.png ...
`-- ...
```

Copy `poses_bounds.npy` and `points3D_downsample2.ply` from the original first-stage dataset. The original `camXX.mp4` files are optional when `camXX/images` directories already exist. PNG, JPG, and JPEG frames are supported. Do not mix frames from a different scene or change the camera indices.

## 3. Locate the first-stage model

`--pretrained_model` may point to any of these:

- the first-stage output root containing `point_cloud/iteration_*`;
- a specific `iteration_*` or `fine_iteration_*` directory;
- the iteration's `point_cloud.ply` file.

The selected iteration directory must contain both `point_cloud.ply` and `deformation.pth`. When an output root is supplied, the latest valid fine iteration is selected automatically.

## 4. Run training

From `Second Stage`, run:

```bash
python train.py \
  -s /path/to/stylized_scene \
  -m /path/to/stage2_output \
  --pretrained_model /path/to/stage1_output \
  --configs arguments/dynerf/default.py \
  --disable_gui
```

The equivalent helper script is:

```bash
bash scripts/train_dynerf.sh \
  /path/to/stylized_scene \
  /path/to/stage1_output \
  /path/to/stage2_output \
  arguments/dynerf/default.py
```

By default, only the per-Gaussian style MLP is optimized. Add `--optimize_base_appearance` if SH coefficients and base opacity should also be fine-tuned. Geometry, scale, rotation, and the first-stage deformation field remain frozen in both modes. Keep `convert_SHs_python` disabled because the custom rasterizer consumes the appended MLP parameters together with the SH coefficients.

Useful optional arguments include `--num_workers 0` for debugging data-loading issues, `--save_iterations 3000 7000 14000`, and `--start_checkpoint /path/to/chkpnt_fine_N.pth` for resuming a second-stage training checkpoint. `--start_checkpoint` accepts only a second-stage checkpoint; a first-stage result belongs in `--pretrained_model`.

## 5. Outputs and PLY compatibility

Second-stage models are written to:

```text
stage2_output/point_cloud/iteration_N/
|-- point_cloud.ply
|-- deformation.pth
|-- deformation_table.pth
`-- deformation_accum.pth
```

The unified `GaussianModel.load_ply()` detects the PLY format from its properties:

- stage-one PLY: scalar `opacity`, no `quadrant_*`; the 60 MLP parameters are initialized with a zero-output residual;
- second-stage PLY: `opacity_0` through `opacity_3` and `quadrant_0` through `quadrant_59`; all expanded parameters are restored.

The legacy `load_ply_orj()` name remains as a deprecated compatibility alias and calls the unified loader.

To render the trained model:

```bash
python render.py \
  --model_path /path/to/stage2_output \
  -s /path/to/stylized_scene \
  --configs arguments/dynerf/default.py \
  --skip_train
```
