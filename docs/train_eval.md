# Training and Evaluation Guide

These commands use the 3s MomentumDiff config included in this repository. A 3s checkpoint and the SparseDrive stage-1 initialization must be obtained separately. The config expects nuScenes at `data/nuscenes/`.

## 1. Data Conversion

```bash
mkdir -p data
ln -s /data/nuscenes data/nuscenes

PYTHONPATH=. python tools/data_converter/nuscenes_converter.py nuscenes \
    --root-path data/nuscenes \
    --canbus data/nuscenes \
    --out-dir ./data/infos/ \
    --extra-tag nuscenes \
    --version v1.0 \
    --skip-test
```

`--skip-test` avoids loading `v1.0-test`, which is unnecessary for train/val evaluation and may not be installed. Omit it only when test-set infos are needed and the test tables are available.

## 2. Anchor Generation

```bash
python tools/kmeans/gen_hierarchical_kmeans.py
python tools/kmeans/kmeans_det.py
python tools/kmeans/kmeans_map.py
python tools/kmeans/kmeans_motion.py
python tools/kmeans/kmeans_plan.py
```

## 3. Training

```bash
chmod +x tools/dist_train.sh

./tools/dist_train.sh \
    projects/configs/diffusiondrive_configs/diffusiondrive_small_stage2.py \
    8 \
    --cfg-options \
    load_from=ckpts/sparsedrive_nuscenes_stage1_r50.pth
```

Continue from a checkpoint:

```bash
./tools/dist_train.sh \
    projects/configs/diffusiondrive_configs/diffusiondrive_small_stage2.py \
    8 \
    --resume-from work_dirs/diffusiondrive_small_stage2/latest.pth
```

For this IterBasedRunner config, use `runner.max_iters=<N>` when changing total training length.
The default schedule assumes eight GPUs with global batch size 48. Place `resnet50-19c8e357.pth` from the [PyTorch model archive](https://download.pytorch.org/models/resnet50-19c8e357.pth) in `ckpts/` before building the model.

## 4. Evaluation

### nuScenes 3s

```bash
./tools/dist_test.sh \
    projects/configs/diffusiondrive_configs/diffusiondrive_small_stage2.py \
    ckpts/planning_nuscenes_3s_r50.pth \
    6 \
    --eval bbox \
    --out work_dirs/diffusiondrive_small_stage2/results.pkl \
    --cfg-options \
    evaluation.eval_mode.with_det=False \
    evaluation.eval_mode.with_tracking=False \
    evaluation.eval_mode.with_map=False \
    evaluation.eval_mode.with_motion=False \
    evaluation.eval_mode.with_planning=True
```

### nuScenes 6s

The included converter, planning anchors and config use six ego-planning steps (3s at 2 Hz). The reported 6s results require 12-step infos, anchors, a matching config and the 6s checkpoint. Those artifacts are not in this snapshot, so the 3s config must not be used to evaluate the 6s checkpoint.

### Turning-nuScenes 6s

Create the turning subset from 12-step validation infos and evaluate with the matching 6s config and checkpoint. The current source snapshot contains only the 3s pipeline.

### Turning-nuScenes 3s

Use a separately distributed 3s checkpoint. Apply the Turning subset to both `data.test.ann_file` and `data.test.eval_config.ann_file`.

```bash
python tools/create_turning_subset.py \
    --data-root data/nuscenes/ \
    --input data/infos/nuscenes_infos_val.pkl \
    --output data/infos/nuscenes_infos_val_turning.pkl

mkdir -p work_dirs/diffusiondrive_small_stage2
./tools/dist_test.sh \
    projects/configs/diffusiondrive_configs/diffusiondrive_small_stage2.py \
    ckpts/planning_nuscenes_3s_r50.pth \
    6 \
    --eval bbox \
    --out work_dirs/diffusiondrive_small_stage2/turning_results_3s_tpc.pkl \
    --cfg-options \
    data.test.ann_file=data/infos/nuscenes_infos_val_turning.pkl \
    data.test.eval_config.ann_file=data/infos/nuscenes_infos_val_turning.pkl \
    evaluation.eval_mode.with_det=False \
    evaluation.eval_mode.with_tracking=False \
    evaluation.eval_mode.with_map=False \
    evaluation.eval_mode.with_motion=False \
    evaluation.eval_mode.with_planning=True \
    2>&1 | tee work_dirs/diffusiondrive_small_stage2/eval_turning_3s_tpc.log
```

## 5. Visualization

```bash
PYTHONPATH=. python tools/visualization/visualize.py \
    projects/configs/diffusiondrive_configs/diffusiondrive_small_stage2.py \
    --result-path work_dirs/diffusiondrive_small_stage2/results.pkl \
    --out-dir vis_results \
    --start 0 \
    --end 200 \
    --interval 2 \
    --skip-video
```
