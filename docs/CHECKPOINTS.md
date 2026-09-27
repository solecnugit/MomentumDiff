# Checkpoints

Checkpoints are hosted as GitHub Release assets, outside the Git source tree. For this private preview, see the [existing private checkpoint release](https://github.com/momentumdiff/planning-artifact/releases/tag/v0.1-checkpoints). The final public repository needs its own accessible checkpoint links.

Recommended local layout:

```text
ckpts/sparsedrive_nuscenes_stage1_r50.pth
ckpts/planning_nuscenes_3s_r50.pth
ckpts/planning_nuscenes_6s_r50.pth
```

When moving checkpoints to another repository, verify:

1. The file is the intended model state only.
2. No TensorBoard logs or local paths are bundled with it.
3. The license of upstream pretrained weights permits redistribution.
4. The README link points to a stable public download location.
