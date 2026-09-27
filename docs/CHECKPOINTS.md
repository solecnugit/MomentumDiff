# Checkpoints

This repository intentionally does not ship checkpoints.

Recommended local layout:

```text
ckpts/sparsedrive_nuscenes_stage1_r50.pth
ckpts/planning_nuscenes_3s_r50.pth
ckpts/planning_nuscenes_6s_r50.pth
```

Before publishing a checkpoint link, verify:

1. The file is the intended model state only.
2. No TensorBoard logs or local paths are bundled with it.
3. The license of upstream pretrained weights permits redistribution.
4. The README link points to a stable public download location.
