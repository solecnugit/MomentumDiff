# Privacy and Release Checklist

Run this checklist before making the repository public.

## Must Not Be Committed

- nuScenes raw data or generated `data/infos/*.pkl` files.
- Model checkpoints: `*.pth`, `*.pt`, `*.ckpt`.
- TensorBoard logs, evaluation outputs, videos, visualizations, and `work_dirs/`.
- Compiled CUDA extensions: `*.so`, object files, `build/`, `*.egg-info/`.
- Local absolute paths such as `/home/<user>/...`, `/data/<user>/...`, `C:\Users\...`.
- API keys, tokens, cloud credentials, SSH keys, or private URLs.

## Suggested Checks

```bash
rg -n --hidden "C:\\Users|/home/|/data/[^ ]+|password|passwd|token=|api[_-]?key|secret|wandb"
find . -type f \( -name "*.pth" -o -name "*.pt" -o -name "*.ckpt" -o -name "*.pkl" -o -name "*.so" -o -name "events.out.tfevents.*" \)
```

`token` appears in normal nuScenes sample-token code. Treat credential-like strings as sensitive; dataset sample tokens are expected.

## Checkpoints

Host checkpoints outside GitHub, for example in a release asset, institutional storage, Hugging Face, or Google Drive. Put the public URL in `README.md` only after verifying that the file contains no private training logs or local metadata.
