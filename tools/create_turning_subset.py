# tools/create_turning_subset.py
"""Filter nuScenes validation infos to the Turning-nuScenes scene list."""

import argparse

import mmcv
from nuscenes.nuscenes import NuScenes


DEFAULT_TURNING_SCENES = [
    "scene-0778", "scene-1069", "scene-0926", "scene-0928", "scene-0632",
    "scene-0916", "scene-0636", "scene-1064", "scene-0552", "scene-0907",
    "scene-0272", "scene-0634", "scene-0018", "scene-0971", "scene-0919",
    "scene-0561", "scene-0917",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Create Turning-nuScenes infos")
    parser.add_argument("--data-root", default="data/nuscenes/", help="nuScenes root")
    parser.add_argument(
        "--input",
        default="data/infos/nuscenes_infos_val.pkl",
        help="Input nuScenes validation info file",
    )
    parser.add_argument(
        "--output",
        default="data/infos/nuscenes_infos_val_turning.pkl",
        help="Output Turning-nuScenes info file",
    )
    parser.add_argument("--version", default="v1.0-trainval")
    return parser.parse_args()


def create_turning_subset(args):
    nusc = NuScenes(version=args.version, dataroot=args.data_root, verbose=False)

    target_scenes = set(DEFAULT_TURNING_SCENES)
    valid_tokens = set()
    scene_count = 0

    for scene in nusc.scene:
        if scene["name"] not in target_scenes:
            continue
        scene_count += 1
        sample_token = scene["first_sample_token"]
        while sample_token:
            valid_tokens.add(sample_token)
            sample = nusc.get("sample", sample_token)
            sample_token = sample["next"]

    data = mmcv.load(args.input)
    original_infos = data["infos"]
    data["infos"] = [info for info in original_infos if info["token"] in valid_tokens]
    data.setdefault("metadata", {})["version"] = "v1.0-turning-subset"
    mmcv.dump(data, args.output)

    print(
        f"Turning subset saved to {args.output}: "
        f"{scene_count} scenes, {len(original_infos)} -> {len(data['infos'])} samples"
    )


if __name__ == "__main__":
    create_turning_subset(parse_args())
