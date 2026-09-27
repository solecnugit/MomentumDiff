from inspect import signature
import numpy as np
import torch

from mmcv.runner import force_fp32, auto_fp16
from mmcv.utils import build_from_cfg
from mmcv.cnn.bricks.registry import PLUGIN_LAYERS
from mmdet.models import (
    DETECTORS,
    BaseDetector,
    build_backbone,
    build_head,
    build_neck,
)
from .grid_mask import GridMask

try:
    from ..ops import feature_maps_format
    DAF_VALID = True
except:
    DAF_VALID = False

__all__ = ["SparseDrive"]


@DETECTORS.register_module()
class SparseDrive(BaseDetector):
    def __init__(
        self,
        img_backbone,
        head,
        img_neck=None,
        init_cfg=None,
        train_cfg=None,
        test_cfg=None,
        pretrained=None,
        use_grid_mask=True,
        use_deformable_func=False,
        depth_branch=None,
    ):
        super(SparseDrive, self).__init__(init_cfg=init_cfg)
        if pretrained is not None:
            backbone.pretrained = pretrained
        self.img_backbone = build_backbone(img_backbone)
        if img_neck is not None:
            self.img_neck = build_neck(img_neck)
        self.head = build_head(head)
        self.use_grid_mask = use_grid_mask
        if use_deformable_func:
            assert DAF_VALID, "deformable_aggregation needs to be set up."
        self.use_deformable_func = use_deformable_func
        if depth_branch is not None:
            self.depth_branch = build_from_cfg(depth_branch, PLUGIN_LAYERS)
        else:
            self.depth_branch = None
        if use_grid_mask:
            self.grid_mask = GridMask(
                True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7
            )
            
        # [MomentumFlow] Initialize memory for momentum prior
        self.prev_traj = None
        self.prev_l2g = None
        self.prev_scene_token = None

    @auto_fp16(apply_to=("img",), out_fp32=True)
    def extract_feat(self, img, return_depth=False, metas=None):
        bs = img.shape[0]
        if img.dim() == 5:  # multi-view
            num_cams = img.shape[1]
            img = img.flatten(end_dim=1)
        else:
            num_cams = 1
        if self.use_grid_mask:
            img = self.grid_mask(img)
        if "metas" in signature(self.img_backbone.forward).parameters:
            feature_maps = self.img_backbone(img, num_cams, metas=metas)
        else:
            feature_maps = self.img_backbone(img)
        if self.img_neck is not None:
            feature_maps = list(self.img_neck(feature_maps))
        for i, feat in enumerate(feature_maps):
            feature_maps[i] = torch.reshape(
                feat, (bs, num_cams) + feat.shape[1:]
            )
        if return_depth and self.depth_branch is not None:
            depths = self.depth_branch(feature_maps, metas.get("focal"))
        else:
            depths = None
        if self.use_deformable_func:
            feature_maps = feature_maps_format(feature_maps)
        if return_depth:
            return feature_maps, depths
        return feature_maps

    @force_fp32(apply_to=("img",))
    def forward(self, img, **data):
        if self.training:
            return self.forward_train(img, **data)
        else:
            return self.forward_test(img, **data)

    def forward_train(self, img, **data):
        feature_maps, depths = self.extract_feat(img, True, data)
        model_outs = self.head(feature_maps, data)
        output = self.head.loss(model_outs, data)
        if depths is not None and "gt_depth" in data:
            output["loss_dense_depth"] = self.depth_branch.loss(
                depths, data["gt_depth"]
            )
        return output

    def forward_test(self, img, **data):
        if isinstance(img, list):
            return self.aug_test(img, **data)
        else:
            return self.simple_test(img, **data)

    def simple_test(self, img, **data):
        feature_maps = self.extract_feat(img)

        # [MomentumFlow] Calculate Momentum Prior
        # Assume batch_size=1 during test
        img_metas = data['img_metas'][0]
        curr_l2g = img_metas['lidar2global'] # 4x4 numpy array
        curr_scene_token = img_metas.get('scene_token', None)
        
        # Reset if new scene
        if curr_scene_token != self.prev_scene_token:
            self.prev_traj = None
            self.prev_l2g = None
            self.prev_scene_token = curr_scene_token

        momentum_prior = None
        if self.prev_traj is not None and self.prev_l2g is not None:
            try:
                # 1. Previous Local -> Global
                # prev_traj: (ts, 2) or (ts, 3) -> (ts, 4) homogeneous
                ts = self.prev_traj.shape[0]
                prev_traj_homo = np.concatenate([self.prev_traj[:, :2], np.zeros((ts, 1)), np.ones((ts, 1))], axis=-1) # (ts, 4)
                
                # Transform to global
                global_traj = (self.prev_l2g @ prev_traj_homo.T).T # (ts, 4)
                
                # 2. Global -> Current Local
                curr_g2l = np.linalg.inv(curr_l2g)
                curr_traj_homo = (curr_g2l @ global_traj.T).T # (ts, 4)
                curr_traj = curr_traj_homo[:, :2] # (ts, 2)
                
                # 3. Time Shift (t -> t-1)
                # Discard first point (t=0 which is now past), append extrapolation
                shifted_traj = np.zeros_like(curr_traj)
                shifted_traj[:-1] = curr_traj[1:]
                # Extrapolate last point (constant velocity assumption)
                delta = curr_traj[-1] - curr_traj[-2]
                shifted_traj[-1] = curr_traj[-1] + delta
                
                momentum_prior = torch.tensor(shifted_traj, dtype=torch.float32, device=img.device)
                
            except Exception as e:
                print(f"Momentum prior calculation failed: {e}")
                momentum_prior = None

        # Inject prior into metas for the head to use
        # We assume the head knows how to extract 'momentum_prior' from metas
        if momentum_prior is not None:
            data['img_metas'][0]['momentum_prior'] = momentum_prior

        # Forward pass
        model_outs = self.head(feature_maps, data)
        results = self.head.post_process(model_outs, data)
        
        # [MomentumFlow] Update History
        # Extract best predicted trajectory
        # model_outs structure depends on SparseDriveHead -> V13MotionPlanningHead
        # Usually model_outs is a tuple: (motion_output, planning_output)
        # planning_output['prediction'][-1] is the final refined trajectory
        
        try:
            # Note: This structure assumes V13MotionPlanningHead output format
            if isinstance(model_outs, tuple) and len(model_outs) >= 2:
                planning_output = model_outs[1]
                # prediction: list of (B, modes, ts, 2), we take the last layer
                pred_trajs = planning_output['prediction'][-1] 
                cls_scores = planning_output['classification'][-1]
                
                # Select best mode
                best_mode_idx = cls_scores[0].argmax()
                best_traj = pred_trajs[0, 0, best_mode_idx].detach().cpu().numpy() # (ts, 2)
                
                self.prev_traj = best_traj
                self.prev_l2g = curr_l2g
        except Exception as e:
             # print(f"Failed to update momentum history: {e}")
             pass

        output = [{"img_bbox": result} for result in results]
        return output

    def aug_test(self, img, **data):
        # fake test time augmentation
        for key in data.keys():
            if isinstance(data[key], list):
                data[key] = data[key][0]
        return self.simple_test(img[0], **data)