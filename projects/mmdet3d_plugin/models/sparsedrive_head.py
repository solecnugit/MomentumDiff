import torch
import torch.nn as nn
from mmcv.runner import BaseModule
from mmdet.models import HEADS
from mmcv.utils import build_from_cfg
from mmdet.core.bbox.builder import BBOX_SAMPLERS, BBOX_CODERS
from projects.mmdet3d_plugin.models.instance_bank import InstanceBank

@HEADS.register_module()
class SparseDriveHead(BaseModule):
    def __init__(
        self,
        det_head=None,
        map_head=None,
        motion_head=None,
        anchor_encoder=None,
        instance_bank=None,
        init_cfg=None,
    ):
        super(SparseDriveHead, self).__init__(init_cfg=init_cfg)
        
        # Build sub-heads
        if det_head is not None:
            self.det_head = build_from_cfg(det_head, HEADS)
        else:
            self.det_head = None
            
        if map_head is not None:
            self.map_head = build_from_cfg(map_head, HEADS)
        else:
            self.map_head = None
            
        if motion_head is not None:
            self.motion_head = build_from_cfg(motion_head, HEADS)
        else:
            self.motion_head = None

        # Build shared modules
        if anchor_encoder is not None:
            self.anchor_encoder = build_from_cfg(anchor_encoder, HEADS)
        else:
            self.anchor_encoder = None
            
        if instance_bank is not None:
            self.instance_bank = build_from_cfg(instance_bank, PLUGIN_LAYERS) if 'PLUGIN_LAYERS' in globals() else InstanceBank(**instance_bank)
        else:
            self.instance_bank = None

    def init_weights(self):
        if self.det_head: self.det_head.init_weights()
        if self.map_head: self.map_head.init_weights()
        if self.motion_head: self.motion_head.init_weights()
        if self.anchor_encoder: 
            for p in self.anchor_encoder.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)

    def forward(self, feature_maps, data):
        # Extract metadata (contains momentum_prior injected in sparsedrive.py)
        metas = data['img_metas']
        
        # 1. Detection Head
        if self.det_head is not None:
            det_output = self.det_head(feature_maps, metas, self.anchor_encoder)
        else:
            det_output = None

        # 2. Map Head
        if self.map_head is not None:
            map_output = self.map_head(feature_maps, metas, self.anchor_encoder)
        else:
            map_output = None

        # 3. Motion Planning Head
        if self.motion_head is not None:
            # Prepare Anchor Handler & Mask from Instance Bank
            if hasattr(self, 'instance_bank') and self.instance_bank is not None:
                mask = self.instance_bank.mask
                anchor_handler = self.instance_bank
                
                # Retrieve cached anchors/features if needed
                # (Logic matches standard SparseDrive flow)
                cached_anchors = self.instance_bank.get_anchors()
            else:
                mask = None
                anchor_handler = None

            # [MomentumFlow Critical Fix]
            # Ensure 'metas' is passed to motion_head so it can access 'momentum_prior'
            motion_output, planning_output = self.motion_head(
                det_output,
                map_output,
                feature_maps,
                metas,          # <--- 必须传递此参数
                self.anchor_encoder,
                mask,
                anchor_handler,
            )
        else:
            motion_output, planning_output = None, None

        return {
            'det_output': det_output,
            'map_output': map_output,
            'motion_output': motion_output,
            'planning_output': planning_output
        }

    def loss(self, model_outs, data):
        losses = {}
        
        # Detection Loss
        if self.det_head is not None and model_outs['det_output'] is not None:
            det_loss = self.det_head.loss(model_outs['det_output'], data)
            losses.update(det_loss)
            
        # Map Loss
        if self.map_head is not None and model_outs['map_output'] is not None:
            map_loss = self.map_head.loss(model_outs['map_output'], data)
            losses.update(map_loss)
            
        # Motion & Planning Loss
        if self.motion_head is not None:
            # Pass motion_loss_cache from instance bank if available
            motion_loss_cache = self.instance_bank.motion_loss_cache if self.instance_bank else None
            
            motion_loss = self.motion_head.loss(
                model_outs['motion_output'],
                model_outs['planning_output'],
                data,
                motion_loss_cache
            )
            losses.update(motion_loss)
            
        return losses

    def post_process(self, model_outs, data):
        results = []
        # Usually we only care about the final planning trajectory for evaluation
        if self.motion_head is not None:
            motion_res, planning_res = self.motion_head.post_process(
                model_outs['det_output'],
                model_outs['motion_output'],
                model_outs['planning_output'],
                data
            )
            # Combine or format results as needed by nuScenes evaluator
            # Typically returns a list of dicts
            return planning_res 
            
        return results