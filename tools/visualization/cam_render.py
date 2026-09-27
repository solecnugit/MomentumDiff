import os
import numpy as np
import cv2
from PIL import Image

import matplotlib
import matplotlib.pyplot as plt
from pyquaternion import Quaternion
from nuscenes.utils.data_classes import Box as NuScenesBox
from nuscenes.utils.geometry_utils import view_points, box_in_image, BoxVisibility, transform_matrix

from tools.visualization.bev_render import (
    color_mapping, 
    SCORE_THRESH, 
    MAP_SCORE_THRESH,
    CMD_LIST
)


CAM_NAMES_NUSC = [
    'CAM_FRONT_LEFT',
    'CAM_FRONT',
    'CAM_FRONT_RIGHT',
    'CAM_BACK_RIGHT',
    'CAM_BACK',
    'CAM_BACK_LEFT',
]
CAM_NAMES_NUSC_converter = [
    'CAM_FRONT',
    'CAM_FRONT_RIGHT',
    'CAM_FRONT_LEFT',
    'CAM_BACK',
    'CAM_BACK_LEFT',
    'CAM_BACK_RIGHT',
]

class CamRender:
    def __init__(
        self, 
        plot_choices,
        out_dir,
    ):
        self.plot_choices = plot_choices
        self.pred_dir = os.path.join(out_dir, "cam_pred")
        os.makedirs(self.pred_dir, exist_ok=True)

    def reset_canvas(self):
        plt.close()
        plt.gca().set_axis_off()
        plt.axis('off')
        self.fig, self.axes = plt.subplots(2, 3, figsize=(160 /3 , 20))
        plt.tight_layout()

    def render(
        self,
        data, 
        result,
        index,
    ):
        self.reset_canvas()
        self.render_image_data(data, index)
        self.draw_detection_pred(data, result)
        # self.draw_motion_pred(data, result)
        self.draw_planning_pred(data, result)
        save_path = os.path.join(self.pred_dir, str(index).zfill(4) + '.png')
        self.save_fig(save_path)
        return save_path

    def load_image(self, data_path, cam):
        """Update the axis of the plot with the provided image."""
        image = np.array(Image.open(data_path))
        font = cv2.FONT_HERSHEY_SIMPLEX
        org = (50, 60)
        fontScale = 2
        color = (0, 0, 0)
        thickness = 4
        return cv2.putText(image, cam, org, font, fontScale, color, thickness, cv2.LINE_AA)

    def update_image(self, image, index, cam):
        """Render image data for each camera."""
        ax = self.get_axis(index)
        ax.imshow(image)
        plt.axis('off')
        ax.axis('off')
        ax.grid(False)

    def get_axis(self, index):
        """Retrieve the corresponding axis based on the index."""
        return self.axes[index//3, index % 3]

    def save_fig(self, filename):
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                            hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.savefig(filename, format='png', dpi=500)

    def render_image_data(self, data, index):
        """Load and annotate image based on the provided path."""
        for i, cam in enumerate(CAM_NAMES_NUSC):
            idx = CAM_NAMES_NUSC_converter.index(cam)
            img_path = data['img_filename'][idx]
            image = self.load_image(img_path, cam)
            self.update_image(image, i, cam)
    
    def draw_detection_pred(self, data, result):
        if not (self.plot_choices['draw_pred'] and self.plot_choices['det'] and "boxes_3d" in result):
            return

        bboxes = result['boxes_3d'].numpy()
        for j, cam in enumerate(CAM_NAMES_NUSC):
            idx = CAM_NAMES_NUSC_converter.index(cam)
            cam_intrinsic = data['cam_intrinsic'][idx]
            lidar2cam = data['lidar2cam']
            extrinsic = lidar2cam[idx]
            trans = extrinsic[3, :3]
            rot = Quaternion(matrix=extrinsic[:3, :3]).inverse
            imsize = (1600, 900)

            for i in range(result['labels_3d'].shape[0]):
                score = result['scores_3d'][i]
                if score < SCORE_THRESH: 
                    continue
                color = color_mapping[result['instance_ids'][i] % len(color_mapping)]
                
                center = bboxes[i, 0 : 3]
                box_dims = bboxes[i, 3 : 6]
                nusc_dims = box_dims[..., [1, 0, 2]]
                quat = Quaternion(axis=[0, 0, 1], radians=bboxes[i, 6])
                box = NuScenesBox(
                    center,
                    nusc_dims,
                    quat
                )
                box.rotate(rot)
                box.translate(trans)
                if box_in_image(box, cam_intrinsic, imsize):
                    box.render(
                        self.axes[j // 3, j % 3], 
                        view=cam_intrinsic, 
                        normalize=True, 
                        colors=(color, color, color),
                        linewidth=4,
                    )
            
            self.axes[j//3, j % 3].set_xlim(0, imsize[0])
            self.axes[j//3, j % 3].set_ylim(imsize[1], 0)

    def draw_motion_pred(self, data, result, points_per_step=10):
        if not (self.plot_choices['draw_pred'] and self.plot_choices['motion'] and "trajs_3d" in result):
            return

        bboxes = result['boxes_3d'].numpy()
        for j, cam in enumerate(CAM_NAMES_NUSC):
            idx = CAM_NAMES_NUSC_converter.index(cam)
            cam_intrinsic = data['cam_intrinsic'][idx]
            lidar2cam = data['lidar2cam']
            extrinsic = lidar2cam[idx]
            trans = extrinsic[3, :3]
            rot = Quaternion(matrix=extrinsic[:3, :3]).inverse
            imsize = (1600, 900)

            for i in range(result['labels_3d'].shape[0]):
                score = result['scores_3d'][i]
                if score < SCORE_THRESH: 
                    continue
                color = color_mapping[result['instance_ids'][i] % len(color_mapping)]
                
                traj_score = result['trajs_score'][i].numpy()
                traj = result['trajs_3d'][i].numpy()
                
                mode_idx = traj_score.argmax()
                traj = traj[mode_idx]
                origin = bboxes[i, :2][None]
                traj = np.concatenate([origin, traj], axis=0)
                traj_expand = np.ones((traj.shape[0], 1)) 
                traj_expand[:] = bboxes[i, 2] - bboxes[i, 5] / 2
                traj = np.concatenate([traj, traj_expand], axis=1)

                center = bboxes[i, 0 : 3]
                box_dims = bboxes[i, 3 : 6]
                nusc_dims = box_dims[..., [1, 0, 2]]
                quat = Quaternion(axis=[0, 0, 1], radians=bboxes[i, 6])
                box = NuScenesBox(
                    center,
                    nusc_dims,
                    quat
                )
                box.rotate(rot)
                box.translate(trans)
                if not box_in_image(box, cam_intrinsic, imsize):
                    continue
                traj_points = traj @ extrinsic[:3, :3] + trans
                self._render_traj(traj_points, cam_intrinsic, j, color=color, s=15)

    '''
    def draw_planning_pred(self, data, result):
        if not (self.plot_choices['draw_pred'] and self.plot_choices['planning'] and "planning" in result):
            return
        # for j, cam in enumerate(CAM_NAMES_NUSC[1]):
        #     idx = CAM_NAMES_NUSC_converter.index(cam)
        #     cam_intrinsic = data['cam_intrinsic'][idx]
        #     lidar2cam = data['lidar2cam']
        #     extrinsic = lidar2cam[idx]
        #     trans = extrinsic[3, :3]
        #     rot = Quaternion(matrix=extrinsic[:3, :3]).inverse
        #     imsize = (1600, 900)

        #     plan_trajs = result['planning'][0].cpu().numpy()
        #     plan_trajs = plan_trajs.reshape(3, -1, 6, 2)
        #     num_cmd = len(CMD_LIST)
        #     num_mode = plan_trajs.shape[1]
        #     plan_trajs = np.concatenate((np.zeros((num_cmd, num_mode, 1, 2)), plan_trajs), axis=2)
        #     plan_trajs = plan_trajs.cumsum(axis=-2)
        #     plan_score = result['planning_score'][0].cpu().numpy()
        #     plan_score = plan_score.reshape(3, -1)

        #     cmd = data['gt_ego_fut_cmd'].argmax()
        #     plan_trajs = plan_trajs[cmd]
        #     plan_score = plan_score[cmd]

        #     mode_idx = plan_score.argmax()
        #     plan_traj = plan_trajs[mode_idx]
        #     traj_expand = np.ones((plan_traj.shape[0], 1)) * -2
        #     # traj_expand[:] = bboxes[i, 2] - bboxes[i, 5] / 2
        #     plan_traj = np.concatenate([plan_traj, traj_expand], axis=1)

        #     traj_points = plan_traj @ extrinsic[:3, :3] + trans
        #     self._render_traj(traj_points, cam_intrinsic, j)

        idx = 0 ## front camera
        cam_intrinsic = data['cam_intrinsic'][idx]
        lidar2cam = data['lidar2cam']
        extrinsic = lidar2cam[idx]
        trans = extrinsic[3, :3]
        rot = Quaternion(matrix=extrinsic[:3, :3]).inverse
        # plan_trajs = result['planning'][0].cpu().numpy()
        # plan_trajs = plan_trajs.reshape(3, -1, 6, 2)
        # num_cmd = len(CMD_LIST)
        # num_mode = plan_trajs.shape[1]
        # plan_trajs = np.concatenate((np.zeros((num_cmd, num_mode, 1, 2)), plan_trajs), axis=2)
        # plan_trajs = plan_trajs.cumsum(axis=-2)
        # plan_score = result['planning_score'][0].cpu().numpy()
        # plan_score = plan_score.reshape(3, -1)

        # cmd = data['gt_ego_fut_cmd'].argmax()
        # plan_trajs = plan_trajs[cmd]
        # plan_score = plan_score[cmd]

        # mode_idx = plan_score.argmax()
        # plan_traj = plan_trajs[mode_idx]
        plan_traj = result["final_planning"]
        plan_traj = np.concatenate((np.zeros((1, 2)), plan_traj), axis=0)
        traj_expand = np.ones((plan_traj.shape[0], 1)) * -1.8
        plan_traj = np.concatenate([plan_traj, traj_expand], axis=1)

        traj_points = plan_traj @ extrinsic[:3, :3] + trans
        self._render_traj(traj_points, cam_intrinsic, j=1)
    '''
    def draw_planning_pred(self, data, result):

        target_key = 'planning' 
        if "final_planning" in result:
             target_key = "final_planning"
             
        if not (self.plot_choices['draw_pred'] and self.plot_choices['planning'] and target_key in result):
            return

        idx = 0 ## front camera
        cam_intrinsic = data['cam_intrinsic'][idx]
        lidar2cam = data['lidar2cam']
        extrinsic = lidar2cam[idx]
        trans = extrinsic[3, :3]
        rot = Quaternion(matrix=extrinsic[:3, :3]).inverse

        # [修复 3] 获取轨迹数据
        # 注意：result['planning'] 通常是 Tensor 或 numpy array
        # 如果是原始输出，形状可能是 (1, 6, 6, 2) 或 (6, 2)
        plan_traj = result[target_key]
        
        # 如果 plan_traj 是 Tensor，转为 numpy
        if hasattr(plan_traj, 'cpu'):
            plan_traj = plan_traj.cpu().numpy()
            
        # 兼容性处理：如果取出来是多模态 (N, T, 2) 或带 Batch (1, ...)，需要取最优的一条
        if len(plan_traj.shape) > 2: 
             # 假设取 score 最高的一条，或者如果已经是 final 就取第0条
             # 这里假设您已经处理好它是 (T, 2) 形状，如果不是，需要额外处理
             # 例如: plan_traj = plan_traj[0] 
             pass

        # 拼接原点 (0,0)
        plan_traj = np.concatenate((np.zeros((1, 2)), plan_traj), axis=0)
        
        # 扩展维度以进行投影 (x, y, z)，假设地面高度
        # 这里您之前写的是 -1.8，如果觉得不准可以调成 -1.5 或 0.0 试试
        traj_expand = np.ones((plan_traj.shape[0], 1)) * -1.8 
        plan_traj = np.concatenate([plan_traj, traj_expand], axis=1)

        # 投影到相机坐标系
        traj_points = plan_traj @ extrinsic[:3, :3] + trans
        
        # 绘制 (j=1 对应 CAM_FRONT)
        self._render_traj(traj_points, cam_intrinsic, j=1)

    def _render_traj(self, traj_points, cam_intrinsic, j, color=(1, 0.5, 0), s=150, points_per_step=10):
        # 插值生成平滑轨迹点
        total_steps = (len(traj_points)-1) * points_per_step + 1
        total_xy = np.zeros((total_steps, 3))
        for k in range(total_steps-1):
            unit_vec = traj_points[k//points_per_step +
                                    1] - traj_points[k//points_per_step]
            total_xy[k] = (k/points_per_step - k//points_per_step) * \
                unit_vec + traj_points[k//points_per_step]
        # 最后一个点赋值
        total_xy[-1] = traj_points[-1] # 注意：原代码这里可能漏了给最后一个点赋值，虽然zeros初始化了

        # [关键修复] 先筛选，再投影
        # 1. 找出位于相机前方(Z > 0.1m)的点
        in_range_mask = total_xy[:, 2] > 0.1
        
        # 2. 如果没有点在视野内，直接返回
        if not np.any(in_range_mask):
            return

        # 3. 只选取有效的 3D 点进行投影 (转置为 3xN)
        valid_points_3d = total_xy[in_range_mask].T 

        # 4. 投影到 2D 像素坐标 (view_points 内部会除以 Z)
        points_2d = view_points(
            valid_points_3d, cam_intrinsic, normalize=True)[:2, :]
        
        # 5. 绘制
        self.axes[j // 3, j % 3].scatter(points_2d[0], points_2d[1], color=color, s=s)