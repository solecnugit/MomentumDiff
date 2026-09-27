# tools/offline_eval_turning.py
import sys
import os
import mmcv
from os import path as osp
import importlib

# ================= 1. Patch 必须最先运行 =================
print(">>> 正在初始化评测环境...")
from nuscenes.utils import splits

# 定义转弯场景列表 (Top 17)
my_turning_scenes = [
    'scene-0778', 'scene-1069', 'scene-0926', 'scene-0928', 'scene-0632', 
    'scene-0916', 'scene-0636', 'scene-1064', 'scene-0552', 'scene-0907', 
    'scene-0272', 'scene-0634', 'scene-0018', 'scene-0971', 'scene-0919', 
    'scene-0561', 'scene-0917'
]

# 劫持 create_splits_scenes
_original_create_splits = splits.create_splits_scenes
def custom_create_splits():
    data = _original_create_splits()
    data['val'] = my_turning_scenes 
    return data

splits.create_splits_scenes = custom_create_splits
splits.val = my_turning_scenes
print(f">>> [Patch] 验证集已被锁定为 {len(splits.val)} 个转弯场景。")
# =============================================================

from mmcv import Config
from mmdet.datasets import build_dataset

def main():
    # --- 配置路径 ---
    config_path = 'projects/configs/diffusiondrive_configs/diffusiondrive_small_stage2.py'
    pkl_results_path = 'work_dirs/diffusiondrive_small_stage2/turning_results.pkl'
    ann_file_path = 'data/infos/nuscenes_infos_val_turning.pkl'
    
    # 设定工作目录
    work_dir = 'work_dirs/turning_eval_logs'
    mmcv.mkdir_or_exist(work_dir)

    if not osp.exists(pkl_results_path):
        print(f"错误：找不到结果文件 {pkl_results_path}")
        return

    # 1. 加载 Config
    print(f">>> 正在加载配置文件: {config_path}")
    cfg = Config.fromfile(config_path)

    # ================= 2. 手动加载插件 =================
    if hasattr(cfg, "plugin") and cfg.plugin:
        if hasattr(cfg, "plugin_dir"):
            plugin_dir = cfg.plugin_dir
            _module_dir = os.path.dirname(plugin_dir)
            _module_dir = _module_dir.split("/")
            _module_path = _module_dir[0]
            for m in _module_dir[1:]:
                _module_path = _module_path + "." + m
            print(f">>> [Plugin] 正在加载插件: {_module_path}")
            importlib.import_module(_module_path)

    # 3. 准备构建 Dataset
    print(f">>> 正在修改 Dataset 配置 (使用 {ann_file_path})...")
    
    # [关键修复] 同时修改 data.test.ann_file 和 data.test.eval_config.ann_file
    if isinstance(cfg.data.test, dict):
        # 1. 修改主 Dataset 配置
        cfg.data.test.ann_file = ann_file_path
        cfg.data.test.work_dir = work_dir
        
        # 2. 修改嵌套的 eval_config (这步是关键！)
        if 'eval_config' in cfg.data.test:
            print(">>> [Fix] 同步修改 eval_config 中的 ann_file...")
            cfg.data.test.eval_config['ann_file'] = ann_file_path
            # 如果需要，也可以同步修改 version，防止打印混乱
            # cfg.data.test.eval_config['version'] = 'v1.0-turning-subset'
            
    elif isinstance(cfg.data.test, list):
        for ds in cfg.data.test:
            ds.ann_file = ann_file_path
            ds.work_dir = work_dir
            if 'eval_config' in ds:
                ds.eval_config['ann_file'] = ann_file_path
            
    print(f">>> 正在构建 Dataset...")
    dataset = build_dataset(cfg.data.test)
    
    # 4. 加载推理结果
    print(f">>> 正在加载推理结果: {pkl_results_path}")
    outputs = mmcv.load(pkl_results_path)
    
    print(f">>> 数据集长度: {len(dataset)}, 结果长度: {len(outputs)}")
    
    # 5. 开始评测
    print(">>> 开始运行 Evaluation...")
    
    eval_cfg = cfg.get('evaluation', {}).copy()
    for key in ['interval', 'tmpdir', 'start', 'gpu_collect', 'save_best', 'rule']:
        eval_cfg.pop(key, None)
        
    results = dataset.evaluate(outputs, **eval_cfg)
    
    print("\n>>> 转弯场景评测完成！结果如下：")
    print(results)

if __name__ == '__main__':
    main()