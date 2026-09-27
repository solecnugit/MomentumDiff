import numpy as np
import pickle
from sklearn.cluster import KMeans
import os

def get_hierarchical_kmeans(info_path, output_path, modes_per_cmd=6):
    print(f"Loading infos from {info_path}...")
    with open(info_path, 'rb') as f:
        data = pickle.load(f)
    
    infos = data['infos']
    
    # 根据 nuscenes_converter.py 的定义：
    # Index 0: Turn Right ([1, 0, 0])
    # Index 1: Turn Left  ([0, 1, 0])
    # Index 2: Go Straight([0, 0, 1])
    trajs_by_cmd = {0: [], 1: [], 2: []}
    cmd_names = {0: "Right", 1: "Left", 2: "Straight"}
    
    print(f"Extracting trajectories from {len(infos)} samples...")
    count = 0
    for info in infos:
        if 'gt_ego_fut_trajs' not in info or 'gt_ego_fut_cmd' not in info:
            continue
            
        traj = info['gt_ego_fut_trajs'] # Shape: (TS, 2)
        cmd_raw = info['gt_ego_fut_cmd'] # Shape: (3,) One-Hot Vector
        
        # [核心修正] 使用 argmax 解析 One-Hot 向量
        if isinstance(cmd_raw, np.ndarray):
            cmd_idx = np.argmax(cmd_raw)
        else:
            # 兼容可能的列表格式
            cmd_idx = np.argmax(np.array(cmd_raw))
        
        # 验证解析出的索引是否有效 (0, 1, 2)
        if cmd_idx in trajs_by_cmd:
            # 过滤掉静止或无效轨迹 (位移过小通常是噪音)
            if np.abs(traj).sum() > 0.1: 
                trajs_by_cmd[cmd_idx].append(traj)
                count += 1

    print(f"Valid trajectories found: {count}")
    
    # 按照索引顺序 0->1->2 依次聚类
    # 这样生成的锚点 tensor 形状为 (3, 6, TS, 2)，分别对应 [Right, Left, Straight]
    target_order = [0, 1, 2]
    
    final_anchors = []
    
    for cmd_idx in target_order:
        data_list = trajs_by_cmd[cmd_idx]
        cmd_name = cmd_names[cmd_idx]
        
        # 处理数据不足的情况 (Mini 数据集常见)
        if len(data_list) < modes_per_cmd:
            print(f"Warning: Not enough data for command '{cmd_name}' (idx={cmd_idx}, count={len(data_list)}). Padding with noise.")
            # 假设轨迹长度为 6 (ego_fut_ts)
            ts = 6 
            if len(data_list) > 0:
                ts = data_list[0].shape[0]
            
            # 生成随机数据填补，避免 KMeans 报错
            dummy_data = np.random.randn(modes_per_cmd, ts, 2) * 0.5 
            # 如果是直行(2)，加一点向前的速度让它更真实
            if cmd_idx == 2: 
                dummy_data[:, :, 0] += np.linspace(0, 5, ts) # x轴向前
                
            data = dummy_data
        else:
            data = np.array(data_list) # (N, TS, 2)

        # 展平时间维度进行聚类: (N, TS*2)
        N, TS, C = data.shape
        data_flat = data.reshape(N, -1)
        
        # 防止样本数少于聚类数报错
        n_clusters = min(modes_per_cmd, len(data))
        
        print(f"Clustering '{cmd_name}' (idx={cmd_idx}) with {len(data)} samples into {modes_per_cmd} modes...")
        kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(data_flat)
        centers = kmeans.cluster_centers_
        
        # 如果聚类出的中心不够 6 个（因为样本太少），复制补充
        if len(centers) < modes_per_cmd:
            padding = np.tile(centers[-1:], (modes_per_cmd - len(centers), 1))
            centers = np.concatenate([centers, padding], axis=0)
            
        # 恢复形状 (modes, TS, 2)
        centers = centers.reshape(modes_per_cmd, TS, C)
        final_anchors.append(centers)
    
    # Stack 得到 (3, 6, TS, 2)
    all_anchors = np.stack(final_anchors, axis=0)
    
    # 保存结果
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.save(output_path, all_anchors)
    print(f"✅ Saved hierarchical kmeans anchors to {output_path}")
    print(f"Final Shape: {all_anchors.shape} (Expected: 3, {modes_per_cmd}, {TS}, 2)")
    print(f"Order: Index 0=Right, 1=Left, 2=Straight")

if __name__ == "__main__":
    # 配置路径
    info_path = 'data/infos/nuscenes_infos_train.pkl'
    output_path = 'data/kmeans/kmeans_plan_6.npy'
    
    get_hierarchical_kmeans(info_path, output_path, modes_per_cmd=6)