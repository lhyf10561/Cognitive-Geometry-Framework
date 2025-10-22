import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.font_manager as fm
from matplotlib import image as mpimg
import time

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

"""
Structural Control of Attractor Dimension: Destruction vs. Fusion

A 3x3 comparison showing three distinct mechanisms of dimension collapse:
- Paradigm 1: Functional reorganization (resource conservation)
- Paradigm 2: Physical lesion (resource loss)  
- Paradigm 3: Functional integration (fusion)

每次运行使用不同的噪声，但三种范式使用相同的噪声序列
"""

# ==================== 参数配置区域 ====================

EXPERIMENT_CONFIG = {
    # 动力学参数
    'plunder_rate': 1,
    'total_resource': 100.0,
    'noise_strength': 4,
    
    # 时间参数
    'intervention_time': 500,    # 干预时间（摧毁或融合）
    't_end': 1000,
    'dt': 0.1,
    
    # 融合参数
    'fusion_strength': 0.1,     # 融合后的强双向分享系数
    
    # 初始条件（绿色阵营占优）
    'X0': [0.0, 0.0, 0.0, 33.3, 33.3, 33.4],
    
    # 可视化参数
    'dpi': 300,
    'figsize': (24, 24),         # 3x3需要正方形
    'view_elev': 20,
    'view_azim': 45,
    
    # 图像文件路径
    'destruction_image': '毁灭.jpeg',
    'fusion_image': '融合.jpeg',
}

# =====================================================


def generate_noise_sequence(n_steps, noise_strength, n_nodes=3):
    """
    预先生成整个模拟过程的噪声序列
    
    Args:
        n_steps: 时间步数
        noise_strength: 噪声强度
        n_nodes: 节点数（默认3个绿色阵营节点）
    
    Returns:
        noise_seq: shape (n_steps, n_nodes) 的噪声数组
    """
    # 使用当前时间作为随机种子，保证每次不同
    seed = int(time.time() * 1000) % (2**32)
    rng = np.random.RandomState(seed)
    
    noise_seq = rng.randn(n_steps, n_nodes) * noise_strength
    
    # 每个时间步的噪声求和为0（资源守恒）
    for i in range(n_steps):
        noise_seq[i, :] -= noise_seq[i, :].mean()
    
    return noise_seq, seed


def dynamics_model_with_noise_pregenerated(t, X, W, noise_strength=0.0, active_green_nodes=None, 
                                           noise_seq=None, step_idx=0):
    """
    使用预生成噪声的资源动力学模型
    
    Args:
        noise_seq: 预生成的噪声序列 shape (n_steps, 3)
        step_idx: 当前时间步索引
    """
    N = len(X)
    X = np.maximum(X, 0)
    dXdt = np.zeros(N)
    
    S = np.maximum(0, -W)
    P = np.maximum(0, W)

    # Sharing mechanism
    sharing_outflow = X * np.sum(S, axis=1)
    sharing_inflow = S.T @ X
    dXdt += sharing_inflow - sharing_outflow

    # Plunder mechanism
    for j in range(N):
        plunderers_indices = np.where(P[:, j] > 0)[0]
        if not len(plunderers_indices):
            continue
        
        total_plunder_demand = np.sum(P[plunderers_indices, j] * X[plunderers_indices])
        
        if total_plunder_demand > 0:
            max_outflow = X[j] / 0.01
            actual_outflow = min(total_plunder_demand, max_outflow)
            dXdt[j] -= actual_outflow
            
            for i_plunderer in plunderers_indices:
                plunder_share = P[i_plunderer, j] * X[i_plunderer] / total_plunder_demand
                dXdt[i_plunderer] += plunder_share * actual_outflow
    
    # Add pre-generated noise to active green camp nodes
    if noise_strength > 0 and active_green_nodes is not None and noise_seq is not None:
        n_active = len(active_green_nodes)
        if n_active > 1 and step_idx < len(noise_seq):
            # 使用预生成的噪声
            if n_active == 3:
                # 三个节点都活跃（融合情况或干预前）
                for i, node_idx in enumerate(active_green_nodes):
                    dXdt[node_idx] += noise_seq[step_idx, i]
            elif n_active == 2:
                # 两个节点活跃（Node6被毁灭后）
                # Node6的噪声按照当前资源比例分配给Node4和Node5
                noise_4 = noise_seq[step_idx, 0]  # Node4原本的噪声
                noise_5 = noise_seq[step_idx, 1]  # Node5原本的噪声
                noise_6 = noise_seq[step_idx, 2]  # Node6的噪声，需要重分配
                
                # 获取Node4和Node5的当前资源
                X4_current = X[3]
                X5_current = X[4]
                total_45 = X4_current + X5_current
                
                if total_45 > 0:
                    # 按资源比例分配Node6的噪声
                    ratio_4 = X4_current / total_45
                    ratio_5 = X5_current / total_45
                    
                    dXdt[3] += noise_4 + noise_6 * ratio_4
                    dXdt[4] += noise_5 + noise_6 * ratio_5
                else:
                    # 如果两个节点资源都为0，平均分配
                    dXdt[3] += noise_4 + noise_6 * 0.5
                    dXdt[4] += noise_5 + noise_6 * 0.5
    
    return dXdt


def dynamics_fusion_constrained_pregenerated(t, X, W, noise_strength=0.0, active_green_nodes=None, 
                                            noise_seq=None, step_idx=0):
    """使用预生成噪声的融合约束动力学"""
    N = len(X)
    X = np.maximum(X, 0)
    dXdt = np.zeros(N)
    
    S = np.maximum(0, -W)
    P = np.maximum(0, W)

    # Sharing mechanism
    sharing_outflow = X * np.sum(S, axis=1)
    sharing_inflow = S.T @ X
    dXdt += sharing_inflow - sharing_outflow

    # Plunder mechanism
    for j in range(N):
        plunderers_indices = np.where(P[:, j] > 0)[0]
        if not len(plunderers_indices):
            continue
        
        total_plunder_demand = np.sum(P[plunderers_indices, j] * X[plunderers_indices])
        
        if total_plunder_demand > 0:
            max_outflow = X[j] / 0.01
            actual_outflow = min(total_plunder_demand, max_outflow)
            dXdt[j] -= actual_outflow
            
            for i_plunderer in plunderers_indices:
                plunder_share = P[i_plunderer, j] * X[i_plunderer] / total_plunder_demand
                dXdt[i_plunderer] += plunder_share * actual_outflow
    
    # Add noise with fusion constraint
    if noise_strength > 0 and noise_seq is not None and step_idx < len(noise_seq):
        # Node 4 gets independent noise
        noise_4 = noise_seq[step_idx, 0]
        # Node 5 and 6 get shared noise (to maintain synchrony)
        noise_56_shared = (noise_seq[step_idx, 1] + noise_seq[step_idx, 2]) / 2
        
        dXdt[3] += noise_4
        dXdt[4] += noise_56_shared
        dXdt[5] += noise_56_shared
        
        # Remove mean to conserve total
        mean_noise = (dXdt[3] + dXdt[4] + dXdt[5]) / 3
        dXdt[3] -= mean_noise
        dXdt[4] -= mean_noise
        dXdt[5] -= mean_noise
    
    return dXdt


def build_green_camp_matrix(N=6, plunder_rate=0.1):
    """Build interaction matrix with Green camp dominant"""
    W = np.zeros((N, N))
    
    for green_node in [3, 4, 5]:
        for other_node in [0, 1, 2]:
            W[green_node, other_node] = plunder_rate
            W[other_node, green_node] = plunder_rate
    
    return W


def run_paradigm_experiment(config, paradigm='reorganization', noise_seq=None):
    """
    Run experiment with specified paradigm using pre-generated noise
    
    Args:
        config: Experiment configuration
        paradigm: 'reorganization', 'lesion', or 'fusion'
        noise_seq: Pre-generated noise sequence
    """
    W = build_green_camp_matrix(plunder_rate=config['plunder_rate'])
    X0 = np.array(config['X0'])
    intervention_time = config['intervention_time']
    t_end = config['t_end']
    dt = config['dt']
    noise_strength = config['noise_strength']
    
    # === Phase 1: Before intervention ===
    print(f"Phase 1 ({paradigm}): Random walk on 2D attractor...")
    
    t_before = []
    X_before = []
    
    X_current = X0.copy()
    t_current = 0
    step_idx = 0
    active_nodes_before = [3, 4, 5]
    
    while t_current < intervention_time:
        t_before.append(t_current)
        X_before.append(X_current.copy())
        
        dXdt = dynamics_model_with_noise_pregenerated(
            t_current, X_current, W, noise_strength, 
            active_nodes_before, noise_seq, step_idx)
        
        X_current = X_current + dXdt * dt
        X_current = np.maximum(X_current, 0)
        
        total = np.sum(X_current)
        if total > 0:
            X_current = X_current * config['total_resource'] / total
        
        t_current += dt
        step_idx += 1
    
    t_before = np.array(t_before)
    X_before = np.array(X_before).T
    
    # === Intervention Event ===
    print(f"\nIntervention at t={intervention_time} ({paradigm}):")
    print(f"  Before: Node4={X_current[3]:.2f}, Node5={X_current[4]:.2f}, Node6={X_current[5]:.2f}")
    
    X_intervention = X_current.copy()
    
    if paradigm == 'reorganization':
        # Paradigm 1: Functional reorganization (resource conservation)
        resource_node6 = X_current[5]
        X_current[3] += 0.5 * resource_node6
        X_current[4] += 0.5 * resource_node6
        X_current[5] = 0
        print(f"  Paradigm 1: Resource redistributed ({resource_node6:.2f} → Node4+Node5)")
        print(f"  After:  Node4={X_current[3]:.2f}, Node5={X_current[4]:.2f}, Node6={X_current[5]:.2f}")
        print(f"  Total:  {np.sum(X_current):.2f} (conserved)")
        
        # Remove connections to Node6
        W[:, 5] = 0
        W[5, :] = 0
        active_nodes_after = [3, 4]
        
    elif paradigm == 'lesion':
        # Paradigm 2: Physical lesion (resource loss)
        resource_node6 = X_current[5]
        X_current[5] = 0  # Node6 resource vanishes
        print(f"  Paradigm 2: Resource lost ({resource_node6:.2f} vanished)")
        print(f"  After:  Node4={X_current[3]:.2f}, Node5={X_current[4]:.2f}, Node6={X_current[5]:.2f}")
        print(f"  Total:  {np.sum(X_current):.2f} (net loss)")
        
        # Remove connections to Node6
        W[:, 5] = 0
        W[5, :] = 0
        active_nodes_after = [3, 4]
        
    elif paradigm == 'fusion':
        # Paradigm 3: Functional integration (fusion)
        fusion_strength = config['fusion_strength']
        # Add strong bidirectional sharing between Node5 and Node6
        W[4, 5] = -fusion_strength  # Node5 shares with Node6
        W[5, 4] = -fusion_strength  # Node6 shares with Node5
        print(f"  Paradigm 3: Fusion initiated (Node5 ↔ Node6, strength={fusion_strength})")
        print(f"  After:  Node4={X_current[3]:.2f}, Node5={X_current[4]:.2f}, Node6={X_current[5]:.2f}")
        print(f"  Total:  {np.sum(X_current):.2f} (conserved)")
        
        # All three nodes remain active, but 5 and 6 will synchronize
        active_nodes_after = [3, 4, 5]
    
    # === Phase 2: After intervention ===
    print(f"Phase 2 ({paradigm}): Random walk on collapsed attractor...")
    
    t_after = []
    X_after = []
    
    t_current = intervention_time
    
    while t_current < t_end:
        t_after.append(t_current)
        X_after.append(X_current.copy())
        
        # Use fusion-constrained dynamics for fusion paradigm
        if paradigm == 'fusion':
            dXdt = dynamics_fusion_constrained_pregenerated(
                t_current, X_current, W, noise_strength, 
                active_nodes_after, noise_seq, step_idx)
        else:
            dXdt = dynamics_model_with_noise_pregenerated(
                t_current, X_current, W, noise_strength, 
                active_nodes_after, noise_seq, step_idx)
        
        X_current = X_current + dXdt * dt
        X_current = np.maximum(X_current, 0)
        
        # Normalization depends on paradigm
        total = np.sum(X_current)
        if paradigm in ['reorganization', 'fusion']:
            # Maintain total=100
            if total > 0:
                X_current = X_current * config['total_resource'] / total
        else:  # lesion
            # No renormalization - accept reduced total
            pass
        
        t_current += dt
        step_idx += 1
    
    t_after = np.array(t_after)
    X_after = np.array(X_after).T
    
    return {
        't_before': t_before,
        'X_before': X_before,
        't_after': t_after,
        'X_after': X_after,
        'X_intervention': X_intervention,
        'paradigm': paradigm
    }


def plot_nine_panel_comparison(results_reorg, results_lesion, results_fusion, config):
    """Create 3x3 nine-panel comparison figure with topology images in first column"""
    
    intervention_time = config['intervention_time']
    
    # Color scheme - Light Green, Green, Dark Green
    color_node4 = '#81C784'  # Light Green (浅绿)
    color_node5 = '#4CAF50'  # Green (绿)
    color_node6 = '#2E7D32'  # Dark Green (深绿)
    
    # Load topology images
    try:
        img_destruction = mpimg.imread(config['destruction_image'])
        img_fusion = mpimg.imread(config['fusion_image'])
        images_loaded = True
    except Exception as e:
        print(f"Warning: Could not load images: {e}")
        images_loaded = False
    
    fig = plt.figure(figsize=config['figsize'])
    
    # ===== Row 1: Paradigm 1 (Functional Reorganization) =====
    
    # Panel 1: Topology - Destruction
    ax_topo1 = plt.subplot(3, 3, 1)
    if images_loaded:
        ax_topo1.imshow(img_destruction)
        ax_topo1.axis('off')
        ax_topo1.set_title('Topology Representation\n(Destruction)', 
                          fontsize=12, fontweight='bold', pad=10)
        
        # Add legend for camps and arrows
        from matplotlib.patches import Patch
        from matplotlib.lines import Line2D
        
        legend_elements = [
            Patch(facecolor='#FF9800', label='Red Camp (Nodes 1,2)'),
            Patch(facecolor='#2196F3', label='Node 3'),
            Patch(facecolor='#4CAF50', label='Green Camp (Nodes 4,5,6)'),
            Line2D([0], [0], color='red', linewidth=2, label='Plunder Arrow'),
            Line2D([0], [0], color='green', linewidth=2, label='Share Arrow'),
        ]
        
        ax_topo1.legend(handles=legend_elements, loc='upper left', 
                       bbox_to_anchor=(0, 1), ncol=1, fontsize=8,
                       frameon=True, fancybox=True, shadow=True)
    else:
        ax_topo1.text(0.5, 0.5, 'Destruction\nTopology', 
                     ha='center', va='center', fontsize=14, fontweight='bold')
        ax_topo1.axis('off')
    
    # Panel 2: Time Series
    ax_a = plt.subplot(3, 3, 2)
    ax_a.plot(results_reorg['t_before'], results_reorg['X_before'][3, :], 
             color=color_node4, linewidth=2, label='Node 4', alpha=0.8)
    ax_a.plot(results_reorg['t_before'], results_reorg['X_before'][4, :], 
             color=color_node5, linewidth=2, label='Node 5', alpha=0.8)
    ax_a.plot(results_reorg['t_before'], results_reorg['X_before'][5, :], 
             color=color_node6, linewidth=2, label='Node 6', alpha=0.8)
    
    ax_a.plot(results_reorg['t_after'], results_reorg['X_after'][3, :], 
             color=color_node4, linewidth=2, alpha=0.8)
    ax_a.plot(results_reorg['t_after'], results_reorg['X_after'][4, :], 
             color=color_node5, linewidth=2, alpha=0.8)
    ax_a.plot(results_reorg['t_after'], results_reorg['X_after'][5, :], 
             color='#999999', linewidth=2, linestyle='--', alpha=0.4)
    
    ax_a.axvline(x=intervention_time, color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax_a.text(intervention_time + 30, 75, 'Destroy Node 6\n(Redistribute)', 
             fontsize=10, color='red', fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax_a.axhline(y=33.3, color='gray', linestyle=':', linewidth=1, alpha=0.3)
    ax_a.axhline(y=50, color='gray', linestyle=':', linewidth=1, alpha=0.3)
    
    ax_a.set_xlabel('Time', fontsize=11)
    ax_a.set_ylabel('Resource Amount', fontsize=11)
    ax_a.set_title('(A) Functional Reorganization (Temporal Dynamics)', 
                   fontsize=12, fontweight='bold', pad=10)
    ax_a.legend(loc='upper right', fontsize=9)
    ax_a.grid(True, alpha=0.3)
    ax_a.set_ylim([0, 110])
    
    # Panel 3: State Space
    ax_b = fig.add_subplot(3, 3, 3, projection='3d')
    ax_b.plot(results_reorg['X_before'][3, :], 
             results_reorg['X_before'][4, :], 
             results_reorg['X_before'][5, :],
             color='blue', linewidth=2, alpha=0.7, label='Before (2D)')
    
    ax_b.plot(results_reorg['X_after'][3, :], 
             results_reorg['X_after'][4, :], 
             results_reorg['X_after'][5, :],
             color='red', linewidth=2.5, alpha=0.9, label='After (1D)')
    
    X_d = results_reorg['X_intervention']
    ax_b.scatter([X_d[3]], [X_d[4]], [X_d[5]], 
                color='red', marker='*', s=500, edgecolors='darkred', 
                linewidths=2, zorder=10, label='Destruction')
    
    ax_b.plot_trisurf([100, 0, 0], [0, 100, 0], [0, 0, 100],
                     triangles=[[0, 1, 2]], alpha=0.15, color='green')
    
    line_x = np.linspace(0, 100, 50)
    line_y = 100 - line_x
    line_z = np.zeros_like(line_x)
    ax_b.plot(line_x, line_y, line_z, 'k--', linewidth=1.5, alpha=0.4, 
             label='Collapsed attractor')
    
    ax_b.set_xlabel('Node 4', fontsize=10)
    ax_b.set_ylabel('Node 5', fontsize=10)
    ax_b.set_zlabel('Node 6', fontsize=10)
    ax_b.set_title('(B) Functional Reorganization (State Space, Resource Conserved)', 
                   fontsize=12, fontweight='bold', pad=10)
    ax_b.legend(loc='upper left', fontsize=8)
    ax_b.view_init(elev=config['view_elev'], azim=config['view_azim'])
    ax_b.set_xlim([0, 110])
    ax_b.set_ylim([0, 110])
    ax_b.set_zlim([0, 110])
    
    # ===== Row 2: Paradigm 2 (Physical Lesion) =====
    
    # Panel 4: Topology - Destruction (same as row 1)
    ax_topo2 = plt.subplot(3, 3, 4)
    if images_loaded:
        ax_topo2.imshow(img_destruction)
        ax_topo2.axis('off')
        ax_topo2.set_title('Topology Representation\n(Destruction)', 
                          fontsize=12, fontweight='bold', pad=10)
    else:
        ax_topo2.text(0.5, 0.5, 'Destruction\nTopology', 
                     ha='center', va='center', fontsize=14, fontweight='bold')
        ax_topo2.axis('off')
    
    # Panel 5: Time Series
    ax_c = plt.subplot(3, 3, 5)
    ax_c.plot(results_lesion['t_before'], results_lesion['X_before'][3, :], 
             color=color_node4, linewidth=2, label='Node 4', alpha=0.8)
    ax_c.plot(results_lesion['t_before'], results_lesion['X_before'][4, :], 
             color=color_node5, linewidth=2, label='Node 5', alpha=0.8)
    ax_c.plot(results_lesion['t_before'], results_lesion['X_before'][5, :], 
             color=color_node6, linewidth=2, label='Node 6', alpha=0.8)
    
    ax_c.plot(results_lesion['t_after'], results_lesion['X_after'][3, :], 
             color=color_node4, linewidth=2, alpha=0.8)
    ax_c.plot(results_lesion['t_after'], results_lesion['X_after'][4, :], 
             color=color_node5, linewidth=2, alpha=0.8)
    ax_c.plot(results_lesion['t_after'], results_lesion['X_after'][5, :], 
             color='#999999', linewidth=2, linestyle='--', alpha=0.4)
    
    ax_c.axvline(x=intervention_time, color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax_c.text(intervention_time + 30, 75, 'Destroy Node 6\n(Resource lost)', 
             fontsize=10, color='red', fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax_c.axhline(y=33.3, color='gray', linestyle=':', linewidth=1, alpha=0.3)
    
    ax_c.set_xlabel('Time', fontsize=11)
    ax_c.set_ylabel('Resource Amount', fontsize=11)
    ax_c.set_title('(C) Physical Lesion (Temporal Dynamics)', 
                   fontsize=12, fontweight='bold', pad=10)
    ax_c.legend(loc='upper right', fontsize=9)
    ax_c.grid(True, alpha=0.3)
    ax_c.set_ylim([0, 110])
    
    # Panel 6: State Space
    ax_d = fig.add_subplot(3, 3, 6, projection='3d')
    ax_d.plot(results_lesion['X_before'][3, :], 
             results_lesion['X_before'][4, :], 
             results_lesion['X_before'][5, :],
             color='blue', linewidth=2, alpha=0.7, label='Before (2D)')
    
    ax_d.plot(results_lesion['X_after'][3, :], 
             results_lesion['X_after'][4, :], 
             results_lesion['X_after'][5, :],
             color='red', linewidth=2.5, alpha=0.9, label='After (1D)')
    
    X_d = results_lesion['X_intervention']
    ax_d.scatter([X_d[3]], [X_d[4]], [X_d[5]], 
                color='red', marker='*', s=500, edgecolors='darkred', 
                linewidths=2, zorder=10, label='Destruction')
    
    ax_d.plot_trisurf([100, 0, 0], [0, 100, 0], [0, 0, 100],
                     triangles=[[0, 1, 2]], alpha=0.15, color='green')
    
    final_total = np.mean(results_lesion['X_after'][3, -100:] + results_lesion['X_after'][4, -100:])
    line2_x = np.linspace(0, final_total, 50)
    line2_y = final_total - line2_x
    line2_z = np.zeros_like(line2_x)
    ax_d.plot(line2_x, line2_y, line2_z, 'r--', linewidth=2, alpha=0.6, 
             label=f'Collapsed attractor (sum≈{final_total:.0f})')
    
    ax_d.plot(line_x, line_y, line_z, 'k--', linewidth=1, alpha=0.2)
    
    ax_d.set_xlabel('Node 4', fontsize=10)
    ax_d.set_ylabel('Node 5', fontsize=10)
    ax_d.set_zlabel('Node 6', fontsize=10)
    ax_d.set_title('(D) Physical Lesion (State Space, Resource Lost)', 
                   fontsize=12, fontweight='bold', pad=10)
    ax_d.legend(loc='upper left', fontsize=8)
    ax_d.view_init(elev=config['view_elev'], azim=config['view_azim'])
    ax_d.set_xlim([0, 110])
    ax_d.set_ylim([0, 110])
    ax_d.set_zlim([0, 110])
    
    # ===== Row 3: Paradigm 3 (Functional Integration) =====
    
    # Panel 7: Topology - Fusion
    ax_topo3 = plt.subplot(3, 3, 7)
    if images_loaded:
        ax_topo3.imshow(img_fusion)
        ax_topo3.axis('off')
        ax_topo3.set_title('Topology Representation\n(Fusion)', 
                          fontsize=12, fontweight='bold', pad=10)
    else:
        ax_topo3.text(0.5, 0.5, 'Fusion\nTopology', 
                     ha='center', va='center', fontsize=14, fontweight='bold')
        ax_topo3.axis('off')
    
    # Panel 8: Time Series
    ax_e = plt.subplot(3, 3, 8)
    ax_e.plot(results_fusion['t_before'], results_fusion['X_before'][3, :], 
             color=color_node4, linewidth=2, label='Node 4', alpha=0.8)
    ax_e.plot(results_fusion['t_before'], results_fusion['X_before'][4, :], 
             color=color_node5, linewidth=2, label='Node 5', alpha=0.8)
    ax_e.plot(results_fusion['t_before'], results_fusion['X_before'][5, :], 
             color=color_node6, linewidth=2, label='Node 6', alpha=0.8)
    
    ax_e.plot(results_fusion['t_after'], results_fusion['X_after'][3, :], 
             color=color_node4, linewidth=2, alpha=0.8)
    ax_e.plot(results_fusion['t_after'], results_fusion['X_after'][4, :], 
             color=color_node5, linewidth=2, alpha=0.8)
    ax_e.plot(results_fusion['t_after'], results_fusion['X_after'][5, :], 
             color=color_node6, linewidth=2, alpha=0.8)
    
    ax_e.axvline(x=intervention_time, color='purple', linestyle='--', linewidth=2, alpha=0.7)
    ax_e.text(intervention_time + 30, 75, 'Fusion 5 & 6', 
             fontsize=10, color='purple', fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax_e.axhline(y=33.3, color='gray', linestyle=':', linewidth=1, alpha=0.3)
    ax_e.axhline(y=50, color='gray', linestyle=':', linewidth=1, alpha=0.3)
    ax_e.axhline(y=66.7, color='gray', linestyle=':', linewidth=1, alpha=0.3)
    
    ax_e.set_xlabel('Time', fontsize=11)
    ax_e.set_ylabel('Resource Amount', fontsize=11)
    ax_e.set_title('(E) Functional Integration (Temporal Dynamics)', 
                   fontsize=12, fontweight='bold', pad=10)
    ax_e.legend(loc='upper right', fontsize=8)
    ax_e.grid(True, alpha=0.3)
    ax_e.set_ylim([0, 110])
    
    # Panel 9: State Space
    ax_f = fig.add_subplot(3, 3, 9, projection='3d')
    ax_f.plot(results_fusion['X_before'][3, :], 
             results_fusion['X_before'][4, :], 
             results_fusion['X_before'][5, :],
             color='blue', linewidth=2, alpha=0.7, label='Before (2D)')
    
    n_skip = 5
    ax_f.plot(results_fusion['X_after'][3, ::n_skip], 
             results_fusion['X_after'][4, ::n_skip], 
             results_fusion['X_after'][5, ::n_skip],
             color='red', linewidth=1.5, alpha=0.8, label='After (1D fusion)', marker='.', markersize=2)
    
    X_f = results_fusion['X_intervention']
    ax_f.scatter([X_f[3]], [X_f[4]], [X_f[5]], 
                color='purple', marker='*', s=500, edgecolors='darkmagenta', 
                linewidths=2, zorder=10, label='Fusion point')
    
    ax_f.plot_trisurf([100, 0, 0], [0, 100, 0], [0, 0, 100],
                     triangles=[[0, 1, 2]], alpha=0.15, color='green')
    
    t_param = np.linspace(0, 50, 100)
    fusion_line_x4 = 100 - 2 * t_param
    fusion_line_x5 = t_param
    fusion_line_x6 = t_param
    ax_f.plot(fusion_line_x4, fusion_line_x5, fusion_line_x6, 
             'purple', linewidth=2.5, alpha=0.7, linestyle='--',
             label='Fusion attractor (X5≈X6)', zorder=5)
    
    ax_f.plot(line_x, line_y, line_z, 'k--', linewidth=1, alpha=0.2)
    
    ax_f.set_xlabel('Node 4', fontsize=10)
    ax_f.set_ylabel('Node 5', fontsize=10)
    ax_f.set_zlabel('Node 6', fontsize=10)
    ax_f.set_title('(F) Functional Integration (State Space, N_eff reduced)', 
                   fontsize=12, fontweight='bold', pad=10)
    ax_f.legend(loc='upper left', fontsize=7)
    ax_f.view_init(elev=config['view_elev'], azim=config['view_azim'])
    ax_f.set_xlim([0, 110])
    ax_f.set_ylim([0, 110])
    ax_f.set_zlim([0, 110])
    
    # Main title
    fig.suptitle('Structural Control of Attractor Dimension: Destruction vs. Fusion', 
                 fontsize=18, fontweight='bold', y=0.995)
    
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    filename = 'structural_control_3x3_comparison.png'
    plt.savefig(filename, dpi=config['dpi'], bbox_inches='tight')
    print(f"\n✓ Figure saved as {filename}")
    plt.show()


def print_comparative_analysis(results_reorg, results_lesion, results_fusion):
    """Print comparative analysis of all three paradigms"""
    print("\n" + "="*80)
    print("COMPARATIVE ANALYSIS: THREE PARADIGMS".center(80))
    print("="*80)
    
    # Paradigm 1
    X_reorg_before = results_reorg['X_before'][:, -100:]
    X_reorg_after = results_reorg['X_after'][:, -100:]
    total_reorg_before = np.mean(np.sum(X_reorg_before, axis=0))
    total_reorg_after = np.mean(np.sum(X_reorg_after, axis=0))
    
    print("\n【Paradigm 1: Functional Reorganization】")
    print(f"  Mechanism:       Resource redistribution (Node6 → Node4+5)")
    print(f"  Before collapse: Total = {total_reorg_before:.2f}")
    print(f"  After collapse:  Total = {total_reorg_after:.2f}")
    print(f"  Resource change: {total_reorg_after - total_reorg_before:+.2f} (conserved)")
    print(f"  Attractor:       2D plane → 1D line (edge of original plane)")
    print(f"  Constraint:      Node4 + Node5 = 100, Node6 = 0")
    print(f"  N_effective:     3 → 2")
    
    # Paradigm 2
    X_lesion_before = results_lesion['X_before'][:, -100:]
    X_lesion_after = results_lesion['X_after'][:, -100:]
    total_lesion_before = np.mean(np.sum(X_lesion_before, axis=0))
    total_lesion_after = np.mean(np.sum(X_lesion_after, axis=0))
    
    print("\n【Paradigm 2: Physical Lesion】")
    print(f"  Mechanism:       Resource vanishing (Node6 → void)")
    print(f"  Before collapse: Total = {total_lesion_before:.2f}")
    print(f"  After collapse:  Total = {total_lesion_after:.2f}")
    print(f"  Resource change: {total_lesion_after - total_lesion_before:+.2f} (net loss)")
    print(f"  Attractor:       2D plane → 1D line (parallel, inside)")
    print(f"  Constraint:      Node4 + Node5 ≈ {total_lesion_after:.0f}, Node6 = 0")
    print(f"  N_effective:     3 → 2")
    
    # Paradigm 3
    X_fusion_before = results_fusion['X_before'][:, -100:]
    X_fusion_after = results_fusion['X_after'][:, -100:]
    total_fusion_before = np.mean(np.sum(X_fusion_before, axis=0))
    total_fusion_after = np.mean(np.sum(X_fusion_after, axis=0))
    
    ratio_56 = np.mean(results_fusion['X_after'][4, -100:] / (results_fusion['X_after'][5, -100:] + 1e-6))
    
    print("\n【Paradigm 3: Functional Integration】")
    print(f"  Mechanism:       Strong bidirectional sharing (Node5 ↔ Node6)")
    print(f"  Before fusion:   Total = {total_fusion_before:.2f}")
    print(f"  After fusion:    Total = {total_fusion_after:.2f}")
    print(f"  Resource change: {total_fusion_after - total_fusion_before:+.2f} (conserved)")
    print(f"  Attractor:       2D plane → 1D line (diagonal, inside)")
    print(f"  Constraint:      Node5 ≈ Node6 (ratio ≈ {ratio_56:.2f})")
    print(f"  Effective state: Node4 vs (Node5+Node6)")
    print(f"  N_effective:     3 → 2")
    
    print("\n" + "─"*80)
    print("KEY DIFFERENCES:")
    print("  • Paradigm 1 (Reorganization): Collapse to EDGE - Node6 eliminated")
    print("  • Paradigm 2 (Lesion):         Collapse to PARALLEL LINE - resource reduced")
    print("  • Paradigm 3 (Fusion):         Collapse to DIAGONAL LINE - nodes synchronized")
    print("\n  All three achieve dimension reduction (3D → 1D effective dynamics),")
    print("  but through fundamentally different structural mechanisms:")
    print("    → Destruction + redistribution")
    print("    → Destruction + loss")
    print("    → Integration + synchronization")
    print("="*80 + "\n")


# ==================== 主程序 ====================
if __name__ == "__main__":
    config = EXPERIMENT_CONFIG
    
    print("\n" + "="*80)
    print("STRUCTURAL CONTROL: THREE PARADIGMS COMPARISON (3x3 Layout)".center(80))
    print("="*80)
    
    # 生成统一的噪声序列
    n_steps = int(config['t_end'] / config['dt'])
    noise_seq, seed = generate_noise_sequence(n_steps, config['noise_strength'], n_nodes=3)
    
    print("\n【Experimental Design】")
    print(f"  Random seed:     {seed} (generated from current time)")
    print(f"  Intervention at: t = {config['intervention_time']}")
    print(f"  Noise strength:  {config['noise_strength']}")
    print(f"  Total steps:     {n_steps}")
    print(f"  ⚠ Using unified noise sequence for all three paradigms")
    print(f"  ⚠ Destroyed node's noise redistributed proportionally")
    
    print("\n【Three Paradigms】")
    print("  Paradigm 1: Functional Reorganization")
    print("  Paradigm 2: Physical Lesion")
    print("  Paradigm 3: Functional Integration")
    
    # Run all three paradigms with the same noise sequence
    print("\n" + "─"*80)
    print("Running Paradigm 1 (Functional Reorganization)...")
    results_reorg = run_paradigm_experiment(config, paradigm='reorganization', noise_seq=noise_seq)
    
    print("\n" + "─"*80)
    print("Running Paradigm 2 (Physical Lesion)...")
    results_lesion = run_paradigm_experiment(config, paradigm='lesion', noise_seq=noise_seq)
    
    print("\n" + "─"*80)
    print("Running Paradigm 3 (Functional Integration)...")
    results_fusion = run_paradigm_experiment(config, paradigm='fusion', noise_seq=noise_seq)
    
    # Comparative analysis
    print_comparative_analysis(results_reorg, results_lesion, results_fusion)
    
    # Visualization
    print("Generating 3x3 nine-panel comparison figure with topology images...")
    plot_nine_panel_comparison(results_reorg, results_lesion, results_fusion, config)
    
    print("="*80)
    print("EXPERIMENT COMPLETE".center(80))
    print("="*80)
    print("\nThis nine-panel figure demonstrates:")
    print("  • Topology representations in the first column")
    print("  • Three distinct mechanisms of dimension collapse")
    print("  • Destruction (with/without resource conservation) vs. Fusion")
    print("  • Different geometric manifestations in state space")
    print("  • Unified noise ensures fair comparison across paradigms")
    print("  • Destroyed node's noise redistributed by resource proportion")
    print("="*80 + "\n")
