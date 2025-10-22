# -*- coding: utf-8 -*-
"""
Created on Wed Oct 22 21:31:45 2025

@author: 奶宝
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Oct 21 19:09:00 2025

@author: MLTZ
Modified to show multi-view only for green camp
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
import pickle
import os

# 设置Nature风格参数
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 9
plt.rcParams['axes.linewidth'] = 0.6
plt.rcParams['xtick.major.width'] = 0.6
plt.rcParams['ytick.major.width'] = 0.6

CACHE_FILE = 'camp_dynamics_cache.pkl'

class CampDynamics:
    def __init__(self, plunder_strength=0.0015, total_resource=100, n_nodes=6):
        self.plunder_strength = plunder_strength
        self.total_resource = total_resource
        self.n_nodes = n_nodes
        
        self.node_map = {
            0: {'camp': 0, 'name': 'blue'},
            1: {'camp': 1, 'name': 'orange'},
            2: {'camp': 1, 'name': 'orange'},
            3: {'camp': 2, 'name': 'green'},
            4: {'camp': 2, 'name': 'green'},
            5: {'camp': 2, 'name': 'green'}
        }
        
        self.colors = {'blue': '#3498db', 'orange': '#e67e22', 'green': '#2ecc71'}
        self.W = self._build_interaction_matrix()
    
    def _build_interaction_matrix(self):
        W = np.zeros((self.n_nodes, self.n_nodes))
        for i in range(self.n_nodes):
            for j in range(self.n_nodes):
                if self.node_map[i]['camp'] != self.node_map[j]['camp']:
                    W[i, j] = self.plunder_strength
        return W
    
    def dynamics(self, X):
        N = len(X)
        dXdt = np.zeros(N)
        
        for j in range(N):
            plunderers = [i for i in range(N) if self.W[i, j] > 0]
            if len(plunderers) == 0:
                continue
            
            total_demand = sum(self.W[i, j] * X[i] for i in plunderers)
            
            if total_demand > 0:
                max_outflow = X[j] / 0.01
                actual_outflow = min(total_demand, max_outflow)
                dXdt[j] -= actual_outflow
                
                for i in plunderers:
                    share = self.W[i, j] * X[i] / total_demand
                    dXdt[i] += share * actual_outflow
        
        return dXdt
    
    def dynamics_with_noise(self, X, noise_strength):
        dXdt = self.dynamics(X)
        noise = np.random.uniform(-1, 1, len(X))
        noise = noise - noise.mean()
        dXdt += noise * noise_strength
        return dXdt
    
    def simulate(self, X0, t_span, n_points):
        dt = (t_span[1] - t_span[0]) / n_points
        t = np.linspace(t_span[0], t_span[1], n_points)
        X_traj = np.zeros((n_points, self.n_nodes))
        
        X = np.array(X0, dtype=float)
        
        for i in range(n_points):
            X_traj[i] = X.copy()
            dXdt = self.dynamics(X)
            X = np.maximum(0, X + dXdt * dt)
            
            total = X.sum()
            if total > 0:
                X = X * self.total_resource / total
        
        return t, X_traj
    
    def random_walk_on_attractor(self, X0, n_steps=3000, noise_strength=1.5):
        trajectory = np.zeros((n_steps, self.n_nodes))
        dt = 0.1
        X = np.array(X0, dtype=float)
        
        for i in range(n_steps):
            trajectory[i] = X.copy()
            dXdt = self.dynamics_with_noise(X, noise_strength)
            X = np.maximum(0, X + dXdt * dt)
            
            total = X.sum()
            if total > 0:
                X = X * self.total_resource / total
        
        return trajectory
    
    def sample_attractor(self, camp_name):
        if camp_name == 'blue':
            return [np.array([100, 0, 0, 0, 0, 0])]
        elif camp_name == 'orange':
            samples = []
            for i in range(7):
                a = (i / 6) * 100
                samples.append(np.array([0, a, 100 - a, 0, 0, 0]))
            return samples
        else:
            points = [[100, 0, 0], [0, 100, 0], [0, 0, 100],
                     [50, 50, 0], [50, 0, 50], [0, 50, 50],
                     [33.3, 33.3, 33.4], [66.7, 16.7, 16.6], [16.7, 66.7, 16.6]]
            samples = []
            for p in points:
                samples.append(np.array([0, 0, 0] + p))
            return samples
    
    def sqrt_transform_with_signs(self, points):
        transformed = np.sqrt(np.abs(points))
        signs = np.random.choice([-1, 1], size=transformed.shape)
        return transformed * signs


def create_sphere_wireframe(radius, resolution=20):
    u = np.linspace(0, 2 * np.pi, resolution)
    v = np.linspace(0, np.pi, resolution)
    x = radius * np.outer(np.cos(u), np.sin(v))
    y = radius * np.outer(np.sin(u), np.sin(v))
    z = radius * np.outer(np.ones(np.size(u)), np.cos(v))
    return x, y, z


def compute_depth_alpha(points, elev, azim, alpha_range=(0.05, 0.4)):
    elev_rad = np.radians(elev)
    azim_rad = np.radians(azim)
    
    view_dir = np.array([
        np.cos(elev_rad) * np.cos(azim_rad),
        np.cos(elev_rad) * np.sin(azim_rad),
        np.sin(elev_rad)
    ])
    
    depths = np.dot(points, view_dir)
    depth_min, depth_max = depths.min(), depths.max()
    
    if depth_max > depth_min:
        normalized_depths = (depths - depth_min) / (depth_max - depth_min)
    else:
        normalized_depths = np.ones_like(depths) * 0.5
    
    alphas = alpha_range[0] + normalized_depths * (alpha_range[1] - alpha_range[0])
    return alphas


def generate_or_load_data(force_regenerate=False):
    if not force_regenerate and os.path.exists(CACHE_FILE):
        print(f"✓ 发现缓存文件，正在加载...")
        with open(CACHE_FILE, 'rb') as f:
            data = pickle.load(f)
        print("✓ 数据加载完成!")
        return data
    
    print("⚙ 开始生成模拟数据...")
    np.random.seed(42)
    system = CampDynamics()
    
    camps = {
        'blue': {'X0': [40, 13, 17, 5, 10, 15], 'time': 300, 'noise': 0.5,
                'nodes': [0, 1, 2], 'labels': ['Node 3\n(Blue)', 'Node 1\n(Orange)', 'Node 2\n(Orange)']},
        'orange': {'X0': [10, 18, 22, 15, 10, 25], 'time': 1000, 'noise': 1.5,
                  'nodes': [0, 1, 2], 'labels': ['Node 3\n(Blue)', 'Node 1\n(Orange)', 'Node 2\n(Orange)']},
        'green': {'X0': [10, 15, 5, 20, 23, 27], 'time': 300, 'noise': 2.0,
                 'nodes': [3, 4, 5], 'labels': ['Node 4\n(Green)', 'Node 5\n(Green)', 'Node 6\n(Green)']}
    }
    
    data = {'system': system, 'camps': camps}
    
    for camp_name, config in camps.items():
        print(f"  - 模拟{camp_name}阵营...")
        t, X_traj = system.simulate(config['X0'], [0, config['time']], 500)
        attractor_samples = system.sample_attractor(camp_name)
        all_points = []
        for sample in attractor_samples:
            traj = system.random_walk_on_attractor(sample, n_steps=3000, noise_strength=config['noise'])
            all_points.append(traj)
        all_points = np.vstack(all_points)
        data[camp_name] = {'t': t, 'X_traj': X_traj, 'all_points': all_points}
    
    print(f"✓ 保存数据到缓存...")
    with open(CACHE_FILE, 'wb') as f:
        pickle.dump(data, f)
    print("✓ 数据生成完成!")
    
    return data


def create_nature_figure(force_regenerate=False):
    data = generate_or_load_data(force_regenerate)
    system = data['system']
    camps = data['camps']
    
    print("\n⚙ 开始绘制图像...")
    
    # 创建图像 (4行3列)
    fig = plt.figure(figsize=(10.8, 12))
    gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.45, wspace=0.35,
                          left=0.06, right=0.98, top=0.98, bottom=0.04,
                          height_ratios=[1, 1, 1, 1])  # 调整第0行高度为1
    
    # ==== 第0行：拓扑图 ====
    # 如果想让图更宽，可以跨多列：gs[0, :2] 占2列，gs[0, :] 占3列
    ax_topo = fig.add_subplot(gs[0, :1])  # 占据前两列，更宽
    try:
        img = plt.imread('TUOPU.jpeg')
        ax_topo.imshow(img, aspect='auto')
        ax_topo.axis('off')
        ax_topo.set_title('(A)  Network Topology', fontsize=11, fontweight='bold', loc='left', pad=10)
        print("✓ 成功加载 TUOPU.jpeg")
    except FileNotFoundError:
        ax_topo.text(0.5, 0.5, 'TUOPU.jpeg not found', 
                    ha='center', va='center', fontsize=12, color='red')
        ax_topo.set_xlim([0, 1])
        ax_topo.set_ylim([0, 1])
        ax_topo.axis('off')
        print("⚠ 警告: 未找到 TUOPU.jpeg")
    
    # ==== 第0行第2格：图例 ====
    ax_legend = fig.add_subplot(gs[0, 1])
    ax_legend.axis('off')
    
    # 添加图例
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    
    legend_elements = [
        Patch(facecolor='#3498db', label='Blue Camp (Node 3)'),
        Patch(facecolor='#e67e22', label='Orange Camp (Nodes 1,2)'),
        Patch(facecolor='#2ecc71', label='Green Camp (Nodes 4,5,6)'),
        Line2D([0], [0], color='red', linewidth=2, label='Plunder Arrow'),
    ]
    
    legend = ax_legend.legend(handles=legend_elements, loc='center', 
                             ncol=1, fontsize=9,
                             frameon=True, fancybox=True, shadow=True,
                             title='Legend', title_fontsize=10)
    
    # ==== 三个阵营 ====
    camp_configs = [
        ('blue', 1, '(B)', 'Zero-dimensional Attractor (Point)'),
        ('orange', 2, '(C)', 'One-dimensional Attractor (Line or Ring)'),
        ('green', 3, '(D)', 'Two-dimensional plane attractor (Plane or Sphere)')
    ]
    
    for camp_name, row_idx, label, title in camp_configs:
        config = camps[camp_name]
        camp_data = data[camp_name]
        
        # (i) 时间序列
        ax_time = fig.add_subplot(gs[row_idx, 0])
        
        # 用于收集图例的handles和labels
        legend_lines = []
        legend_labels = []
        added_camps = set()
        
        for i in range(6):
            node_camp = system.node_map[i]['name']
            is_winner = node_camp == camp_name
            color = system.colors[node_camp]
            alpha = 1.0 if is_winner else 0.25
            lw = 1.5 if is_winner else 0.8
            line, = ax_time.plot(camp_data['t'], camp_data['X_traj'][:, i], 
                        color=color, alpha=alpha, linewidth=lw)
            
            # 为每个阵营添加一个图例条目（避免重复）
            if node_camp not in added_camps:
                legend_lines.append(line)
                camp_label = f'{node_camp.capitalize()} Camp'
                if node_camp == 'blue':
                    camp_label += ' (Node 0)'
                elif node_camp == 'orange':
                    camp_label += ' (Nodes 1,2)'
                elif node_camp == 'green':
                    camp_label += ' (Nodes 3,4,5)'
                legend_labels.append(camp_label)
                added_camps.add(node_camp)
        
        ax_time.set_xlabel('Time', fontsize=9)
        ax_time.set_ylabel('Resource amount', fontsize=9)
        ax_time.set_ylim([0, 105])
        ax_time.set_title(f'{label}  {title}\n(1)', 
                         fontsize=10, fontweight='bold', loc='left', pad=10)
        ax_time.grid(True, alpha=0.2, linewidth=0.5)
        
        # 蓝色阵营单独调整图例位置
        if camp_name == 'blue':
            ax_time.legend(legend_lines, legend_labels, loc='center right', 
                          fontsize=7, framealpha=0.9)
        else:
            ax_time.legend(legend_lines, legend_labels, loc='upper right', 
                          fontsize=7, framealpha=0.9)
        
        ax_time.spines['top'].set_visible(False)
        ax_time.spines['right'].set_visible(False)
        ax_time.tick_params(labelsize=8)
        
        # (ii) 几何图 - 根据阵营选择布局
        nodes = config['nodes']
        x = camp_data['all_points'][:, nodes[0]]
        y = camp_data['all_points'][:, nodes[1]]
        z = camp_data['all_points'][:, nodes[2]]
        vertices = np.array([[100, 0, 0], [0, 100, 0], [0, 0, 100], [100, 0, 0]])
        
        if camp_name == 'green':
            # 绿色阵营：一大两小布局
            gs_geom = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=gs[row_idx, 1],
                                                        hspace=0.25, wspace=0.25,
                                                        height_ratios=[1, 1], width_ratios=[1.8, 1])
            ax_geom_main = fig.add_subplot(gs_geom[:, 0], projection='3d')
        else:
            # 蓝色和橙色阵营：单幅大图
            ax_geom_main = fig.add_subplot(gs[row_idx, 1], projection='3d')
        ax_geom_main.scatter(x, y, z, c=system.colors[camp_name], alpha=0.15, s=4)
        
        # 只有绿色阵营画参考线
        if camp_name == 'green':
            ax_geom_main.plot(vertices[:, 0], vertices[:, 1], vertices[:, 2],
                             color=system.colors[camp_name], linewidth=1.5, linestyle='--', alpha=0.6)
        
        ax_geom_main.set_xlabel(config['labels'][0], fontsize=7, labelpad=4)
        ax_geom_main.set_ylabel(config['labels'][1], fontsize=7, labelpad=4)
        ax_geom_main.set_zlabel(config['labels'][2], fontsize=7, labelpad=4)
        ax_geom_main.set_xlim([0, 105])
        ax_geom_main.set_ylim([0, 105])
        ax_geom_main.set_zlim([0, 105])
        ax_geom_main.set_title('(2) ', fontsize=8, fontweight='bold', loc='left', pad=5)
        ax_geom_main.tick_params(labelsize=6, pad=0.5)
        ax_geom_main.view_init(elev=20, azim=45)
        for axis in [ax_geom_main.xaxis, ax_geom_main.yaxis, ax_geom_main.zaxis]:
            axis.line.set_linewidth(0.4)
        
        # 只有绿色阵营显示俯视图和侧视图
        if camp_name == 'green':
            # 俯视图
            ax_geom_top = fig.add_subplot(gs_geom[0, 1], projection='3d')
            ax_geom_top.scatter(x, y, z, c=system.colors[camp_name], alpha=0.15, s=1)
            ax_geom_top.plot(vertices[:, 0], vertices[:, 1], vertices[:, 2],
                           color=system.colors[camp_name], linewidth=1, linestyle='--', alpha=0.5)
            ax_geom_top.set_xlim([0, 105])
            ax_geom_top.set_ylim([0, 105])
            ax_geom_top.set_zlim([0, 105])
            ax_geom_top.set_xticks([])
            ax_geom_top.set_yticks([])
            ax_geom_top.set_zticks([])
            ax_geom_top.set_title('Top', fontsize=7, fontweight='bold')
            ax_geom_top.view_init(elev=89, azim=0)
            for axis in [ax_geom_top.xaxis, ax_geom_top.yaxis, ax_geom_top.zaxis]:
                axis.line.set_linewidth(0.3)
            
            # 侧视图
            ax_geom_side = fig.add_subplot(gs_geom[1, 1], projection='3d')
            ax_geom_side.scatter(x, y, z, c=system.colors[camp_name], alpha=0.15, s=1)
            ax_geom_side.plot(vertices[:, 0], vertices[:, 1], vertices[:, 2],
                            color=system.colors[camp_name], linewidth=1, linestyle='--', alpha=0.5)
            ax_geom_side.set_xlim([0, 105])
            ax_geom_side.set_ylim([0, 105])
            ax_geom_side.set_zlim([0, 105])
            ax_geom_side.set_xticks([])
            ax_geom_side.set_yticks([])
            ax_geom_side.set_zticks([])
            ax_geom_side.set_title('Side', fontsize=7, fontweight='bold')
            ax_geom_side.view_init(elev=130, azim=45)
            for axis in [ax_geom_side.xaxis, ax_geom_side.yaxis, ax_geom_side.zaxis]:
                axis.line.set_linewidth(0.3)
        
        # (iii) PCA - 根据阵营选择布局
        transformed_data = system.sqrt_transform_with_signs(camp_data['all_points'])
        pca = PCA(n_components=3)
        pca_result = pca.fit_transform(transformed_data)
        
        distances = np.sqrt(np.sum(pca_result**2, axis=1))
        avg_radius = np.mean(distances)
        sphere_x, sphere_y, sphere_z = create_sphere_wireframe(avg_radius, resolution=15)
        
        if camp_name == 'green':
            # 绿色阵营：一大两小布局
            gs_pca = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=gs[row_idx, 2],
                                                       hspace=0.25, wspace=0.25,
                                                       height_ratios=[1, 1], width_ratios=[1.8, 1])
            ax_pca_main = fig.add_subplot(gs_pca[:, 0], projection='3d')
        else:
            # 蓝色和橙色阵营：单幅大图
            ax_pca_main = fig.add_subplot(gs[row_idx, 2], projection='3d')
        ax_pca_main.plot_wireframe(sphere_x, sphere_y, sphere_z, 
                                   color='gray', alpha=0.12, linewidth=0.25, 
                                   linestyle='--', rcount=8, ccount=8)
        
        elev_angle, azim_angle = 25, 45
        alphas = compute_depth_alpha(pca_result, elev_angle, azim_angle, alpha_range=(0.05, 0.5))
        from matplotlib.colors import to_rgba
        base_color = to_rgba(system.colors[camp_name])
        colors = np.array([list(base_color[:3]) + [alpha] for alpha in alphas])
        
        ax_pca_main.scatter(pca_result[:, 0], pca_result[:, 1], pca_result[:, 2],
                           c=colors, s=3)
        
        lim = max(abs(pca_result).max(), avg_radius * 1.2)
        ax_pca_main.set_xlim([-lim, lim])
        ax_pca_main.set_ylim([-lim, lim])
        ax_pca_main.set_zlim([-lim, lim])
        ax_pca_main.set_xlabel('PC1', fontsize=7, labelpad=4)
        ax_pca_main.set_ylabel('PC2', fontsize=7, labelpad=4)
        ax_pca_main.set_zlabel('PC3', fontsize=7, labelpad=4)
        ax_pca_main.set_title('(3)', fontsize=8, fontweight='bold', loc='left', pad=5)
        ax_pca_main.tick_params(labelsize=6, pad=0.5)
        ax_pca_main.view_init(elev=elev_angle, azim=azim_angle)
        for axis in [ax_pca_main.xaxis, ax_pca_main.yaxis, ax_pca_main.zaxis]:
            axis.line.set_linewidth(0.4)
        
        # 只有绿色阵营显示额外的两个视角
        if camp_name == 'green':
            # PCA视角2
            ax_pca_v2 = fig.add_subplot(gs_pca[0, 1], projection='3d')
            ax_pca_v2.plot_wireframe(sphere_x, sphere_y, sphere_z, 
                                    color='gray', alpha=0.1, linewidth=0.2, 
                                    linestyle='--', rcount=6, ccount=6)
            alphas2 = compute_depth_alpha(pca_result, 85, 0, alpha_range=(0.05, 0.5))
            colors2 = np.array([list(base_color[:3]) + [alpha] for alpha in alphas2])
            ax_pca_v2.scatter(pca_result[:, 0], pca_result[:, 1], pca_result[:, 2],
                             c=colors2, s=1.5)
            ax_pca_v2.set_xlim([-lim, lim])
            ax_pca_v2.set_ylim([-lim, lim])
            ax_pca_v2.set_zlim([-lim, lim])
            ax_pca_v2.set_xticks([])
            ax_pca_v2.set_yticks([])
            ax_pca_v2.set_zticks([])
            ax_pca_v2.set_title('Top', fontsize=7, fontweight='bold')
            ax_pca_v2.view_init(elev=85, azim=0)
            for axis in [ax_pca_v2.xaxis, ax_pca_v2.yaxis, ax_pca_v2.zaxis]:
                axis.line.set_linewidth(0.3)
            
            # PCA视角3
            ax_pca_v3 = fig.add_subplot(gs_pca[1, 1], projection='3d')
            ax_pca_v3.plot_wireframe(sphere_x, sphere_y, sphere_z, 
                                    color='gray', alpha=0.1, linewidth=0.2, 
                                    linestyle='--', rcount=6, ccount=6)
            alphas3 = compute_depth_alpha(pca_result, 5, 90, alpha_range=(0.05, 0.5))
            colors3 = np.array([list(base_color[:3]) + [alpha] for alpha in alphas3])
            ax_pca_v3.scatter(pca_result[:, 0], pca_result[:, 1], pca_result[:, 2],
                             c=colors3, s=1.5)
            ax_pca_v3.set_xlim([-lim, lim])
            ax_pca_v3.set_ylim([-lim, lim])
            ax_pca_v3.set_zlim([-lim, lim])
            ax_pca_v3.set_xticks([])
            ax_pca_v3.set_yticks([])
            ax_pca_v3.set_zticks([])
            ax_pca_v3.set_title('Side', fontsize=7, fontweight='bold')
            ax_pca_v3.view_init(elev=130, azim=45)
            for axis in [ax_pca_v3.xaxis, ax_pca_v3.yaxis, ax_pca_v3.zaxis]:
                axis.line.set_linewidth(0.3)
    
    # 保存
    print("\n⚙ 保存图像...")
    plt.savefig('camp_dynamics_modified.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    #plt.savefig('camp_dynamics_modified.pdf', bbox_inches='tight', 
                #facecolor='white', edgecolor='none')
    
    print("\n✓ 图像已生成!")
    print("  - camp_dynamics_modified.png (300 DPI)")
    print("  - camp_dynamics_modified.pdf (矢量)")
    
    plt.show()


if __name__ == "__main__":
    import sys
    force_regenerate = '--regenerate' in sys.argv or '-r' in sys.argv
    
    print("=" * 70)
    print("生成Nature风格科学图像 - 修改版")
    print("=" * 70)
    create_nature_figure(force_regenerate)
    print("=" * 70)
    print("✓ 完成!")
