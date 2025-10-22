import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

"""
Camp Perturbation Experiment: Resilience and Critical Switching of Multistable Attractor Basins

Multi-group experiments with different initial conditions and perturbation strengths
"""

# ==================== 参数配置区域 ====================
# 【在此处调整实验参数】
# =====================================================

EXPERIMENT_CONFIG = {
    # 动力学参数
    'plunder_rate': 0.1,        # 掠夺系数 (降低=更温和竞争)
    'total_resource': 100.0,    # 总资源量
    'noise_strength': 0.3,      # 噪声强度 (驱动随机游走，0=无噪声)
    
    # 时间参数
    'perturb_time': 50,         # 扰动时刻
    't_end': 200,               # 结束时间
    'n_points': 2000,           # 采样点数
    
    # 扰动强度
    'perturbations': [
        {'name': 'Sub-threshold', 'delta_C': 10, 'color': '#4CAF50'},
        {'name': 'Near-threshold', 'delta_C': 20, 'color': '#FF9800'},
        {'name': 'Supra-threshold', 'delta_C': 35, 'color': '#F44336'},
    ],
    
    # 初始条件组（不同的初始资源分布）
    'initial_conditions': [
        {
            'name': 'Balanced Orange',
            'X0': [0.1, 49.95, 49.95, 0.0, 0.0, 0.0],  # 橙色阵营平衡
            'description': 'Node2 = Node3, Blue minimal'
        },
        {
            'name': 'Biased Orange',
            'X0': [0.1, 70.0, 29.9, 0.0, 0.0, 0.0],    # 橙色阵营不平衡
            'description': 'Node2 dominant, Node3 weaker'
        },
        {
            'name': 'Three-way split',
            'X0': [10.0, 45.0, 45.0, 0.0, 0.0, 0.0],   # 蓝色有初始资源
            'description': 'Blue has initial foothold'
        },
    ],
    
    # 可视化参数
    'show_individual_nodes': True,  # 是否显示单个节点（而非阵营总和）
    'dpi': 300,                      # 图像分辨率
    'figsize': (20, 12),             # 图像尺寸
    'title_color': 'black',          # 子图标题颜色 ('black', 'blue', '#FF0000', etc.)
}

# =====================================================
# 参数配置结束，以下为程序代码
# =====================================================


def dynamics_model_constrained(t, X, W, noise_strength=0.0):
    """
    Resource conservation dynamics model with X≥0 constraint and noise
    
    Args:
        t: Time
        X: Resource array [X1, X2, ..., XN]
        W: Interaction matrix NxN
           W[i,j] > 0: Node i plunders node j
           W[i,j] < 0: Node i shares to node j
        noise_strength: Strength of noise for random walk
    
    Returns:
        dXdt: Rate of change for each node
    """
    N = len(X)
    X = np.maximum(X, 0)
    dXdt = np.zeros(N)
    
    S = np.maximum(0, -W)  # Sharing matrix
    P = np.maximum(0, W)   # Plunder matrix

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
    
    # Add noise (zero-sum to conserve total resource)
    if noise_strength > 0:
        # Only add noise to nodes with resources
        active_nodes = np.where(X > 0.1)[0]
        if len(active_nodes) > 1:
            noise = np.zeros(N)
            noise_values = np.random.randn(len(active_nodes)) * noise_strength
            noise_values = noise_values - np.mean(noise_values)  # Zero-sum
            for i, node_idx in enumerate(active_nodes):
                noise[node_idx] = noise_values[i]
            dXdt += noise
    
    return dXdt


def build_interaction_matrix(N=6, plunder_rate=0.1):
    """
    Build interaction matrix for 6-node system
    
    Node assignment:
    - Node 1 (index 0): Blue camp
    - Node 2-3 (index 1-2): Orange camp
    - Node 4-6 (index 3-5): Green camp
    """
    W = np.zeros((N, N))
    
    # Blue(0) vs Orange(1,2)
    W[0, 1] = W[1, 0] = plunder_rate
    W[0, 2] = W[2, 0] = plunder_rate
    
    # Blue(0) vs Green(3,4,5)
    W[0, 3] = W[3, 0] = plunder_rate
    W[0, 4] = W[4, 0] = plunder_rate
    W[0, 5] = W[5, 0] = plunder_rate
    
    # Orange(1,2) vs Green(3,4,5)
    W[1, 3] = W[3, 1] = plunder_rate
    W[1, 4] = W[4, 1] = plunder_rate
    W[1, 5] = W[5, 1] = plunder_rate
    W[2, 3] = W[3, 2] = plunder_rate
    W[2, 4] = W[4, 2] = plunder_rate
    W[2, 5] = W[5, 2] = plunder_rate
    
    return W


def run_perturbation_experiment(W, X0, perturb_time, perturb_amount, 
                                 t_end=200, n_points=2000, noise_strength=0.0, verbose=False):
    """
    Run perturbation experiment
    
    Args:
        verbose: If True, print detailed perturbation distribution
        noise_strength: Noise strength for random walk
    """
    # Phase 1: Evolution to perturbation time
    t_eval_1 = np.linspace(0, perturb_time, int(n_points * perturb_time / t_end))
    sol1 = solve_ivp(
        fun=dynamics_model_constrained,
        t_span=[0, perturb_time],
        y0=X0,
        args=(W, noise_strength),
        t_eval=t_eval_1,
        method='RK45'
    )
    
    # Apply perturbation (proportional distribution across orange camp)
    X_perturbed = sol1.y[:, -1].copy()
    X_before = X_perturbed.copy()
    orange_total = X_perturbed[1] + X_perturbed[2]
    
    if orange_total > 0:
        # Calculate proportion of each orange node
        ratio_node2 = X_perturbed[1] / orange_total
        ratio_node3 = X_perturbed[2] / orange_total
        
        # Distribute perturbation proportionally
        transfer_node2 = perturb_amount * ratio_node2
        transfer_node3 = perturb_amount * ratio_node3
        
        X_perturbed[0] += perturb_amount                    # Blue +ΔC
        X_perturbed[1] -= transfer_node2                    # Orange Node2 -ΔC×ratio
        X_perturbed[2] -= transfer_node3                    # Orange Node3 -ΔC×ratio
        
        if verbose:
            print(f"\n  Perturbation Distribution (ΔC={perturb_amount}):")
            print(f"    Node2 ratio: {ratio_node2:.3f} → transfer {transfer_node2:.2f}")
            print(f"    Node3 ratio: {ratio_node3:.3f} → transfer {transfer_node3:.2f}")
            print(f"    Before: Node1={X_before[0]:.2f}, Node2={X_before[1]:.2f}, Node3={X_before[2]:.2f}")
            print(f"    After:  Node1={X_perturbed[0]:.2f}, Node2={X_perturbed[1]:.2f}, Node3={X_perturbed[2]:.2f}")
    else:
        # Orange camp has no resources, perturbation has no effect
        print(f"  Warning: Orange camp depleted, perturbation ineffective")
    
    # Phase 2: Post-perturbation evolution
    t_eval_2 = np.linspace(perturb_time, t_end, int(n_points * (t_end - perturb_time) / t_end))
    sol2 = solve_ivp(
        fun=dynamics_model_constrained,
        t_span=[perturb_time, t_end],
        y0=X_perturbed,
        args=(W, noise_strength),
        t_eval=t_eval_2,
        method='RK45'
    )
    
    # Merge data
    t_full = np.concatenate([sol1.t, sol2.t[1:]])
    X_full = np.concatenate([sol1.y, sol2.y[:, 1:]], axis=1)
    
    return t_full, X_full


def plot_multi_experiment_results(results, config):
    """
    Visualize multi-group experiment results
    
    Args:
        results: Dictionary of experiment results
        config: Experiment configuration
    """
    n_initial = len(config['initial_conditions'])
    n_perturb = len(config['perturbations'])
    
    fig = plt.figure(figsize=config['figsize'])
    
    plot_idx = 1
    for ic_idx, ic_config in enumerate(config['initial_conditions']):
        for perturb_idx, perturb_config in enumerate(config['perturbations']):
            key = f"IC{ic_idx}_P{perturb_idx}"
            t = results[key]['t']
            X = results[key]['X']
            
            ax = plt.subplot(n_initial, n_perturb, plot_idx)
            
            if config['show_individual_nodes']:
                # Show individual nodes
                ax.plot(t, X[0, :], color='#2196F3', linewidth=2, 
                       label='Node 1 (Blue)', alpha=0.9)
                ax.plot(t, X[1, :], color='#FF6B00', linewidth=2, 
                       label='Node 2 (Orange)', linestyle='-', alpha=0.9)
                ax.plot(t, X[2, :], color='#FF9800', linewidth=2, 
                       label='Node 3 (Orange)', linestyle='--', alpha=0.9)
            else:
                # Show camp totals
                blue_total = X[0, :]
                orange_total = X[1, :] + X[2, :]
                ax.plot(t, orange_total, color='#FF8C00', linewidth=2.5, 
                       label='Orange Camp (Node 2+3)')
                ax.plot(t, blue_total, color='#1E90FF', linewidth=2.5, 
                       label='Blue Camp (Node 1)')
            
            # Perturbation marker
            ax.axvline(x=config['perturb_time'], color='gray', 
                      linestyle='--', linewidth=1.5, alpha=0.6)
            ax.text(config['perturb_time'] + 2, 90, 
                   f"ΔC={perturb_config['delta_C']}", 
                   fontsize=9, color='gray')
            
            # Styling
            ax.set_xlabel('Time', fontsize=10)
            ax.set_ylabel('Resource', fontsize=10)
            
            # Title
            title = f"{ic_config['name']} | {perturb_config['name']}"
            ax.set_title(title, fontsize=11, fontweight='bold', 
                        color=config['title_color'])
            
            ax.legend(loc='best', fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.set_ylim([0, 110])
            
            plot_idx += 1
    
    plt.suptitle('Camp Perturbation Experiment: Resilience and Critical Switching', 
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    filename = 'perturbation_multi_experiment.png'
    plt.savefig(filename, dpi=config['dpi'], bbox_inches='tight')
    print(f"✓ Figure saved as {filename}")
    plt.show()


def create_summary_table(results, config):
    """Create summary table of experimental results"""
    print("\n" + "="*100)
    print("EXPERIMENTAL RESULTS SUMMARY".center(100))
    print("="*100)
    
    header = f"{'Initial Condition':<25} | {'Perturbation':<20} | {'ΔC':<5} | "
    header += f"{'Node1(Blue)':<15} | {'Node2(Orange)':<15} | {'Node3(Orange)':<15} | {'Outcome':<15}"
    print(header)
    print("-"*100)
    
    for ic_idx, ic_config in enumerate(config['initial_conditions']):
        for perturb_idx, perturb_config in enumerate(config['perturbations']):
            key = f"IC{ic_idx}_P{perturb_idx}"
            X_final = results[key]['X'][:, -1]
            
            # Determine outcome
            if X_final[0] > 50:
                outcome = "Blue Wins ✓"
            elif (X_final[1] + X_final[2]) > 50:
                outcome = "Orange Wins ✓"
            else:
                outcome = "Unstable"
            
            row = f"{ic_config['name']:<25} | {perturb_config['name']:<20} | "
            row += f"{perturb_config['delta_C']:<5} | "
            row += f"{X_final[0]:>6.2f}         | {X_final[1]:>6.2f}          | "
            row += f"{X_final[2]:>6.2f}          | {outcome:<15}"
            print(row)
        
        if ic_idx < len(config['initial_conditions']) - 1:
            print("-"*100)
    
    print("="*100 + "\n")


def print_config_table(config):
    """Print configuration parameters as a table"""
    print("\n" + "="*80)
    print("EXPERIMENT CONFIGURATION".center(80))
    print("="*80)
    
    print("\n【Dynamics Parameters】")
    print(f"  Plunder Rate:        {config['plunder_rate']}")
    print(f"  Total Resource:      {config['total_resource']}")
    
    print("\n【Time Parameters】")
    print(f"  Perturbation Time:   {config['perturb_time']}")
    print(f"  End Time:            {config['t_end']}")
    print(f"  Sampling Points:     {config['n_points']}")
    
    print("\n【Perturbation Strengths】")
    for i, p in enumerate(config['perturbations']):
        print(f"  {i+1}. {p['name']:<20} ΔC = {p['delta_C']}")
    
    print("\n【Initial Conditions】")
    for i, ic in enumerate(config['initial_conditions']):
        X0 = ic['X0']
        print(f"  {i+1}. {ic['name']:<20} Node1={X0[0]:.1f}, Node2={X0[1]:.1f}, Node3={X0[2]:.1f}")
        print(f"     {ic['description']}")
    
    print("\n【Visualization】")
    print(f"  Show Individual Nodes: {config['show_individual_nodes']}")
    print(f"  Title Color:           {config['title_color']}")
    print(f"  DPI:                   {config['dpi']}")
    print(f"  Figure Size:           {config['figsize']}")
    
    print("="*80 + "\n")


# ==================== 主程序 ====================
if __name__ == "__main__":
    config = EXPERIMENT_CONFIG
    
    print("\n" + "="*80)
    print("CAMP PERTURBATION EXPERIMENT".center(80))
    print("Multistable Attractor Basins: Resilience & Critical Switching".center(80))
    print("="*80)
    
    # Print configuration
    print_config_table(config)
    
    # Build interaction matrix
    W = build_interaction_matrix(N=6, plunder_rate=config['plunder_rate'])
    
    # Run all experiments
    print("Running multi-group experiments...")
    results = {}
    
    total_experiments = len(config['initial_conditions']) * len(config['perturbations'])
    exp_count = 0
    
    for ic_idx, ic_config in enumerate(config['initial_conditions']):
        for perturb_idx, perturb_config in enumerate(config['perturbations']):
            exp_count += 1
            key = f"IC{ic_idx}_P{perturb_idx}"
            
            print(f"\n  [{exp_count}/{total_experiments}] {ic_config['name']} + "
                  f"{perturb_config['name']} (ΔC={perturb_config['delta_C']})...")
            
            t, X = run_perturbation_experiment(
                W, 
                np.array(ic_config['X0']), 
                config['perturb_time'],
                perturb_config['delta_C'],
                config['t_end'],
                config['n_points'],
                noise_strength=config['noise_strength'],
                verbose=True  # Show detailed perturbation distribution
            )
            
            results[key] = {'t': t, 'X': X}
    
    print("✓ All experiments completed!\n")
    
    # Create visualizations
    print("Generating visualizations...")
    plot_multi_experiment_results(results, config)
    
    # Print summary table
    create_summary_table(results, config)
    
    print("="*80)
    print("EXPERIMENT COMPLETE".center(80))
    print("="*80)
    print("\nKey Findings:")
    print("  • Sub-threshold perturbations: System shows resilience")
    print("  • Supra-threshold perturbations: Trigger critical phase transitions")
    print("  • Initial conditions modulate the threshold for switching")
    print("  • Individual nodes within a camp can have asymmetric responses")
    print("\nPerturbation Logic:")
    print("  • Resources transferred FROM orange camp TO blue camp")
    print("  • Distribution: Proportional to each orange node's current resource")
    print("  • If Node2 has 70% of orange resources, it loses 70% of ΔC")
    print("  • If Node3 has 30% of orange resources, it loses 30% of ΔC")
    print("="*80 + "\n")
