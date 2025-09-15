"""
Figure 3 Maker - Hub's double-edged role

Panels (from figures.md):
  A: Hubs as signal centers — Degree centrality vs Fisher information (scatter + density + regression)
  B: Hubs' decisive role — Decoding vs %neurons (Hub-first vs Periphery-first)
  C: Hubs as noise centers — Noise correlation by pair type (Hub-Hub / Hub-Periphery / Periphery-Periphery)
  D: Mechanism schematic — Minimal illustrative network (hub highlighted)

Usage examples:
  python figure3_maker.py --panel A --mouse m27
  python figure3_maker.py --panel B --mouse m27
  python figure3_maker.py --panel C --mouse m27
  python figure3_maker.py --panel D

Notes:
  - Uses RR neurons and stimulus window.
  - Reuses functions from degree.py when available for Fisher info and correlation networks.
"""

import os
import json
import glob
import argparse
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from scipy.stats import pearsonr, spearmanr

from loaddata import (
    cfg as global_cfg,
    cfg,  # Also import as cfg for compatibility
    reclassify_labels,
    fast_rr_selection,
)

# Import from degree.py for consistent analysis
from degree import (
    calculate_fisher_information_per_neuron,
    calculate_multivariate_fisher_per_level,
    stratify_neurons_by_centrality,
    calculate_centrality_metrics
)

try:
    from loaddata import load_data, segment_neuron_data, load_old_version_data
except Exception:
    load_data = None
    segment_neuron_data = None
    load_old_version_data = None

try:
    from degree import (
        build_correlation_network,
        calculate_centrality_metrics,
        calculate_fisher_information_per_neuron,
    )
except Exception:
    build_correlation_network = None
    calculate_centrality_metrics = None
    calculate_fisher_information_per_neuron = None


# ---------------------------
# Helpers
# ---------------------------
def setup_style():
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.family': 'Arial',
        'font.size': 10,
        'axes.titlesize': 12,
        'axes.labelsize': 11,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'figure.titlesize': 14,
        'axes.linewidth': 1.2,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
    })


COLORS = {
    'ordered': '#FF7F50',
    'noise': '#4682B4',
    'neutral': '#708090',
    'hub': '#E74C3C',
    'node': '#2E86AB',
    'edge': '#6C757D',
    'accent': '#F18F01',
}


def list_config_files() -> List[str]:
    return sorted(glob.glob(os.path.join('config', '*.json')))


def get_config_by_mouse(mouse: Optional[str]) -> Optional[str]:
    if not mouse:
        return None
    key = str(mouse).lower().strip()
    name = None
    if key in {'m27', '27'}:
        name = 'm27.json'
    elif key in {'m30', '30'}:
        name = 'm30.json'
    elif key in {'m65', '65'}:
        name = 'm65.json'
    elif key in {'m74', '74'}:
        name = 'm74.json'
    if not name:
        return None
    path = os.path.join('config', name)
    return path if os.path.exists(path) else None


def load_session(config_path: str):
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg_json = json.load(f)
    loader = cfg_json.get('LOADER_VERSION', 'new')
    data_path = cfg_json.get('DATA_PATH')
    oldp = cfg_json.get('OLD_VERSION_PATHS', {})
    if loader == 'new':
        neuron_data, neuron_pos, trigger_data, stimulus_data = load_data(data_path)
        segments, labels = segment_neuron_data(neuron_data, trigger_data, stimulus_data)
    else:
        neuron_index, segments, labels, neuron_pos = load_old_version_data(
            oldp['neurons'], oldp['trials'], oldp['location']
        )
        stimulus_data = np.column_stack([labels, np.zeros(len(labels))])
    return segments, labels, neuron_pos, stimulus_data


def compute_rr_neurons(segments: np.ndarray, stimulus_data: np.ndarray) -> List[int]:
    rr_labels = reclassify_labels(stimulus_data)
    rr = fast_rr_selection(segments, rr_labels)
    return list(rr.get('rr_neurons', []))


def extract_activity(segments: np.ndarray, neuron_idx: List[int], use_stimulus_window: bool = True) -> np.ndarray:
    data = segments[:, neuron_idx, :]
    if use_stimulus_window:
        t0 = getattr(global_cfg, 'PRE_FRAMES', 10)
        t1 = t0 + getattr(global_cfg, 'STIMULUS_DURATION', 20)
        data = data[:, :, t0:min(t1, data.shape[2])]
    X = data.mean(axis=2)
    return X


def corr_matrix_fast(X: np.ndarray) -> np.ndarray:
    Xc = X - X.mean(axis=0, keepdims=True)
    std = X.std(axis=0, keepdims=True)
    std = np.where(std == 0, 1, std)
    Z = Xc / std
    C = (Z.T @ Z) / max(1, (X.shape[0] - 1))
    np.fill_diagonal(C, 1.0)
    return np.clip(C, -1, 1)


def adjacency_by_density(C: np.ndarray, density: float = 0.1) -> np.ndarray:
    n = C.shape[0]
    iu = np.triu_indices(n, 1)
    vals = np.abs(C[iu])
    m = len(vals)
    k = max(1, int(m * float(density)))
    thresh = np.partition(vals, -k)[-k]
    keep = vals >= thresh
    A = np.zeros_like(C, dtype=int)
    A[iu] = keep.astype(int)
    A = A + A.T
    np.fill_diagonal(A, 0)
    return A


def avg_corr_within_trial(segments: np.ndarray, neuron_idx: List[int], use_stimulus_window: bool = True) -> np.ndarray:
    """Compute per-trial correlation (across time) and average across trials."""
    t0 = getattr(global_cfg, 'PRE_FRAMES', 10)
    t1 = t0 + getattr(global_cfg, 'STIMULUS_DURATION', 20)
    if not use_stimulus_window:
        t0 = 0
        t1 = segments.shape[2]
    n_trials = segments.shape[0]
    n_neurons = len(neuron_idx)
    Csum = np.zeros((n_neurons, n_neurons), dtype=float)
    cnt = 0
    for tr in range(n_trials):
        X = segments[tr, neuron_idx, t0:min(t1, segments.shape[2])].T  # (time, neurons)
        if X.shape[0] < 3:
            continue
        mu = X.mean(axis=0, keepdims=True)
        sd = X.std(axis=0, keepdims=True)
        sd = np.where(sd == 0, 1, sd)
        Z = (X - mu) / sd
        C = np.corrcoef(Z, rowvar=False)
        if C.shape != (n_neurons, n_neurons):
            continue
        C = np.nan_to_num(C, nan=0.0, posinf=0.0, neginf=0.0)
        np.fill_diagonal(C, 1.0)
        Csum += np.clip(C, -1, 1)
        cnt += 1
    if cnt == 0:
        return np.eye(n_neurons)
    Cmean = Csum / cnt
    np.fill_diagonal(Cmean, 1.0)
    return Cmean

def cross_val_performance(X: np.ndarray, y: np.ndarray, n_splits: int = 5) -> float:
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    clf = SVC(kernel='rbf', C=1.0, gamma='scale')
    accs = []
    for tr, te in skf.split(X, y):
        clf.fit(X[tr], y[tr])
        y_pred = clf.predict(X[te])
        accs.append(accuracy_score(y[te], y_pred))
    return float(np.mean(accs))


def compute_multivariate_fisher_info(segments: np.ndarray, stimulus_data: np.ndarray, 
                                   neuron_subset: List[int]) -> float:
    """
    Compute multivariate Fisher information for a subset of neurons.
    
    This considers the covariance structure among neurons in the subset,
    providing a measure of the information content of the neural population.
    """
    try:
        # Try to use loaddata's window-based implementation with PCA support
        from loaddata import calculate_fisher_information_window
        y = reclassify_labels(stimulus_data)
        
        # 提取指定神经元的数据
        neuron_data = segments[:, neuron_subset, :]
        
        # 使用loaddata的实现（包含PCA降维）
        fisher_score = calculate_fisher_information_window(neuron_data, y)
        
        return max(0.0, float(fisher_score))
        
    except Exception as e:
        print(f"Loaddata Fisher calculation failed: {e}")
        print(f"Falling back to local implementation for {len(neuron_subset)} neurons")
        # Fallback: compute multivariate Fisher using covariance matrices
        result = compute_multivariate_fisher_fallback(segments, stimulus_data, neuron_subset)
        print(f"Fallback Fisher result: {result}")
        return result


def compute_multivariate_fisher_fallback(segments: np.ndarray, stimulus_data: np.ndarray, 
                                       neuron_subset: List[int]) -> float:
    """
    Fallback method to compute multivariate Fisher information using 
    between-class and within-class covariance matrices.
    """
    try:
        print(f"Fallback: processing {len(neuron_subset)} neurons")
        y = reclassify_labels(stimulus_data)
        valid = y != 0
        seg = segments[valid][:, neuron_subset, :]
        yv = y[valid]
        print(f"Fallback: valid trials={np.sum(valid)}, unique labels={np.unique(yv)}")
        
        # Extract stimulus window activity
        t0 = getattr(global_cfg, 'PRE_FRAMES', 10)
        t1 = t0 + getattr(global_cfg, 'STIMULUS_DURATION', 20)
        X = seg[:, :, t0:t1].mean(axis=2)  # (trials, neurons)
        
        if X.shape[0] < 3 or X.shape[1] == 0:
            return 0.0
        
        # Standardize features
        X = StandardScaler().fit_transform(X)
        
        # PCA降维处理（参考loaddata.py实现）
        n_trials, n_neurons = X.shape
        if n_neurons > n_trials * 0.5:  # 当神经元数 > 试次数的50%时进行降维
            # 目标维度：试次数的1/3，但至少保留2维，最多不超过15维
            target_dim = max(2, min(15, n_trials // 3))
            
            print(f"使用PCA降维: {n_neurons}维 -> {target_dim}维 (试次数: {n_trials})")
            
            # 执行PCA降维
            pca = PCA(n_components=target_dim, random_state=42)
            X = pca.fit_transform(X)
            
            print(f"PCA解释方差比: {np.sum(pca.explained_variance_ratio_):.3f}")
        
        classes = np.unique(yv)
        if len(classes) < 2:
            return 0.0
        
        # Compute multivariate Fisher information: trace(Sw^-1 * Sb)
        grand_mean = X.mean(axis=0)
        
        # Between-class scatter matrix
        Sb = np.zeros((X.shape[1], X.shape[1]))
        # Within-class scatter matrix  
        Sw = np.zeros((X.shape[1], X.shape[1]))
        
        for c in classes:
            X_c = X[yv == c]
            if X_c.shape[0] == 0:
                continue
                
            class_mean = X_c.mean(axis=0)
            n_c = X_c.shape[0]
            
            # Between-class contribution
            diff = (class_mean - grand_mean).reshape(-1, 1)
            Sb += n_c * (diff @ diff.T)
            
            # Within-class contribution
            X_c_centered = X_c - class_mean
            Sw += X_c_centered.T @ X_c_centered
        
        # 自适应正则化：基于数据规模和条件数（参考loaddata.py）
        from scipy.linalg import pinv, eigvals
        try:
            eigenvals = eigvals(Sw).real.astype(np.float64)
            eigenvals = eigenvals[eigenvals > 0]
            if len(eigenvals) > 1:
                condition_number = float(np.max(eigenvals) / np.min(eigenvals))
                reg_strength = max(1e-6, float(np.max(eigenvals)) * 1e-10 * condition_number)
            else:
                reg_strength = 1e-3
        except:
            reg_strength = 1e-3
        
        Sw += reg_strength * np.eye(Sw.shape[0])
        
        # Compute Fisher information as trace(Sw^-1 * Sb)
        try:
            Sw_inv = pinv(Sw).astype(np.float64)
            fisher_matrix = np.dot(Sw_inv, Sb).astype(np.float64)
            fisher_info = float(np.trace(fisher_matrix).real)
            
            # 数值稳定性检查
            if np.isnan(fisher_info) or np.isinf(fisher_info) or fisher_info < 0:
                # 备选方案：使用广义特征值问题求解
                from scipy.linalg import eigh
                try:
                    eigenvals, _ = eigh(Sb, Sw)
                    eigenvals_real = eigenvals.real.astype(np.float64)
                    fisher_info = float(np.sum(eigenvals_real[eigenvals_real > 0]))
                except:
                    # 最简化方案
                    trace_s_b = float(np.trace(Sb).real)
                    trace_s_w = float(np.trace(Sw).real)
                    fisher_info = trace_s_b / (trace_s_w + 1e-10)
            
            return max(0.0, fisher_info)
        except Exception:
            return 0.0
            
    except Exception as e:
        print(f"Warning: Fisher computation failed: {e}")
        return 0.0


# ---------------------------
# Panels
# ---------------------------
def _compute_multivariate_curve(cfg_path: str, density: float, n_levels: int):
    """Compute level-wise degree means and multivariate FI for one mouse/config."""
    segments, labels, neuron_pos, stimulus_data = load_session(cfg_path)
    rr = compute_rr_neurons(segments, stimulus_data)
    if len(rr) == 0:
        rr = list(range(segments.shape[1]))

    # Build correlation network and degree centrality
    C = avg_corr_within_trial(segments, rr, use_stimulus_window=True)
    A = adjacency_by_density(C, density=density)
    G = nx.from_numpy_array(A)
    G.remove_edges_from(nx.selfloop_edges(G))

    centrality_dict = calculate_centrality_metrics(G)
    degree_centrality_scores = centrality_dict['degree']

    # Reclassify labels
    labels = reclassify_labels(stimulus_data)

    # Prepare degree list aligned to rr
    n = len(rr)
    degree_values = np.array([degree_centrality_scores.get(i, 0.0) for i in range(n)])

    # Valid mask for possible NaNs (should be none)
    valid_mask = np.isfinite(degree_values)
    sorted_indices = np.argsort(degree_values[valid_mask])
    level_size = max(1, len(sorted_indices) // n_levels)

    level_degree_means: List[float] = []
    level_multivariate_fishers: List[float] = []
    level_names: List[str] = []

    for i in range(n_levels):
        start_idx = i * level_size
        end_idx = len(sorted_indices) if i == n_levels - 1 else (i + 1) * level_size
        idx_slice = sorted_indices[start_idx:end_idx]
        if idx_slice.size == 0:
            level_degree_means.append(0.0)
            level_multivariate_fishers.append(0.0)
            level_names.append(f'L{i+1}')
            continue

        level_degree_mean = float(np.mean(degree_values[valid_mask][idx_slice]))

        # Map back to original neuron indices
        valid_neuron_indices = np.arange(len(rr))[valid_mask]
        level_neurons = [rr[valid_neuron_indices[idx]] for idx in idx_slice if idx < len(valid_neuron_indices)]

        mv_fi = calculate_multivariate_fisher_per_level(segments, labels, level_neurons)

        level_degree_means.append(level_degree_mean)
        level_multivariate_fishers.append(float(mv_fi))
        level_names.append(f'L{i+1}')

    return level_degree_means, level_multivariate_fishers, level_names


def panel_a_multivariate_only(output_dir: str, mice: List[str], density: float = 0.1, n_levels: int = 5):
    """Plot only the multivariate FI subplot. If multiple mice provided, overlay all in one plot."""
    setup_style()
    os.makedirs(output_dir, exist_ok=True)

    # Resolve mouse keys to config paths
    cfg_paths: List[Tuple[str, str]] = []
    for m in mice:
        cfg_path = get_config_by_mouse(m)
        if cfg_path is not None:
            cfg_paths.append((m, cfg_path))

    # Prepare figure
    fig, ax = plt.subplots(figsize=(8.5, 6.0))

    colors = plt.cm.tab10(np.linspace(0, 1, max(3, len(cfg_paths))))

    for (color, (mouse_key, path)) in zip(colors, cfg_paths):
        level_deg, level_mv_fi, level_names = _compute_multivariate_curve(path, density, n_levels)
        # Normalize FI within each mouse to improve cross-mouse comparability
        mv = np.array(level_mv_fi, dtype=float)
        mv_max = float(np.max(mv)) if len(mv) > 0 else 0.0
        mv_norm = (mv / (mv_max + 1e-12)) if mv_max > 0 else mv
        ax.plot(level_deg, mv_norm.tolist(), '-o', color=color, linewidth=2.8, markersize=7,
                markerfacecolor='white', markeredgewidth=2, label=mouse_key.upper())

    ax.set_xlabel('Degree Centrality (level mean)')
    ax.set_ylabel('Normalized Multivariate Fisher Information')
    ax.set_title('Multivariate Fisher Information vs Degree Centrality (Levels)')
    ax.grid(True, alpha=0.35)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylim(0, 1.1)
    if len(cfg_paths) > 1:
        ax.legend(frameon=True)

    out_path = os.path.join(output_dir, 'figure3_panel_a.png')
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'[Saved] {out_path}')

def panel_a(output_dir: str, cfg_path: str, density: float = 0.1, n_levels: int = 5):
    """
    Multi-layer composite plot showing hub neurons as signal information carriers.
    
    Layer 1: 2D density scatter plot (background)
    Layer 2: Binned averages overlay (5 levels)
    Layer 3: Statistical annotation box (Spearman correlation)
    """
    setup_style()
    import pandas as pd

    segments, labels, neuron_pos, stimulus_data = load_session(cfg_path)
    rr = compute_rr_neurons(segments, stimulus_data)
    if len(rr) == 0:
        rr = list(range(segments.shape[1]))

    # Use degree.py's standard analysis pipeline for consistency
    # Build correlation network
    C = avg_corr_within_trial(segments, rr, use_stimulus_window=True)
    A = adjacency_by_density(C, density=density)
    G = nx.from_numpy_array(A)
    G.remove_edges_from(nx.selfloop_edges(G))

    # Calculate centrality metrics using degree.py's function
    centrality_dict = calculate_centrality_metrics(G)
    degree_centrality_scores = centrality_dict['degree']  # This is already normalized
    
    # Calculate Fisher information using degree.py's function
    labels = reclassify_labels(stimulus_data)
    fisher_scores = calculate_fisher_information_per_neuron(segments, labels, rr)
    
    # Prepare data for plotting (ensure matching indices)
    n = len(rr)
    degree_values = [degree_centrality_scores.get(i, 0.0) for i in range(n)]
    fisher_values = [fisher_scores.get(rr[i], 0.0) for i in range(n)]
    
    # Remove invalid values
    valid_mask = np.isfinite(degree_values) & np.isfinite(fisher_values)
    degree_values = np.array(degree_values)[valid_mask]
    fisher_values = np.array(fisher_values)[valid_mask]
    
    # Stratify neurons into levels by degree centrality
    sorted_indices = np.argsort(degree_values)
    level_size = len(sorted_indices) // n_levels
    
    level_degree_means = []
    level_multivariate_fishers = []
    level_names = []
    
    print(f"计算各层级的多变量Fisher信息...")
    
    for i in range(n_levels):
        start_idx = i * level_size
        if i == n_levels - 1:  # Last level gets remaining neurons
            end_idx = len(sorted_indices)
        else:
            end_idx = (i + 1) * level_size
        
        level_indices = sorted_indices[start_idx:end_idx]
        level_degree_mean = np.mean(degree_values[level_indices])
        
        # Map sorted indices back to RR neuron indices
        # level_indices are indices into the valid_mask filtered arrays
        valid_neuron_indices = np.arange(len(rr))[valid_mask]
        level_neurons = [rr[valid_neuron_indices[idx]] for idx in level_indices if idx < len(valid_neuron_indices)]
        
        # Calculate BOTH individual and multivariate Fisher information
        if len(level_neurons) > 0:
            # Method 1: Individual Fisher information average
            level_individual_fi_values = []
            for neuron_idx in level_neurons:
                if neuron_idx in fisher_scores:
                    level_individual_fi_values.append(fisher_scores[neuron_idx])
            level_avg_individual_fi = np.mean(level_individual_fi_values) if level_individual_fi_values else 0
            
            # Method 2: Multivariate Fisher information
            level_multivariate_fi = calculate_multivariate_fisher_per_level(segments, labels, level_neurons)
    else:
            level_avg_individual_fi = 0.0
            level_multivariate_fi = 0.0
        
        level_degree_means.append(level_degree_mean)
        level_multivariate_fishers.append(level_multivariate_fi)  # Store multivariate FI
        level_names.append(f'L{i+1}')
        
        print(f"  Level {i+1}: {len(level_neurons)} neurons, "
              f"avg_centrality={level_degree_mean:.3f}, "
              f"avg_individual_FI={level_avg_individual_fi:.3f}, "
              f"multivariate_FI={level_multivariate_fi:.3f}")
        
        # Store individual FI for second plot
        if i == 0:
            level_individual_fishers = []
        level_individual_fishers.append(level_avg_individual_fi)
    
    # Calculate Spearman correlation for original data
    spearman_corr, spearman_p = spearmanr(degree_values, fisher_values)
    
    # Create figure with two comparison plots: Multivariate vs Individual Fisher Information
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(2, 2, height_ratios=[3, 1], width_ratios=[1, 1], hspace=0.3, wspace=0.3)
    
    # Left plot: Multivariate Fisher Information (monotonic trend)
    ax1 = fig.add_subplot(gs[0, 0])
    # Right plot: Individual Fisher Information Average (non-monotonic trend)
    ax2 = fig.add_subplot(gs[0, 1])
    # Bottom: Distribution comparison subplot (spans both columns)
    ax_dist = fig.add_subplot(gs[1, :])
    
    # ===== LEFT PLOT: Multivariate Fisher Information =====
    # Layer 1: 2D density scatter plot (background)
    hb1 = ax1.hexbin(degree_values, fisher_values, gridsize=30, cmap='Blues', alpha=0.6, mincnt=1)
    ax1.scatter(degree_values, fisher_values, s=8, color='lightblue', alpha=0.4, edgecolors='none')
    
    # Layer 2: Multivariate Fisher Information (red squares)
    ax1.scatter(level_degree_means, level_multivariate_fishers, s=150, c='#E74C3C', 
               marker='s', alpha=0.9, edgecolors='black', linewidth=2, zorder=10, label='Multivariate FI')
    ax1.plot(level_degree_means, level_multivariate_fishers, '--', color='#E74C3C', 
            linewidth=2.5, alpha=0.8, zorder=9)
    
    # Add level labels
    for i, (x, y, name) in enumerate(zip(level_degree_means, level_multivariate_fishers, level_names)):
        ax1.annotate(name, (x, y), xytext=(5, 5), textcoords='offset points',
                   fontsize=10, fontweight='bold', color='darkred')
    
    # Statistics annotation
    stats_text1 = f"Spearman's ρ = {spearman_corr:.3f}\np = {spearman_p:.3f}"
    ax1.text(0.05, 0.95, stats_text1, transform=ax1.transAxes, fontsize=11, fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.4", facecolor='white', alpha=0.9, edgecolor='gray'),
            verticalalignment='top')
    
    # Formatting for left plot
    ax1.set_xlabel('Degree Centrality', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Multivariate Fisher Information', fontsize=12, fontweight='bold')
    ax1.set_title('Method 1: Multivariate Fisher Information\n(Monotonic Trend)', 
                 fontsize=13, fontweight='bold', pad=15)
    ax1.grid(True, alpha=0.3)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # ===== RIGHT PLOT: Individual Fisher Information Average =====
    # Layer 1: Same background
    hb2 = ax2.hexbin(degree_values, fisher_values, gridsize=30, cmap='Blues', alpha=0.6, mincnt=1)
    ax2.scatter(degree_values, fisher_values, s=8, color='lightblue', alpha=0.4, edgecolors='none')
    
    # Layer 2: Individual Fisher Information Average (blue squares)
    ax2.scatter(level_degree_means, level_individual_fishers, s=150, c='#3498DB', 
               marker='s', alpha=0.9, edgecolors='black', linewidth=2, zorder=10, label='Individual FI Avg')
    ax2.plot(level_degree_means, level_individual_fishers, '--', color='#3498DB', 
            linewidth=2.5, alpha=0.8, zorder=9)
    
    # Add level labels
    for i, (x, y, name) in enumerate(zip(level_degree_means, level_individual_fishers, level_names)):
        ax2.annotate(name, (x, y), xytext=(5, 5), textcoords='offset points',
                   fontsize=10, fontweight='bold', color='darkblue')
    
    # Statistics annotation (same background correlation)
    ax2.text(0.05, 0.95, stats_text1, transform=ax2.transAxes, fontsize=11, fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.4", facecolor='white', alpha=0.9, edgecolor='gray'),
            verticalalignment='top')
    
    # Formatting for right plot
    ax2.set_xlabel('Degree Centrality', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Individual Fisher Information (Average)', fontsize=12, fontweight='bold')
    ax2.set_title('Method 2: Individual Fisher Information Average\n(Non-monotonic Trend)', 
                 fontsize=13, fontweight='bold', pad=15)
    ax2.grid(True, alpha=0.3)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    # === DISTRIBUTION COMPARISON SUBPLOT (spans both columns) ===
    # Create comparison bar plot for both methods
    x_positions = np.arange(len(level_names))
    width = 0.35
    
    # Plot multivariate Fisher information
    bars1 = ax_dist.bar(x_positions - width/2, level_multivariate_fishers, width, 
                       label='Multivariate Fisher Info', color='#E74C3C', alpha=0.8, edgecolor='black')
    
    # Plot individual Fisher information average
    bars2 = ax_dist.bar(x_positions + width/2, level_individual_fishers, width,
                       label='Individual Fisher Info (Avg)', color='#3498DB', alpha=0.8, edgecolor='black')
    
    # Add value labels on bars
    for i, (bar1, bar2, mv_val, ind_val) in enumerate(zip(bars1, bars2, level_multivariate_fishers, level_individual_fishers)):
        ax_dist.text(bar1.get_x() + bar1.get_width()/2, bar1.get_height() + 0.01,
                    f'{mv_val:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold', color='darkred')
        ax_dist.text(bar2.get_x() + bar2.get_width()/2, bar2.get_height() + 0.001,
                    f'{ind_val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold', color='darkblue')
    
    ax_dist.set_xlabel('Centrality Level', fontsize=12, fontweight='bold')
    ax_dist.set_ylabel('Fisher Information', fontsize=12, fontweight='bold')
    ax_dist.set_title('Comparison: Multivariate vs Individual Fisher Information by Level', 
                     fontsize=13, fontweight='bold')
    ax_dist.set_xticks(x_positions)
    ax_dist.set_xticklabels(level_names)
    ax_dist.legend(fontsize=11)
    ax_dist.grid(True, alpha=0.3, axis='y')
    ax_dist.spines['top'].set_visible(False)
    ax_dist.spines['right'].set_visible(False)
    
    # Overall figure title
    fig.suptitle('Hub Neurons as Information Carriers: Two Analysis Methods Comparison',
                fontsize=16, fontweight='bold', y=0.98)

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, 'figure3_panel_a_comparison.png')
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'[Saved] {out_path}')
    print(f'Analysis complete: {n_levels} levels comparison')
    print(f'Multivariate FI range: {min(level_multivariate_fishers):.3f} - {max(level_multivariate_fishers):.3f}')
    print(f'Individual FI range: {min(level_individual_fishers):.3f} - {max(level_individual_fishers):.3f}')
    print(f'Background correlation: Spearman ρ = {spearman_corr:.3f}, p = {spearman_p:.3f}')


def panel_b(output_dir: str, cfg_path: str, density: float = 0.1):
    """Compare Hub-first vs Periphery-first as a function of %neurons using two metrics:
    - Decoding accuracy (5-fold CV, RBF-SVM)
    - Multivariate Fisher information (loaddata.calculate_fisher_information)

    Saves two figures:
    - figures/figure3_panel_b_accuracy.png
    - figures/figure3_panel_b_fisher.png
    """
    setup_style()

    segments, labels, neuron_pos, stimulus_data = load_session(cfg_path)
    rr = compute_rr_neurons(segments, stimulus_data)
    if len(rr) == 0:
        rr = list(range(segments.shape[1]))
    # Build network to rank by degree (average of per-trial correlations)
    C = avg_corr_within_trial(segments, rr, use_stimulus_window=True)
    A = adjacency_by_density(C, density=density)
    G = nx.from_numpy_array(A)
    G.remove_edges_from(nx.selfloop_edges(G))
    deg = dict(G.degree())
    order_desc = [rr[i] for i, _ in sorted(deg.items(), key=lambda kv: kv[1], reverse=True)]
    order_asc = list(reversed(order_desc))

    # Multivariate FI function using loaddata implementation
    try:
        from loaddata import calculate_fisher_information as calc_fi
    except Exception:
        calc_fi = None

    # Helper: features for accuracy
    def make_features(sel):
        t0 = getattr(global_cfg, 'PRE_FRAMES', 10)
        t1 = t0 + getattr(global_cfg, 'STIMULUS_DURATION', 20)
        data = segments[:, sel, :][:, :, t0:t1]
        Xf = data.reshape(data.shape[0], -1)
        Xf = StandardScaler().fit_transform(Xf)
        y = reclassify_labels(stimulus_data)
        return Xf, y

    def compute_fi_for_subset(sel):
        y = reclassify_labels(stimulus_data)
        if calc_fi is not None:
            fi = calc_fi(segments, y, sel)
            fi = np.asarray(fi)
            t0 = getattr(global_cfg, 'PRE_FRAMES', 10)
            t1 = t0 + getattr(global_cfg, 'STIMULUS_DURATION', 20)
            if fi.ndim == 1 and fi.size >= t1:
                return float(np.mean(fi[t0:t1]))
            return float(np.mean(fi)) if fi.size > 0 else 0.0
        # Fallback: simple Fisher ratio on projected space
        valid = y != 0
        seg = segments[valid][:, sel, :]
        yv = y[valid]
        t0 = getattr(global_cfg, 'PRE_FRAMES', 10)
        t1 = t0 + getattr(global_cfg, 'STIMULUS_DURATION', 20)
        Xw = seg[:, :, t0:t1].mean(axis=2)
        Xw = StandardScaler().fit_transform(Xw)
        # Reduce dimensionality if needed
        if Xw.shape[1] > max(2, Xw.shape[0]//2):
            Xw = PCA(n_components=max(2, min(10, Xw.shape[0]//3)), random_state=42).fit_transform(Xw)
        classes = np.unique(yv)
        grand = Xw.mean(axis=0)
        Sb = np.zeros((Xw.shape[1], Xw.shape[1]))
        Sw = np.zeros_like(Sb)
        for c in classes:
            Xc = Xw[yv == c]
            if Xc.size == 0:
                continue
            mc = Xc.mean(axis=0)
            diff = (mc - grand).reshape(-1, 1)
            Sb += Xc.shape[0] * (diff @ diff.T)
            Xc0 = Xc - mc
            Sw += Xc0.T @ Xc0
        Sw += 1e-6 * np.eye(Sw.shape[0])
        from scipy.linalg import pinv
        try:
            val = float(np.trace(pinv(Sw) @ Sb))
            return max(0.0, val)
        except Exception:
            return 0.0

    fracs = np.linspace(0.1, 1.0, 10)
    # Accuracy
    acc_hub, acc_per = [], []
    # FI
    fi_hub, fi_per = [], []
    for f in fracs:
        k = max(1, int(len(order_desc) * f))
        # Accuracy
        X1, y1 = make_features(order_desc[:k])
        X2, y2 = make_features(order_asc[:k])
        acc_hub.append(cross_val_performance(X1, y1))
        acc_per.append(cross_val_performance(X2, y2))
        # FI
        fi_hub.append(compute_fi_for_subset(order_desc[:k]))
        fi_per.append(compute_fi_for_subset(order_asc[:k]))

    # Plot accuracy
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.plot(fracs*100, acc_hub, '-o', color=COLORS['ordered'], label='Hub-first')
    ax.plot(fracs*100, acc_per, '-o', color=COLORS['neutral'], label='Periphery-first')
    ax.set_xlabel('Percentage of neurons included (%)')
    ax.set_ylabel('Decoding accuracy')
    ax.set_ylim(0.0, 1.05)
    ax.set_title('Figure 3B-Acc. Hub vs Periphery (Accuracy)', loc='left', fontweight='bold')
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.3)
    os.makedirs(output_dir, exist_ok=True)
    out_path_acc = os.path.join(output_dir, 'figure3_panel_b_accuracy.png')
    plt.savefig(out_path_acc)
    plt.close(fig)
    print(f'[Saved] {out_path_acc}')

    # Plot FI
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.plot(fracs*100, fi_hub, '-o', color=COLORS['ordered'], label='Hub-first')
    ax.plot(fracs*100, fi_per, '-o', color=COLORS['neutral'], label='Periphery-first')
    ax.set_xlabel('Percentage of neurons included (%)')
    ax.set_ylabel('Fisher information (multivariate)')
    ax.set_title('Figure 3B-FI. Hub vs Periphery (Fisher)', loc='left', fontweight='bold')
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.3)
    out_path_fi = os.path.join(output_dir, 'figure3_panel_b_fisher.png')
    plt.savefig(out_path_fi)
    plt.close(fig)
    print(f'[Saved] {out_path_fi}')


def panel_c(output_dir: str, cfg_path: str, density: float = 0.1):
    """Noise correlation by pair type (Hub-Hub / Hub-Periphery / Periphery-Periphery)."""
    setup_style()
    import pandas as pd

    segments, labels, neuron_pos, stimulus_data = load_session(cfg_path)
    rr = compute_rr_neurons(segments, stimulus_data)
    if len(rr) == 0:
        rr = list(range(segments.shape[1]))

    # Define hubs via degree based on averaged per-trial correlations
    C = avg_corr_within_trial(segments, rr, use_stimulus_window=True)
    A = adjacency_by_density(C, density=density)
    G = nx.from_numpy_array(A)
    G.remove_edges_from(nx.selfloop_edges(G))
    deg_vals = np.array([d for _, d in G.degree()])
    p90 = np.quantile(deg_vals, 0.9) if len(deg_vals) else 0
    p10 = np.quantile(deg_vals, 0.1) if len(deg_vals) else 0
    hubs = set([i for i, d in G.degree() if d >= p90])
    perip = set([i for i, d in G.degree() if d <= p10])

    # Compute noise correlations: subtract per-condition means
    y = reclassify_labels(stimulus_data)
    valid = y != 0
    seg = segments[valid][:, rr, :]
    yv = y[valid]
    t0 = getattr(global_cfg, 'PRE_FRAMES', 10)
    t1 = t0 + getattr(global_cfg, 'STIMULUS_DURATION', 20)
    m = seg[:, :, t0:t1].mean(axis=2)  # (trials, neurons)
    # remove condition mean
    for c in np.unique(yv):
        mask = (yv == c)
        m[mask] -= m[mask].mean(axis=0, keepdims=True)

    # pair groups
    pairs = {'Hub-Hub': [], 'Hub-Periphery': [], 'Periphery-Periphery': []}
    for i in range(m.shape[1]):
        for j in range(i+1, m.shape[1]):
            r = np.corrcoef(m[:, i], m[:, j])[0, 1]
            if i in hubs and j in hubs:
                pairs['Hub-Hub'].append(r)
            elif (i in hubs and j in perip) or (j in hubs and i in perip):
                pairs['Hub-Periphery'].append(r)
            elif i in perip and j in perip:
                pairs['Periphery-Periphery'].append(r)

    data = []
    for k, vals in pairs.items():
        for v in vals:
            if np.isfinite(v):
                data.append({'pair_type': k, 'noise_corr': v})
    df = pd.DataFrame(data)

    fig, ax = plt.subplots(figsize=(7.5, 5))
    sns.violinplot(data=df, x='pair_type', y='noise_corr', palette=[COLORS['hub'], COLORS['neutral'], COLORS['node']], inner='box', ax=ax)
    ax.set_xlabel('Pair type')
    ax.set_ylabel('Noise correlation')
    ax.set_title('Figure 3C. Noise correlation by pair type', loc='left', fontweight='bold')

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, 'figure3_panel_c.png')
    plt.savefig(out_path)
    plt.close(fig)
    print(f'[Saved] {out_path}')


def panel_d(output_dir: str):
    """Illustrative mechanism schematic (simple)."""
    setup_style()

    # Create toy graph
    G = nx.erdos_renyi_graph(40, 0.08, seed=42)
    deg = dict(G.degree())
    thresh = np.quantile(list(deg.values()), 0.9)
    hubs = [n for n, d in deg.items() if d >= thresh]
    others = [n for n in G.nodes() if n not in hubs]
    pos = nx.spring_layout(G, seed=42)

    fig, ax = plt.subplots(figsize=(7.5, 5))
    nx.draw_networkx_nodes(G, pos, nodelist=others, node_size=40, node_color=COLORS['node'], alpha=0.8, ax=ax)
    nx.draw_networkx_nodes(G, pos, nodelist=hubs, node_size=100, node_color=COLORS['hub'], alpha=0.9, ax=ax)
    nx.draw_networkx_edges(G, pos, width=0.6, edge_color=COLORS['edge'], alpha=0.5, ax=ax)
    ax.set_title('Figure 3D. Hub-centric mechanism (schematic)', loc='left', fontweight='bold')
    ax.axis('off')

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, 'figure3_panel_d.png')
    plt.savefig(out_path)
    plt.close(fig)
    print(f'[Saved] {out_path}')


# ---------------------------
# CLI
# ---------------------------
def main():
    parser = argparse.ArgumentParser(description='Generate Figure 3 panels')
    parser.add_argument('--panel', type=str, required=True, help='Panel letter (A/B/C/D)')
    parser.add_argument('--mouse', type=str, default='m27', help='Mouse/session key (m27/m30/m65/m74)')
    parser.add_argument('--config', type=str, default=None, help='Optional config JSON (overrides --mouse)')
    parser.add_argument('--outdir', type=str, default='figures', help='Output directory')
    parser.add_argument('--density', type=float, default=0.1, help='Network density for thresholding (0-1)')
    parser.add_argument('--n-levels', type=int, default=5, help='Number of centrality levels for panel A binned averages')
    parser.add_argument('--mice', type=str, default=None, help='Comma-separated mice keys or "all" for m27,m30,m65,m74')
    args = parser.parse_args()

    cfg_path = args.config if args.config else get_config_by_mouse(args.mouse)
    if cfg_path is None and args.panel.upper() != 'D':
        raise RuntimeError('No valid config found. Please provide --config or a known --mouse key.')

    panel = args.panel.strip().upper()
    if panel == 'A':
        # Aggregate view if requested
        if args.mice:
            mice_keys = [k.strip() for k in (['m27','m30','m65','m74'] if args.mice.lower() == 'all' else args.mice.split(','))]
        else:
            mice_keys = [args.mouse]
        panel_a_multivariate_only(args.outdir, mice=mice_keys, density=args.density, n_levels=args.n_levels)
    elif panel == 'B':
        panel_b(args.outdir, cfg_path, density=args.density)
    elif panel == 'C':
        panel_c(args.outdir, cfg_path, density=args.density)
    elif panel == 'D':
        panel_d(args.outdir)
    else:
        raise ValueError('Unsupported panel. Use A/B/C/D.')


if __name__ == '__main__':
    main()
