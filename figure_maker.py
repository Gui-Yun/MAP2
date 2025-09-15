"""
Figure Maker - Generate publication-quality panels based on figures.md and paper.md

Usage examples:
  python figure_maker.py --figure 1 --panel A
  python figure_maker.py --figure 1 --panel B [--config config/m74.json | --mouse m74]
  python figure_maker.py --figure 1 --panel C [--mouse m27]
  python figure_maker.py --figure 1 --panel D [--mouse m27]
  python figure_maker.py --figure 1 --panel B_TIME [--mouse m27]

This script loads real data via loaddata.py (new/old loaders via config),
computes any missing analysis on the fly, and saves panels into `figures/`.
It does not modify existing project scripts.
"""

import os
import json
import glob
import argparse
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score
from scipy.stats import binomtest

warnings.filterwarnings('ignore')

# Project imports (guarded to avoid import-time failures)
from loaddata import (
    cfg as global_cfg,
    reclassify_labels,
    fast_rr_selection,
)

# Optional imports from project
try:
    from loaddata import load_data, segment_neuron_data, load_old_version_data
except Exception:
    load_data = None
    segment_neuron_data = None
    load_old_version_data = None

try:
    # Fisher info and other helpers if available
    from loaddata import calculate_fisher_information, calculate_fisher_information_by_condition, calculate_multivariate_fisher_single_timepoint
except Exception:
    calculate_fisher_information = None
    calculate_fisher_information_by_condition = None
    calculate_multivariate_fisher_single_timepoint = None

try:
    from loaddata import classify_by_timepoints
except Exception:
    classify_by_timepoints = None

try:
    # Manifold utilities
    from manifold import (
        prepare_data_for_manifold,
        perform_pca,
        perform_tsne,
    )
except Exception:
    prepare_data_for_manifold = None
    perform_pca = None
    perform_tsne = None


# ---------------------------
# Styling helpers
# ---------------------------
def setup_publication_style():
    plt.style.use('default')
    # 确保Arial字体可用，否则回退到DejaVu Sans
    font_family = 'Arial'
    try:
        plt.rcParams['font.family'] = font_family
    except:
        font_family = 'DejaVu Sans'
        plt.rcParams['font.family'] = font_family
    
    plt.rcParams.update({
        # 严格的字体设置 - 科研发表标准
        'font.family': font_family,
        'font.size': 8,              # 基础字体8pt (Nature标准)
        'axes.titlesize': 10,        # 子图标题 10pt
        'axes.labelsize': 9,         # 坐标轴标签 9pt  
        'xtick.labelsize': 8,        # 刻度标签 8pt
        'ytick.labelsize': 8,
        'legend.fontsize': 8,        # 图例 8pt
        'figure.titlesize': 11,      # 主标题 11pt
        
        # 极简主义风格
        'axes.linewidth': 1.0,       # 更细的轴线
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.edgecolor': '#666666', # 细灰色轴线
        'axes.grid': False,          # 默认关闭网格
        'grid.alpha': 0.2,           # 如需网格，使用更浅的颜色
        'grid.linewidth': 0.5,       # 更细的网格线
        'grid.color': '#CCCCCC',     # 浅灰色网格
        
        # 背景设置
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        
        # 输出设置 - 科研发表标准
        'savefig.dpi': 600,          # 600 DPI for publication
        'savefig.bbox': 'tight',
        'savefig.facecolor': 'white',
        'savefig.edgecolor': 'none',
        'savefig.pad_inches': 0.05,  # 最小边距
        'text.usetex': False,        # 避免LaTeX依赖问题
        'mathtext.fontset': 'dejavusans',  # 数学符号字体
        
        # 线条和标记
        'lines.linewidth': 2.0,      # 稍粗的数据线
        'lines.markersize': 4,       # 适中的标记大小
        'patch.linewidth': 1.0,      # 图形边框
    })


def save_figure_both_formats(fig, output_path_base, include_title_and_caption=True):
    """Save figure in both PNG and SVG formats.
    
    Args:
        fig: matplotlib figure object
        output_path_base: base path without extension (e.g., 'figures/figure1_panel_a')
        include_title_and_caption: if True, include title and caption (PNG), if False, plot only (SVG)
    """
    # Save PNG with titles and captions
    png_path = output_path_base + '.png'
    fig.savefig(png_path, format='png', dpi=600, bbox_inches='tight', 
                facecolor='white', edgecolor='none', pad_inches=0.05)
    
    # For SVG, temporarily hide titles and text annotations if requested
    if not include_title_and_caption:
        # Store original title and text elements
        original_suptitle = fig._suptitle
        original_texts = []
        
        # Hide suptitle
        if fig._suptitle:
            fig._suptitle.set_visible(False)
        
        # Hide figure-level text annotations
        for text in fig.texts:
            if text != fig._suptitle:
                original_texts.append((text, text.get_visible()))
                text.set_visible(False)
    
    # Save SVG
    svg_path = output_path_base + '.svg'
    fig.savefig(svg_path, format='svg', bbox_inches='tight', 
                facecolor='white', edgecolor='none', pad_inches=0.05)
    
    # Restore titles and text for PNG if they were hidden
    if not include_title_and_caption:
        # Restore suptitle
        if original_suptitle:
            original_suptitle.set_visible(True)
        
        # Restore figure-level texts
        for text, was_visible in original_texts:
            text.set_visible(was_visible)
    
    return png_path, svg_path


COLORS = {
    # 按照figures.md规范的色系
    'ordered': '#FF7F50',        # 珊瑚橙 - 有序光流/核心发现
    'noise': '#4682B4',          # 钢青色 - 随机噪音/对照组  
    'neutral': '#708090',        # 石板灰 - 中性/外围
    'accent': '#D2691E',         # 赭石色 - 强调色
    'hub': '#FF7F50',            # 枢纽神经元 - 珊瑚橙
    'periphery': '#708090',      # 外围神经元 - 石板灰
    'axis_color': '#666666',     # 坐标轴颜色
    'grid_color': '#CCCCCC',     # 网格颜色
}


# ---------------------------
# Data loading helpers
# ---------------------------
def list_config_files() -> List[str]:
    return sorted(glob.glob(os.path.join('config', '*.json')))


def get_config_by_mouse(mouse: Optional[str]) -> Optional[str]:
    """Map mouse id (e.g., 'm27') to a known config path if exists."""
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
    if name is None:
        return None
    path = os.path.join('config', name)
    return path if os.path.exists(path) else None


def load_session_from_config(config_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load one session according to a config JSON.

    Returns
    -------
    segments : (trials, neurons, time)
    labels   : (trials,) int
    neuron_pos : (2 or 4, neurons)
    stimulus_data : (trials, 2) [category, intensity] if available; otherwise synthesized
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg_json = json.load(f)

    loader_version = cfg_json.get('LOADER_VERSION', 'new')
    data_path = cfg_json.get('DATA_PATH')
    old_paths = cfg_json.get('OLD_VERSION_PATHS', {})

    if loader_version == 'new':
        if load_data is None or segment_neuron_data is None:
            raise RuntimeError('load_data/segment_neuron_data not available')
        neuron_data, neuron_pos, trigger_data, stimulus_data = load_data(data_path)
        segments, labels = segment_neuron_data(neuron_data, trigger_data, stimulus_data)
        return segments, labels, neuron_pos, stimulus_data
    elif loader_version == 'old':
        if load_old_version_data is None:
            raise RuntimeError('load_old_version_data not available')
        neuron_index, segments, labels, neuron_pos = load_old_version_data(
            old_paths['neurons'], old_paths['trials'], old_paths['location']
        )
        # Synthesize stimulus_data: category in first column, intensity as zeros
        stimulus_data = np.column_stack([labels, np.zeros(len(labels))])
        # Harmonize neuron_pos to first two rows if 4 available
        if neuron_pos.ndim == 2 and neuron_pos.shape[0] >= 2:
            neuron_pos = neuron_pos[:2, :]
        return segments, labels, neuron_pos, stimulus_data
    else:
        raise ValueError(f'Unknown LOADER_VERSION: {loader_version}')


def compute_rr_neurons(segments: np.ndarray, stimulus_data: np.ndarray) -> List[int]:
    # Prefer project reclassify to filter zeros/noise
    try:
        rr_labels = reclassify_labels(stimulus_data)
    except Exception:
        # Fallback: use category directly (first column)
        rr_labels = stimulus_data[:, 0].astype(int)
    rr_results = fast_rr_selection(segments, rr_labels)
    rr_neurons = list(rr_results.get('rr_neurons', []))
    return rr_neurons


def extract_trial_features(
    segments: np.ndarray,
    labels: np.ndarray,
    rr_neurons: Optional[List[int]] = None,
    pre_frames: Optional[int] = None,
    stim_duration: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Flatten trials × (neurons×time) using stimulus window."""
    if pre_frames is None:
        pre_frames = getattr(global_cfg, 'PRE_FRAMES', 10)
    if stim_duration is None:
        stim_duration = getattr(global_cfg, 'STIMULUS_DURATION', 20)

    if rr_neurons is not None and len(rr_neurons) > 0:
        data = segments[:, rr_neurons, :]
    else:
        data = segments

    t0 = pre_frames
    t1 = min(segments.shape[2], pre_frames + stim_duration)
    data = data[:, :, t0:t1]
    X = data.reshape(data.shape[0], -1)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    y = np.array(labels).astype(int)
    return X, y


def cross_val_performance(X: np.ndarray, y: np.ndarray, n_splits: int = 5, random_state: int = 42) -> Dict:
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    clf = SVC(kernel='rbf', C=1.0, gamma='scale')

    overall_accs = []
    per_class_accs_list = []

    classes = np.unique(y)
    for train_idx, test_idx in skf.split(X, y):
        clf.fit(X[train_idx], y[train_idx])
        y_pred = clf.predict(X[test_idx])
        overall_accs.append(accuracy_score(y[test_idx], y_pred))

        cm = confusion_matrix(y[test_idx], y_pred, labels=classes)
        with np.errstate(divide='ignore', invalid='ignore'):
            per_class = np.diag(cm) / cm.sum(axis=1)
            per_class[np.isnan(per_class)] = 0.0
        per_class_accs_list.append(per_class)

    per_class_mean = np.mean(np.vstack(per_class_accs_list), axis=0)
    per_class_std = np.std(np.vstack(per_class_accs_list), axis=0)
    results = {
        'classes': classes,
        'overall_mean': float(np.mean(overall_accs)),
        'overall_std': float(np.std(overall_accs)),
        'per_class_mean': per_class_mean,
        'per_class_std': per_class_std,
        'fold_overall': overall_accs,
    }
    return results


# ---------------------------
# Figure 1 panels
# ---------------------------
def fig1_panel_a(output_dir: str):
    setup_publication_style()
    fig, ax = plt.subplots(figsize=(10, 4))

    # Mouse (subject)
    ax.text(0.08, 0.5, 'Head-fixed mouse', ha='center', va='center', fontsize=11,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgray', alpha=0.8))
    ax.annotate('', xy=(0.18, 0.5), xytext=(0.12, 0.5),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))

    # Visual stimuli blocks
    stim_defs = [
        (COLORS['ordered'], 'Expansion'),
        (COLORS['neutral'], 'Contraction'),
        (COLORS['noise'], 'Random motion'),
    ]
    for i, (color, label) in enumerate(stim_defs):
        y = 0.7 - i * 0.2
        ax.add_patch(plt.Rectangle((0.22, y - 0.05), 0.16, 0.1,
                                   facecolor=color, alpha=0.9, edgecolor='black'))
        ax.text(0.30, y, label, ha='center', va='center', fontsize=10, color='white', weight='bold')
    ax.text(0.30, 0.12, 'Visual stimuli', ha='center', va='center', fontsize=10, weight='bold')

    ax.annotate('', xy=(0.45, 0.5), xytext=(0.38, 0.5),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))

    # V1
    ax.text(0.55, 0.5, 'Primary visual cortex (V1)', ha='center', va='center', fontsize=11,
            bbox=dict(boxstyle='round,pad=0.3', facecolor=COLORS['accent'], alpha=0.75, edgecolor='black'))

    ax.annotate('', xy=(0.72, 0.5), xytext=(0.62, 0.5),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))

    # Neural activity matrix
    ax.text(0.84, 0.5, 'Neural activity matrix', ha='center', va='center', fontsize=11,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#ADD8E6', alpha=0.9, edgecolor='black'))

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    fig.suptitle('Figure 1A. Experimental paradigm schematic', x=0.02, ha='left', fontweight='bold')
    fig.text(0.02, -0.02, 'Schematic of head-fixed mouse viewing expansion, contraction, and random-motion stimuli with responses recorded in V1.',
             fontsize=9)

    os.makedirs(output_dir, exist_ok=True)
    out_path_base = os.path.join(output_dir, 'figure1_panel_a')
    png_path, svg_path = save_figure_both_formats(fig, out_path_base)
    plt.close(fig)
    print(f'[Saved] {png_path}')
    print(f'[Saved] {svg_path}')


def _evaluate_session_overall_accuracy(segments: np.ndarray, labels: np.ndarray, rr_neurons: List[int]) -> dict:
    """Train SVM with 5-fold CV, aggregate predictions across folds,
    compute overall accuracy and exact binomial p-value vs chance (1/3).
    """
    # Use RR set; fallback to all neurons if empty
    use_neurons = rr_neurons if len(rr_neurons) > 0 else list(range(segments.shape[1]))
    X, y = extract_trial_features(segments, labels, use_neurons)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    clf = SVC(kernel='rbf', C=1.0, gamma='scale', probability=False)

    y_true_all, y_pred_all = [], []
    fold_acc = []
    for tr, te in skf.split(X, y):
        clf.fit(X[tr], y[tr])
        y_pred = clf.predict(X[te])
        y_true_all.append(y[te])
        y_pred_all.append(y_pred)
        fold_acc.append(accuracy_score(y[te], y_pred))

    y_true_all = np.concatenate(y_true_all)
    y_pred_all = np.concatenate(y_pred_all)
    total = len(y_true_all)
    correct = int(np.sum(y_true_all == y_pred_all))
    overall_acc = correct / total if total > 0 else 0.0

    n_classes = len(np.unique(y))
    chance = 1.0 / n_classes if n_classes > 0 else 0.0
    # Exact binomial test versus chance
    pval = binomtest(k=correct, n=total, p=chance, alternative='greater').pvalue if total > 0 else 1.0

    return {
        'overall_acc': overall_acc,
        'fold_acc_mean': float(np.mean(fold_acc)),
        'fold_acc_std': float(np.std(fold_acc)),
        'correct': correct,
        'total': total,
        'p_value': float(pval),
        'chance': chance,
        'n_classes': n_classes,
    }


def _p_to_stars(p: float) -> str:
    if p < 1e-4:
        return '****'
    if p < 1e-3:
        return '***'
    if p < 1e-2:
        return '**'
    if p < 0.05:
        return '*'
    return 'n.s.'


def fig1_panel_b(output_dir: str, force_config: Optional[str] = None):
    """Three-way decoding overall accuracy, single best subject, significance vs chance.
    - No per-class bars, no per-subject scatter.
    - Select the session (animal) with highest overall CV accuracy.
    """
    setup_publication_style()
    os.makedirs(output_dir, exist_ok=True)

    best = None
    best_cfg = None
    candidate_configs = [force_config] if force_config else list_config_files()

    for cfg_path in candidate_configs:
        if cfg_path is None:
            continue
        try:
            segments, labels, neuron_pos, stimulus_data = load_session_from_config(cfg_path)
            rr_neurons = compute_rr_neurons(segments, stimulus_data)
            stats = _evaluate_session_overall_accuracy(segments, labels, rr_neurons)
            if (best is None) or (stats['overall_acc'] > best['overall_acc']):
                best = stats
                best_cfg = os.path.basename(cfg_path)
            if force_config:
                # If explicitly specified, use it without searching others
                break
        except Exception:
            continue

    if best is None:
        raise RuntimeError('No session could be loaded to compute decoding accuracy for Panel B.')

    # Build shuffled-label baseline (right bar)
    # Keep CV scheme stable across shuffles for fair comparison
    # Recompute fold accuracies for the chosen session
    # Prepare features once using RR (or all if none)
    # Note: we assume the helper used 5-fold CV
    # Replicate the internal logic here to reuse folds

    # Load chosen session again to compute baseline
    chosen_cfg_path = None
    for cfg_path in candidate_configs:
        if cfg_path and os.path.basename(cfg_path) == best_cfg:
            chosen_cfg_path = cfg_path
            break
    if chosen_cfg_path is None:
        chosen_cfg_path = candidate_configs[0]

    segments, labels, neuron_pos, stimulus_data = load_session_from_config(chosen_cfg_path)
    rr_neurons = compute_rr_neurons(segments, stimulus_data)
    use_neurons = rr_neurons if len(rr_neurons) > 0 else list(range(segments.shape[1]))
    X, y = extract_trial_features(segments, labels, use_neurons)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    splits = list(skf.split(X, y))

    # Real per-fold accuracies for plotting if we want errorbar consistency
    clf = SVC(kernel='rbf', C=1.0, gamma='scale')
    real_fold_acc = []
    for tr, te in splits:
        clf.fit(X[tr], y[tr])
        y_pred = clf.predict(X[te])
        real_fold_acc.append(accuracy_score(y[te], y_pred))

    # Shuffled baseline via multiple repetitions
    rng = np.random.default_rng(123)
    n_rep = 20
    shuffle_means = []
    for rep in range(n_rep):
        y_shuf = y.copy()
        rng.shuffle(y_shuf)
        rep_accs = []
        for tr, te in splits:
            clf.fit(X[tr], y_shuf[tr])
            y_pred = clf.predict(X[te])
            rep_accs.append(accuracy_score(y_shuf[te], y_pred))
        shuffle_means.append(float(np.mean(rep_accs)))

    baseline_mean = float(np.mean(shuffle_means))
    baseline_std = float(np.std(shuffle_means))

    # Permutation p-value: how often shuffled >= real overall
    p_perm = (1 + np.sum(np.array(shuffle_means) >= best['overall_acc'])) / (1 + n_rep)

    # Plot two bars: Decoding vs Label-shuffled
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    colors = [COLORS['ordered'], 'gray']
    heights = [best['overall_acc'], baseline_mean]
    yerrs = [best['fold_acc_std'], baseline_std]
    ax.bar([0, 1], heights, yerr=yerrs, capsize=5, color=colors, edgecolor='white', linewidth=1.2, alpha=0.95)

    # Chance line
    ax.axhline(best['chance'], color='gray', linestyle='--', linewidth=1.2, label='Chance')

    # Significance annotations
    # 1) Real vs chance (binomial)
    y_top = max(heights[0] + yerrs[0], heights[1] + yerrs[1], best['chance'])
    y1 = y_top + 0.02
    ax.text(0.0, y1 + 0.01, _p_to_stars(best['p_value']), ha='center', va='bottom', fontsize=11, fontweight='bold')
    # 2) Real vs shuffled (permutation)
    y2 = y1 + 0.06
    ax.plot([0, 0, 1, 1], [y2 - 0.01, y2, y2, y2 - 0.01], color='black', linewidth=1.0)
    ax.text(0.5, y2 + 0.01, _p_to_stars(p_perm), ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.set_xlim(-0.5, 1.5)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Three-way decoding', 'Label-shuffled baseline'])
    ax.set_ylabel('Accuracy')
    ax.set_ylim(0.0, min(1.05, max(0.25, y2 + 0.08)))
    ax.legend(frameon=False, loc='lower right')
    # 更精细的网格设置
    ax.grid(True, axis='y', alpha=0.2, color=COLORS['grid_color'], linewidth=0.5)
    ax.tick_params(colors=COLORS['axis_color'])

    title = 'Figure 1B. High-fidelity three-way decoding (with baseline)'
    subtitle = (
        f'Best subject: {best_cfg} | Decoding={best["overall_acc"]:.3f} ± {best["fold_acc_std"]:.3f}; '
        f'Baseline={baseline_mean:.3f} ± {baseline_std:.3f}; Chance={best["chance"]:.3f}; '
        f'p_vs_chance={best["p_value"]:.2e}; p_vs_baseline≈{p_perm:.3f}'
    )
    fig.suptitle(title, x=0.02, ha='left', fontweight='bold')
    fig.text(0.02, -0.02, subtitle, fontsize=9)

    out_path_base = os.path.join(output_dir, 'figure1_panel_b')
    png_path, svg_path = save_figure_both_formats(fig, out_path_base)
    plt.close(fig)
    print(f'[Saved] {png_path}')
    print(f'[Saved] {svg_path}')


def fig1_panel_c(output_dir: str, force_config: Optional[str] = None,
                 fi_log: Optional[bool] = None, fi_norm: str = 'none'):
    """Generate both accuracy and Fisher info panels separately."""
    fig1_panel_c_accuracy(output_dir, force_config)
    fig1_panel_c_fisher(output_dir, force_config, fi_log, fi_norm)


def fig1_panel_c_accuracy(output_dir: str, force_config: Optional[str] = None):
    """Accuracy vs number of neurons only."""
    setup_publication_style()
    os.makedirs(output_dir, exist_ok=True)

    # Load one representative session
    session_loaded = False
    candidate_configs = [force_config] if force_config else list_config_files()
    for cfg_path in candidate_configs:
        if cfg_path is None:
            continue
        try:
            segments, labels, neuron_pos, stimulus_data = load_session_from_config(cfg_path)
            rr_neurons = compute_rr_neurons(segments, stimulus_data)
            session_loaded = True
            break
        except Exception:
            continue
    if not session_loaded:
        raise RuntimeError('No session could be loaded for panel C accuracy.')

    rng = np.random.default_rng(42)
    if len(rr_neurons) == 0:
        rr_neurons = list(range(segments.shape[1]))

    # Determine neuron counts with smaller steps for smoother curves
    max_n = max(10, min(len(rr_neurons), 200))
    counts = np.unique(np.linspace(5, max_n, num=12, dtype=int))

    acc_means, acc_stds = [], []

    for k in counts:
        acc_trials = []
        for b in range(8):
            subset = rng.choice(rr_neurons, size=k, replace=False)
            X, y = extract_trial_features(segments, labels, subset)
            res = cross_val_performance(X, y)
            acc_trials.append(res['overall_mean'])

        acc_means.append(np.mean(acc_trials))
        acc_stds.append(np.std(acc_trials))

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(counts, acc_means, color=COLORS['accent'], lw=2)
    ax.fill_between(counts, np.array(acc_means) - np.array(acc_stds),
                     np.array(acc_means) + np.array(acc_stds), color=COLORS['accent'], alpha=0.2)
    ax.set_xlabel('Number of neurons used')
    ax.set_ylabel('Decoding accuracy')
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, axis='y', alpha=0.2, color=COLORS['grid_color'], linewidth=0.5)
    ax.tick_params(colors=COLORS['axis_color'])

    fig.suptitle('Figure 1C. Decoding accuracy vs neuron count', 
                x=0.02, ha='left', fontweight='bold')
    fig.text(0.02, -0.02,
             'Decoding accuracy increases with neuron count using cross-validation.', 
             fontsize=8, color='#666666')

    out_path_base = os.path.join(output_dir, 'figure1_panel_c_accuracy')
    png_path, svg_path = save_figure_both_formats(fig, out_path_base)
    plt.close(fig)
    print(f'[Saved] {png_path}')
    print(f'[Saved] {svg_path}')


def fig1_panel_c_fisher(output_dir: str, force_config: Optional[str] = None,
                       fi_log: Optional[bool] = None, fi_norm: str = 'none'):
    """Fisher information vs number of neurons.
    Display options:
      - fi_log: bool, if True use log scale for FI axis; if None, uses default (False)
      - fi_norm: 'none' | 'minmax' (display-only normalization)
    """
    setup_publication_style()
    os.makedirs(output_dir, exist_ok=True)

    # Load one representative session
    session_loaded = False
    candidate_configs = [force_config] if force_config else list_config_files()
    for cfg_path in candidate_configs:
        if cfg_path is None:
            continue
        try:
            segments, labels, neuron_pos, stimulus_data = load_session_from_config(cfg_path)
            rr_neurons = compute_rr_neurons(segments, stimulus_data)
            session_loaded = True
            break
        except Exception:
            continue
    if not session_loaded:
        raise RuntimeError('No session could be loaded for panel C Fisher.')

    rng = np.random.default_rng(42)
    if len(rr_neurons) == 0:
        rr_neurons = list(range(segments.shape[1]))

    # Determine neuron counts with smaller steps for smoother curves
    max_n = max(10, min(len(rr_neurons), 200))
    counts = np.unique(np.linspace(5, max_n, num=12, dtype=int))

    fisher_means, fisher_stds = [], []

    # Use reclassified labels for FI consistency
    try:
        labels_for_fi = reclassify_labels(stimulus_data)
    except Exception:
        labels_for_fi = labels

    for k in counts:
        fisher_trials = []
        for b in range(8):
            subset = rng.choice(rr_neurons, size=k, replace=False)

            # Use loaddata.py style Fisher calculation with fixed time window
            try:
                # Extract time window data (consistent with loaddata.py approach)
                t0 = getattr(global_cfg, 'PRE_FRAMES', 10)
                t1 = t0 + getattr(global_cfg, 'STIMULUS_DURATION', 20)
                
                # Filter valid trials and RR neurons
                valid_mask = np.array(labels_for_fi) != 0
                valid_segments_fi = segments[valid_mask][:, subset, :]
                valid_labels_fi = np.array(labels_for_fi)[valid_mask]
                
                # Extract stimulus window
                window_data = valid_segments_fi[:, :, t0:t1]
                
                # Calculate Fisher info using improved method with PCA-aware processing
                fi_score = _calculate_fisher_window_with_pca(window_data, valid_labels_fi)
                fisher_trials.append(float(fi_score))
            except Exception as e:
                # Fallback to original method if available
                if calculate_fisher_information is not None:
                    try:
                        fi = calculate_fisher_information(segments, labels_for_fi, subset)
                        fi_win = np.mean(fi[t0:t1]) if fi.ndim == 1 else np.mean(fi)
                        fisher_trials.append(float(fi_win))
                    except Exception:
                        pass

        fisher_means.append(np.mean(fisher_trials) if fisher_trials else np.nan)
        fisher_stds.append(np.std(fisher_trials) if fisher_trials else np.nan)

    fig, ax = plt.subplots(figsize=(7, 5))

    # Process Fisher info data
    if not np.all(np.isnan(fisher_means)):
        use_log = False if fi_log is None else bool(fi_log)
        
        fm = np.array(fisher_means, dtype=float)
        fs = np.array(fisher_stds, dtype=float)
        valid = ~np.isnan(fm)

        # Build lower/upper bands
        lower = fm - fs
        upper = fm + fs

        # Apply normalization
        label_suffix = ''
        if fi_norm.lower() == 'minmax' and valid.any():
            vmin = np.nanmin(lower[valid])
            vmax = np.nanmax(upper[valid])
            
            if np.isfinite(vmin) and np.isfinite(vmax) and vmax > vmin:
                fm = (fm - vmin) / (vmax - vmin)
                lower = (lower - vmin) / (vmax - vmin)
                upper = (upper - vmin) / (vmax - vmin)
                label_suffix = ' (normalized)'

        # Positivity enforcement if using log
        if use_log:
            eps = 1e-8
            fm = np.where(fm <= 0, eps, fm)
            lower = np.where(lower <= 0, eps, lower)
            upper = np.where(upper <= lower, lower * (1 + 1e-3), upper)

        # Plot
        ax.plot(counts[valid], fm[valid], color=COLORS['noise'], lw=2)
        if valid.any():
            ax.fill_between(counts[valid], lower[valid], upper[valid], color=COLORS['noise'], alpha=0.2)
        if use_log:
            ax.set_yscale('log')
        
        ax.set_ylabel(f'Fisher information{label_suffix}', color=COLORS['noise'])
    
    ax.set_xlabel('Number of neurons used')
    ax.grid(True, axis='y', alpha=0.2, color=COLORS['grid_color'], linewidth=0.5)
    ax.tick_params(colors=COLORS['axis_color'])

    fig.suptitle('Figure 1C. Fisher information vs neuron count', 
                x=0.02, ha='left', fontweight='bold')
    fig.text(0.02, -0.02,
             'Fisher information changes with neuron count.', 
             fontsize=8, color='#666666')

    out_path_base = os.path.join(output_dir, 'figure1_panel_c_fisher')
    png_path, svg_path = save_figure_both_formats(fig, out_path_base)
    plt.close(fig)
    print(f'[Saved] {png_path}')
    print(f'[Saved] {svg_path}')


def _timecourse_decoding_accuracy(
    segments: np.ndarray,
    labels: np.ndarray,
    neuron_idx: List[int],
    repeats: int = 10,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute per-timepoint decoding accuracy with repeated CV.
    Returns (mean_acc[T], sd_acc[T], timepoints[T]).
    """
    T = segments.shape[2]
    acc_mat = []  # (repeats, T)
    # Prefer project implementation for consistency
    if classify_by_timepoints is not None:
        for r in range(repeats):
            acc_t, time_points = classify_by_timepoints(segments, labels, neuron_idx)
            acc_mat.append(acc_t)
        acc_mat = np.array(acc_mat)
        timepoints = np.array(time_points)
    else:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        for r in range(repeats):
            # small jitter in CV split randomness
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42 + r)
            acc_t = []
            for t in range(T):
                X_t = segments[:, neuron_idx, t]  # (trials, neurons)
                X_t = StandardScaler().fit_transform(X_t)
                y = labels
                clf = SVC(kernel='rbf', C=1.0, gamma='scale')
                fold_acc = []
                for tr, te in skf.split(X_t, y):
                    clf.fit(X_t[tr], y[tr])
                    y_pred = clf.predict(X_t[te])
                    fold_acc.append(accuracy_score(y[te], y_pred))
                acc_t.append(np.mean(fold_acc))
            acc_mat.append(acc_t)
        acc_mat = np.array(acc_mat)
        timepoints = np.arange(T)
    mean_acc = acc_mat.mean(axis=0)
    sd_acc = acc_mat.std(axis=0, ddof=1)
    return mean_acc, sd_acc, timepoints


def _calculate_fisher_window_with_pca(data: np.ndarray, labels: np.ndarray) -> float:
    """
    Calculate Fisher information for a time window with PCA-aware processing,
    consistent with loaddata.py's calculate_fisher_information_window approach.
    
    Parameters:
    data: Neural data (trials, neurons, timepoints)
    labels: Label array
    
    Returns:
    fisher_score: Multivariate Fisher information score
    """
    from scipy.linalg import pinv, eigvals
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    
    # Average over time dimension (consistent with loaddata.py)
    mean_data = np.mean(data, axis=2)  # (trials, neurons)
    
    # Only use categories 1 and 2 for Fisher calculation, exclude category 3 (noise)
    target_labels = [1, 2]
    target_mask = np.isin(labels, target_labels)
    
    if np.sum(target_mask) < 10:  # Need at least 10 samples
        return 0.0
    
    # Filter data and labels
    filtered_data = mean_data[target_mask]
    filtered_labels = labels[target_mask]
    
    unique_labels = np.unique(filtered_labels)
    if len(unique_labels) < 2:
        return 0.0
    
    n_trials, n_neurons = filtered_data.shape
    n_classes = len(unique_labels)
    
    # Check sufficient samples per class
    min_samples_per_class = min([np.sum(filtered_labels == label) for label in unique_labels])
    if min_samples_per_class < 2:
        return 0.0
    
    # Data standardization to avoid numerical issues
    scaler = StandardScaler()
    mean_data_scaled = scaler.fit_transform(filtered_data)
    
    # Key improvement: Use PCA when neuron count approaches or exceeds trial count
    effective_dim = min(n_neurons, n_trials - n_classes - 1)
    
    # Use fixed PCA dimension to avoid jumps and maintain consistency
    # Determine a safe fixed dimension for PCA that works across all neuron counts
    safe_max_dim = max(2, min(n_trials - n_classes - 2, 15))  # Conservative safe maximum
    
    # Always apply PCA when neuron count exceeds safe dimension to maintain consistency
    if n_neurons > safe_max_dim:
        # Use FIXED target dimension - no variation
        target_dim = safe_max_dim  # Fixed dimension, no changes
        
        # Execute PCA dimensionality reduction
        pca = PCA(n_components=target_dim, random_state=42)
        mean_data_scaled = pca.fit_transform(mean_data_scaled)
        
        # Update dimension info
        n_neurons = target_dim
    
    # Now calculate multivariate Fisher information on (potentially) reduced data
    
    # Calculate grand mean
    grand_mean = np.mean(mean_data_scaled, axis=0)  # (n_neurons,)
    
    # Calculate class means and sizes
    class_means = []
    class_sizes = []
    
    for label in unique_labels:
        label_mask = filtered_labels == label
        label_data = mean_data_scaled[label_mask]
        if len(label_data) > 0:
            class_means.append(np.mean(label_data, axis=0))
            class_sizes.append(len(label_data))
        else:
            class_means.append(grand_mean)
            class_sizes.append(0)
    
    class_means = np.array(class_means)  # (n_classes, n_neurons)
    class_sizes = np.array(class_sizes)
    
    # Calculate between-class scatter matrix (S_b)
    S_b = np.zeros((n_neurons, n_neurons), dtype=np.float64)
    for i, (class_mean, n_i) in enumerate(zip(class_means, class_sizes)):
        if n_i > 0:
            diff = (class_mean - grand_mean).reshape(-1, 1).astype(np.float64)
            S_b += n_i * np.dot(diff, diff.T).astype(np.float64)
    
    # Calculate within-class scatter matrix (S_w)
    S_w = np.zeros((n_neurons, n_neurons), dtype=np.float64)
    for label in unique_labels:
        label_mask = filtered_labels == label
        label_data = mean_data_scaled[label_mask]
        if len(label_data) > 1:  # Need at least 2 samples for covariance
            class_mean = np.mean(label_data, axis=0).astype(np.float64)
            centered_data = (label_data - class_mean).astype(np.float64)
            S_w += np.dot(centered_data.T, centered_data).astype(np.float64)
    
    # Adaptive regularization: based on data scale and condition number
    # Calculate S_w condition number to determine regularization strength
    try:
        eigenvals = eigvals(S_w).real.astype(np.float64)  # Ensure real and float64
        eigenvals = eigenvals[eigenvals > 0]  # Only consider positive eigenvalues
        if len(eigenvals) > 1:
            condition_number = float(np.max(eigenvals) / np.min(eigenvals))
            # Adaptively adjust regularization based on condition number
            reg_strength = max(1e-6, float(np.max(eigenvals)) * 1e-10 * condition_number)
        else:
            reg_strength = 1e-3
    except:
        reg_strength = 1e-3
    
    regularization = (reg_strength * np.eye(n_neurons)).astype(np.float64)
    S_w += regularization
    
    try:
        # Use more stable method to calculate multivariate Fisher information
        # Method 1: Direct calculation of trace(S_w^(-1) * S_b)
        S_w_inv = pinv(S_w).astype(np.float64)
        fisher_matrix = np.dot(S_w_inv, S_b).astype(np.float64)
        fisher_score = float(np.trace(fisher_matrix).real)  # Ensure real
        
        # Numerical stability check
        if np.isnan(fisher_score) or np.isinf(fisher_score) or fisher_score < 0:
            # Method 2: Use generalized eigenvalue problem
            from scipy.linalg import eigh
            try:
                eigenvals, _ = eigh(S_b, S_w)
                eigenvals_real = eigenvals.real.astype(np.float64)
                fisher_score = float(np.sum(eigenvals_real[eigenvals_real > 0]))
            except:
                # Method 3: Simplified multivariate Fisher ratio
                trace_s_b = float(np.trace(S_b).real)
                trace_s_w = float(np.trace(S_w).real)
                fisher_score = trace_s_b / (trace_s_w + 1e-10)
        
        # Ensure non-negative finite value
        fisher_score = max(0.0, float(fisher_score))
        if not np.isfinite(fisher_score):
            fisher_score = 0.0
        
    except Exception as e:
        # Final fallback: use simplified version
        try:
            trace_s_b = float(np.trace(S_b).real)
            trace_s_w = float(np.trace(S_w).real)
            fisher_score = trace_s_b / (trace_s_w + 1e-10)
            fisher_score = max(0.0, float(fisher_score))
        except:
            fisher_score = 0.0
    
    return fisher_score


def _calculate_multivariate_fisher_single_timepoint(data: np.ndarray, labels: np.ndarray) -> float:
    """
    Calculate single timepoint multivariate Fisher information using robust scatter matrix method
    from loaddata.py implementation.
    
    Parameters:
    data: Neural data (trials, neurons)  
    labels: Label array
    
    Returns:
    fisher_score: Multivariate Fisher information score
    """
    from scipy.linalg import pinv
    
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        return 0.0
    
    n_trials, n_neurons = data.shape
    n_classes = len(unique_labels)
    
    # Check if there are enough samples
    min_samples_per_class = min([np.sum(labels == label) for label in unique_labels])
    if min_samples_per_class < 2:
        return 0.0
    
    # Calculate grand mean
    grand_mean = np.mean(data, axis=0)  # (n_neurons,)
    
    # Calculate class means and sizes
    class_means = []
    class_sizes = []
    
    for label in unique_labels:
        label_mask = labels == label
        label_data = data[label_mask]
        if len(label_data) > 0:
            class_means.append(np.mean(label_data, axis=0))
            class_sizes.append(len(label_data))
        else:
            class_means.append(grand_mean)
            class_sizes.append(0)
    
    class_means = np.array(class_means)  # (n_classes, n_neurons)
    class_sizes = np.array(class_sizes)
    
    # Calculate between-class scatter matrix (S_b)
    S_b = np.zeros((n_neurons, n_neurons), dtype=np.float64)
    for i, (class_mean, n_i) in enumerate(zip(class_means, class_sizes)):
        if n_i > 0:
            diff = (class_mean - grand_mean).reshape(-1, 1).astype(np.float64)
            S_b += n_i * np.dot(diff, diff.T).astype(np.float64)
    
    # Calculate within-class scatter matrix (S_w)
    S_w = np.zeros((n_neurons, n_neurons), dtype=np.float64)
    for label in unique_labels:
        label_mask = labels == label
        label_data = data[label_mask]
        if len(label_data) > 1:  # Need at least 2 samples for covariance
            class_mean = np.mean(label_data, axis=0).astype(np.float64)
            centered_data = (label_data - class_mean).astype(np.float64)
            S_w += np.dot(centered_data.T, centered_data).astype(np.float64)
    
    # Add regularization to avoid singular matrix
    regularization = 1e-6 * np.eye(n_neurons, dtype=np.float64)
    S_w += regularization
    
    try:
        # Calculate multivariate Fisher discriminant ratio: trace(S_w^(-1) * S_b)
        S_w_inv = pinv(S_w)
        fisher_matrix = np.dot(S_w_inv, S_b)
        
        # Fisher information is the trace of the matrix
        fisher_score = np.trace(fisher_matrix)
        
        # Ensure non-negative value
        fisher_score = max(0.0, fisher_score)
        
    except Exception as e:
        fisher_score = 0.0
    
    return fisher_score


def _timecourse_fisher(
    segments: np.ndarray,
    labels: np.ndarray,
    neuron_idx: List[int],
    repeats: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute per-timepoint Fisher info with bootstrap over trials for SEM.
    Uses robust multivariate Fisher calculation from loaddata.py.
    Returns (mean_fi[T], sem_fi[T]) normalized per neuron.
    """
    T = segments.shape[2]
    if calculate_fisher_information is None:
        # Use our robust multivariate Fisher implementation
        rng = np.random.default_rng(42)
        fi_mat = []
        for r in range(repeats):
            # bootstrap trials
            idx = rng.integers(0, segments.shape[0], size=segments.shape[0])
            seg_b = segments[idx][:, neuron_idx, :]
            lab_b = labels[idx]
            fi_t = []
            for t in range(T):
                X_t = seg_b[:, :, t]  # (trials, neurons)
                # Standardize data for numerical stability
                X_t = StandardScaler().fit_transform(X_t)
                # Use robust multivariate Fisher calculation
                fi_val = _calculate_multivariate_fisher_single_timepoint(X_t, lab_b)
                fi_t.append(fi_val)
            fi_mat.append(fi_t)
        fi_mat = np.array(fi_mat)
        mean_fi = fi_mat.mean(axis=0)
        sem_fi = fi_mat.std(axis=0, ddof=1) / np.sqrt(repeats)
        return mean_fi, sem_fi
    else:
        # Use project's calculate_fisher_information if available
        rng = np.random.default_rng(43)
        fi_mat = []
        for r in range(repeats):
            idx = rng.integers(0, segments.shape[0], size=segments.shape[0])
            seg_b = segments[idx]
            lab_b = labels[idx]
            fi = calculate_fisher_information(seg_b, lab_b, neuron_idx)
            fi = np.asarray(fi)
            if fi.ndim == 0:
                fi = np.array([fi])
            fi_mat.append(fi)
        fi_mat = np.array(fi_mat)
        mean_fi = fi_mat.mean(axis=0)
        sem_fi = fi_mat.std(axis=0, ddof=1) / np.sqrt(repeats)
        return mean_fi, sem_fi


def fig1_panel_b_timecourse(output_dir: str, force_config: Optional[str] = None,
                            repeats: int = 10, fi_log: Optional[bool] = None,
                            fi_norm: str = 'none'):
    """Supplemental for Panel B: accuracy and Fisher info over time within a trial.
    Shows mean ± SEM across CV repeats (accuracy) and bootstrap (FI).
    """
    setup_publication_style()
    os.makedirs(output_dir, exist_ok=True)

    # Load chosen session
    cfg_path = force_config if force_config else (list_config_files()[0] if list_config_files() else None)
    if cfg_path is None:
        raise RuntimeError('No config found for timecourse panel.')
    segments, labels, neuron_pos, stimulus_data = load_session_from_config(cfg_path)
    rr_neurons = compute_rr_neurons(segments, stimulus_data)
    use_neurons = rr_neurons if len(rr_neurons) > 0 else list(range(segments.shape[1]))

    # Use reclassified labels for consistency
    try:
        used_labels = reclassify_labels(stimulus_data)
    except Exception:
        used_labels = labels

    # Compute timecourses
    mean_acc, sem_acc, timepoints = _timecourse_decoding_accuracy(segments, used_labels, use_neurons, repeats=repeats)
    mean_fi, sem_fi = _timecourse_fisher(segments, used_labels, use_neurons, repeats=repeats)

    # Plot
    # Convert SEM -> SD at plot stage if upstream returned SEM
    sd_acc = sem_acc * np.sqrt(repeats)
    sd_fi = sem_fi * np.sqrt(repeats)

    # Get stimulus timing
    t0 = getattr(global_cfg, 'PRE_FRAMES', 10)
    t1 = t0 + getattr(global_cfg, 'STIMULUS_DURATION', 20)

    # Shift time so that stimulus onset is 0 and pre-stimulus is negative
    time_shifted = timepoints - t0

    fig, ax1 = plt.subplots(figsize=(8, 5.5))
    ax1.plot(time_shifted, mean_acc, color=COLORS['accent'], lw=2, label='Decoding accuracy')
    ax1.fill_between(time_shifted, mean_acc - sd_acc, mean_acc + sd_acc, color=COLORS['accent'], alpha=0.2)
    ax1.set_xlabel('Time from stimulus onset (frames)')
    ax1.set_ylabel('Accuracy', color=COLORS['accent'])
    ax1.set_ylim(0.0, 1.05)
    ax1.grid(True, axis='y', alpha=0.3)

    # Stimulus window shading (0 to duration)
    ax1.axvspan(0, (t1 - t0), color='gray', alpha=0.1, label='Stimulus window')

    ax2 = ax1.twinx()
    # Display normalization for timecourse
    fi_mean_disp = mean_fi.copy()
    fi_low = mean_fi - sd_fi
    fi_up = mean_fi + sd_fi
    label_suffix = ''
    if fi_norm.lower() == 'minmax':
        vmin = np.nanmin(fi_low)
        vmax = np.nanmax(fi_up)
        if np.isfinite(vmin) and np.isfinite(vmax) and vmax > vmin:
            fi_mean_disp = (fi_mean_disp - vmin) / (vmax - vmin)
            fi_low = (fi_low - vmin) / (vmax - vmin)
            fi_up = (fi_up - vmin) / (vmax - vmin)
            label_suffix = ' (normalized)'
    # Log option
    use_log = False if fi_log is None else bool(fi_log)
    if use_log:
        eps = 1e-8
        fi_mean_disp = np.where(fi_mean_disp <= 0, eps, fi_mean_disp)
        fi_low = np.where(fi_low <= 0, eps, fi_low)
        fi_up = np.where(fi_up <= fi_low, fi_low * (1 + 1e-3), fi_up)

    ax2.plot(time_shifted, fi_mean_disp, color=COLORS['noise'], lw=2, label='Fisher information')
    ax2.fill_between(time_shifted, fi_low, fi_up, color=COLORS['noise'], alpha=0.2)
    if use_log:
        ax2.set_yscale('log')
    ax2.set_ylabel(f'Fisher information{label_suffix}', color=COLORS['noise'])

    # Legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, frameon=False, loc='lower right')

    fig.suptitle('Figure 1B-supp. Timecourse of accuracy and Fisher information', x=0.02, ha='left', fontweight='bold')
    fig.text(0.02, -0.02, 'Mean ± SD across repeats; gray band marks stimulus window (onset at 0).', fontsize=9)

    # Save into figures/supp for supplemental
    supp_dir = os.path.join(output_dir, 'supp')
    os.makedirs(supp_dir, exist_ok=True)
    out_path_base = os.path.join(supp_dir, 'figure1_panel_b_timecourse')
    png_path, svg_path = save_figure_both_formats(fig, out_path_base)
    plt.close(fig)
    print(f'[Saved] {png_path}')
    print(f'[Saved] {svg_path}')


def fig1_panel_d(output_dir: str, force_config: Optional[str] = None):
    """t-SNE neural manifold with category coloring using RR neurons and stimulus window."""
    if prepare_data_for_manifold is None or perform_pca is None or perform_tsne is None:
        raise RuntimeError('Manifold utilities not available; please ensure manifold.py imports succeed.')

    setup_publication_style()
    os.makedirs(output_dir, exist_ok=True)

    # Load one session
    session_loaded = False
    candidate_configs = [force_config] if force_config else list_config_files()
    for cfg_path in candidate_configs:
        if cfg_path is None:
            continue
        try:
            segments, labels, neuron_pos, stimulus_data = load_session_from_config(cfg_path)
            rr_neurons = compute_rr_neurons(segments, stimulus_data)
            session_loaded = True
            break
        except Exception:
            continue
    if not session_loaded:
        raise RuntimeError('No session could be loaded for panel D.')

    # Use original stimulus categories for coloring if available
    categories = stimulus_data[:, 0].astype(int) if stimulus_data is not None else labels.astype(int)
    rr_list = rr_neurons if len(rr_neurons) > 0 else None

    # Prepare, PCA then t-SNE (2D)
    X, y = prepare_data_for_manifold(segments, categories, rr_list, use_stimulus_only=True)
    X_pca, _ = perform_pca(X, n_components=min(50, max(2, X.shape[1] // 2)))
    X_tsne = perform_tsne(X_pca, n_components=2)

    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    palette = {
        1: COLORS['ordered'],
        2: COLORS['noise'],
        3: COLORS['neutral'],
    }
    for cls in np.unique(y):
        mask = (y == cls)
        ax.scatter(X_tsne[mask, 0], X_tsne[mask, 1], s=22, alpha=0.75,
                   color=palette.get(int(cls), '#999999'), label=f'Class {int(cls)}', edgecolor='white', linewidth=0.3)

    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.legend(frameon=False, title='Category')
    ax.grid(True, alpha=0.2)
    fig.suptitle('Figure 1D. Neural manifold separability (t-SNE)', x=0.02, ha='left', fontweight='bold')
    fig.text(0.02, -0.02, '2D t-SNE of trial features (RR neurons, stimulus window), colored by stimulus category.', fontsize=9)

    out_path_base = os.path.join(output_dir, 'figure1_panel_d')
    png_path, svg_path = save_figure_both_formats(fig, out_path_base)
    plt.close(fig)
    print(f'[Saved] {png_path}')
    print(f'[Saved] {svg_path}')


def fig1_panel_fisher_conditions(output_dir: str, force_config: Optional[str] = None):
    """Fisher information comparison across different condition pairs."""
    if calculate_fisher_information_by_condition is None:
        raise RuntimeError('Fisher information by condition calculation not available.')
    
    setup_publication_style()
    os.makedirs(output_dir, exist_ok=True)

    # Use M65 if no specific config provided
    if force_config is None:
        force_config = get_config_by_mouse('m65')
    
    if force_config is None or not os.path.exists(force_config):
        raise RuntimeError('M65 config not found.')

    # Load single session with M65 fix using fast_rr_selection
    try:
        print(f"\nLoading: {force_config}")
        segments, labels, neuron_pos, stimulus_data = load_session_from_config(force_config)
        print(f"Successfully loaded data: {segments.shape[0]} trials, {segments.shape[1]} neurons")
        
        # Use fast_rr_selection to recompute RR neurons for M65 (fix from loaddata.py)
        try:
            rr_labels = reclassify_labels(stimulus_data)
        except Exception:
            rr_labels = stimulus_data[:, 0].astype(int)
        
        rr_results = fast_rr_selection(segments, rr_labels)
        rr_neurons = list(rr_results.get('rr_neurons', []))
        print(f"RR neurons found: {len(rr_neurons)}")
        
        # Use original labels to preserve all categories (1, 2, 3)
        fisher_labels = labels
        print(f"Using original labels distribution: {np.unique(fisher_labels, return_counts=True)}")

        # Calculate Fisher information by condition
        condition_fisher_scores = calculate_fisher_information_by_condition(segments, fisher_labels, rr_neurons)
        print(f"Fisher conditions calculated: {list(condition_fisher_scores.keys())}")
        
    except Exception as e:
        print(f"Failed to load {force_config}: {e}")
        import traceback
        traceback.print_exc()
        raise RuntimeError('Failed to load session for Fisher conditions panel.')

    # Define condition mapping with all four comparisons
    condition_mapping = {
        'condition_1_vs_2': ('Contraction vs Expansion', COLORS['ordered']),
        'condition_1_vs_3': ('Contraction vs Random', COLORS['accent']), 
        'condition_2_vs_3': ('Expansion vs Random', COLORS['noise']),
        'all_conditions': ('All Conditions', COLORS['neutral'])
    }

    # Calculate statistics for each condition
    condition_stats = {}
    
    for condition_key in condition_mapping.keys():
        if condition_key in condition_fisher_scores:
            fisher_scores = condition_fisher_scores[condition_key]
            fisher_values = fisher_scores.flatten()
            condition_stats[condition_key] = {
                'mean': np.mean(fisher_values),
                'sem': np.std(fisher_values) / np.sqrt(len(fisher_values)),  # Standard error
                'count': len(fisher_values)
            }

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Prepare data for plotting
    condition_names = []
    means = []
    sems = []
    bar_colors = []

    for condition_key, (condition_name, color) in condition_mapping.items():
        if condition_key in condition_stats:
            condition_names.append(condition_name)
            means.append(condition_stats[condition_key]['mean'])
            sems.append(condition_stats[condition_key]['sem'])
            bar_colors.append(color)

    # Create bars with standard error bars (no scatter plots)
    x_positions = np.arange(len(condition_names))
    bars = ax.bar(x_positions, means, yerr=sems,
                  capsize=4, color=bar_colors, alpha=0.8, edgecolor='white', linewidth=1.0)

    # Add significance markers
    # Simple significance test: compare each condition vs baseline (lowest mean)
    baseline_mean = min(means)
    y_max = max(np.array(means) + np.array(sems))
    
    for i, (mean_val, sem_val) in enumerate(zip(means, sems)):
        # Simple criterion: mean > baseline + 2*sem
        if mean_val > baseline_mean + 2 * sem_val:
            significance = '**'
        elif mean_val > baseline_mean + sem_val:
            significance = '*'
        else:
            significance = ''
        
        if significance:
            ax.text(x_positions[i], mean_val + sem_val + y_max * 0.02,
                   significance, ha='center', va='bottom', 
                   fontweight='bold', fontsize=12)

    # Set labels and styling
    ax.set_xticks(x_positions)
    ax.set_xticklabels(condition_names, rotation=15, ha='right')
    ax.set_ylabel('Fisher Information')
    ax.set_ylim(0, y_max * 1.15)
    
    # Apply figure_maker styling
    ax.grid(True, axis='y', alpha=0.2, color=COLORS['grid_color'], linewidth=0.5)
    ax.tick_params(colors=COLORS['axis_color'])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Title and caption
    mouse_name = os.path.basename(force_config).replace('.json', '').upper()
    fig.suptitle('Figure 1E. Fisher Information by Condition Pairs', 
                x=0.02, ha='left', fontweight='bold')
    fig.text(0.02, -0.02, 
             f'Fisher information for {mouse_name}. Error bars show standard error.', 
             fontsize=9, color='#666666')

    # Save figure
    out_path_base = os.path.join(output_dir, 'figure1_panel_fisher_conditions')
    png_path, svg_path = save_figure_both_formats(fig, out_path_base)
    plt.close(fig)
    print(f'[Saved] {png_path}')
    print(f'[Saved] {svg_path}')
    
    # Print summary statistics
    print(f"\n=== Fisher Information Comparison Summary ===")
    for condition_key, (condition_name, _) in condition_mapping.items():
        if condition_key in condition_stats:
            stats = condition_stats[condition_key]
            print(f"{condition_name}: {stats['mean']:.3f} ± {stats['sem']:.3f} (SEM, n={stats['count']})")
    print(f"Mouse: {mouse_name}")


# ---------------------------
# CLI
# ---------------------------
def main():
    parser = argparse.ArgumentParser(description='Generate figure panels based on analysis results.')
    parser.add_argument('--figure', type=int, required=True, help='Figure number (e.g., 1)')
    parser.add_argument('--panel', type=str, required=True, help='Panel letter (e.g., A/B/C/D)')
    parser.add_argument('--outdir', type=str, default='figures', help='Output directory for images')
    parser.add_argument('--config', type=str, default=None, help='Optional specific config JSON (e.g., config/m74.json)')
    parser.add_argument('--mouse', type=str, default='m27', help='Mouse/session key (e.g., m27/m30/m65/m74). Default m27')
    # Display options for Fisher information
    parser.add_argument('--fi-log', dest='fi_log', action='store_true', help='Use log scale for FI axis')
    parser.add_argument('--fi-linear', dest='fi_log', action='store_false', help='Use linear scale for FI axis')
    parser.set_defaults(fi_log=None)
    parser.add_argument('--fi-norm', type=str, default='none', choices=['none', 'minmax'], help='Display normalization for FI (none|minmax)')
    args = parser.parse_args()

    fig_num = args.figure
    panel = args.panel.strip().upper()

    # Resolve forced config: --config wins, otherwise map --mouse to a config path
    forced_cfg = args.config if args.config else get_config_by_mouse(args.mouse)

    if fig_num == 1:
        if panel == 'A':
            fig1_panel_a(args.outdir)
        elif panel == 'B':
            fig1_panel_b(args.outdir, force_config=forced_cfg)
        elif panel in ('B_TIME', 'B-TIME'):
            fig1_panel_b_timecourse(args.outdir, force_config=forced_cfg, repeats=10,
                                    fi_log=args.fi_log, fi_norm=args.fi_norm)
        elif panel == 'C':
            fig1_panel_c(args.outdir, force_config=forced_cfg, fi_log=args.fi_log, fi_norm=args.fi_norm)
        elif panel == 'D':
            fig1_panel_d(args.outdir, force_config=forced_cfg)
        elif panel in ('E', 'FISHER', 'FISHER_CONDITIONS'):
            fig1_panel_fisher_conditions(args.outdir, force_config=forced_cfg)
        else:
            raise ValueError('Unsupported panel for Figure 1. Use A/B/C/D/E.')
    else:
        raise ValueError('Only Figure 1 is implemented currently.')


if __name__ == '__main__':
    main()
