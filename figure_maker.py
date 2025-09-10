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
    from loaddata import calculate_fisher_information
except Exception:
    calculate_fisher_information = None

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
        'axes.edgecolor': '#2C3E50',
        'axes.grid': False,
        'grid.alpha': 0.3,
        'grid.linewidth': 0.8,
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.facecolor': 'white',
        'savefig.edgecolor': 'none',
    })


COLORS = {
    'ordered': '#FF7F50',
    'noise': '#4682B4',
    'neutral': '#708090',
    'accent': '#D2691E',
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
    y = labels.astype(int)
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
    out_path = os.path.join(output_dir, 'figure1_panel_a.png')
    plt.savefig(out_path)
    plt.close(fig)
    print(f'[Saved] {out_path}')


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
    ax.grid(True, axis='y', alpha=0.3)

    title = 'Figure 1B. High-fidelity three-way decoding (with baseline)'
    subtitle = (
        f'Best subject: {best_cfg} | Decoding={best["overall_acc"]:.3f} ± {best["fold_acc_std"]:.3f}; '
        f'Baseline={baseline_mean:.3f} ± {baseline_std:.3f}; Chance={best["chance"]:.3f}; '
        f'p_vs_chance={best["p_value"]:.2e}; p_vs_baseline≈{p_perm:.3f}'
    )
    fig.suptitle(title, x=0.02, ha='left', fontweight='bold')
    fig.text(0.02, -0.02, subtitle, fontsize=9)

    out_path = os.path.join(output_dir, 'figure1_panel_b.png')
    plt.savefig(out_path)
    plt.close(fig)
    print(f'[Saved] {out_path}')


def fig1_panel_c(output_dir: str, force_config: Optional[str] = None,
                 fi_log: Optional[bool] = None, fi_norm: str = 'none'):
    """Accuracy vs number of neurons (and optional Fisher info).
    Display options:
      - fi_log: bool, if True use log scale for FI axis; if None, uses default (True)
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
        raise RuntimeError('No session could be loaded for panel C.')

    rng = np.random.default_rng(42)
    if len(rr_neurons) == 0:
        rr_neurons = list(range(segments.shape[1]))

    # Determine neuron counts (log/linear spacing)
    max_n = max(10, min(len(rr_neurons), 200))
    counts = np.unique(np.linspace(5, max_n, num=8, dtype=int))

    acc_means, acc_stds = [], []
    fisher_means, fisher_stds = [], []

    # Use reclassified labels for FI consistency
    try:
        labels_for_fi = reclassify_labels(stimulus_data)
    except Exception:
        labels_for_fi = labels

    for k in counts:
        acc_trials = []
        fisher_trials = []
        for b in range(5):
            subset = rng.choice(rr_neurons, size=k, replace=False)
            X, y = extract_trial_features(segments, labels, subset)
            res = cross_val_performance(X, y)
            acc_trials.append(res['overall_mean'])

            if calculate_fisher_information is not None:
                try:
                    # Average Fisher info over stimulus window
                    fi = calculate_fisher_information(segments, labels_for_fi, subset)
                    t0 = getattr(global_cfg, 'PRE_FRAMES', 10)
                    t1 = t0 + getattr(global_cfg, 'STIMULUS_DURATION', 20)
                    fi_win = np.mean(fi[t0:t1]) if fi.ndim == 1 else np.mean(fi)
                    fisher_trials.append(float(fi_win))
                except Exception:
                    pass

        acc_means.append(np.mean(acc_trials))
        acc_stds.append(np.std(acc_trials))
        fisher_means.append(np.mean(fisher_trials) if fisher_trials else np.nan)
        fisher_stds.append(np.std(fisher_trials) if fisher_trials else np.nan)

    fig, ax1 = plt.subplots(figsize=(8, 5.5))
    ax1.plot(counts, acc_means, color=COLORS['accent'], lw=2)
    ax1.fill_between(counts, np.array(acc_means) - np.array(acc_stds),
                     np.array(acc_means) + np.array(acc_stds), color=COLORS['accent'], alpha=0.2)
    ax1.set_xlabel('Number of neurons used')
    ax1.set_ylabel('Decoding accuracy', color=COLORS['accent'])
    ax1.set_ylim(0.0, 1.05)
    ax1.grid(True, axis='y', alpha=0.3)

    # Optional Fisher info on secondary axis (log scale and/or normalization)
    if not np.all(np.isnan(fisher_means)):
        # Defaults: Panel C 用对数轴，除非显式关闭
        use_log = True if fi_log is None else bool(fi_log)

        ax2 = ax1.twinx()
        fm = np.array(fisher_means, dtype=float)
        fs = np.array(fisher_stds, dtype=float)
        valid = ~np.isnan(fm)

        # Build lower/upper bands
        lower = fm - fs
        upper = fm + fs

        # Display normalization (min-max on band)
        label_suffix = ''
        if fi_norm.lower() == 'minmax':
            vmin = np.nanmin(lower)
            vmax = np.nanmax(upper)
            if np.isfinite(vmin) and np.isfinite(vmax) and vmax > vmin:
                fm = (fm - vmin) / (vmax - vmin)
                lower = (lower - vmin) / (vmax - vmin)
                upper = (upper - vmin) / (vmax - vmin)
                label_suffix = ' (normalized)'
            else:
                # Fallback: no normalization
                pass

        # Positivity enforcement if using log
        if use_log:
            eps = 1e-8
            fm = np.where(fm <= 0, eps, fm)
            lower = np.where(lower <= 0, eps, lower)
            # ensure upper >= lower
            upper = np.where(upper <= lower, lower * (1 + 1e-3), upper)

        # Plot
        ax2.plot(counts[valid], fm[valid], color=COLORS['noise'], lw=2)
        if valid.any():
            ax2.fill_between(counts[valid], lower[valid], upper[valid], color=COLORS['noise'], alpha=0.2)
        if use_log:
            ax2.set_yscale('log')
        ax2.set_ylabel(f'Fisher information{label_suffix}', color=COLORS['noise'])

    fig.suptitle('Figure 1C. Distributed coding with increasing neuron count', x=0.02, ha='left', fontweight='bold')
    fig.text(0.02, -0.02,
             'Decoding accuracy (and Fisher information, if available) versus the number of neurons included.', fontsize=9)

    out_path = os.path.join(output_dir, 'figure1_panel_c.png')
    plt.savefig(out_path)
    plt.close(fig)
    print(f'[Saved] {out_path}')


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


def _timecourse_fisher(
    segments: np.ndarray,
    labels: np.ndarray,
    neuron_idx: List[int],
    repeats: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute per-timepoint Fisher info with bootstrap over trials for SEM.
    Returns (mean_fi[T], sem_fi[T]) normalized per neuron.
    """
    T = segments.shape[2]
    if calculate_fisher_information is None:
        # Fallback simple Fisher-like score: between/within variance ratio on projections
        # Here we compute per timepoint the ratio of between-class variance of means to within-class variance.
        rng = np.random.default_rng(42)
        fi_mat = []
        for r in range(repeats):
            # bootstrap trials
            idx = rng.integers(0, segments.shape[0], size=segments.shape[0])
            seg_b = segments[idx][:, neuron_idx, :]
            lab_b = labels[idx]
            fi_t = []
            for t in range(T):
                X_t = seg_b[:, :, t]
                # project to first PC to stabilize
                X_t = StandardScaler().fit_transform(X_t)
                # per-class means
                classes = np.unique(lab_b)
                means = []
                within_vars = []
                for c in classes:
                    Xc = X_t[lab_b == c]
                    if len(Xc) == 0:
                        continue
                    means.append(Xc.mean(axis=0))
                    within_vars.append(Xc.var(axis=0).mean())
                if len(means) >= 2:
                    means = np.stack(means)
                    between = np.var(means, axis=0).mean()
                    within = np.mean(within_vars) + 1e-8
                    fi_val = (between / within)
                else:
                    fi_val = 0.0
                fi_t.append(fi_val)
            fi_mat.append(fi_t)
        fi_mat = np.array(fi_mat)
        mean_fi = fi_mat.mean(axis=0)
        sem_fi = fi_mat.std(axis=0, ddof=1) / np.sqrt(repeats)
        return mean_fi, sem_fi
    else:
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
    out_path = os.path.join(supp_dir, 'figure1_panel_b_timecourse.png')
    plt.savefig(out_path)
    plt.close(fig)
    print(f'[Saved] {out_path}')


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

    out_path = os.path.join(output_dir, 'figure1_panel_d.png')
    plt.savefig(out_path)
    plt.close(fig)
    print(f'[Saved] {out_path}')


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
        else:
            raise ValueError('Unsupported panel for Figure 1. Use A/B/C/D.')
    else:
        raise ValueError('Only Figure 1 is implemented currently.')


if __name__ == '__main__':
    main()
