import sys, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    import noise_correlation_analysis as n

    # 1) Existing scientific figures from module
    n.render_scientific_noise_signal_from_saved()
    n.render_scientific_hub_peripheral_from_saved()
    if hasattr(n, 'render_scientific_shuffle_from_saved'):
        n.render_scientific_shuffle_from_saved()
    else:
        # Fallback: render shuffle figure directly from saved npz
        try:
            import numpy as _np
            import os as _os
            save_dir = n.cfg.get_figures_dir() if hasattr(n.cfg, 'get_figures_dir') else 'figures'
            _os.makedirs(save_dir, exist_ok=True)
            npz_path = _os.path.join(n.ncfg.get_results_dir(), 'shuffle_fisher_results.npz')
            d = _np.load(npz_path, allow_pickle=True)
            fractions = _np.array(d['shuffle_fractions']).astype(float)
            fi_means = _np.array(d['fisher_means']).astype(float)
            fi_stds = _np.array(d['fisher_stds']).astype(float)
            orig_fi = float(d['original_fisher_mean']) if 'original_fisher_mean' in d else None
            if 'degradation_percents' in d:
                deg_perc = _np.array(d['degradation_percents']).astype(float)
            else:
                deg_perc = _np.zeros_like(fi_means) if not orig_fi else (orig_fi - fi_means)/orig_fi*100.0

            fig, axes = plt.subplots(1, 2, figsize=(11, 6))
            ax = axes[0]
            ax.plot(fractions, fi_means, '-o', color='#2ECC71', linewidth=2.2, markersize=5)
            ax.fill_between(fractions, fi_means - fi_stds, fi_means + fi_stds, color='#2ECC71', alpha=0.25)
            if orig_fi is not None:
                ax.axhline(orig_fi, linestyle='--', color='#F5B041', linewidth=1.4, label=f'Original: {orig_fi:.3f}')
                ax.legend(frameon=False, loc='upper left')
            ax.set_xlabel('Shuffling Fraction'); ax.set_ylabel('Fisher Information')
            ax.set_title('Fisher Information Degradation', loc='left', fontweight='bold')
            ax.grid(True, alpha=0.3)

            ax2 = axes[1]
            bars = ax2.bar(fractions, deg_perc, width=0.06, color='#A23B72', edgecolor='black', linewidth=1.0, alpha=0.9)
            for b, v in zip(bars, deg_perc):
                ax2.text(b.get_x()+b.get_width()/2, v - (3 if v>0 else -3), f"{v:+.1f}%", ha='center', va='bottom' if v>0 else 'top', color='white', fontsize=9)
            ax2.set_xlabel('Shuffling Fraction'); ax2.set_ylabel('Information Degradation (%)')
            ax2.set_title('Information Loss Percentage', loc='left', fontweight='bold')
            ax2.grid(True, axis='y', alpha=0.3)
            fig.suptitle('Neuron Shuffling Effects on Fisher Information', y=0.98)
            fig.tight_layout()
            out = _os.path.join(save_dir, 'shuffle_fisher_scientific.png')
            plt.savefig(out, dpi=n.ncfg.DPI, bbox_inches='tight')
            plt.close(fig)
        except Exception as e3:
            print('WARN: shuffle fisher rendering fallback failed:', repr(e3))

    # Ensure a copy exists in project ./figures for convenience
    try:
        data_fig_dir = n.cfg.get_figures_dir() if hasattr(n.cfg, 'get_figures_dir') else None
        if data_fig_dir and os.path.isdir(data_fig_dir):
            src = os.path.join(data_fig_dir, 'shuffle_fisher_scientific.png')
            dst_dir = 'figures'
            os.makedirs(dst_dir, exist_ok=True)
            dst = os.path.join(dst_dir, 'shuffle_fisher_scientific.png')
            if os.path.exists(src):
                import shutil
                shutil.copyfile(src, dst)
    except Exception as e4:
        print('WARN: copy shuffle figure to ./figures failed:', repr(e4))

    # 2) Additional scatter without regression line
    try:
        noise_npz = os.path.join(n.ncfg.get_results_dir(), 'noise_correlation_matrices.npz')
        signal_npz = os.path.join(n.ncfg.get_results_dir(), 'signal_correlation_matrices.npz')
        noise = np.load(noise_npz)
        signal = np.load(signal_npz)
        save_dir = n.cfg.get_figures_dir() if hasattr(n.cfg, 'get_figures_dir') else 'figures'
        os.makedirs(save_dir, exist_ok=True)

        for key in noise.files:
            cond = int(key.split('_')[-1])
            noise_mat = noise[key]
            sig_key = f'condition_{cond}'
            if sig_key not in signal.files:
                continue
            signal_mat = signal[sig_key]
            iu = np.triu_indices_from(noise_mat, k=1)
            x = noise_mat[iu]
            y = signal_mat[iu]
            fig, ax = plt.subplots(figsize=(6.5, 5.5))
            hb = ax.hexbin(x, y, gridsize=60, cmap='Blues', bins='log', mincnt=1)
            cbar = fig.colorbar(hb, ax=ax); cbar.set_label('Count')
            lim = (-1.0, 1.0)
            ax.plot(lim, lim, linestyle=':', color='gray', linewidth=1.2)
            from scipy.stats import pearsonr
            r, p = pearsonr(x, y)
            ax.text(0.03, 0.97, f"r = {r:.3f}\np = {p:.1e}\nn = {x.size}", transform=ax.transAxes,
                    va='top', ha='left', bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.85, edgecolor='#cccccc'))
            ax.set_xlim(lim); ax.set_ylim(lim)
            ax.set_xlabel('Noise Correlation'); ax.set_ylabel('Signal Correlation')
            ax.set_title(f'Noise vs Signal Correlation — Condition {cond}', loc='left', fontweight='bold')
            out = os.path.join(save_dir, f'noise_signal_scatter_no_reg_condition_{cond}.png')
            plt.savefig(out, dpi=n.ncfg.DPI, bbox_inches='tight')
            plt.close(fig)
    except Exception as e2:
        print('WARN: extra scatter w/o regression failed:', repr(e2))

    print('Rendered scientific noise figures into figures/.')
except Exception as e:
    print('FAILED rendering scientific noise figures:', repr(e))
    sys.exit(1)
