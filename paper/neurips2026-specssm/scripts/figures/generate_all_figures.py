"""Generate all paper figures for SpecSSM NeurIPS 2026 submission."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures', 'main')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# NeurIPS-friendly style
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
})

COLORS = {
    'blue': '#2563EB',
    'red': '#DC2626',
    'green': '#16A34A',
    'orange': '#EA580C',
    'purple': '#9333EA',
    'gray': '#6B7280',
}

# ============================================================
# Figure 1: Activation Replay vs Re-prefill Scaling (log-log)
# ============================================================
def plot_replay_scaling():
    ctx_lens = [32, 64, 128, 512, 1024]
    reprefill = [299, 602, 1019, 3644, 8036]
    replay = [47, 46, 37, 36, 40]

    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.plot(ctx_lens, reprefill, 'o-', color=COLORS['red'], linewidth=2,
            markersize=7, label='Re-prefill $O(T)$', zorder=3)
    ax.plot(ctx_lens, replay, 's-', color=COLORS['blue'], linewidth=2,
            markersize=7, label='Activation replay $O(K)$', zorder=3)

    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.set_xlabel('Context length (tokens)')
    ax.set_ylabel('Latency (ms)')
    ax.set_xticks(ctx_lens)
    ax.set_xticklabels([str(c) for c in ctx_lens])

    # Annotate speedups
    for i, (c, rp, rl) in enumerate(zip(ctx_lens, reprefill, replay)):
        speedup = rp / rl
        if i in [0, 2, 4]:  # annotate 32, 128, 1024
            ax.annotate(f'{speedup:.0f}×',
                       xy=(c, (rp * rl)**0.5),
                       fontsize=9, ha='center', color=COLORS['green'],
                       fontweight='bold')

    ax.legend(loc='upper left', framealpha=0.9)
    ax.grid(True, alpha=0.3, which='both')
    ax.set_title('SSM State Resynchronization (CPU, Mamba2-27M)')

    fig.savefig(os.path.join(OUTPUT_DIR, 'replay_scaling.pdf'))
    fig.savefig(os.path.join(OUTPUT_DIR, 'replay_scaling.png'))
    plt.close(fig)
    print('  -> replay_scaling.pdf')

# ============================================================
# Figure 2: CPU Kernel Optimization Stack (bar chart)
# ============================================================
def plot_kernel_stack():
    engines = ['HF PyTorch\n(baseline)', 'AVX-512\nFP32 SSM', 'Fused BF16\n(full fwd)', 'INT8 VNNI\n(fused)']
    latency = [79, 57, 32, 9]
    speedup = [1.0, 1.1, 2.8, 7.5]
    colors = [COLORS['gray'], COLORS['blue'], COLORS['orange'], COLORS['green']]

    fig, ax1 = plt.subplots(figsize=(5.5, 3.5))

    bars = ax1.bar(range(len(engines)), latency, color=colors, width=0.6,
                   edgecolor='white', linewidth=0.5, zorder=3)
    ax1.set_ylabel('8-token draft latency (ms)')
    ax1.set_xticks(range(len(engines)))
    ax1.set_xticklabels(engines)
    ax1.set_ylim(0, 95)

    # Add latency labels on bars
    for bar, lat, sp in zip(bars, latency, speedup):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f'{lat} ms\n({sp}×)', ha='center', va='bottom', fontsize=9,
                fontweight='bold')

    # Add dashed line for GPU verify window
    ax1.axhline(y=16, color=COLORS['red'], linestyle='--', linewidth=1.5,
               label='GPU verify window (16 ms)', zorder=2)
    ax1.legend(loc='upper right', fontsize=9)

    ax1.grid(True, alpha=0.2, axis='y')
    ax1.set_title('CPU Kernel Optimization (Mamba2-27M, 8 tokens)')

    fig.savefig(os.path.join(OUTPUT_DIR, 'kernel_stack.pdf'))
    fig.savefig(os.path.join(OUTPUT_DIR, 'kernel_stack.png'))
    plt.close(fig)
    print('  -> kernel_stack.pdf')

# ============================================================
# Figure 3: Acceptance by Dataset (grouped bars for LLaMA-8B)
# ============================================================
def plot_acceptance_by_dataset():
    datasets = ['HumanEval', 'GSM8K', 'Alpaca', 'UltraChat', 'XSum']

    # Data from experiments (LLaMA-8B verifier)
    data = {
        'LLaMA-1B':       [4.88, 2.60, 4.35, 2.73, 2.39],
        '45M guided':     [2.45, 3.37, 3.42, 2.62, 2.23],
        '27M guided':     [2.07, 2.91, 2.75, 2.20, 1.72],
        '27M pretrained': [1.49, 2.09, 2.64, 1.87, 1.18],
    }

    x = np.arange(len(datasets))
    width = 0.19
    offsets = [-1.5, -0.5, 0.5, 1.5]
    colors = [COLORS['purple'], COLORS['green'], COLORS['blue'], COLORS['gray']]

    fig, ax = plt.subplots(figsize=(7, 3.5))

    for i, (label, vals) in enumerate(data.items()):
        ax.bar(x + offsets[i] * width, vals, width, label=label,
               color=colors[i], edgecolor='white', linewidth=0.5, zorder=3)

    ax.set_ylabel('Mean accepted tokens ($\\bar{n}_a$)')
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.legend(loc='upper right', ncol=2, fontsize=9)
    ax.grid(True, alpha=0.2, axis='y')
    ax.set_ylim(0, 5.5)
    ax.set_title('Acceptance by Dataset (LLaMA-3.1-8B verifier, greedy, $K=8$)')

    fig.savefig(os.path.join(OUTPUT_DIR, 'acceptance_by_dataset.pdf'))
    fig.savefig(os.path.join(OUTPUT_DIR, 'acceptance_by_dataset.png'))
    plt.close(fig)
    print('  -> acceptance_by_dataset.pdf')

# ============================================================
# Figure 4: Pipeline Overlap Timeline (Gantt-style)
# ============================================================
def plot_pipeline_timeline():
    fig, ax = plt.subplots(figsize=(7, 2.5))

    # Three rounds of pipeline overlap
    rounds = [
        # (device, start, duration, label, color)
        ('GPU', 0, 16, 'Verify #1', COLORS['red']),
        ('CPU', 0, 9, 'Draft #2', COLORS['blue']),
        ('GPU', 16, 16, 'Verify #2', COLORS['red']),
        ('CPU', 16, 9, 'Draft #3', COLORS['blue']),
        ('GPU', 32, 16, 'Verify #3', COLORS['red']),
        ('CPU', 32, 9, 'Draft #4', COLORS['blue']),
    ]

    device_y = {'GPU': 1, 'CPU': 0}

    for device, start, dur, label, color in rounds:
        y = device_y[device]
        ax.barh(y, dur, left=start, height=0.5, color=color, alpha=0.85,
                edgecolor='white', linewidth=1, zorder=3)
        ax.text(start + dur/2, y, label, ha='center', va='center',
               fontsize=8, color='white', fontweight='bold')

    ax.set_yticks([0, 1])
    ax.set_yticklabels(['CPU\n(Mamba2-27M)', 'GPU\n(LLaMA-8B)'])
    ax.set_xlabel('Time (ms)')
    ax.set_xlim(-1, 50)
    ax.set_ylim(-0.5, 1.8)
    ax.grid(True, alpha=0.2, axis='x')

    # Add annotation for overlap
    ax.annotate('Draft fits within\nverify window',
               xy=(8, 0.35), xytext=(25, 1.5),
               fontsize=9, ha='center',
               arrowprops=dict(arrowstyle='->', color=COLORS['green'], lw=1.5),
               color=COLORS['green'], fontweight='bold')

    ax.set_title('Asynchronous CPU-GPU Pipeline (projected)')

    fig.savefig(os.path.join(OUTPUT_DIR, 'pipeline_timeline.pdf'))
    fig.savefig(os.path.join(OUTPUT_DIR, 'pipeline_timeline.png'))
    plt.close(fig)
    print('  -> pipeline_timeline.pdf')

# ============================================================
# Figure 5: Guidance Improvement Factor (radar/bar per verifier)
# ============================================================
def plot_guidance_improvement():
    """Bar chart showing guidance improvement ratio across verifiers."""
    verifiers = ['LLaMA-8B\n(27M)', 'LLaMA-8B\n(45M)', 'LLaMA-70B\n(27M)', 'Gemma-4B\n(45M)']
    pretrained = [1.86, 1.52, 2.51, 0.63]
    guided = [2.33, 2.82, None, 0.94]  # 70B guided = TBD
    improvement = [25, 85, None, 49]

    fig, ax = plt.subplots(figsize=(5.5, 3.5))

    x = np.arange(len(verifiers))
    width = 0.35

    bars1 = ax.bar(x - width/2, pretrained, width, label='Pretrained',
                   color=COLORS['gray'], edgecolor='white', zorder=3)
    guided_vals = [g if g is not None else 0 for g in guided]
    guided_colors = [COLORS['green'] if g is not None else 'white' for g in guided]
    bars2 = ax.bar(x + width/2, guided_vals, width, label='Guided',
                   color=guided_colors, edgecolor=['white']*3 + ['white'],
                   zorder=3)

    # Hatch the 70B guided bar as TODO
    bars2[2].set_hatch('///')
    bars2[2].set_edgecolor(COLORS['green'])
    bars2[2].set_facecolor('white')

    # Annotate improvement percentages
    for i, imp in enumerate(improvement):
        if imp is not None:
            ax.text(x[i] + width/2, guided_vals[i] + 0.08,
                   f'+{imp}%', ha='center', fontsize=9,
                   color=COLORS['green'], fontweight='bold')
        else:
            ax.text(x[i] + width/2, 0.3,
                   'TODO', ha='center', fontsize=8,
                   color=COLORS['green'], fontstyle='italic')

    ax.set_ylabel('Mean accepted tokens ($\\bar{n}_a$)')
    ax.set_xticks(x)
    ax.set_xticklabels(verifiers)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.2, axis='y')
    ax.set_ylim(0, 3.2)
    ax.set_title('Guidance Improvement Across Verifiers')

    fig.savefig(os.path.join(OUTPUT_DIR, 'guidance_improvement.pdf'))
    fig.savefig(os.path.join(OUTPUT_DIR, 'guidance_improvement.png'))
    plt.close(fig)
    print('  -> guidance_improvement.pdf')


if __name__ == '__main__':
    print('Generating SpecSSM paper figures...')
    plot_replay_scaling()
    plot_kernel_stack()
    plot_acceptance_by_dataset()
    plot_pipeline_timeline()
    plot_guidance_improvement()
    print(f'All figures saved to {os.path.abspath(OUTPUT_DIR)}/')
