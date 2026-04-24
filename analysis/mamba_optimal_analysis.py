#!/usr/bin/env python3
"""
Mamba Speculative Decoding: Optimal Configuration Analysis

Focus: Deep analysis of Mamba model performance at optimal K values
Goal: Understand best achievable performance and guide training improvements
Baseline: Llama-3.2-1B for reference only
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def load_results(json_path):
    """Load sweep results from JSON file."""
    with open(json_path, 'r') as f:
        data = json.load(f)
        if isinstance(data, list):
            return {'experiments': data}
        return data


def extract_draft_model_name(full_name):
    """Extract shorter name from full draft model path."""
    if 'llama-3.2-1B' in full_name:
        return 'Llama-3.2-1B'
    elif 'mamba-65m-pretrained' in full_name:
        return 'Mamba-65m-pretrained'
    elif 'checkpoint-1000' in full_name:
        return 'Mamba-1k'
    elif 'checkpoint-2000' in full_name:
        return 'Mamba-2k'
    elif 'checkpoint-3000' in full_name:
        return 'Mamba-3k'
    elif 'distilled-best' in full_name:
        return 'Mamba-best'
    return full_name


def analyze_mamba_optimal_performance(results, output_dir, target_filter=None, tf32_filter=None):
    """
    Deep analysis of Mamba models at optimal K configurations.
    Focus on understanding best achievable performance vs current distillation.
    """
    
    target_name = target_filter if target_filter else "All Targets"
    tf32_name = "TF32 ON" if tf32_filter is True else "TF32 OFF" if tf32_filter is False else "TF32 All"
    
    print("\n" + "="*120)
    print(f"MAMBA OPTIMAL PERFORMANCE ANALYSIS")
    print(f"Target: {target_name} | TF32: {tf32_name}")
    print("="*120)
    
    experiments = results['experiments']
    spec_exps = [e for e in experiments if e['method'] != 'autoregressive']
    
    if target_filter:
        spec_exps = [e for e in spec_exps if e['target_model'] == target_filter]
    if tf32_filter is not None:
        spec_exps = [e for e in spec_exps if e['tf32_enabled'] == tf32_filter]
    
    # Separate Mamba and baseline
    mamba_exps = [e for e in spec_exps if 'mamba' in e['draft_model'].lower()]
    llama_exps = [e for e in spec_exps if 'llama-3.2-1B' in e['draft_model'].lower()]
    
    # Group Mamba by (variant, sample_type) to find best K
    mamba_grouped = defaultdict(list)
    for exp in mamba_exps:
        variant = extract_draft_model_name(exp['draft_model'])
        key = (variant, exp['sample_set'])
        mamba_grouped[key].append(exp)
    
    # Find best config for each group
    mamba_best = {}
    for key, exps_list in mamba_grouped.items():
        best = max(exps_list, key=lambda x: x['speedup_vs_auto'])
        mamba_best[key] = best
    
    # Calculate Llama-3.2-1B baseline (for reference)
    llama_grouped = defaultdict(list)
    for exp in llama_exps:
        llama_grouped[exp['sample_set']].append(exp)
    
    llama_best = {}
    for sample_type, exps_list in llama_grouped.items():
        best = max(exps_list, key=lambda x: x['speedup_vs_auto'])
        llama_best[sample_type] = best
    
    print(f"\n1. BEST ACHIEVABLE PERFORMANCE (Optimal K per Configuration)")
    print(f"{'-'*120}")
    print(f"Baseline Reference - Llama-3.2-1B (Transformer):")
    for sample_type in ['short', 'combined', 'long']:
        if sample_type in llama_best:
            best = llama_best[sample_type]
            print(f"  {sample_type.capitalize():<10}: K={best['lookahead']:<2} → {best['speedup_vs_auto']:.3f}x "
                  f"@ {best['acceptance_rate']*100:.1f}% accept")
    
    print(f"\n{'-'*120}")
    print(f"Mamba Variants (SSM Architecture):")
    print(f"{'-'*120}")
    
    # Organize by sample type for clear comparison
    mamba_variants = ['Mamba-65m-pretrained', 'Mamba-1k', 'Mamba-2k', 'Mamba-3k', 'Mamba-best']
    
    for sample_type in ['short', 'combined', 'long']:
        print(f"\n{sample_type.upper()} Prompts:")
        print(f"  {'Variant':<25} | {'Best K':<8} | {'Speedup':<10} | {'Accept %':<10} | "
              f"{'vs Llama':<12} | {'Gap':<8}")
        print(f"  {'-'*25}+{'-'*10}+{'-'*12}+{'-'*12}+{'-'*14}+{'-'*10}")
        
        llama_ref = llama_best[sample_type]['speedup_vs_auto'] if sample_type in llama_best else 1.0
        
        for variant in mamba_variants:
            key = (variant, sample_type)
            if key in mamba_best:
                best = mamba_best[key]
                gap = llama_ref - best['speedup_vs_auto']
                gap_pct = (best['speedup_vs_auto'] / llama_ref - 1) * 100
                
                print(f"  {variant:<25} | K={best['lookahead']:<6} | {best['speedup_vs_auto']:>8.3f}x | "
                      f"{best['acceptance_rate']*100:>8.1f}% | {gap_pct:>+10.1f}% | {gap:>6.3f}x")
    
    print(f"\n2. DISTILLATION EFFECTIVENESS AT OPTIMAL K")
    print(f"{'-'*120}")
    
    # Group by variant for overall statistics
    variant_stats = defaultdict(lambda: {'best_speedups': [], 'avg_all_k': []})
    
    for variant in mamba_variants:
        # Get best speedups (optimal K)
        for sample_type in ['short', 'combined', 'long']:
            key = (variant, sample_type)
            if key in mamba_best:
                variant_stats[variant]['best_speedups'].append(mamba_best[key]['speedup_vs_auto'])
        
        # Get average across all K for comparison
        variant_exps = [e for e in mamba_exps if extract_draft_model_name(e['draft_model']) == variant]
        variant_stats[variant]['avg_all_k'] = [e['speedup_vs_auto'] for e in variant_exps]
    
    print(f"{'Variant':<25} | {'Avg Best K':<12} | {'Peak':<10} | {'Avg All K':<12} | "
          f"{'Gain from K*':<14} | {'vs Pretrain':<12}")
    print(f"{'-'*25}+{'-'*14}+{'-'*12}+{'-'*14}+{'-'*16}+{'-'*14}")
    
    pretrain_best = np.mean(variant_stats['Mamba-65m-pretrained']['best_speedups'])
    
    for variant in mamba_variants:
        stats = variant_stats[variant]
        if stats['best_speedups']:
            avg_best = np.mean(stats['best_speedups'])
            peak = np.max(stats['best_speedups'])
            avg_all = np.mean(stats['avg_all_k'])
            gain = (avg_best - avg_all) / avg_all * 100
            vs_pretrain = (avg_best - pretrain_best) / pretrain_best * 100 if variant != 'Mamba-65m-pretrained' else 0
            
            print(f"{variant:<25} | {avg_best:>10.3f}x | {peak:>8.3f}x | {avg_all:>10.3f}x | "
                  f"{gain:>+12.1f}% | {vs_pretrain:>+10.1f}%")
    
    return mamba_best, llama_best, variant_stats


def analyze_k_value_sensitivity(results, output_dir, target_filter=None, tf32_filter=None):
    """
    Analyze how different K values affect Mamba performance.
    Key question: Which K values work best for which scenarios?
    """
    
    target_name = target_filter if target_filter else "All Targets"
    tf32_name = "TF32 ON" if tf32_filter is True else "TF32 OFF" if tf32_filter is False else "TF32 All"
    
    print(f"\n{'='*120}")
    print(f"K-VALUE SENSITIVITY ANALYSIS FOR MAMBA MODELS")
    print(f"Target: {target_name} | TF32: {tf32_name}")
    print(f"{'='*120}")
    
    experiments = results['experiments']
    spec_exps = [e for e in experiments if e['method'] != 'autoregressive']
    
    if target_filter:
        spec_exps = [e for e in spec_exps if e['target_model'] == target_filter]
    if tf32_filter is not None:
        spec_exps = [e for e in spec_exps if e['tf32_enabled'] == tf32_filter]
    
    mamba_exps = [e for e in spec_exps if 'mamba' in e['draft_model'].lower()]
    
    # Analyze K-value patterns
    k_analysis = defaultdict(lambda: defaultdict(list))
    
    for exp in mamba_exps:
        variant = extract_draft_model_name(exp['draft_model'])
        k = exp['lookahead']
        k_analysis[variant][k].append({
            'speedup': exp['speedup_vs_auto'],
            'accept': exp['acceptance_rate'] * 100,
            'sample': exp['sample_set']
        })
    
    mamba_variants = ['Mamba-65m-pretrained', 'Mamba-1k', 'Mamba-2k', 'Mamba-3k', 'Mamba-best']
    
    print(f"\n1. K-VALUE PERFORMANCE RANKING (Best to Worst)")
    print(f"{'-'*120}")
    
    for variant in mamba_variants:
        if variant not in k_analysis:
            continue
            
        print(f"\n{variant}:")
        
        k_stats = []
        for k in sorted(k_analysis[variant].keys()):
            data = k_analysis[variant][k]
            avg_speedup = np.mean([d['speedup'] for d in data])
            success_rate = len([d for d in data if d['speedup'] > 1.0]) / len(data) * 100
            avg_accept = np.mean([d['accept'] for d in data])
            std_speedup = np.std([d['speedup'] for d in data])
            k_stats.append((k, avg_speedup, success_rate, avg_accept, std_speedup))
        
        # Sort by average speedup
        k_stats.sort(key=lambda x: x[1], reverse=True)
        
        print(f"  {'Rank':<6} | {'K':<4} | {'Avg Speedup':<12} | {'Success %':<10} | "
              f"{'Avg Accept':<12} | {'Stability':<12}")
        print(f"  {'-'*6}+{'-'*6}+{'-'*14}+{'-'*12}+{'-'*14}+{'-'*14}")
        
        for rank, (k, avg_speedup, success_rate, avg_accept, std_speedup) in enumerate(k_stats, 1):
            stability = "Stable" if std_speedup < 0.15 else "Variable" if std_speedup < 0.25 else "Unstable"
            marker = "⭐" if rank == 1 else "✓" if rank <= 2 else "○"
            print(f"  {marker} {rank:<3} | K={k:<2} | {avg_speedup:>10.3f}x | {success_rate:>8.1f}% | "
                  f"{avg_accept:>10.1f}% | {stability:<12}")
    
    print(f"\n2. K-VALUE RECOMMENDATIONS BY SCENARIO")
    print(f"{'-'*120}")
    
    # Analyze by sample type
    scenario_k = defaultdict(lambda: defaultdict(list))
    
    for exp in mamba_exps:
        variant = extract_draft_model_name(exp['draft_model'])
        key = (variant, exp['sample_set'])
        scenario_k[key][exp['lookahead']].append(exp['speedup_vs_auto'])
    
    for sample_type in ['short', 'combined', 'long']:
        print(f"\n{sample_type.upper()} Prompts - Optimal K:")
        print(f"  {'Variant':<25} | {'Best K':<8} | {'Speedup':<10} | {'2nd Best K':<10} | "
              f"{'Speedup':<10} | {'K Matters?':<12}")
        print(f"  {'-'*25}+{'-'*10}+{'-'*12}+{'-'*12}+{'-'*12}+{'-'*14}")
        
        for variant in mamba_variants:
            key = (variant, sample_type)
            if key in scenario_k:
                k_perfs = []
                for k, speedups in scenario_k[key].items():
                    k_perfs.append((k, np.mean(speedups)))
                
                k_perfs.sort(key=lambda x: x[1], reverse=True)
                
                if len(k_perfs) >= 2:
                    best_k, best_speedup = k_perfs[0]
                    second_k, second_speedup = k_perfs[1]
                    gap = best_speedup - second_speedup
                    sensitivity = "Critical" if gap > 0.2 else "Important" if gap > 0.1 else "Moderate"
                    
                    print(f"  {variant:<25} | K={best_k:<6} | {best_speedup:>8.3f}x | "
                          f"K={second_k:<8} | {second_speedup:>8.3f}x | {sensitivity:<12}")


def analyze_failure_modes(results, output_dir, target_filter=None, tf32_filter=None):
    """
    Deep dive into when and why Mamba fails (speedup < 1.0).
    Critical for understanding training improvements needed.
    """
    
    target_name = target_filter if target_filter else "All Targets"
    tf32_name = "TF32 ON" if tf32_filter is True else "TF32 OFF" if tf32_filter is False else "TF32 All"
    
    print(f"\n{'='*120}")
    print(f"MAMBA FAILURE MODE ANALYSIS")
    print(f"Target: {target_name} | TF32: {tf32_name}")
    print(f"{'='*120}")
    
    experiments = results['experiments']
    spec_exps = [e for e in experiments if e['method'] != 'autoregressive']
    
    if target_filter:
        spec_exps = [e for e in spec_exps if e['target_model'] == target_filter]
    if tf32_filter is not None:
        spec_exps = [e for e in spec_exps if e['tf32_enabled'] == tf32_filter]
    
    mamba_exps = [e for e in spec_exps if 'mamba' in e['draft_model'].lower()]
    
    # Analyze failures
    failures = [e for e in mamba_exps if e['speedup_vs_auto'] < 1.0]
    successes = [e for e in mamba_exps if e['speedup_vs_auto'] >= 1.0]
    
    print(f"\n1. FAILURE STATISTICS AT SUBOPTIMAL K")
    print(f"{'-'*120}")
    print(f"Total Mamba experiments: {len(mamba_exps)}")
    print(f"Failures (speedup < 1.0): {len(failures)} ({len(failures)/len(mamba_exps)*100:.1f}%)")
    print(f"Successes (speedup ≥ 1.0): {len(successes)} ({len(successes)/len(mamba_exps)*100:.1f}%)")
    
    # Now with optimal K
    mamba_grouped = defaultdict(list)
    for exp in mamba_exps:
        variant = extract_draft_model_name(exp['draft_model'])
        key = (variant, exp['sample_set'])
        mamba_grouped[key].append(exp)
    
    optimal_failures = 0
    optimal_total = 0
    for key, exps_list in mamba_grouped.items():
        best = max(exps_list, key=lambda x: x['speedup_vs_auto'])
        optimal_total += 1
        if best['speedup_vs_auto'] < 1.0:
            optimal_failures += 1
    
    print(f"\nWith OPTIMAL K selection:")
    print(f"Total configurations: {optimal_total}")
    print(f"Failures at optimal K: {optimal_failures} ({optimal_failures/optimal_total*100:.1f}%)")
    print(f"Successes at optimal K: {optimal_total - optimal_failures} "
          f"({(optimal_total-optimal_failures)/optimal_total*100:.1f}%)")
    
    print(f"\n2. FAILURE PATTERNS BY VARIANT")
    print(f"{'-'*120}")
    
    variant_failures = defaultdict(lambda: {'total': 0, 'fail': 0, 'worst': None})
    
    for exp in mamba_exps:
        variant = extract_draft_model_name(exp['draft_model'])
        variant_failures[variant]['total'] += 1
        if exp['speedup_vs_auto'] < 1.0:
            variant_failures[variant]['fail'] += 1
            if (variant_failures[variant]['worst'] is None or 
                exp['speedup_vs_auto'] < variant_failures[variant]['worst']['speedup']):
                variant_failures[variant]['worst'] = {
                    'speedup': exp['speedup_vs_auto'],
                    'k': exp['lookahead'],
                    'sample': exp['sample_set'],
                    'accept': exp['acceptance_rate'] * 100
                }
    
    print(f"{'Variant':<25} | {'Total':<8} | {'Failures':<10} | {'Rate':<8} | "
          f"{'Worst Case':<50}")
    print(f"{'-'*25}+{'-'*10}+{'-'*12}+{'-'*10}+{'-'*52}")
    
    for variant in ['Mamba-65m-pretrained', 'Mamba-1k', 'Mamba-2k', 'Mamba-3k', 'Mamba-best']:
        if variant in variant_failures:
            stats = variant_failures[variant]
            rate = stats['fail'] / stats['total'] * 100
            worst = stats['worst']
            worst_str = (f"{worst['speedup']:.3f}x @ K={worst['k']}, "
                        f"{worst['sample']}, {worst['accept']:.1f}% accept") if worst else "N/A"
            
            print(f"{variant:<25} | {stats['total']:<8} | {stats['fail']:<10} | {rate:>6.1f}% | {worst_str:<50}")
    
    print(f"\n3. ROOT CAUSE ANALYSIS")
    print(f"{'-'*120}")
    
    # Analyze acceptance rates for failures vs successes
    fail_accepts = [e['acceptance_rate'] * 100 for e in failures]
    success_accepts = [e['acceptance_rate'] * 100 for e in successes]
    
    if fail_accepts and success_accepts:
        print(f"Acceptance Rate Comparison:")
        print(f"  Failures: {np.mean(fail_accepts):.1f}% avg (range: {np.min(fail_accepts):.1f}% - "
              f"{np.max(fail_accepts):.1f}%)")
        print(f"  Successes: {np.mean(success_accepts):.1f}% avg (range: {np.min(success_accepts):.1f}% - "
              f"{np.max(success_accepts):.1f}%)")
        print(f"  Gap: {np.mean(success_accepts) - np.mean(fail_accepts):.1f} percentage points")
        
        # Find acceptance threshold
        sorted_accepts = sorted([(e['acceptance_rate']*100, e['speedup_vs_auto']) 
                                for e in mamba_exps], key=lambda x: x[0])
        
        # Find approximate threshold where success rate crosses 50%
        for i in range(len(sorted_accepts)):
            window = sorted_accepts[max(0, i-5):min(len(sorted_accepts), i+6)]
            success_rate = len([x for x in window if x[1] >= 1.0]) / len(window)
            if 0.45 <= success_rate <= 0.55:
                print(f"\nCritical Acceptance Threshold: ~{sorted_accepts[i][0]:.1f}%")
                print(f"  Below this: High failure risk")
                print(f"  Above this: Success likely")
                break


def create_mamba_optimization_plots(results, output_dir, target_filter=None, tf32_filter=None):
    """
    Create focused visualizations for Mamba optimization insights.
    """
    
    target_name = target_filter.replace('/', '_') if target_filter else "all_targets"
    tf32_label = "tf32_on" if tf32_filter is True else "tf32_off" if tf32_filter is False else "tf32_all"
    
    print(f"\nGenerating Mamba optimization plots for {target_filter}, TF32 {'ON' if tf32_filter else 'OFF'}...")
    
    experiments = results['experiments']
    spec_exps = [e for e in experiments if e['method'] != 'autoregressive']
    
    if target_filter:
        spec_exps = [e for e in spec_exps if e['target_model'] == target_filter]
    if tf32_filter is not None:
        spec_exps = [e for e in spec_exps if e['tf32_enabled'] == tf32_filter]
    
    mamba_exps = [e for e in spec_exps if 'mamba' in e['draft_model'].lower()]
    llama_exps = [e for e in spec_exps if 'llama-3.2-1B' in e['draft_model'].lower()]
    
    # Prepare data for plotting
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle(f'Mamba Optimization Analysis\n{target_filter} | TF32 {"ON" if tf32_filter else "OFF"}', 
                 fontsize=16, fontweight='bold')
    
    # Plot 1: Best achievable vs average performance
    ax1 = axes[0, 0]
    
    variants = ['Mamba-65m-pretrained', 'Mamba-1k', 'Mamba-2k', 'Mamba-3k', 'Mamba-best']
    variant_best = []
    variant_avg = []
    
    for variant in variants:
        var_exps = [e for e in mamba_exps if extract_draft_model_name(e['draft_model']) == variant]
        
        # Best per sample type
        grouped = defaultdict(list)
        for e in var_exps:
            grouped[e['sample_set']].append(e)
        
        best_vals = [max(exps, key=lambda x: x['speedup_vs_auto'])['speedup_vs_auto'] 
                     for exps in grouped.values()]
        variant_best.append(np.mean(best_vals))
        variant_avg.append(np.mean([e['speedup_vs_auto'] for e in var_exps]))
    
    x = np.arange(len(variants))
    width = 0.35
    ax1.bar(x - width/2, variant_avg, width, label='Avg All K', alpha=0.7, color='lightcoral')
    ax1.bar(x + width/2, variant_best, width, label='Avg Best K', alpha=0.7, color='seagreen')
    ax1.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, linewidth=1)
    ax1.set_ylabel('Speedup', fontsize=11)
    ax1.set_title('Impact of Optimal K Selection', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([v.replace('Mamba-', '') for v in variants], rotation=45, ha='right')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # Plot 2: K-value heatmap for Mamba-1k
    ax2 = axes[0, 1]
    
    mamba_1k = [e for e in mamba_exps if 'checkpoint-1000' in e['draft_model']]
    k_values = sorted(set(e['lookahead'] for e in mamba_1k))
    samples = ['short', 'combined', 'long']
    
    heatmap_data = np.zeros((len(samples), len(k_values)))
    for i, sample in enumerate(samples):
        for j, k in enumerate(k_values):
            matching = [e for e in mamba_1k if e['sample_set'] == sample and e['lookahead'] == k]
            if matching:
                heatmap_data[i, j] = matching[0]['speedup_vs_auto']
    
    im = ax2.imshow(heatmap_data, cmap='RdYlGn', aspect='auto', vmin=0.5, vmax=2.0)
    ax2.set_xticks(np.arange(len(k_values)))
    ax2.set_yticks(np.arange(len(samples)))
    ax2.set_xticklabels([f'K={k}' for k in k_values])
    ax2.set_yticklabels([s.capitalize() for s in samples])
    ax2.set_title('Mamba-1k: K-Value Sensitivity', fontsize=12, fontweight='bold')
    
    # Add text annotations
    for i in range(len(samples)):
        for j in range(len(k_values)):
            text = ax2.text(j, i, f'{heatmap_data[i, j]:.2f}',
                           ha="center", va="center", color="black", fontsize=9)
    
    plt.colorbar(im, ax=ax2, label='Speedup')
    
    # Plot 3: Distillation progress curve
    ax3 = axes[0, 2]
    
    distill_order = ['Mamba-65m-pretrained', 'Mamba-1k', 'Mamba-2k', 'Mamba-3k', 'Mamba-best']
    distill_steps = [0, 1000, 2000, 3000, -1]  # -1 for "best"
    
    # Get best performance for each
    distill_perf = []
    for variant in distill_order:
        var_exps = [e for e in mamba_exps if extract_draft_model_name(e['draft_model']) == variant]
        grouped = defaultdict(list)
        for e in var_exps:
            grouped[e['sample_set']].append(e)
        best_vals = [max(exps, key=lambda x: x['speedup_vs_auto'])['speedup_vs_auto'] 
                     for exps in grouped.values()]
        distill_perf.append(np.mean(best_vals))
    
    ax3.plot(range(len(distill_order)), distill_perf, 'o-', linewidth=2, markersize=8, color='steelblue')
    ax3.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, linewidth=1)
    ax3.set_xticks(range(len(distill_order)))
    ax3.set_xticklabels([v.replace('Mamba-', '') for v in distill_order], rotation=45, ha='right')
    ax3.set_ylabel('Avg Best Speedup', fontsize=11)
    ax3.set_title('Distillation Progress (Optimal K)', fontsize=12, fontweight='bold')
    ax3.grid(alpha=0.3)
    
    # Annotate values
    for i, (variant, perf) in enumerate(zip(distill_order, distill_perf)):
        ax3.annotate(f'{perf:.2f}x', (i, perf), textcoords="offset points", 
                    xytext=(0,10), ha='center', fontsize=9)
    
    # Plot 4: Acceptance rate vs speedup (scatter)
    ax4 = axes[1, 0]
    
    for variant in variants:
        var_exps = [e for e in mamba_exps if extract_draft_model_name(e['draft_model']) == variant]
        accepts = [e['acceptance_rate'] * 100 for e in var_exps]
        speedups = [e['speedup_vs_auto'] for e in var_exps]
        ax4.scatter(accepts, speedups, alpha=0.6, label=variant.replace('Mamba-', ''), s=30)
    
    ax4.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, linewidth=1)
    ax4.axvline(x=70, color='orange', linestyle=':', alpha=0.5, linewidth=1, label='~70% threshold')
    ax4.set_xlabel('Acceptance Rate (%)', fontsize=11)
    ax4.set_ylabel('Speedup', fontsize=11)
    ax4.set_title('Acceptance Rate vs Performance', fontsize=12, fontweight='bold')
    ax4.legend(fontsize=8, loc='lower right')
    ax4.grid(alpha=0.3)
    
    # Plot 5: Sample type comparison
    ax5 = axes[1, 1]
    
    sample_data = {s: [] for s in ['short', 'combined', 'long']}
    for sample in sample_data.keys():
        for variant in variants:
            var_exps = [e for e in mamba_exps if extract_draft_model_name(e['draft_model']) == variant 
                       and e['sample_set'] == sample]
            if var_exps:
                best = max(var_exps, key=lambda x: x['speedup_vs_auto'])
                sample_data[sample].append(best['speedup_vs_auto'])
    
    x = np.arange(len(variants))
    width = 0.25
    for i, (sample, data) in enumerate(sample_data.items()):
        ax5.bar(x + i*width - width, data, width, label=sample.capitalize(), alpha=0.7)
    
    ax5.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, linewidth=1)
    ax5.set_ylabel('Best Speedup', fontsize=11)
    ax5.set_title('Performance by Prompt Length (Optimal K)', fontsize=12, fontweight='bold')
    ax5.set_xticks(x)
    ax5.set_xticklabels([v.replace('Mamba-', '') for v in variants], rotation=45, ha='right')
    ax5.legend()
    ax5.grid(axis='y', alpha=0.3)
    
    # Plot 6: Gap to Llama-3.2-1B baseline
    ax6 = axes[1, 2]
    
    # Get Llama best per sample
    llama_grouped = defaultdict(list)
    for e in llama_exps:
        llama_grouped[e['sample_set']].append(e)
    
    llama_best_per_sample = {}
    for sample, exps in llama_grouped.items():
        llama_best_per_sample[sample] = max(exps, key=lambda x: x['speedup_vs_auto'])['speedup_vs_auto']
    
    gaps = []
    for variant in variants:
        var_gaps = []
        for sample in ['short', 'combined', 'long']:
            var_exps = [e for e in mamba_exps if extract_draft_model_name(e['draft_model']) == variant 
                       and e['sample_set'] == sample]
            if var_exps and sample in llama_best_per_sample:
                best = max(var_exps, key=lambda x: x['speedup_vs_auto'])['speedup_vs_auto']
                gap_pct = (best / llama_best_per_sample[sample] - 1) * 100
                var_gaps.append(gap_pct)
        gaps.append(np.mean(var_gaps) if var_gaps else 0)
    
    colors = ['green' if g > -10 else 'orange' if g > -30 else 'red' for g in gaps]
    ax6.barh(range(len(variants)), gaps, color=colors, alpha=0.7)
    ax6.axvline(x=0, color='black', linestyle='-', linewidth=1)
    ax6.axvline(x=-20, color='orange', linestyle=':', alpha=0.5, linewidth=1)
    ax6.set_xlabel('Gap to Llama-3.2-1B (%)', fontsize=11)
    ax6.set_title('Performance Gap vs Baseline', fontsize=12, fontweight='bold')
    ax6.set_yticks(range(len(variants)))
    ax6.set_yticklabels([v.replace('Mamba-', '') for v in variants])
    ax6.grid(axis='x', alpha=0.3)
    
    # Annotate values
    for i, gap in enumerate(gaps):
        ax6.text(gap - 2 if gap < 0 else gap + 2, i, f'{gap:.1f}%', 
                va='center', ha='right' if gap < 0 else 'left', fontsize=9)
    
    plt.tight_layout()
    
    # Save figure
    output_path = output_dir / 'mamba_optimization_analysis.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: {output_path.name}")


def main():
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python mamba_optimal_analysis.py <sweep_results.json>")
        sys.exit(1)
    
    json_path = sys.argv[1]
    results = load_results(json_path)
    
    base_output = Path("outputs/sweep_results/mamba_optimal_analysis")
    base_output.mkdir(parents=True, exist_ok=True)
    
    print(f"Loaded {len(results['experiments'])} experiments")
    
    targets = list(set(e['target_model'] for e in results['experiments'] if e['method'] != 'autoregressive'))
    print(f"Target models found: {targets}")
    
    tf32_settings = [False, True]
    tf32_labels = {False: "tf32_off", True: "tf32_on"}
    
    # Run analysis for each combination
    for target in targets:
        for tf32 in tf32_settings:
            print(f"\n{'#'*120}")
            print(f"# MAMBA ANALYSIS: {target} | TF32 {'ON' if tf32 else 'OFF'}")
            print(f"{'#'*120}")
            
            # Create output directory
            target_clean = target.replace('/', '_')
            output_dir = base_output / target_clean / tf32_labels[tf32]
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Run analyses
            analyze_mamba_optimal_performance(results, output_dir, target_filter=target, tf32_filter=tf32)
            analyze_k_value_sensitivity(results, output_dir, target_filter=target, tf32_filter=tf32)
            analyze_failure_modes(results, output_dir, target_filter=target, tf32_filter=tf32)
            create_mamba_optimization_plots(results, output_dir, target_filter=target, tf32_filter=tf32)
    
    print(f"\n{'='*120}")
    print(f"✅ Mamba optimal analysis complete!")
    print(f"{'='*120}")
    print(f"Output directory: {base_output}")


if __name__ == "__main__":
    main()
