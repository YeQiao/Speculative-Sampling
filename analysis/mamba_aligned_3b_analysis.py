#!/usr/bin/env python3
"""
Mamba-Aligned-3B Training Progress Analysis

Focus: Analyzing the mamba-aligned-3b training checkpoints (tuned for llama-3.2-3B)
Goal: Understand training progression and optimal checkpoint selection
Comparison: Baseline pretrained vs aligned-3b training
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict
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


def extract_checkpoint_number(model_name):
    """Extract checkpoint number from model name."""
    if 'aligned-3b-250' in model_name or 'aligned-3b/checkpoint-250' in model_name:
        return 250
    elif 'aligned-3b-500' in model_name or 'aligned-3b/checkpoint-500' in model_name:
        return 500
    elif 'aligned-3b-750' in model_name or 'aligned-3b/checkpoint-750' in model_name:
        return 750
    elif 'aligned-3b-1000' in model_name or 'aligned-3b/checkpoint-1000' in model_name:
        return 1000
    elif 'aligned-3b-1250' in model_name or 'aligned-3b/checkpoint-1250' in model_name:
        return 1250
    elif 'aligned-3b-1500' in model_name or 'aligned-3b/checkpoint-1500' in model_name:
        return 1500
    elif 'aligned-3b-best' in model_name or 'aligned-3b/best_model' in model_name:
        return 9999  # Special marker for best
    elif 'aligned-3b-final' in model_name or 'aligned-3b/final_model' in model_name:
        return 10000  # Special marker for final
    return None


def analyze_training_progression(results, output_dir, target_filter=None, tf32_filter=None):
    """
    Analyze how performance improves during training.
    Track optimal K values and acceptance rates across checkpoints.
    """
    
    target_name = target_filter if target_filter else "All Targets"
    tf32_name = "TF32 OFF" if tf32_filter is False else "TF32 ON" if tf32_filter is True else "TF32 All"
    
    print("\n" + "="*120)
    print(f"TRAINING PROGRESSION ANALYSIS - MAMBA-ALIGNED-3B")
    print(f"Target: {target_name} | {tf32_name}")
    print("="*120)
    
    experiments = results['experiments']
    spec_exps = [e for e in experiments if e['method'] != 'autoregressive']
    
    if target_filter:
        spec_exps = [e for e in spec_exps if e['target_model'] == target_filter]
    if tf32_filter is not None:
        spec_exps = [e for e in spec_exps if e['tf32_enabled'] == tf32_filter]
    
    # Filter for aligned-3b Mamba models only
    improved_exps = [e for e in spec_exps if 'aligned-3b' in e['draft_model'].lower()]
    
    if not improved_exps:
        print("No mamba-aligned-3b experiments found!")
        return
    
    # Group by checkpoint and sample type, find optimal K for each
    checkpoint_stats = defaultdict(list)
    
    for exp in improved_exps:
        ckpt = extract_checkpoint_number(exp['draft_model'])
        if ckpt is None:
            continue
        
        sample_type = exp['sample_set']
        key = (ckpt, sample_type)
        checkpoint_stats[key].append({
            'k': exp['lookahead'],
            'speedup': exp['speedup_vs_auto'],
            'accept': exp['acceptance_rate'] * 100
        })
    
    # Find optimal performance at each checkpoint
    checkpoint_progression = defaultdict(lambda: {
        'short': {'best_k': None, 'speedup': 0, 'accept': 0, 'all_speedups': []},
        'combined': {'best_k': None, 'speedup': 0, 'accept': 0, 'all_speedups': []},
        'long': {'best_k': None, 'speedup': 0, 'accept': 0, 'all_speedups': []}
    })
    
    for (ckpt, sample_type), data_list in checkpoint_stats.items():
        best = max(data_list, key=lambda x: x['speedup'])
        checkpoint_progression[ckpt][sample_type] = {
            'best_k': best['k'],
            'speedup': best['speedup'],
            'accept': best['accept'],
            'all_speedups': [d['speedup'] for d in data_list]
        }
    
    # Sort checkpoints
    sorted_checkpoints = sorted([k for k in checkpoint_progression.keys() if k < 9000])
    
    print(f"\n1. TRAINING PROGRESSION - OPTIMAL K PER CHECKPOINT")
    print(f"{'-'*120}")
    
    for sample_type in ['short', 'combined', 'long']:
        print(f"\n{sample_type.upper()} Prompts:")
        print(f"  {'Checkpoint':<12} | {'Best K':<8} | {'Speedup':<10} | {'Accept %':<10} | "
              f"{'Avg All K':<12} | {'Improvement':<12}")
        print(f"  {'-'*12}+{'-'*10}+{'-'*12}+{'-'*12}+{'-'*14}+{'-'*14}")
        
        prev_speedup = None
        for ckpt in sorted_checkpoints:
            stats = checkpoint_progression[ckpt][sample_type]
            avg_speedup = np.mean(stats['all_speedups'])
            
            if prev_speedup is not None:
                improvement = ((stats['speedup'] - prev_speedup) / prev_speedup) * 100
                improvement_str = f"{improvement:+.1f}%"
            else:
                improvement_str = "baseline"
            
            print(f"  {ckpt:<12} | K={stats['best_k']:<6} | {stats['speedup']:>8.3f}x | "
                  f"{stats['accept']:>8.1f}% | {avg_speedup:>10.3f}x | {improvement_str:>12}")
            
            prev_speedup = stats['speedup']
        
        # Show best/final models
        for special_ckpt, label in [(9999, 'best_model'), (10000, 'final_model')]:
            if special_ckpt in checkpoint_progression:
                stats = checkpoint_progression[special_ckpt][sample_type]
                avg_speedup = np.mean(stats['all_speedups'])
                improvement = ((stats['speedup'] - prev_speedup) / prev_speedup) * 100 if prev_speedup else 0
                print(f"  {label:<12} | K={stats['best_k']:<6} | {stats['speedup']:>8.3f}x | "
                      f"{stats['accept']:>8.1f}% | {avg_speedup:>10.3f}x | {improvement:>+11.1f}%")
    
    print(f"\n2. OVERALL TRAINING SUMMARY")
    print(f"{'-'*120}")
    
    # Calculate average across all sample types for each checkpoint
    overall_stats = {}
    for ckpt in sorted_checkpoints + [9999, 10000]:
        if ckpt not in checkpoint_progression:
            continue
        
        speedups = []
        accepts = []
        for sample_type in ['short', 'combined', 'long']:
            stats = checkpoint_progression[ckpt][sample_type]
            speedups.append(stats['speedup'])
            accepts.append(stats['accept'])
        
        overall_stats[ckpt] = {
            'avg_speedup': np.mean(speedups),
            'peak_speedup': np.max(speedups),
            'avg_accept': np.mean(accepts),
            'stability': np.std(speedups)
        }
    
    print(f"{'Checkpoint':<12} | {'Avg Speedup':<12} | {'Peak Speedup':<12} | "
          f"{'Avg Accept':<12} | {'Stability':<12}")
    print(f"{'-'*12}+{'-'*14}+{'-'*14}+{'-'*14}+{'-'*14}")
    
    best_checkpoint = max(sorted_checkpoints, key=lambda c: overall_stats[c]['avg_speedup'])
    
    for ckpt in sorted_checkpoints:
        stats = overall_stats[ckpt]
        marker = "⭐" if ckpt == best_checkpoint else "  "
        stability = "Stable" if stats['stability'] < 0.2 else "Variable" if stats['stability'] < 0.4 else "Unstable"
        
        print(f"{marker}{ckpt:<10} | {stats['avg_speedup']:>10.3f}x | {stats['peak_speedup']:>10.3f}x | "
              f"{stats['avg_accept']:>10.1f}% | {stability:<12}")
    
    # Show special models
    for special_ckpt, label in [(9999, 'best_model'), (10000, 'final_model')]:
        if special_ckpt in overall_stats:
            stats = overall_stats[special_ckpt]
            stability = "Stable" if stats['stability'] < 0.2 else "Variable" if stats['stability'] < 0.4 else "Unstable"
            print(f"  {label:<10} | {stats['avg_speedup']:>10.3f}x | {stats['peak_speedup']:>10.3f}x | "
                  f"{stats['avg_accept']:>10.1f}% | {stability:<12}")
    
    print(f"\n3. CONVERGENCE ANALYSIS")
    print(f"{'-'*120}")
    
    # Check if training has converged or is still improving
    recent_checkpoints = sorted_checkpoints[-3:]  # Last 3 checkpoints
    if len(recent_checkpoints) >= 2:
        speedup_trend = [overall_stats[c]['avg_speedup'] for c in recent_checkpoints]
        improvement_rate = np.diff(speedup_trend)
        
        print(f"Last 3 checkpoints: {recent_checkpoints}")
        print(f"Speedup progression: {' → '.join([f'{s:.3f}x' for s in speedup_trend])}")
        print(f"Improvement rates: {' → '.join([f'{r:+.3f}x' for r in improvement_rate])}")
        
        if len(improvement_rate) >= 2 and all(abs(r) < 0.05 for r in improvement_rate[-2:]):
            print(f"✓ Training appears CONVERGED (improvements < 0.05x)")
        elif improvement_rate[-1] > 0:
            print(f"→ Training STILL IMPROVING (last change: {improvement_rate[-1]:+.3f}x)")
        else:
            print(f"⚠ Training may be OVERTRAINING (last change: {improvement_rate[-1]:+.3f}x)")
    
    return checkpoint_progression, overall_stats


def compare_with_baseline_and_llama(results, baseline_results, output_dir, target_filter=None, tf32_filter=None):
    """
    Compare mamba-aligned-3b with:
    1. Baseline (mamba-65m-pretrained) 
    2. Llama-3.2-1B (alternative draft model)
    Show if the aligned training approach is better.
    """
    
    target_name = target_filter if target_filter else "All Targets"
    tf32_name = "TF32 OFF" if tf32_filter is False else "TF32 ON" if tf32_filter is True else "TF32 All"
    
    print(f"\n{'='*120}")
    print(f"MAMBA-ALIGNED-3B VS BASELINE & LLAMA-1B COMPARISON")
    print(f"Target: {target_name} | {tf32_name}")
    print(f"{'='*120}")
    
    experiments = results['experiments']
    spec_exps = [e for e in experiments if e['method'] != 'autoregressive']
    
    if target_filter:
        spec_exps = [e for e in spec_exps if e['target_model'] == target_filter]
    if tf32_filter is not None:
        spec_exps = [e for e in spec_exps if e['tf32_enabled'] == tf32_filter]
    
    # Get baseline experiments (baseline JSON is a list, not dict)
    baseline_all = baseline_results if isinstance(baseline_results, list) else baseline_results.get('experiments', [])
    baseline_spec_exps = [e for e in baseline_all if e['method'] != 'autoregressive']
    if target_filter:
        baseline_spec_exps = [e for e in baseline_spec_exps if e['target_model'] == target_filter]
    if tf32_filter is not None:
        baseline_spec_exps = [e for e in baseline_spec_exps if e['tf32_enabled'] == tf32_filter]
    
    # Separate aligned-3b vs baseline vs llama
    improved_exps = [e for e in spec_exps if 'aligned-3b' in e['draft_model'].lower()]
    baseline_mamba = [e for e in baseline_spec_exps if 'mamba-65m-pretrained' in e['draft_model'].lower()]
    llama1b_exps = [e for e in baseline_spec_exps if 'llama-3.2-1b' in e['draft_model'].lower()]
    
    if not improved_exps:
        print("Missing mamba-aligned-3b data for comparison!")
        return
    
    if not baseline_mamba:
        print("Missing baseline (mamba-65m-pretrained) data for comparison!")
        return
    
    if not llama1b_exps:
        print("Missing Llama-3.2-1B data for comparison!")
        # Continue anyway, just skip Llama comparison
    
    # First, show baseline vs improved comparison
    print(f"\n1. BASELINE (Pretrained) VS ALIGNED-3B TRAINING")
    print(f"{'-'*120}")
    
    # Get baseline performance for each sample type
    baseline_perf = {}
    llama1b_perf = {}
    
    for sample_type in ['short', 'combined', 'long']:
        sample_baseline = [e for e in baseline_mamba if e['sample_set'] == sample_type]
        if sample_baseline:
            best_baseline = max(sample_baseline, key=lambda x: x['speedup_vs_auto'])
            baseline_perf[sample_type] = {
                'speedup': best_baseline['speedup_vs_auto'],
                'k': best_baseline['lookahead'],
                'accept': best_baseline['acceptance_rate']
            }
        
        # Get Llama-1B performance
        sample_llama = [e for e in llama1b_exps if e['sample_set'] == sample_type]
        if sample_llama:
            best_llama = max(sample_llama, key=lambda x: x['speedup_vs_auto'])
            llama1b_perf[sample_type] = {
                'speedup': best_llama['speedup_vs_auto'],
                'k': best_llama['lookahead'],
                'accept': best_llama['acceptance_rate']
            }
    
    # Compare improved checkpoints vs baseline
    improved_checkpoints = [250, 500, 750, 1000, 1250, 1500]
    
    for sample_type in ['short', 'combined', 'long']:
        if sample_type not in baseline_perf:
            continue
            
        baseline = baseline_perf[sample_type]
        print(f"\n{sample_type.upper()} Prompts (Baseline: {baseline['speedup']:.3f}x @ K={baseline['k']}):")
        print(f"  {'Checkpoint':<12} | {'Speedup':<15} | {'Gain vs Baseline':<20} | {'Accept Rate':<15}")
        print(f"  {'-'*12}+{'-'*17}+{'-'*22}+{'-'*17}")
        
        for ckpt in improved_checkpoints:
            ckpt_exps = [e for e in improved_exps 
                        if extract_checkpoint_number(e['draft_model']) == ckpt 
                        and e['sample_set'] == sample_type]
            if ckpt_exps:
                best = max(ckpt_exps, key=lambda x: x['speedup_vs_auto'])
                gain = best['speedup_vs_auto'] - baseline['speedup']
                gain_pct = (gain / baseline['speedup']) * 100
                
                print(f"  {ckpt:<12} | {best['speedup_vs_auto']:.3f}x (K={best['lookahead']:<2}) | "
                      f"{gain:+.3f}x ({gain_pct:+5.1f}%) | {best['acceptance_rate']*100:.1f}%")
        
        # Show special models
        for special_name, label in [('best', 'best_model'), ('final', 'final_model')]:
            special_exps = [e for e in improved_exps 
                           if special_name in e['draft_model'].lower() 
                           and e['sample_set'] == sample_type]
            if special_exps:
                best = max(special_exps, key=lambda x: x['speedup_vs_auto'])
                gain = best['speedup_vs_auto'] - baseline['speedup']
                gain_pct = (gain / baseline['speedup']) * 100
                
                print(f"  {label:<12} | {best['speedup_vs_auto']:.3f}x (K={best['lookahead']:<2}) | "
                      f"{gain:+.3f}x ({gain_pct:+5.1f}%) | {best['acceptance_rate']*100:.1f}%")
    
    # Now Llama-1B comparison
    if llama1b_exps:
        print(f"\n2. LLAMA-3.2-1B VS MAMBA-ALIGNED-3B COMPARISON")
        print(f"{'-'*120}")
        print(f"Compare aligned-3b Mamba training with Llama-1B (different architecture)")
        print(f"{'-'*120}")
        
        for sample_type in ['short', 'combined', 'long']:
            if sample_type not in llama1b_perf:
                continue
            
            llama = llama1b_perf[sample_type]
            print(f"\n{sample_type.upper()} Prompts (Llama-1B: {llama['speedup']:.3f}x @ K={llama['k']}):")
            print(f"  {'Model':<12} | {'Speedup':<15} | {'vs Llama-1B':<20} | {'Accept Rate':<15}")
            print(f"  {'-'*12}+{'-'*17}+{'-'*22}+{'-'*17}")
            
            # Compare each improved checkpoint with Llama-1B
            for ckpt in improved_checkpoints:
                ckpt_exps = [e for e in improved_exps 
                            if extract_checkpoint_number(e['draft_model']) == ckpt 
                            and e['sample_set'] == sample_type]
                if ckpt_exps:
                    best = max(ckpt_exps, key=lambda x: x['speedup_vs_auto'])
                    diff = best['speedup_vs_auto'] - llama['speedup']
                    diff_pct = (diff / llama['speedup']) * 100
                    marker = "✓" if diff > 0 else "✗"
                    
                    print(f"{marker} {ckpt:<10} | {best['speedup_vs_auto']:.3f}x (K={best['lookahead']:<2}) | "
                          f"{diff:+.3f}x ({diff_pct:+5.1f}%) | {best['acceptance_rate']*100:.1f}%")
            
            # Show special models
            for special_name, label in [('best', 'best_model'), ('final', 'final_model')]:
                special_exps = [e for e in improved_exps 
                               if special_name in e['draft_model'].lower() 
                               and e['sample_set'] == sample_type]
                if special_exps:
                    best = max(special_exps, key=lambda x: x['speedup_vs_auto'])
                    diff = best['speedup_vs_auto'] - llama['speedup']
                    diff_pct = (diff / llama['speedup']) * 100
                    marker = "✓" if diff > 0 else "✗"
                    
                    print(f"{marker} {label:<10} | {best['speedup_vs_auto']:.3f}x (K={best['lookahead']:<2}) | "
                          f"{diff:+.3f}x ({diff_pct:+5.1f}%) | {best['acceptance_rate']*100:.1f}%")


def create_improved_mamba_plots(results, baseline_results, output_dir, target_filter=None, tf32_filter=None):
    """
    Create visualizations for mamba-aligned-3b training progression with baseline and Llama-1B comparison.
    """
    
    target_name = target_filter.replace('/', '_') if target_filter else "all_targets"
    tf32_label = "tf32_on" if tf32_filter is True else "tf32_off" if tf32_filter is False else "tf32_all"
    
    print(f"\nGenerating mamba-aligned-3b training plots for {target_filter}, TF32 {'ON' if tf32_filter else 'OFF'}...")
    
    experiments = results['experiments']
    spec_exps = [e for e in experiments if e['method'] != 'autoregressive']
    
    if target_filter:
        spec_exps = [e for e in spec_exps if e['target_model'] == target_filter]
    if tf32_filter is not None:
        spec_exps = [e for e in spec_exps if e['tf32_enabled'] == tf32_filter]
    
    improved_exps = [e for e in spec_exps if 'aligned-3b' in e['draft_model'].lower()]
    
    # Get baseline experiments
    baseline_all = baseline_results if isinstance(baseline_results, list) else baseline_results.get('experiments', [])
    baseline_spec_exps = [e for e in baseline_all if e['method'] != 'autoregressive']
    if target_filter:
        baseline_spec_exps = [e for e in baseline_spec_exps if e['target_model'] == target_filter]
    if tf32_filter is not None:
        baseline_spec_exps = [e for e in baseline_spec_exps if e['tf32_enabled'] == tf32_filter]
    baseline_mamba = [e for e in baseline_spec_exps if 'mamba-65m-pretrained' in e['draft_model'].lower()]
    llama1b_exps = [e for e in baseline_spec_exps if 'llama-3.2-1b' in e['draft_model'].lower()]
    
    if not improved_exps:
        print("No mamba-aligned-3b experiments found for plotting!")
        return
    
    # Prepare data
    fig, axes = plt.subplots(3, 3, figsize=(20, 18))
    fig.suptitle(f'Mamba-Aligned-3B vs Llama-1B Training Progression\n{target_filter} | TF32 {"ON" if tf32_filter else "OFF"}', 
                 fontsize=16, fontweight='bold')
    
    # Group by checkpoint
    checkpoints = sorted(list(set([extract_checkpoint_number(e['draft_model']) 
                                   for e in improved_exps if extract_checkpoint_number(e['draft_model'])])))
    checkpoints = [c for c in checkpoints if c < 9000]  # Regular checkpoints only for main plots
    
    # Add checkpoint 0 for baseline
    checkpoints_with_baseline = [0] + checkpoints
    
    # Get baseline performance for each sample type
    baseline_perf = {}
    llama1b_perf = {}
    
    for sample_type in ['short', 'combined', 'long']:
        sample_baseline = [e for e in baseline_mamba if e['sample_set'] == sample_type]
        if sample_baseline:
            best_baseline = max(sample_baseline, key=lambda x: x['speedup_vs_auto'])
            baseline_perf[sample_type] = {
                'speedup': best_baseline['speedup_vs_auto'],
                'accept': best_baseline['acceptance_rate'],
                'k': best_baseline['lookahead']
            }
        
        # Get Llama-1B performance
        sample_llama = [e for e in llama1b_exps if e['sample_set'] == sample_type]
        if sample_llama:
            best_llama = max(sample_llama, key=lambda x: x['speedup_vs_auto'])
            llama1b_perf[sample_type] = {
                'speedup': best_llama['speedup_vs_auto'],
                'accept': best_llama['acceptance_rate'],
                'k': best_llama['lookahead']
            }
    
    # Plot 1: Training curve - best speedup per checkpoint (with baseline as checkpoint 0, Llama as reference line)
    ax1 = axes[0, 0]
    
    for sample_type, color, marker in [('short', 'red', 'o'), ('combined', 'blue', 's'), ('long', 'green', '^')]:
        speedups = []
        
        # Add baseline as checkpoint 0
        if sample_type in baseline_perf:
            speedups.append(baseline_perf[sample_type]['speedup'])
        else:
            speedups.append(None)
        
        # Add improved checkpoints
        for ckpt in checkpoints:
            ckpt_exps = [e for e in improved_exps 
                        if extract_checkpoint_number(e['draft_model']) == ckpt 
                        and e['sample_set'] == sample_type]
            if ckpt_exps:
                best = max(ckpt_exps, key=lambda x: x['speedup_vs_auto'])
                speedups.append(best['speedup_vs_auto'])
            else:
                speedups.append(None)
        
        ax1.plot(checkpoints_with_baseline, speedups, marker=marker, linewidth=2, markersize=8, 
                label=f'Mamba {sample_type.capitalize()}', color=color)
        
        # Add Llama-1B as horizontal reference line
        if sample_type in llama1b_perf:
            ax1.axhline(y=llama1b_perf[sample_type]['speedup'], color=color, linestyle=':', 
                       alpha=0.5, linewidth=2, label=f'Llama-1B {sample_type.capitalize()}')
    
    ax1.axhline(y=1.0, color='black', linestyle='--', alpha=0.3, linewidth=1, label='Autoregressive')
    ax1.set_xlabel('Training Step (0=Baseline Pretrained)', fontsize=11)
    ax1.set_ylabel('Best Speedup (Optimal K)', fontsize=11)
    ax1.set_title('Mamba Training vs Llama-1B', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=8, ncol=2)
    ax1.grid(alpha=0.3)
    
    # Plot 2: Acceptance rate progression
    ax2 = axes[0, 1]
    
    for sample_type, color, marker in [('short', 'red', 'o'), ('combined', 'blue', 's'), ('long', 'green', '^')]:
        accepts = []
        
        # Add baseline as checkpoint 0
        if sample_type in baseline_perf:
            accepts.append(baseline_perf[sample_type]['accept'] * 100)
        else:
            accepts.append(None)
        
        # Add improved checkpoints
        for ckpt in checkpoints:
            ckpt_exps = [e for e in improved_exps 
                        if extract_checkpoint_number(e['draft_model']) == ckpt 
                        and e['sample_set'] == sample_type]
            if ckpt_exps:
                best = max(ckpt_exps, key=lambda x: x['speedup_vs_auto'])
                accepts.append(best['acceptance_rate'] * 100)
            else:
                accepts.append(None)
        
        ax2.plot(checkpoints_with_baseline, accepts, marker=marker, linewidth=2, markersize=8, 
                label=f'Mamba {sample_type.capitalize()}', color=color)
        
        # Add Llama-1B as horizontal reference line
        if sample_type in llama1b_perf:
            ax2.axhline(y=llama1b_perf[sample_type]['accept'] * 100, color=color, linestyle=':', 
                       alpha=0.5, linewidth=2, label=f'Llama-1B {sample_type.capitalize()}')
    
    ax2.set_xlabel('Training Step (0=Baseline Pretrained)', fontsize=11)
    ax2.set_ylabel('Acceptance Rate (%)', fontsize=11)
    ax2.set_title('Acceptance Rate: Mamba vs Llama-1B', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=8, ncol=2)
    ax2.grid(alpha=0.3)
    
    # Plot 3: K-value evolution
    ax3 = axes[0, 2]
    
    for sample_type, color, marker in [('short', 'red', 'o'), ('combined', 'blue', 's'), ('long', 'green', '^')]:
        best_ks = []
        
        # Add baseline as checkpoint 0
        if sample_type in baseline_perf:
            best_ks.append(baseline_perf[sample_type]['k'])
        else:
            best_ks.append(None)
        
        # Add improved checkpoints
        for ckpt in checkpoints:
            ckpt_exps = [e for e in improved_exps 
                        if extract_checkpoint_number(e['draft_model']) == ckpt 
                        and e['sample_set'] == sample_type]
            if ckpt_exps:
                best = max(ckpt_exps, key=lambda x: x['speedup_vs_auto'])
                best_ks.append(best['lookahead'])
            else:
                best_ks.append(None)
        
        ax3.plot(checkpoints_with_baseline, best_ks, marker=marker, linewidth=2, markersize=8, 
                label=sample_type.capitalize(), color=color)
    
    ax3.set_xlabel('Training Step (0=Baseline Pretrained)', fontsize=11)
    ax3.set_ylabel('Optimal K Value', fontsize=11)
    ax3.set_title('Optimal K Evolution', fontsize=12, fontweight='bold')
    ax3.legend()
    ax3.grid(alpha=0.3)
    ax3.set_yticks([2, 3, 4, 5, 6, 8])
    
    # Plot 4: Performance variance (stability)
    ax4 = axes[1, 0]
    
    variances = []
    
    # Add baseline stability
    if baseline_mamba:
        baseline_speedups = [e['speedup_vs_auto'] for e in baseline_mamba]
        variances.append(np.std(baseline_speedups))
    else:
        variances.append(None)
    
    # Add improved checkpoints
    for ckpt in checkpoints:
        ckpt_exps = [e for e in improved_exps 
                    if extract_checkpoint_number(e['draft_model']) == ckpt]
        if ckpt_exps:
            speedups = [e['speedup_vs_auto'] for e in ckpt_exps]
            variances.append(np.std(speedups))
        else:
            variances.append(None)
    
    ax4.bar(range(len(checkpoints_with_baseline)), variances, alpha=0.7, color='steelblue')
    ax4.set_xticks(range(len(checkpoints_with_baseline)))
    ax4.set_xticklabels(['0(B)'] + checkpoints, rotation=45, fontsize=9)
    ax4.set_xlabel('Training Step (0=Baseline)', fontsize=11)
    ax4.set_ylabel('Performance Std Dev', fontsize=11)
    ax4.set_title('Training Stability (Lower = More Stable)', fontsize=12, fontweight='bold')
    ax4.grid(axis='y', alpha=0.3)
    
    # Plot 5: Heatmap - speedup by checkpoint and K value for long prompts
    ax5 = axes[1, 1]
    
    k_values = sorted(set(e['lookahead'] for e in improved_exps))
    heatmap_data = np.zeros((len(checkpoints_with_baseline), len(k_values)))
    
    # Add baseline as row 0
    for j, k in enumerate(k_values):
        matching = [e for e in baseline_mamba 
                   if e['lookahead'] == k 
                   and e['sample_set'] == 'long']
        if matching:
            heatmap_data[0, j] = matching[0]['speedup_vs_auto']
        else:
            heatmap_data[0, j] = np.nan
    
    # Add improved checkpoints
    for i, ckpt in enumerate(checkpoints, start=1):
        for j, k in enumerate(k_values):
            matching = [e for e in improved_exps 
                       if extract_checkpoint_number(e['draft_model']) == ckpt 
                       and e['lookahead'] == k 
                       and e['sample_set'] == 'long']
            if matching:
                heatmap_data[i, j] = matching[0]['speedup_vs_auto']
            else:
                heatmap_data[i, j] = np.nan
    
    im = ax5.imshow(heatmap_data, cmap='RdYlGn', aspect='auto', vmin=0.5, vmax=2.5)
    ax5.set_xticks(np.arange(len(k_values)))
    ax5.set_yticks(np.arange(len(checkpoints_with_baseline)))
    ax5.set_xticklabels([f'K={k}' for k in k_values])
    ax5.set_yticklabels(['0(B)'] + checkpoints, fontsize=9)
    ax5.set_xlabel('Lookahead K', fontsize=11)
    ax5.set_ylabel('Training Step (0=Baseline)', fontsize=11)
    ax5.set_title('Speedup Heatmap (Long Prompts)', fontsize=12, fontweight='bold')
    
    # Add text annotations
    for i in range(len(checkpoints_with_baseline)):
        for j in range(len(k_values)):
            if not np.isnan(heatmap_data[i, j]):
                text = ax5.text(j, i, f'{heatmap_data[i, j]:.2f}',
                               ha="center", va="center", color="black", fontsize=7)
    
    plt.colorbar(im, ax=ax5, label='Speedup')
    
    # Plot 6: Average improvement over baseline
    ax6 = axes[1, 2]
    
    avg_speedups = []
    
    # Add baseline average
    if baseline_mamba:
        sample_bests = []
        for sample_type in ['short', 'combined', 'long']:
            sample_exps = [e for e in baseline_mamba if e['sample_set'] == sample_type]
            if sample_exps:
                best = max(sample_exps, key=lambda x: x['speedup_vs_auto'])
                sample_bests.append(best['speedup_vs_auto'])
        avg_speedups.append(np.mean(sample_bests))
    else:
        avg_speedups.append(None)
    
    # Add improved checkpoints
    for ckpt in checkpoints:
        ckpt_exps = [e for e in improved_exps 
                    if extract_checkpoint_number(e['draft_model']) == ckpt]
        if ckpt_exps:
            # Get best per sample type
            sample_bests = []
            for sample_type in ['short', 'combined', 'long']:
                sample_exps = [e for e in ckpt_exps if e['sample_set'] == sample_type]
                if sample_exps:
                    best = max(sample_exps, key=lambda x: x['speedup_vs_auto'])
                    sample_bests.append(best['speedup_vs_auto'])
            avg_speedups.append(np.mean(sample_bests))
        else:
            avg_speedups.append(None)
    
    colors = ['green' if s and s > 1.5 else 'orange' if s and s > 1.2 else 'red' 
              for s in avg_speedups]
    ax6.barh(range(len(checkpoints_with_baseline)), avg_speedups, color=colors, alpha=0.7)
    ax6.axvline(x=1.0, color='black', linestyle='--', linewidth=1)
    ax6.set_yticks(range(len(checkpoints_with_baseline)))
    ax6.set_yticklabels(['0(B)'] + checkpoints, fontsize=9)
    ax6.set_xlabel('Average Best Speedup', fontsize=11)
    ax6.set_title('Overall Performance by Checkpoint', fontsize=12, fontweight='bold')
    ax6.grid(axis='x', alpha=0.3)
    
    # Plot 7: Mamba vs Llama-1B comparison by prompt type (side-by-side bars)
    ax7 = axes[2, 0]
    
    sample_types = ['short', 'combined', 'long']
    x = np.arange(len(sample_types))
    width = 0.25
    
    # Get baseline Mamba, best improved Mamba, and Llama-1B for each sample type
    baseline_vals = []
    improved_vals = []
    llama_vals = []
    
    for sample_type in sample_types:
        # Baseline Mamba
        if sample_type in baseline_perf:
            baseline_vals.append(baseline_perf[sample_type]['speedup'])
        else:
            baseline_vals.append(0)
        
        # Best improved Mamba (from all checkpoints)
        improved_sample = [e for e in improved_exps if e['sample_set'] == sample_type]
        if improved_sample:
            best_improved = max(improved_sample, key=lambda x: x['speedup_vs_auto'])
            improved_vals.append(best_improved['speedup_vs_auto'])
        else:
            improved_vals.append(0)
        
        # Llama-1B
        if sample_type in llama1b_perf:
            llama_vals.append(llama1b_perf[sample_type]['speedup'])
        else:
            llama_vals.append(0)
    
    bars1 = ax7.bar(x - width, baseline_vals, width, label='Baseline Mamba', color='lightcoral', alpha=0.8)
    bars2 = ax7.bar(x, improved_vals, width, label='Best Aligned-3B Mamba', color='forestgreen', alpha=0.8)
    bars3 = ax7.bar(x + width, llama_vals, width, label='Llama-3.2-1B', color='steelblue', alpha=0.8)
    
    # Add value labels on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax7.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.2f}x',
                        ha='center', va='bottom', fontsize=8)
    
    ax7.axhline(y=1.0, color='black', linestyle='--', alpha=0.3, linewidth=1)
    ax7.set_xlabel('Prompt Type', fontsize=11)
    ax7.set_ylabel('Speedup', fontsize=11)
    ax7.set_title('Mamba vs Llama-1B Comparison', fontsize=12, fontweight='bold')
    ax7.set_xticks(x)
    ax7.set_xticklabels(sample_types)
    ax7.legend(fontsize=9)
    ax7.grid(axis='y', alpha=0.3)
    
    # Hide unused subplots
    for idx in [(2, 1), (2, 2)]:
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    # Save figure
    output_path = output_dir / 'mamba_aligned_3b_training_analysis.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: {output_path.name}")


def main():
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python mamba_aligned_3b_analysis.py <sweep_results.json>")
        sys.exit(1)
    
    json_path = sys.argv[1]
    results = load_results(json_path)
    
    base_output = Path("outputs/mamba_aligned_3b_analysis")
    base_output.mkdir(parents=True, exist_ok=True)
    
    print(f"Loaded {len(results['experiments'])} experiments")
    
    targets = list(set(e['target_model'] for e in results['experiments'] if e['method'] != 'autoregressive'))
    print(f"Target models found: {targets}")
    
    # TF32 OFF only (as requested)
    tf32_filter = False
    
    # Run analysis for each target
    for target in targets:
        print(f"\n{'#'*120}")
        print(f"# MAMBA-ALIGNED-3B ANALYSIS: {target} | TF32 OFF")
        print(f"{'#'*120}")
        
        # Create output directory
        target_clean = target.replace('/', '_')
        output_dir = base_output / target_clean / "tf32_off"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Run analyses
        checkpoint_prog, overall_stats = analyze_training_progression(
            results, output_dir, target_filter=target, tf32_filter=tf32_filter
        )
        
        # Load baseline results for comparison
        baseline_path = Path("outputs/sweep_results/sweep_results_20251031_041935_final.json")
        baseline_results = None
        if baseline_path.exists():
            with open(baseline_path, 'r') as f:
                baseline_results = json.load(f)
            
            compare_with_baseline_and_llama(
                results, baseline_results, output_dir, target_filter=target, tf32_filter=tf32_filter
            )
            
            create_improved_mamba_plots(
                results, baseline_results, output_dir, target_filter=target, tf32_filter=tf32_filter
            )
        else:
            print(f"\n⚠ Baseline results not found at {baseline_path}, skipping comparison and plots")
    
    print(f"\n{'='*120}")
    print(f"✅ Mamba-Aligned-3B training analysis complete!")
    print(f"{'='*120}")
    print(f"Output directory: {base_output}")


if __name__ == "__main__":
    main()
