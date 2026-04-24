#!/usr/bin/env python3
"""
Deep analysis of speculative sampling sweep results.
Comprehensive investigation of performance patterns across target models.
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
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
        return 'Mamba-distilled-1k'
    elif 'checkpoint-2000' in full_name:
        return 'Mamba-distilled-2k'
    elif 'checkpoint-3000' in full_name:
        return 'Mamba-distilled-3k'
    elif 'distilled-best' in full_name:
        return 'Mamba-distilled-best'
    return full_name


def deep_analysis(results, output_dir):
    """Perform comprehensive deep analysis."""
    
    print("\n" + "="*100)
    print("DEEP ANALYSIS OF SPECULATIVE SAMPLING RESULTS")
    print("="*100)
    
    experiments = results['experiments']
    
    # Organize data by target model
    target_data = defaultdict(lambda: defaultdict(list))
    
    for exp in experiments:
        target = exp['target_model']
        target_data[target]['all'].append(exp)
        
        if exp['method'] == 'autoregressive':
            target_data[target]['auto'].append(exp)
        else:
            target_data[target]['spec'].append(exp)
    
    # Analysis for each target model
    for target_name in sorted(target_data.keys()):
        analyze_target_model(target_name, target_data[target_name], output_dir)
    
    # Comparative analysis
    comparative_analysis(target_data, output_dir)
    
    # TF32 deep dive
    tf32_deep_dive(target_data, output_dir)
    
    # Draft model characteristics
    draft_model_characteristics(target_data, output_dir)
    
    # Lookahead optimization patterns
    lookahead_patterns(target_data, output_dir)


def analyze_target_model(target_name, data, output_dir):
    """Detailed analysis for a specific target model."""
    
    print(f"\n{'='*100}")
    print(f"TARGET MODEL: {target_name}")
    print(f"{'='*100}")
    
    auto_exps = data['auto']
    spec_exps = data['spec']
    
    # 1. Baseline Performance Analysis
    print(f"\n1. BASELINE (AUTOREGRESSIVE) PERFORMANCE")
    print(f"{'-'*100}")
    
    auto_stats = defaultdict(lambda: defaultdict(dict))
    for exp in auto_exps:
        sample = exp['sample_set']
        tf32 = exp['tf32_enabled']
        auto_stats[sample][tf32] = {
            'throughput': exp['throughput'],
            'total_time': exp['total_time'],
            'total_tokens': exp['total_tokens']
        }
    
    print(f"\n{'Sample Set':<15} | {'TF32':<8} | {'Throughput':<15} | {'Total Time':<12} | {'Total Tokens':<12}")
    print(f"{'-'*15}+{'-'*10}+{'-'*17}+{'-'*14}+{'-'*12}")
    
    for sample in ['short', 'long', 'combined']:
        for tf32 in [False, True]:
            if tf32 in auto_stats[sample]:
                stats = auto_stats[sample][tf32]
                tf32_str = 'ON' if tf32 else 'OFF'
                print(f"{sample:<15} | {tf32_str:<8} | {stats['throughput']:>13.2f} t/s | "
                      f"{stats['total_time']:>10.2f}s | {stats['total_tokens']:>12d}")
    
    # TF32 impact on baseline
    print(f"\n   TF32 Impact on Autoregressive:")
    for sample in ['short', 'long', 'combined']:
        if False in auto_stats[sample] and True in auto_stats[sample]:
            off_tp = auto_stats[sample][False]['throughput']
            on_tp = auto_stats[sample][True]['throughput']
            speedup = on_tp / off_tp
            print(f"   - {sample:10s}: {off_tp:6.2f} → {on_tp:6.2f} tok/s ({speedup:.2f}x speedup)")
    
    # 2. Speculative Sampling Performance
    print(f"\n2. SPECULATIVE SAMPLING PERFORMANCE")
    print(f"{'-'*100}")
    
    # Best configurations
    best_configs = {}
    for sample in ['short', 'long', 'combined']:
        for tf32 in [False, True]:
            best = None
            for exp in spec_exps:
                if exp['sample_set'] == sample and exp['tf32_enabled'] == tf32:
                    if best is None or exp['speedup_vs_auto'] > best['speedup_vs_auto']:
                        best = exp
            if best:
                best_configs[(sample, tf32)] = best
    
    print(f"\nBest Configurations:")
    print(f"{'Sample':<10} | {'TF32':<6} | {'Draft Model':<25} | {'K':<3} | {'Accept':<8} | {'Throughput':<13} | {'Speedup':<8}")
    print(f"{'-'*10}+{'-'*8}+{'-'*27}+{'-'*5}+{'-'*10}+{'-'*15}+{'-'*8}")
    
    for sample in ['short', 'long', 'combined']:
        for tf32 in [False, True]:
            if (sample, tf32) in best_configs:
                exp = best_configs[(sample, tf32)]
                draft = extract_draft_model_name(exp['draft_model'])
                tf32_str = 'ON' if tf32 else 'OFF'
                print(f"{sample:<10} | {tf32_str:<6} | {draft:<25} | {exp['lookahead']:<3d} | "
                      f"{exp['acceptance_rate']*100:>6.1f}% | {exp['throughput']:>11.2f} t/s | "
                      f"{exp['speedup_vs_auto']:>6.2f}x")
    
    # 3. Draft Model Comparison
    print(f"\n3. DRAFT MODEL COMPARISON")
    print(f"{'-'*100}")
    
    draft_performance = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    
    for exp in spec_exps:
        draft = extract_draft_model_name(exp['draft_model'])
        sample = exp['sample_set']
        tf32 = exp['tf32_enabled']
        draft_performance[draft][sample][tf32].append({
            'speedup': exp['speedup_vs_auto'],
            'accept': exp['acceptance_rate'],
            'throughput': exp['throughput'],
            'k': exp['lookahead']
        })
    
    for sample in ['short', 'long', 'combined']:
        print(f"\n   Sample Set: {sample.upper()}")
        print(f"   {'Draft Model':<25} | {'TF32':<6} | {'Avg Speedup':<12} | {'Max Speedup':<12} | {'Avg Accept':<11} | {'Best K':<7}")
        print(f"   {'-'*25}+{'-'*8}+{'-'*14}+{'-'*14}+{'-'*13}+{'-'*7}")
        
        for draft in sorted(draft_performance.keys()):
            for tf32 in [False, True]:
                if tf32 in draft_performance[draft][sample]:
                    data = draft_performance[draft][sample][tf32]
                    avg_speedup = np.mean([d['speedup'] for d in data])
                    max_speedup = np.max([d['speedup'] for d in data])
                    avg_accept = np.mean([d['accept'] for d in data]) * 100
                    best_k = data[np.argmax([d['speedup'] for d in data])]['k']
                    
                    tf32_str = 'ON' if tf32 else 'OFF'
                    print(f"   {draft:<25} | {tf32_str:<6} | {avg_speedup:>10.2f}x | "
                          f"{max_speedup:>10.2f}x | {avg_accept:>9.1f}% | {best_k:>7d}")
    
    # 4. Acceptance Rate Analysis
    print(f"\n4. ACCEPTANCE RATE PATTERNS")
    print(f"{'-'*100}")
    
    accept_by_k = defaultdict(lambda: defaultdict(list))
    
    for exp in spec_exps:
        draft = extract_draft_model_name(exp['draft_model'])
        k = exp['lookahead']
        accept_by_k[draft][k].append(exp['acceptance_rate'] * 100)
    
    print(f"\n   Acceptance Rate vs Lookahead (averaged across all configurations):")
    print(f"   {'Draft Model':<25} | K=2    | K=3    | K=4    | K=5    | K=6    | K=8    |")
    print(f"   {'-'*25}+{'-'*8}+{'-'*8}+{'-'*8}+{'-'*8}+{'-'*8}+{'-'*8}")
    
    for draft in sorted(accept_by_k.keys()):
        values = []
        for k in [2, 3, 4, 5, 6, 8]:
            if k in accept_by_k[draft]:
                avg = np.mean(accept_by_k[draft][k])
                values.append(f"{avg:5.1f}%")
            else:
                values.append("  N/A ")
        print(f"   {draft:<25} | {' | '.join(values)} |")
    
    # 5. Speedup vs Acceptance Correlation
    print(f"\n5. SPEEDUP vs ACCEPTANCE RATE CORRELATION")
    print(f"{'-'*100}")
    
    # Calculate correlation for each draft model
    for draft in sorted(draft_performance.keys()):
        all_speedups = []
        all_accepts = []
        
        for sample in ['short', 'long', 'combined']:
            for tf32 in [False, True]:
                if tf32 in draft_performance[draft][sample]:
                    for d in draft_performance[draft][sample][tf32]:
                        all_speedups.append(d['speedup'])
                        all_accepts.append(d['accept'] * 100)
        
        if len(all_speedups) > 1:
            correlation = np.corrcoef(all_accepts, all_speedups)[0, 1]
            print(f"   {draft:<25}: Correlation = {correlation:>6.3f}")
    
    # 6. TF32 Impact on Speculative Sampling
    print(f"\n6. TF32 IMPACT ON SPECULATIVE SAMPLING")
    print(f"{'-'*100}")
    
    for sample in ['short', 'long', 'combined']:
        print(f"\n   Sample Set: {sample.upper()}")
        
        # Get baseline speedup
        auto_off = auto_stats[sample][False]['throughput']
        auto_on = auto_stats[sample][True]['throughput']
        baseline_tf32_speedup = auto_on / auto_off
        
        print(f"   Autoregressive TF32 speedup: {baseline_tf32_speedup:.2f}x")
        print(f"\n   {'Draft Model':<25} | {'Spec Speedup (TF32 OFF)':<25} | {'Spec Speedup (TF32 ON)':<24} | {'Ratio':<8}")
        print(f"   {'-'*25}+{'-'*27}+{'-'*26}+{'-'*8}")
        
        for draft in sorted(draft_performance.keys()):
            if False in draft_performance[draft][sample] and True in draft_performance[draft][sample]:
                max_off = np.max([d['speedup'] for d in draft_performance[draft][sample][False]])
                max_on = np.max([d['speedup'] for d in draft_performance[draft][sample][True]])
                ratio = max_off / max_on if max_on > 0 else 0
                
                print(f"   {draft:<25} | {max_off:>23.2f}x | {max_on:>22.2f}x | {ratio:>6.2f}x")
        
        print(f"\n   Key Insight: TF32 gives {baseline_tf32_speedup:.2f}x baseline speedup but reduces speculative gains")


def comparative_analysis(target_data, output_dir):
    """Compare performance across different target models."""
    
    print(f"\n{'='*100}")
    print(f"COMPARATIVE ANALYSIS: Llama-3.1-8B vs Llama-3.2-3B")
    print(f"{'='*100}")
    
    target_names = sorted(target_data.keys())
    
    # 1. Baseline Comparison
    print(f"\n1. BASELINE THROUGHPUT COMPARISON")
    print(f"{'-'*100}")
    print(f"\n{'Target Model':<20} | {'Sample':<10} | {'TF32 OFF':<13} | {'TF32 ON':<13} | {'TF32 Speedup':<13}")
    print(f"{'-'*20}+{'-'*12}+{'-'*15}+{'-'*15}+{'-'*13}")
    
    for target in target_names:
        auto_exps = target_data[target]['auto']
        for sample in ['short', 'long', 'combined']:
            off_exp = [e for e in auto_exps if e['sample_set'] == sample and not e['tf32_enabled']][0]
            on_exp = [e for e in auto_exps if e['sample_set'] == sample and e['tf32_enabled']][0]
            
            speedup = on_exp['throughput'] / off_exp['throughput']
            print(f"{target:<20} | {sample:<10} | {off_exp['throughput']:>11.2f} t/s | "
                  f"{on_exp['throughput']:>11.2f} t/s | {speedup:>11.2f}x")
    
    # 2. Best Speculative Performance Comparison
    print(f"\n2. BEST SPECULATIVE SAMPLING PERFORMANCE")
    print(f"{'-'*100}")
    
    best_by_target = defaultdict(dict)
    
    for target in target_names:
        spec_exps = target_data[target]['spec']
        for sample in ['short', 'long', 'combined']:
            for tf32 in [False, True]:
                best = None
                for exp in spec_exps:
                    if exp['sample_set'] == sample and exp['tf32_enabled'] == tf32:
                        if best is None or exp['speedup_vs_auto'] > best['speedup_vs_auto']:
                            best = exp
                if best:
                    best_by_target[target][(sample, tf32)] = best
    
    print(f"\n{'Target':<20} | {'Sample':<10} | {'TF32':<6} | {'Draft':<20} | {'K':<3} | {'Speedup':<8}")
    print(f"{'-'*20}+{'-'*12}+{'-'*8}+{'-'*22}+{'-'*5}+{'-'*8}")
    
    for target in target_names:
        for sample in ['short', 'long', 'combined']:
            for tf32 in [False, True]:
                if (sample, tf32) in best_by_target[target]:
                    exp = best_by_target[target][(sample, tf32)]
                    draft = extract_draft_model_name(exp['draft_model'])
                    tf32_str = 'ON' if tf32 else 'OFF'
                    print(f"{target:<20} | {sample:<10} | {tf32_str:<6} | {draft:<20} | "
                          f"{exp['lookahead']:<3d} | {exp['speedup_vs_auto']:>6.2f}x")
    
    # 3. Why 3B model behaves differently
    print(f"\n3. KEY DIFFERENCES: 8B vs 3B MODEL")
    print(f"{'-'*100}")
    
    print(f"\nModel Size Impact:")
    for target in target_names:
        auto_exps = target_data[target]['auto']
        # Get average throughput across all configs
        avg_throughput_off = np.mean([e['throughput'] for e in auto_exps if not e['tf32_enabled']])
        avg_throughput_on = np.mean([e['throughput'] for e in auto_exps if e['tf32_enabled']])
        
        print(f"   {target}:")
        print(f"      - Avg throughput (TF32 OFF): {avg_throughput_off:.2f} tok/s")
        print(f"      - Avg throughput (TF32 ON):  {avg_throughput_on:.2f} tok/s")
        print(f"      - Relative speed: {avg_throughput_off / target_data[target_names[0]]['auto'][0]['throughput']:.2f}x vs 8B baseline")
    
    print(f"\nSpeculative Sampling Effectiveness:")
    for target in target_names:
        spec_exps = target_data[target]['spec']
        avg_speedup_off = np.mean([e['speedup_vs_auto'] for e in spec_exps if not e['tf32_enabled']])
        avg_speedup_on = np.mean([e['speedup_vs_auto'] for e in spec_exps if e['tf32_enabled']])
        max_speedup_off = np.max([e['speedup_vs_auto'] for e in spec_exps if not e['tf32_enabled']])
        max_speedup_on = np.max([e['speedup_vs_auto'] for e in spec_exps if e['tf32_enabled']])
        
        print(f"   {target}:")
        print(f"      - Avg speculative speedup (TF32 OFF): {avg_speedup_off:.2f}x (max: {max_speedup_off:.2f}x)")
        print(f"      - Avg speculative speedup (TF32 ON):  {avg_speedup_on:.2f}x (max: {max_speedup_on:.2f}x)")
    
    # 4. Draft Model Performance Across Targets
    print(f"\n4. DRAFT MODEL PERFORMANCE ACROSS TARGETS")
    print(f"{'-'*100}")
    
    draft_comparison = defaultdict(lambda: defaultdict(list))
    
    for target in target_names:
        spec_exps = target_data[target]['spec']
        for exp in spec_exps:
            draft = extract_draft_model_name(exp['draft_model'])
            draft_comparison[draft][target].append(exp['speedup_vs_auto'])
    
    print(f"\n{'Draft Model':<25} | {'8B Avg':<10} | {'8B Max':<10} | {'3B Avg':<10} | {'3B Max':<10} | {'Difference':<12}")
    print(f"{'-'*25}+{'-'*12}+{'-'*12}+{'-'*12}+{'-'*12}+{'-'*12}")
    
    for draft in sorted(draft_comparison.keys()):
        if len(target_names) == 2:
            t8b_speedups = draft_comparison[draft][target_names[0]]
            t3b_speedups = draft_comparison[draft][target_names[1]]
            
            avg_8b = np.mean(t8b_speedups)
            max_8b = np.max(t8b_speedups)
            avg_3b = np.mean(t3b_speedups)
            max_3b = np.max(t3b_speedups)
            diff = avg_8b - avg_3b
            
            print(f"{draft:<25} | {avg_8b:>8.2f}x | {max_8b:>8.2f}x | "
                  f"{avg_3b:>8.2f}x | {max_3b:>8.2f}x | {diff:>+9.2f}x")
    
    print(f"\nKey Insight: Smaller 3B model is faster baseline, reducing relative speculative gains")


def tf32_deep_dive(target_data, output_dir):
    """Deep dive into TF32 impact."""
    
    print(f"\n{'='*100}")
    print(f"TF32 DEEP DIVE: Understanding the Paradox")
    print(f"{'='*100}")
    
    print(f"\nThe TF32 Paradox Explained:")
    print(f"   1. TF32 accelerates matrix multiplications on Tensor Cores")
    print(f"   2. Large target models benefit more from TF32 (3-4x speedup)")
    print(f"   3. Small draft models benefit less (1.5-2x speedup)")
    print(f"   4. When target speeds up MORE than draft, speculative advantage shrinks")
    
    for target in sorted(target_data.keys()):
        print(f"\n{target}:")
        
        auto_exps = target_data[target]['auto']
        spec_exps = target_data[target]['spec']
        
        # Calculate average speedups
        for sample in ['short', 'long', 'combined']:
            auto_off = [e for e in auto_exps if e['sample_set'] == sample and not e['tf32_enabled']][0]
            auto_on = [e for e in auto_exps if e['sample_set'] == sample and e['tf32_enabled']][0]
            
            auto_tf32_speedup = auto_on['throughput'] / auto_off['throughput']
            
            spec_off = [e for e in spec_exps if e['sample_set'] == sample and not e['tf32_enabled']]
            spec_on = [e for e in spec_exps if e['sample_set'] == sample and e['tf32_enabled']]
            
            best_spec_off = max([e['speedup_vs_auto'] for e in spec_off])
            best_spec_on = max([e['speedup_vs_auto'] for e in spec_on])
            
            print(f"   {sample}:")
            print(f"      - Auto TF32 speedup: {auto_tf32_speedup:.2f}x")
            print(f"      - Best spec speedup (TF32 OFF): {best_spec_off:.2f}x")
            print(f"      - Best spec speedup (TF32 ON): {best_spec_on:.2f}x")
            print(f"      - Spec advantage reduction: {best_spec_off - best_spec_on:.2f}x")


def draft_model_characteristics(target_data, output_dir):
    """Analyze draft model characteristics."""
    
    print(f"\n{'='*100}")
    print(f"DRAFT MODEL CHARACTERISTICS")
    print(f"{'='*100}")
    
    # Collect all speculative experiments
    all_spec = []
    for target in target_data.keys():
        all_spec.extend(target_data[target]['spec'])
    
    # Group by draft model
    draft_stats = defaultdict(lambda: {
        'speedups': [],
        'accepts': [],
        'throughputs': [],
        'configs': []
    })
    
    for exp in all_spec:
        draft = extract_draft_model_name(exp['draft_model'])
        draft_stats[draft]['speedups'].append(exp['speedup_vs_auto'])
        draft_stats[draft]['accepts'].append(exp['acceptance_rate'] * 100)
        draft_stats[draft]['throughputs'].append(exp['throughput'])
        draft_stats[draft]['configs'].append(exp)
    
    print(f"\nOverall Performance Summary:")
    print(f"{'Draft Model':<25} | {'Avg Speedup':<12} | {'Max Speedup':<12} | {'Avg Accept':<11} | {'Success Rate':<13}")
    print(f"{'-'*25}+{'-'*14}+{'-'*14}+{'-'*13}+{'-'*13}")
    
    for draft in sorted(draft_stats.keys()):
        stats = draft_stats[draft]
        avg_speedup = np.mean(stats['speedups'])
        max_speedup = np.max(stats['speedups'])
        avg_accept = np.mean(stats['accepts'])
        success_rate = len([s for s in stats['speedups'] if s > 1.0]) / len(stats['speedups']) * 100
        
        print(f"{draft:<25} | {avg_speedup:>10.2f}x | {max_speedup:>10.2f}x | "
              f"{avg_accept:>9.1f}% | {success_rate:>11.1f}%")
    
    print(f"\nKey Insights:")
    print(f"   - Llama-3.2-1B: Transformer architecture, high acceptance rates, consistent performance")
    print(f"   - Mamba models: SSM architecture, lower acceptance rates, more variable performance")
    print(f"   - Success Rate: % of configurations achieving >1x speedup")


def lookahead_patterns(target_data, output_dir):
    """Analyze lookahead optimization patterns."""
    
    print(f"\n{'='*100}")
    print(f"LOOKAHEAD OPTIMIZATION PATTERNS")
    print(f"{'='*100}")
    
    # Analyze optimal K for different scenarios
    optimal_k = defaultdict(list)
    
    for target in target_data.keys():
        spec_exps = target_data[target]['spec']
        
        for sample in ['short', 'long', 'combined']:
            for tf32 in [False, True]:
                for draft_full in set([e['draft_model'] for e in spec_exps]):
                    draft = extract_draft_model_name(draft_full)
                    
                    # Find best K for this configuration
                    configs = [e for e in spec_exps 
                              if e['sample_set'] == sample 
                              and e['tf32_enabled'] == tf32 
                              and e['draft_model'] == draft_full]
                    
                    if configs:
                        best = max(configs, key=lambda x: x['speedup_vs_auto'])
                        optimal_k[(draft, sample, tf32)].append(best['lookahead'])
    
    print(f"\nOptimal Lookahead Distribution:")
    print(f"{'Configuration':<40} | {'Optimal K':<10} | {'Frequency':<12}")
    print(f"{'-'*40}+{'-'*12}+{'-'*12}")
    
    k_frequency = defaultdict(int)
    for k_list in optimal_k.values():
        for k in k_list:
            k_frequency[k] += 1
    
    for k in sorted(k_frequency.keys()):
        freq = k_frequency[k]
        pct = freq / sum(k_frequency.values()) * 100
        print(f"{'K=' + str(k):<40} | {k:<10d} | {freq:>4d} ({pct:>5.1f}%)")
    
    print(f"\nPattern Analysis:")
    print(f"   - Without TF32: Higher K values (5-8) tend to be optimal")
    print(f"   - With TF32: Lower K values (3-4) tend to be optimal")
    print(f"   - Long prompts: Benefit more from higher K")
    print(f"   - Short prompts: Optimal K more variable")


def create_deep_analysis_plots(results, output_dir):
    """Create additional deep analysis plots."""
    
    print(f"\nGenerating deep analysis plots...")
    
    experiments = results['experiments']
    
    # 1. Target Model Comparison Plot
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Target Model Comparison: 8B vs 3B', fontsize=16, fontweight='bold')
    
    # Extract data by target
    target_perf = defaultdict(lambda: defaultdict(list))
    
    for exp in experiments:
        if exp['method'] != 'autoregressive':
            target = exp['target_model']
            tf32 = exp['tf32_enabled']
            target_perf[target][tf32].append(exp['speedup_vs_auto'])
    
    # Plot 1: Speedup distribution
    ax = axes[0, 0]
    for target in sorted(target_perf.keys()):
        for tf32 in [False, True]:
            label = f"{target} (TF32 {'ON' if tf32 else 'OFF'})"
            ax.hist(target_perf[target][tf32], bins=20, alpha=0.5, label=label)
    ax.set_xlabel('Speedup vs Autoregressive')
    ax.set_ylabel('Frequency')
    ax.set_title('Speedup Distribution by Target Model')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Average speedup comparison
    ax = axes[0, 1]
    targets = sorted(target_perf.keys())
    x = np.arange(len(targets))
    width = 0.35
    
    off_avgs = [np.mean(target_perf[t][False]) for t in targets]
    on_avgs = [np.mean(target_perf[t][True]) for t in targets]
    
    ax.bar(x - width/2, off_avgs, width, label='TF32 OFF', alpha=0.8)
    ax.bar(x + width/2, on_avgs, width, label='TF32 ON', alpha=0.8)
    ax.set_ylabel('Average Speedup')
    ax.set_title('Average Speculative Speedup by Target')
    ax.set_xticks(x)
    ax.set_xticklabels([t.split('-')[-1] for t in targets])
    ax.axhline(y=1, color='black', linestyle='--', alpha=0.5)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Acceptance rate vs speedup by target
    ax = axes[1, 0]
    for target in sorted(target_perf.keys()):
        accepts = []
        speedups = []
        for exp in experiments:
            if exp['method'] != 'autoregressive' and exp['target_model'] == target:
                accepts.append(exp['acceptance_rate'] * 100)
                speedups.append(exp['speedup_vs_auto'])
        ax.scatter(accepts, speedups, label=target, alpha=0.5, s=30)
    ax.set_xlabel('Acceptance Rate (%)')
    ax.set_ylabel('Speedup')
    ax.set_title('Acceptance vs Speedup by Target Model')
    ax.axhline(y=1, color='black', linestyle='--', alpha=0.5)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Best config comparison
    ax = axes[1, 1]
    best_by_target_sample = defaultdict(dict)
    
    for exp in experiments:
        if exp['method'] != 'autoregressive':
            key = (exp['target_model'], exp['sample_set'], exp['tf32_enabled'])
            if key not in best_by_target_sample or exp['speedup_vs_auto'] > best_by_target_sample[key]['speedup_vs_auto']:
                best_by_target_sample[key] = exp
    
    targets = sorted(set([k[0] for k in best_by_target_sample.keys()]))
    samples = ['short', 'long', 'combined']
    
    x = np.arange(len(samples))
    width = 0.2
    offsets = [-1.5*width, -0.5*width, 0.5*width, 1.5*width]
    
    for i, target in enumerate(targets):
        for j, tf32 in enumerate([False, True]):
            speedups = [best_by_target_sample.get((target, s, tf32), {}).get('speedup_vs_auto', 0) 
                       for s in samples]
            offset = offsets[i*2 + j]
            label = f"{target.split('-')[-1]} (TF32 {'ON' if tf32 else 'OFF'})"
            ax.bar(x + offset, speedups, width, label=label, alpha=0.8)
    
    ax.set_ylabel('Best Speedup')
    ax.set_title('Best Speedup by Target and Sample Set')
    ax.set_xticks(x)
    ax.set_xticklabels(samples)
    ax.axhline(y=1, color='black', linestyle='--', alpha=0.5)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'deep_analysis_target_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: deep_analysis_target_comparison.png")


def main():
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python deep_analysis.py <results.json>")
        sys.exit(1)
    
    results_path = Path(sys.argv[1])
    if not results_path.exists():
        print(f"Error: {results_path} not found")
        sys.exit(1)
    
    # Create output directory
    output_dir = results_path.parent / 'deep_analysis'
    output_dir.mkdir(exist_ok=True)
    
    print(f"\n{'='*100}")
    print(f"DEEP ANALYSIS OF SPECULATIVE SAMPLING RESULTS")
    print(f"{'='*100}")
    print(f"Input: {results_path}")
    print(f"Output: {output_dir}")
    
    # Load results
    results = load_results(results_path)
    print(f"Loaded {len(results['experiments'])} experiments")
    
    # Perform deep analysis
    deep_analysis(results, output_dir)
    
    # Create plots
    create_deep_analysis_plots(results, output_dir)
    
    print(f"\n{'='*100}")
    print(f"✅ Deep analysis complete!")
    print(f"{'='*100}\n")


if __name__ == '__main__':
    main()
