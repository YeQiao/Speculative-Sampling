#!/usr/bin/env python3
"""
Comprehensive draft model comparison analysis.
Deep dive into Llama-3.2-1B vs Mamba variants performance.
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
        return 'Mamba-1k'
    elif 'checkpoint-2000' in full_name:
        return 'Mamba-2k'
    elif 'checkpoint-3000' in full_name:
        return 'Mamba-3k'
    elif 'distilled-best' in full_name:
        return 'Mamba-best'
    return full_name


def drafter_overview_analysis(results, output_dir, target_filter=None, tf32_filter=None):
    """Comprehensive overview of draft model performance."""
    
    target_name = target_filter if target_filter else "All Targets"
    tf32_name = "TF32 ON" if tf32_filter is True else "TF32 OFF" if tf32_filter is False else "TF32 All"
    print("\n" + "="*120)
    print(f"DRAFT MODEL COMPARISON: Llama-3.2-1B vs Mamba Variants ({target_name}, {tf32_name})")
    print("="*120)
    
    experiments = results['experiments']
    spec_exps = [e for e in experiments if e['method'] != 'autoregressive']
    
    # Filter by target if specified
    if target_filter:
        spec_exps = [e for e in spec_exps if e['target_model'] == target_filter]
    
    # Filter by TF32 if specified
    if tf32_filter is not None:
        spec_exps = [e for e in spec_exps if e['tf32_enabled'] == tf32_filter]
    
    # Group by draft model
    draft_data = defaultdict(lambda: {
        'speedups': [],
        'accepts': [],
        'throughputs': [],
        'configs': []
    })
    
    for exp in spec_exps:
        draft = extract_draft_model_name(exp['draft_model'])
        draft_data[draft]['speedups'].append(exp['speedup_vs_auto'])
        draft_data[draft]['accepts'].append(exp['acceptance_rate'] * 100)
        draft_data[draft]['throughputs'].append(exp['throughput'])
        draft_data[draft]['configs'].append(exp)
    
    print(f"\n1. OVERALL PERFORMANCE SUMMARY")
    print(f"{'-'*120}")
    print(f"{'Draft Model':<25} | {'Configs':<8} | {'Avg Speedup':<12} | {'Max Speedup':<12} | "
          f"{'Min Speedup':<12} | {'Std Dev':<10} | {'Success %':<10}")
    print(f"{'-'*25}+{'-'*10}+{'-'*14}+{'-'*14}+{'-'*14}+{'-'*12}+{'-'*10}")
    
    for draft in sorted(draft_data.keys()):
        stats = draft_data[draft]
        avg_speedup = np.mean(stats['speedups'])
        max_speedup = np.max(stats['speedups'])
        min_speedup = np.min(stats['speedups'])
        std_speedup = np.std(stats['speedups'])
        success_rate = len([s for s in stats['speedups'] if s > 1.0]) / len(stats['speedups']) * 100
        n_configs = len(stats['configs'])
        
        print(f"{draft:<25} | {n_configs:<8d} | {avg_speedup:>10.2f}x | {max_speedup:>10.2f}x | "
              f"{min_speedup:>10.2f}x | {std_speedup:>10.3f} | {success_rate:>8.1f}%")
    
    print(f"\n2. ACCEPTANCE RATE ANALYSIS")
    print(f"{'-'*120}")
    print(f"{'Draft Model':<25} | {'Avg Accept':<12} | {'Max Accept':<12} | {'Min Accept':<12} | "
          f"{'Std Dev':<10} | {'Range':<10}")
    print(f"{'-'*25}+{'-'*14}+{'-'*14}+{'-'*14}+{'-'*12}+{'-'*10}")
    
    for draft in sorted(draft_data.keys()):
        stats = draft_data[draft]
        avg_accept = np.mean(stats['accepts'])
        max_accept = np.max(stats['accepts'])
        min_accept = np.min(stats['accepts'])
        std_accept = np.std(stats['accepts'])
        range_accept = max_accept - min_accept
        
        print(f"{draft:<25} | {avg_accept:>10.1f}% | {max_accept:>10.1f}% | {min_accept:>10.1f}% | "
              f"{std_accept:>9.2f}% | {range_accept:>8.1f}%")
    
    print(f"\n3. ARCHITECTURE COMPARISON: TRANSFORMER vs SSM")
    print(f"{'-'*120}")
    
    llama_speedups = draft_data['Llama-3.2-1B']['speedups']
    mamba_speedups = []
    for draft in draft_data.keys():
        if 'Mamba' in draft:
            mamba_speedups.extend(draft_data[draft]['speedups'])
    
    print(f"\nLlama-3.2-1B (Transformer Architecture):")
    print(f"   - Total configs: {len(llama_speedups)}")
    print(f"   - Average speedup: {np.mean(llama_speedups):.2f}x")
    print(f"   - Median speedup: {np.median(llama_speedups):.2f}x")
    print(f"   - Success rate: {len([s for s in llama_speedups if s > 1.0]) / len(llama_speedups) * 100:.1f}%")
    print(f"   - 75th percentile: {np.percentile(llama_speedups, 75):.2f}x")
    print(f"   - 25th percentile: {np.percentile(llama_speedups, 25):.2f}x")
    
    print(f"\nMamba Models (SSM Architecture):")
    print(f"   - Total configs: {len(mamba_speedups)}")
    print(f"   - Average speedup: {np.mean(mamba_speedups):.2f}x")
    print(f"   - Median speedup: {np.median(mamba_speedups):.2f}x")
    print(f"   - Success rate: {len([s for s in mamba_speedups if s > 1.0]) / len(mamba_speedups) * 100:.1f}%")
    print(f"   - 75th percentile: {np.percentile(mamba_speedups, 75):.2f}x")
    print(f"   - 25th percentile: {np.percentile(mamba_speedups, 25):.2f}x")
    
    print(f"\nPerformance Gap: {np.mean(llama_speedups) - np.mean(mamba_speedups):.2f}x in favor of Llama")
    
    return draft_data


def mamba_distillation_analysis(results, output_dir, target_filter=None, tf32_filter=None):
    """Analyze Mamba distillation progress."""
    
    target_name = target_filter if target_filter else "All Targets"
    tf32_name = "TF32 ON" if tf32_filter is True else "TF32 OFF" if tf32_filter is False else "TF32 All"
    print(f"\n{'='*120}")
    print(f"MAMBA DISTILLATION ANALYSIS ({target_name}, {tf32_name})")
    print(f"{'='*120}")
    
    experiments = results['experiments']
    spec_exps = [e for e in experiments if e['method'] != 'autoregressive']
    
    # Filter by target if specified
    if target_filter:
        spec_exps = [e for e in spec_exps if e['target_model'] == target_filter]
    
    # Filter by TF32 if specified
    if tf32_filter is not None:
        spec_exps = [e for e in spec_exps if e['tf32_enabled'] == tf32_filter]
    
    # Group Mamba variants
    mamba_variants = {
        'pretrained': 'Mamba-65m-pretrained',
        '1k': 'Mamba-1k',
        '2k': 'Mamba-2k',
        '3k': 'Mamba-3k',
        'best': 'Mamba-best'
    }
    
    mamba_data = defaultdict(lambda: defaultdict(list))
    
    for exp in spec_exps:
        draft = extract_draft_model_name(exp['draft_model'])
        if draft in mamba_variants.values():
            variant = [k for k, v in mamba_variants.items() if v == draft][0]
            target = exp['target_model']
            tf32 = exp['tf32_enabled']
            sample = exp['sample_set']
            
            key = f"{target}_{sample}_tf32{tf32}"
            mamba_data[variant][key].append({
                'speedup': exp['speedup_vs_auto'],
                'accept': exp['acceptance_rate'] * 100,
                'k': exp['lookahead']
            })
    
    print(f"\n1. DISTILLATION PROGRESS: Model Quality Evolution")
    print(f"{'-'*120}")
    print(f"{'Variant':<15} | {'Training':<10} | {'Avg Speedup':<12} | {'Avg Accept':<12} | "
          f"{'vs Pretrained':<15} | {'vs Llama':<12}")
    print(f"{'-'*15}+{'-'*12}+{'-'*14}+{'-'*14}+{'-'*17}+{'-'*12}")
    
    # Get baseline (pretrained)
    pretrained_speedup = np.mean([d['speedup'] for configs in mamba_data['pretrained'].values() 
                                   for d in configs])
    pretrained_accept = np.mean([d['accept'] for configs in mamba_data['pretrained'].values() 
                                  for d in configs])
    
    # Get Llama baseline (filter by target if specified)
    llama_exps = [e for e in spec_exps if 'llama-3.2-1B' in e['draft_model']]
    llama_speedup = np.mean([e['speedup_vs_auto'] for e in llama_exps]) if llama_exps else 0
    
    for variant in ['pretrained', '1k', '2k', '3k', 'best']:
        if variant not in mamba_data:
            continue
        
        training = 'Baseline' if variant == 'pretrained' else f"{variant} steps"
        avg_speedup = np.mean([d['speedup'] for configs in mamba_data[variant].values() 
                               for d in configs])
        avg_accept = np.mean([d['accept'] for configs in mamba_data[variant].values() 
                              for d in configs])
        
        vs_pretrained = ((avg_speedup - pretrained_speedup) / pretrained_speedup * 100) if variant != 'pretrained' else 0
        vs_llama = ((avg_speedup - llama_speedup) / llama_speedup * 100)
        
        vs_pre_str = f"+{vs_pretrained:>6.1f}%" if vs_pretrained > 0 else f"{vs_pretrained:>7.1f}%"
        vs_llama_str = f"{vs_llama:>+7.1f}%"
        
        print(f"{mamba_variants[variant]:<15} | {training:<10} | {avg_speedup:>10.2f}x | "
              f"{avg_accept:>10.1f}% | {vs_pre_str:<15} | {vs_llama_str:<12}")
    
    print(f"\n2. ACCEPTANCE RATE IMPROVEMENT TRACKING")
    print(f"{'-'*120}")
    
    for variant in ['pretrained', '1k', '2k', '3k', 'best']:
        if variant not in mamba_data:
            continue
        
        accepts = [d['accept'] for configs in mamba_data[variant].values() for d in configs]
        improvement = (np.mean(accepts) - pretrained_accept) if variant != 'pretrained' else 0
        
        print(f"{mamba_variants[variant]:<20}: {np.mean(accepts):>5.1f}% (Δ{improvement:>+5.1f}% from pretrained)")
    
    print(f"\n3. BEST CHECKPOINT IDENTIFICATION")
    print(f"{'-'*120}")
    
    for config_key in sorted(set([k for v in mamba_data.values() for k in v.keys()])):
        target, sample, tf32_str = config_key.split('_')
        tf32 = 'ON' if 'True' in tf32_str else 'OFF'
        
        print(f"\n   Config: {target} | {sample} | TF32 {tf32}")
        print(f"   {'Variant':<20} | {'Best Speedup':<12} | {'Best K':<7} | {'Accept':<10}")
        print(f"   {'-'*20}+{'-'*14}+{'-'*9}+{'-'*10}")
        
        for variant in ['pretrained', '1k', '2k', '3k', 'best']:
            if variant in mamba_data and config_key in mamba_data[variant]:
                data = mamba_data[variant][config_key]
                best_idx = np.argmax([d['speedup'] for d in data])
                best = data[best_idx]
                
                print(f"   {mamba_variants[variant]:<20} | {best['speedup']:>10.2f}x | "
                      f"K={best['k']:<5d} | {best['accept']:>8.1f}%")


def detailed_drafter_comparison(results, output_dir):
    """Detailed comparison across different configurations."""
    
    print(f"\n{'='*120}")
    print(f"DETAILED DRAFTER COMPARISON BY CONFIGURATION")
    print(f"{'='*120}")
    
    experiments = results['experiments']
    spec_exps = [e for e in experiments if e['method'] != 'autoregressive']
    
    # Organize by target, sample, tf32
    configs = defaultdict(lambda: defaultdict(list))
    
    for exp in spec_exps:
        draft = extract_draft_model_name(exp['draft_model'])
        key = (exp['target_model'], exp['sample_set'], exp['tf32_enabled'])
        configs[key][draft].append(exp)
    
    targets = sorted(set([k[0] for k in configs.keys()]))
    samples = ['short', 'long', 'combined']
    tf32_settings = [False, True]
    
    for target in targets:
        print(f"\n{'='*120}")
        print(f"TARGET: {target}")
        print(f"{'='*120}")
        
        for sample in samples:
            for tf32 in tf32_settings:
                key = (target, sample, tf32)
                if key not in configs:
                    continue
                
                tf32_str = 'ON' if tf32 else 'OFF'
                print(f"\n   Sample: {sample.upper()} | TF32: {tf32_str}")
                print(f"   {'-'*116}")
                print(f"   {'Draft Model':<25} | {'Best Speedup':<12} | {'Avg Speedup':<12} | "
                      f"{'Best Accept':<12} | {'Best K':<7} | {'Consistency':<12}")
                print(f"   {'-'*25}+{'-'*14}+{'-'*14}+{'-'*14}+{'-'*9}+{'-'*12}")
                
                # Get ranking for this configuration
                draft_stats = {}
                for draft in sorted(configs[key].keys()):
                    exps = configs[key][draft]
                    speedups = [e['speedup_vs_auto'] for e in exps]
                    accepts = [e['acceptance_rate'] * 100 for e in exps]
                    
                    best_idx = np.argmax(speedups)
                    best_speedup = speedups[best_idx]
                    avg_speedup = np.mean(speedups)
                    best_accept = accepts[best_idx]
                    best_k = exps[best_idx]['lookahead']
                    consistency = 1.0 - (np.std(speedups) / np.mean(speedups)) if np.mean(speedups) > 0 else 0
                    
                    draft_stats[draft] = {
                        'best': best_speedup,
                        'avg': avg_speedup,
                        'accept': best_accept,
                        'k': best_k,
                        'consistency': consistency
                    }
                
                # Sort by best speedup
                for draft in sorted(draft_stats.keys(), key=lambda x: draft_stats[x]['best'], reverse=True):
                    stats = draft_stats[draft]
                    print(f"   {draft:<25} | {stats['best']:>10.2f}x | {stats['avg']:>10.2f}x | "
                          f"{stats['accept']:>10.1f}% | K={stats['k']:<5d} | {stats['consistency']:>10.2f}")


def acceptance_vs_lookahead_analysis(results, output_dir):
    """Deep analysis of acceptance rate patterns."""
    
    print(f"\n{'='*120}")
    print(f"ACCEPTANCE RATE vs LOOKAHEAD ANALYSIS")
    print(f"{'='*120}")
    
    experiments = results['experiments']
    spec_exps = [e for e in experiments if e['method'] != 'autoregressive']
    
    # Group by draft and K
    accept_by_draft_k = defaultdict(lambda: defaultdict(list))
    speedup_by_draft_k = defaultdict(lambda: defaultdict(list))
    
    for exp in spec_exps:
        draft = extract_draft_model_name(exp['draft_model'])
        k = exp['lookahead']
        accept_by_draft_k[draft][k].append(exp['acceptance_rate'] * 100)
        speedup_by_draft_k[draft][k].append(exp['speedup_vs_auto'])
    
    print(f"\n1. ACCEPTANCE RATE DEGRADATION WITH LOOKAHEAD")
    print(f"{'-'*120}")
    print(f"{'Draft Model':<25} | {'K=2':<8} | {'K=3':<8} | {'K=4':<8} | {'K=5':<8} | "
          f"{'K=6':<8} | {'K=8':<8} | {'Degradation':<12}")
    print(f"{'-'*25}+{'-'*10}+{'-'*10}+{'-'*10}+{'-'*10}+{'-'*10}+{'-'*10}+{'-'*12}")
    
    for draft in sorted(accept_by_draft_k.keys()):
        values = []
        k_list = [2, 3, 4, 5, 6, 8]
        for k in k_list:
            if k in accept_by_draft_k[draft]:
                avg = np.mean(accept_by_draft_k[draft][k])
                values.append(f"{avg:>6.1f}%")
            else:
                values.append("   N/A ")
        
        # Calculate degradation (K=2 to K=8)
        if 2 in accept_by_draft_k[draft] and 8 in accept_by_draft_k[draft]:
            deg = np.mean(accept_by_draft_k[draft][2]) - np.mean(accept_by_draft_k[draft][8])
            deg_str = f"{deg:>+9.1f}%"
        else:
            deg_str = "      N/A"
        
        print(f"{draft:<25} | {' | '.join(values)} | {deg_str}")
    
    print(f"\n2. OPTIMAL LOOKAHEAD BY DRAFT MODEL")
    print(f"{'-'*120}")
    print(f"{'Draft Model':<25} | {'Most Common K':<15} | {'Highest Avg Speedup K':<22} | "
          f"{'Highest Accept K':<18}")
    print(f"{'-'*25}+{'-'*17}+{'-'*24}+{'-'*18}")
    
    for draft in sorted(speedup_by_draft_k.keys()):
        # Most common optimal K
        optimal_k_counts = defaultdict(int)
        for k in speedup_by_draft_k[draft]:
            if np.mean(speedup_by_draft_k[draft][k]) == max([np.mean(speedup_by_draft_k[draft][kk]) 
                                                              for kk in speedup_by_draft_k[draft]]):
                optimal_k_counts[k] += 1
        
        most_common_k = max(speedup_by_draft_k[draft].keys(), 
                           key=lambda k: np.mean(speedup_by_draft_k[draft][k]))
        
        highest_speedup_k = max(speedup_by_draft_k[draft].keys(),
                               key=lambda k: np.mean(speedup_by_draft_k[draft][k]))
        
        highest_accept_k = max(accept_by_draft_k[draft].keys(),
                              key=lambda k: np.mean(accept_by_draft_k[draft][k]))
        
        avg_at_best = np.mean(speedup_by_draft_k[draft][highest_speedup_k])
        accept_at_best = np.mean(accept_by_draft_k[draft][highest_accept_k])
        
        print(f"{draft:<25} | K={most_common_k:<13d} | K={highest_speedup_k:<3d} ({avg_at_best:>5.2f}x avg) | "
              f"K={highest_accept_k:<3d} ({accept_at_best:>5.1f}%)")
    
    print(f"\n3. ACCEPTANCE RATE vs SPEEDUP EFFICIENCY")
    print(f"{'-'*120}")
    print(f"{'Draft Model':<25} | {'Avg Accept':<12} | {'Avg Speedup':<12} | "
          f"{'Efficiency Ratio':<16} | {'Rank':<6}")
    print(f"{'-'*25}+{'-'*14}+{'-'*14}+{'-'*18}+{'-'*6}")
    
    efficiency = {}
    for draft in sorted(accept_by_draft_k.keys()):
        avg_accept = np.mean([v for values in accept_by_draft_k[draft].values() for v in values])
        avg_speedup = np.mean([v for values in speedup_by_draft_k[draft].values() for v in values])
        
        # Efficiency = speedup / (accept / 100) - measures how well acceptance translates to speedup
        eff_ratio = avg_speedup / (avg_accept / 100) if avg_accept > 0 else 0
        efficiency[draft] = {
            'accept': avg_accept,
            'speedup': avg_speedup,
            'ratio': eff_ratio
        }
    
    # Rank by efficiency
    ranked = sorted(efficiency.items(), key=lambda x: x[1]['ratio'], reverse=True)
    
    for i, (draft, stats) in enumerate(ranked, 1):
        print(f"{draft:<25} | {stats['accept']:>10.1f}% | {stats['speedup']:>10.2f}x | "
              f"{stats['ratio']:>16.3f} | #{i:<5d}")
    
    print(f"\n   Higher efficiency = better speedup per unit of acceptance rate")
    print(f"   Llama-3.2-1B should rank high (good speedup despite high acceptance)")


def failure_mode_analysis(results, output_dir):
    """Analyze when and why draft models fail."""
    
    print(f"\n{'='*120}")
    print(f"FAILURE MODE ANALYSIS: When Speculation Hurts Performance")
    print(f"{'='*120}")
    
    experiments = results['experiments']
    spec_exps = [e for e in experiments if e['method'] != 'autoregressive']
    
    # Find configurations where speedup < 1.0 (slowdown)
    failures = defaultdict(list)
    
    for exp in spec_exps:
        if exp['speedup_vs_auto'] < 1.0:
            draft = extract_draft_model_name(exp['draft_model'])
            failures[draft].append(exp)
    
    print(f"\n1. FAILURE RATES BY DRAFT MODEL")
    print(f"{'-'*120}")
    print(f"{'Draft Model':<25} | {'Total Configs':<13} | {'Failures':<10} | {'Failure Rate':<13} | "
          f"{'Avg Slowdown':<13}")
    print(f"{'-'*25}+{'-'*15}+{'-'*12}+{'-'*15}+{'-'*13}")
    
    for draft in sorted(set([extract_draft_model_name(e['draft_model']) for e in spec_exps])):
        total = len([e for e in spec_exps if extract_draft_model_name(e['draft_model']) == draft])
        fails = len(failures[draft])
        fail_rate = fails / total * 100 if total > 0 else 0
        avg_slowdown = np.mean([e['speedup_vs_auto'] for e in failures[draft]]) if fails > 0 else 0
        
        print(f"{draft:<25} | {total:<13d} | {fails:<10d} | {fail_rate:>11.1f}% | {avg_slowdown:>11.3f}x")
    
    print(f"\n2. FAILURE PATTERNS BY CONFIGURATION")
    print(f"{'-'*120}")
    
    # Analyze which configurations lead to failures
    failure_by_config = defaultdict(lambda: defaultdict(int))
    
    for draft in failures:
        for exp in failures[draft]:
            target = exp['target_model']
            sample = exp['sample_set']
            tf32 = 'ON' if exp['tf32_enabled'] else 'OFF'
            k = exp['lookahead']
            
            config = f"{target}|{sample}|TF32:{tf32}"
            failure_by_config[draft][config] += 1
    
    print(f"   Most Common Failure Configurations:")
    for draft in sorted(failure_by_config.keys()):
        print(f"\n   {draft}:")
        sorted_configs = sorted(failure_by_config[draft].items(), 
                               key=lambda x: x[1], reverse=True)[:3]
        for config, count in sorted_configs:
            target, sample, tf32 = config.split('|')
            print(f"      - {target} | {sample} | {tf32}: {count} failures")
    
    print(f"\n3. TF32 IMPACT ON FAILURES")
    print(f"{'-'*120}")
    
    tf32_failures = defaultdict(lambda: {'ON': 0, 'OFF': 0})
    
    for draft in failures:
        for exp in failures[draft]:
            tf32_key = 'ON' if exp['tf32_enabled'] else 'OFF'
            tf32_failures[draft][tf32_key] += 1
    
    print(f"{'Draft Model':<25} | {'TF32 OFF Failures':<18} | {'TF32 ON Failures':<17} | {'Ratio (ON/OFF)':<16}")
    print(f"{'-'*25}+{'-'*20}+{'-'*19}+{'-'*16}")
    
    for draft in sorted(tf32_failures.keys()):
        off_fails = tf32_failures[draft]['OFF']
        on_fails = tf32_failures[draft]['ON']
        ratio = on_fails / off_fails if off_fails > 0 else float('inf')
        
        print(f"{draft:<25} | {off_fails:<18d} | {on_fails:<17d} | {ratio:>16.2f}x")
    
    print(f"\n   Ratio > 1.0 means TF32 causes MORE failures (expected)")
    print(f"\n4. WORST PERFORMING CONFIGURATIONS")
    print(f"{'-'*120}")
    
    # Find worst speedups
    all_failures = []
    for draft in failures:
        all_failures.extend(failures[draft])
    
    worst = sorted(all_failures, key=lambda x: x['speedup_vs_auto'])[:10]
    
    print(f"   Top 10 Worst Configurations:")
    print(f"   {'Rank':<5} | {'Draft':<20} | {'Target':<15} | {'Sample':<10} | "
          f"{'TF32':<6} | {'K':<3} | {'Speedup':<10} | {'Accept':<10}")
    print(f"   {'-'*5}+{'-'*22}+{'-'*17}+{'-'*12}+{'-'*8}+{'-'*5}+{'-'*12}+{'-'*10}")
    
    for i, exp in enumerate(worst, 1):
        draft = extract_draft_model_name(exp['draft_model'])
        tf32 = 'ON' if exp['tf32_enabled'] else 'OFF'
        print(f"   {i:<5d} | {draft:<20} | {exp['target_model']:<15} | {exp['sample_set']:<10} | "
              f"{tf32:<6} | {exp['lookahead']:<3d} | {exp['speedup_vs_auto']:>10.3f}x | "
              f"{exp['acceptance_rate']*100:>8.1f}%")


def create_drafter_comparison_plots(results, output_dir, target_filter=None, tf32_filter=None):
    """Create comprehensive drafter comparison visualizations."""
    
    target_name = target_filter if target_filter else "All Targets"
    tf32_name = "TF32 ON" if tf32_filter is True else "TF32 OFF" if tf32_filter is False else "TF32 All"
    print(f"\nGenerating drafter comparison plots for {target_name}, {tf32_name}...")
    
    experiments = results['experiments']
    spec_exps = [e for e in experiments if e['method'] != 'autoregressive']
    
    # Filter by target if specified
    if target_filter:
        spec_exps = [e for e in spec_exps if e['target_model'] == target_filter]
    
    # Filter by TF32 if specified
    if tf32_filter is not None:
        spec_exps = [e for e in spec_exps if e['tf32_enabled'] == tf32_filter]
    
    # 1. Performance Distribution Comparison
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle(f'Draft Model Performance Distribution Analysis - {target_name}', fontsize=16, fontweight='bold')
    
    # Extract data by draft
    draft_speedups = defaultdict(list)
    draft_accepts = defaultdict(list)
    draft_throughputs = defaultdict(list)
    
    for exp in spec_exps:
        draft = extract_draft_model_name(exp['draft_model'])
        draft_speedups[draft].append(exp['speedup_vs_auto'])
        draft_accepts[draft].append(exp['acceptance_rate'] * 100)
        draft_throughputs[draft].append(exp['throughput'])
    
    # Plot 1: Speedup distribution (violin plot)
    ax = axes[0, 0]
    data_to_plot = [draft_speedups[d] for d in sorted(draft_speedups.keys())]
    labels = sorted(draft_speedups.keys())
    
    parts = ax.violinplot(data_to_plot, positions=range(len(labels)), showmeans=True, showmedians=True)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Speedup vs Autoregressive')
    ax.set_title('Speedup Distribution by Draft Model')
    ax.axhline(y=1, color='red', linestyle='--', alpha=0.5, linewidth=2)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Acceptance rate distribution (box plot)
    ax = axes[0, 1]
    ax.boxplot(data_to_plot, labels=labels, showmeans=True)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Acceptance Rate (%)')
    ax.set_title('Acceptance Rate Distribution')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Success rate (bar chart)
    ax = axes[0, 2]
    success_rates = []
    for draft in sorted(draft_speedups.keys()):
        rate = len([s for s in draft_speedups[draft] if s > 1.0]) / len(draft_speedups[draft]) * 100
        success_rates.append(rate)
    
    colors = ['#2ecc71' if r > 50 else '#e74c3c' for r in success_rates]
    bars = ax.bar(range(len(labels)), success_rates, color=colors, alpha=0.7)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Success Rate (%)')
    ax.set_title('Configuration Success Rate (Speedup > 1.0x)')
    ax.axhline(y=50, color='black', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for i, (bar, rate) in enumerate(zip(bars, success_rates)):
        ax.text(bar.get_x() + bar.get_width()/2, rate + 2, f'{rate:.0f}%',
                ha='center', va='bottom', fontsize=8)
    
    # Plot 4: Average metrics comparison
    ax = axes[1, 0]
    drafts = sorted(draft_speedups.keys())
    avg_speedups = [np.mean(draft_speedups[d]) for d in drafts]
    avg_accepts = [np.mean(draft_accepts[d]) for d in drafts]
    
    x = np.arange(len(drafts))
    width = 0.35
    
    ax2 = ax.twinx()
    bars1 = ax.bar(x - width/2, avg_speedups, width, label='Avg Speedup', alpha=0.8, color='#3498db')
    bars2 = ax2.bar(x + width/2, avg_accepts, width, label='Avg Accept %', alpha=0.8, color='#e67e22')
    
    ax.set_xlabel('Draft Model')
    ax.set_ylabel('Average Speedup', color='#3498db')
    ax2.set_ylabel('Average Acceptance Rate (%)', color='#e67e22')
    ax.set_title('Average Performance Metrics')
    ax.set_xticks(x)
    ax.set_xticklabels(drafts, rotation=45, ha='right', fontsize=8)
    ax.tick_params(axis='y', labelcolor='#3498db')
    ax2.tick_params(axis='y', labelcolor='#e67e22')
    ax.axhline(y=1, color='red', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 5: Mamba distillation progress
    ax = axes[1, 1]
    mamba_order = ['Mamba-65m-pretrained', 'Mamba-1k', 'Mamba-2k', 'Mamba-3k', 'Mamba-best']
    mamba_labels = ['Pretrained', '1k steps', '2k steps', '3k steps', 'Best']
    mamba_speedups = []
    mamba_accepts = []
    
    for draft in mamba_order:
        if draft in draft_speedups:
            mamba_speedups.append(np.mean(draft_speedups[draft]))
            mamba_accepts.append(np.mean(draft_accepts[draft]))
    
    x = np.arange(len(mamba_speedups))
    ax.plot(x, mamba_speedups, marker='o', linewidth=2, markersize=8, label='Speedup', color='#3498db')
    ax2 = ax.twinx()
    ax2.plot(x, mamba_accepts, marker='s', linewidth=2, markersize=8, label='Accept %', color='#e67e22')
    
    ax.set_xlabel('Distillation Progress')
    ax.set_ylabel('Average Speedup', color='#3498db')
    ax2.set_ylabel('Acceptance Rate (%)', color='#e67e22')
    ax.set_title('Mamba Distillation Progress')
    ax.set_xticks(x)
    ax.set_xticklabels(mamba_labels[:len(x)], rotation=45, ha='right', fontsize=8)
    ax.tick_params(axis='y', labelcolor='#3498db')
    ax2.tick_params(axis='y', labelcolor='#e67e22')
    ax.grid(True, alpha=0.3)
    
    # Plot 6: Acceptance vs Speedup scatter for all drafters
    ax = axes[1, 2]
    
    for draft in sorted(draft_speedups.keys()):
        ax.scatter(draft_accepts[draft], draft_speedups[draft], 
                  label=draft, alpha=0.5, s=30)
    
    ax.set_xlabel('Acceptance Rate (%)')
    ax.set_ylabel('Speedup vs Autoregressive')
    ax.set_title('Acceptance Rate vs Speedup (All Configs)')
    ax.axhline(y=1, color='red', linestyle='--', alpha=0.5)
    ax.axvline(x=70, color='gray', linestyle=':', alpha=0.3)
    ax.legend(fontsize=7, loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'drafter_comparison_comprehensive.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: drafter_comparison_comprehensive.png")
    
    # 2. Acceptance vs K degradation plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f'Acceptance Rate Degradation with Lookahead - {target_name}', fontsize=16, fontweight='bold')
    
    # Group by draft and K
    accept_by_draft_k = defaultdict(lambda: defaultdict(list))
    
    for exp in spec_exps:
        draft = extract_draft_model_name(exp['draft_model'])
        k = exp['lookahead']
        accept_by_draft_k[draft][k].append(exp['acceptance_rate'] * 100)
    
    # Plot 1: All drafters
    ax = axes[0]
    for draft in sorted(accept_by_draft_k.keys()):
        k_vals = sorted(accept_by_draft_k[draft].keys())
        accept_means = [np.mean(accept_by_draft_k[draft][k]) for k in k_vals]
        ax.plot(k_vals, accept_means, marker='o', label=draft, linewidth=2, markersize=6)
    
    ax.set_xlabel('Lookahead Steps (K)')
    ax.set_ylabel('Average Acceptance Rate (%)')
    ax.set_title('All Draft Models')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xticks([2, 3, 4, 5, 6, 8])
    
    # Plot 2: Llama vs Mamba average
    ax = axes[1]
    
    # Llama
    llama_k_accept = defaultdict(list)
    for exp in spec_exps:
        if 'llama-3.2-1B' in exp['draft_model']:
            llama_k_accept[exp['lookahead']].append(exp['acceptance_rate'] * 100)
    
    k_vals = sorted(llama_k_accept.keys())
    llama_means = [np.mean(llama_k_accept[k]) for k in k_vals]
    
    # Mamba average
    mamba_k_accept = defaultdict(list)
    for exp in spec_exps:
        draft = extract_draft_model_name(exp['draft_model'])
        if 'Mamba' in draft:
            mamba_k_accept[exp['lookahead']].append(exp['acceptance_rate'] * 100)
    
    mamba_means = [np.mean(mamba_k_accept[k]) for k in k_vals]
    
    ax.plot(k_vals, llama_means, marker='o', label='Llama-3.2-1B', linewidth=3, markersize=8, color='#2ecc71')
    ax.plot(k_vals, mamba_means, marker='s', label='Mamba (avg)', linewidth=3, markersize=8, color='#e74c3c')
    
    ax.set_xlabel('Lookahead Steps (K)')
    ax.set_ylabel('Average Acceptance Rate (%)')
    ax.set_title('Transformer vs SSM Architecture')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xticks([2, 3, 4, 5, 6, 8])
    
    # Add gap annotation
    for i, k in enumerate(k_vals):
        gap = llama_means[i] - mamba_means[i]
        ax.annotate(f'+{gap:.1f}%', xy=(k, (llama_means[i] + mamba_means[i])/2),
                   ha='center', fontsize=8, color='blue')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'acceptance_degradation_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: acceptance_degradation_analysis.png")


def main():
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python drafter_comparison_analysis.py <results.json>")
        sys.exit(1)
    
    results_path = Path(sys.argv[1])
    if not results_path.exists():
        print(f"Error: {results_path} not found")
        sys.exit(1)
    
    # Create output directory
    output_dir = results_path.parent / 'drafter_analysis'
    output_dir.mkdir(exist_ok=True)
    
    print(f"\n{'='*120}")
    print(f"COMPREHENSIVE DRAFT MODEL COMPARISON ANALYSIS")
    print(f"{'='*120}")
    print(f"Input: {results_path}")
    print(f"Output: {output_dir}")
    
    # Load results
    results = load_results(results_path)
    print(f"Loaded {len(results['experiments'])} experiments")
    
    # Get unique target models
    targets = sorted(set([e['target_model'] for e in results['experiments']]))
    print(f"Target models found: {targets}\n")
    
    # TF32 settings
    tf32_settings = [False, True]
    tf32_labels = {False: "tf32_off", True: "tf32_on"}
    
    # Run analysis for each target model and TF32 setting separately
    for target in targets:
        for tf32 in tf32_settings:
            tf32_label = tf32_labels[tf32]
            tf32_name = "TF32 ON" if tf32 else "TF32 OFF"
            
            print(f"\n{'#'*120}")
            print(f"# ANALYSIS FOR VERIFIER: {target} | {tf32_name}")
            print(f"{'#'*120}")
            
            drafter_overview_analysis(results, output_dir, target_filter=target, tf32_filter=tf32)
            mamba_distillation_analysis(results, output_dir, target_filter=target, tf32_filter=tf32)
    
    # Run combined analyses (these already break down by target internally)
    print(f"\n{'#'*120}")
    print(f"# DETAILED CONFIGURATION BREAKDOWN (ALL TF32 SETTINGS)")
    print(f"{'#'*120}")
    detailed_drafter_comparison(results, output_dir)
    acceptance_vs_lookahead_analysis(results, output_dir)
    failure_mode_analysis(results, output_dir)
    
    # Create plots (separate by target AND TF32)
    for target in targets:
        for tf32 in tf32_settings:
            tf32_label = tf32_labels[tf32]
            tf32_name = "TF32 ON" if tf32 else "TF32 OFF"
            
            print(f"\nGenerating plots for {target} | {tf32_name}...")
            # Create directory: target/tf32_on or target/tf32_off
            target_output_dir = output_dir / target.replace('/', '_') / tf32_label
            target_output_dir.mkdir(parents=True, exist_ok=True)
            create_drafter_comparison_plots(results, target_output_dir, target_filter=target, tf32_filter=tf32)
    
    print(f"\n{'='*120}")
    print(f"✅ Drafter comparison analysis complete!")
    print(f"{'='*120}\n")


if __name__ == '__main__':
    main()
