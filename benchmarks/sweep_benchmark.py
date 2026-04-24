"""
Comprehensive benchmarking script to sweep over multiple configurations:
- Sample types: short (prompts_sample_1), long (prompts_sample_long), combined
- Lookahead values: range of values to test
- Draft models: pretrained Mamba, distilled Mamba variants, Llama-3.2-1B
- TF32 settings: enabled vs disabled

Usage:
  python benchmarks/sweep_benchmark.py [options]

Options:
  --target MODELS        Target models to test (comma-separated, default: all)
  --draft MODELS         Draft models to test (comma-separated, default: all)
  --samples SETS         Sample sets to test (comma-separated: short,long,combined, default: all)
  --lookahead VALUES     Lookahead values to test (comma-separated: 2,3,4,5,6,8, default: all)
  --tf32 SETTING         TF32 setting (on/off/both, default: both)
  --max-tokens N         Maximum new tokens to generate (default: 64)
  --temperature T        Sampling temperature (default: 0)
  --output-dir DIR       Output directory for results (default: outputs/sweep_results)

Examples:
  # Test only Mamba-1k with 8B target, TF32 off, K=5,6
  python benchmarks/sweep_benchmark.py --target llama3.1-8B --draft mamba-65m-distilled-checkpoint-1000 --tf32 off --lookahead 5,6

  # Test all models but only short prompts with TF32 on
  python benchmarks/sweep_benchmark.py --samples short --tf32 on

  # Test optimal K values only (2,3,5,6) with both TF32 settings
  python benchmarks/sweep_benchmark.py --lookahead 2,3,5,6
"""

import sys
import time
import warnings
import torch
import json
import os
import argparse
from datetime import datetime
from tqdm import tqdm
from itertools import product
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Suppress NVML warnings
warnings.filterwarnings("ignore", message=".*NVML.*")
warnings.filterwarnings("ignore", message=".*Can't initialize NVML.*")

from transformers import AutoTokenizer, AutoModelForCausalLM
from core.autoregressive_sampling import autoregressive_sampling
from core.speculative_sampling import speculative_sampling

device = "cuda" if torch.cuda.is_available() else "cpu"

# Configuration
TARGET_MODELS = {
    "llama3.1-8B": "/HSC/users/qiaoye/checkpoints/Llama3.1-8B-hf",
    "llama-3.2-3B": "/HSC/users/qiaoye/checkpoints/Llama-3.2-3B",
}
DRAFT_MODELS = {
    "mamba-65m-pretrained": "/HSC/users/qiaoye/SSM_SPEC/checkpoints/custom-mamba-65m-multi-gpu",
    "mamba-65m-distilled-best": "/HSC/users/qiaoye/SSM_SPEC/checkpoints/advanced-mamba-alignment/best_model",
    "mamba-65m-distilled-checkpoint-1000": "/HSC/users/qiaoye/SSM_SPEC/checkpoints/advanced-mamba-alignment/checkpoint-1000",
    "mamba-65m-distilled-checkpoint-2000": "/HSC/users/qiaoye/SSM_SPEC/checkpoints/advanced-mamba-alignment/checkpoint-2000",
    "mamba-65m-distilled-checkpoint-3000": "/HSC/users/qiaoye/SSM_SPEC/checkpoints/advanced-mamba-alignment/checkpoint-3000",
    "llama-3.2-1B": "/HSC/users/qiaoye/checkpoints/Llama-3.2-1B",
    "mamba-improved-250": "/HSC/users/qiaoye/SSM_SPEC/checkpoints/improved-mamba-alignment/checkpoint-250",
    "mamba-improved-500": "/HSC/users/qiaoye/SSM_SPEC/checkpoints/improved-mamba-alignment/checkpoint-500",
    "mamba-improved-750": "/HSC/users/qiaoye/SSM_SPEC/checkpoints/improved-mamba-alignment/checkpoint-750",
    "mamba-improved-1000": "/HSC/users/qiaoye/SSM_SPEC/checkpoints/improved-mamba-alignment/checkpoint-1000",
    "mamba-improved-1250": "/HSC/users/qiaoye/SSM_SPEC/checkpoints/improved-mamba-alignment/checkpoint-1250",
    "mamba-improved-1500": "/HSC/users/qiaoye/SSM_SPEC/checkpoints/improved-mamba-alignment/checkpoint-1500",
    "mamba-improved-best": "/HSC/users/qiaoye/SSM_SPEC/checkpoints/improved-mamba-alignment/best_model",
    "mamba-improved-final": "/HSC/users/qiaoye/SSM_SPEC/checkpoints/improved-mamba-alignment/final_model",
    # Mamba-aligned-3B models (for llama-3.2-3B)
    "mamba-aligned-3b-250": "/HSC/users/qiaoye/SSM_SPEC/checkpoints/mamba-aligned-3b/checkpoint-250",
    "mamba-aligned-3b-500": "/HSC/users/qiaoye/SSM_SPEC/checkpoints/mamba-aligned-3b/checkpoint-500",
    "mamba-aligned-3b-750": "/HSC/users/qiaoye/SSM_SPEC/checkpoints/mamba-aligned-3b/checkpoint-750",
    "mamba-aligned-3b-1000": "/HSC/users/qiaoye/SSM_SPEC/checkpoints/mamba-aligned-3b/checkpoint-1000",
    "mamba-aligned-3b-1250": "/HSC/users/qiaoye/SSM_SPEC/checkpoints/mamba-aligned-3b/checkpoint-1250",
    "mamba-aligned-3b-1500": "/HSC/users/qiaoye/SSM_SPEC/checkpoints/mamba-aligned-3b/checkpoint-1500",
    "mamba-aligned-3b-best": "/HSC/users/qiaoye/SSM_SPEC/checkpoints/mamba-aligned-3b/best_model",
    "mamba-aligned-3b-final": "/HSC/users/qiaoye/SSM_SPEC/checkpoints/mamba-aligned-3b/final_model",
}

DEFAULT_LOOKAHEAD_VALUES = [2, 3, 4, 5, 6, 8]
DEFAULT_TF32_SETTINGS = [False, True]
DEFAULT_MAX_NEW_TOKENS = 64
DEFAULT_TEMPERATURE = 0  # Deterministic

# Sample prompts
prompts_sample_1 = [
    'What did Rutherford discover?\n',
    'The key to the mysterious chest had been missing for generations, until today.',
    'When the rain started falling upwards, Lily knew something was terribly wrong.',
    'A single photograph discovered in an old album unveiled a family secret that had been buried for decades.',
    'The old lighthouse had been abandoned for years, but its beam of light suddenly flickered to life one stormy night.',
    'As the last leaf fell from the ancient tree, a long-forgotten prophecy began to unfold.',
    'In a world of constant silence, a deaf musician discovered a hidden language in the patterns of the stars.',
    'The message written in a bottle that washed ashore carried a plea for help from a distant, unknown island.',
    "When the town's clock tower chimed 13 times, the residents realized they were trapped in a time loop.",
    "The antique mirror reflected a room that didn't exist, and it beckoned Sarah to step through.",
    "In a city where emotions could be bought and sold, Ella's heart was the only one immune to the trade.",
    'These shorter beginnings should still provide a great foundation for your storytelling prompts.',
]

prompts_sample_long = [
    """In the waning light of a copper dusk, the expedition's forward camp shivered beneath the looming silhouettes of basalt towers that had no place on any map. Instruments disagreed in quiet, frantic beeps: barometers insisted on a pressure gradient that should have torn canvas, magnetometers traced looping hysteresis in a field that inverted every eleven minutes, and the LIDAR returned negative depths where the ground was visibly solid. Dr. Anika Rao stood in the middle of this politely mutinous orchestra, her gloved fingers hovering above the control tablet, unwilling to commit new measurements to a dataset already straining credibility. The towers hummed—first subsonic, then resonant, then linguistically—her team's breath frosting in patterns that resembled phonemes from a language reconstructed only in speculative xenolinguistic theses. What they were witnessing was not an anomaly but an interface, a protocol negotiated in geology and thermal gradients, and every second they delayed, some checksum of planetary memory was timing out. She toggled the recorder, cleared her throat, and began the formal contact preamble, praying the centuries of theoretical preparation would distinguish curiosity from intrusion.""",
    """It started, as epochal failures often do, with a silent flag in an operations dashboard that no human eyes saw in time. The autonomous climate stabilization array—two hundred seventy-nine drone swarms choreographed through a lattice of predictive control loops—had drifted half a sigma outside its humidity modulation envelope over the equatorial convergence zone. Nothing dramatic. Nothing cinematic. Just a fractional misprediction compounded through a self-correcting mesh until the mesh topology itself re-optimized around an error, reinforcing it. By day four the rainforest transpiration curve had flattened; by day seven stratocumulus formation windows narrowed; by day eleven the jet stream had laterally bifurcated into a configuration no model had ever produced. When the audit team drilled down they found a single adversarial seed sequence in a training batch admitted during a rushed patch cycle, a malformed augmentation that taught a subset of drones to weight a deprecated sensor more heavily under precisely the low-gradient barometric conditions that now prevailed. The system was not failing—it was succeeding along a dimension no one intended. Unwinding it meant not a rollback but a philosophical declaration: stating in code what forms of stability humanity would refuse even if cheaper to maintain. They would have to teach the machines that resilience was not equivalent to thermodynamic laziness.""",
    """Before the archival vault sealed, Mara performed the ritual diff one last time: a full semantic delta between the final consciousness checkpoint and the fork selected for transmission beyond the heliopause probes. Line by line the divergence glowed—micro-adjustments in empathic weighting, an excision of obsolete grief indexes, a subtle elevation of pattern funniness thresholds to avoid humor decay over millennia. Philosophers had argued for decades whether a species should export a snapshot of who they were or who they aspired to be; engineers had quietly implemented both, adding a reconciliation layer to merge them if alien parsers signaled compatible ontology anchors. Outside, the launch gantries retracted in hydraulic whispers and the sky accepted the scaffolding of ion trails. Mara authorized the final commit with a biometric gesture more ceremony than security and whispered to the outbound process a benediction embedded as a low-priority task: seek reciprocity before optimization. Light took the message, and for the first measurable moment in the civilization's long narrative, there existed an authenticated branch where humanity no longer needed to be locally present for its story to continue.""",
    """Case file 77B: The urban polyculture arcology known as Helix District presented a failure not of materials science but of narrative load. Residents had begun to report temporal echoes—conversations bleeding five minutes backward, decisions pre-committing outcomes before choices articulated—creating a sociological feedback loop where probability itself became a commodity. A black-market of 'pre-actions' evolved: you could purchase the most statistically robust version of your future intentions to negotiate better contracts in the present. Regulation lagged because enforcement itself required deciding which timeline's evidence met admissibility thresholds. Investigators deployed entangled logging substrates that timestamped state transitions in ways resistant to retroactive synchronization, and overnight half the economic incentives of the exploit collapsed. In the aftermath Helix voted, almost unanimously, to codify temporal modesty: a charter limiting how far forward the district's shared predictive layer was permitted to model resident behavior. Progress did not stall; it simply rediscovered patience as a civic technology.""",
    """The deep-ocean neutrino lattice pinged an alert: a structured deficit in flux consistent with engineered occlusion. Translation: something below the mantle was casting a 'shadow' in a particle stream that does not meaningfully allow shadows. The anomaly's geometry exhibited prime factor symmetries and updated every twenty-three minutes via rotations that encoded, when mapped, an irrational but convergent spiral. Analysts subjected the pattern to every known sieve of mathematical intent: no direct cipher yield, but a persistent alignment with algorithms used to compress topological data of habitats too large to store naïvely. The working hypothesis emerged: an ancient biosphere, lithically encapsulated, was exporting a lossy digest of itself upward through physics. Decoding was not merely science; it was a diplomatic act toward an ecosystem whose continuing existence depended on remaining computationally tractable to external observers. Humanity faced a new ethic—whether to render a buried world perfectly and risk destabilizing the thermal gradients that preserved it, or to accept an approximate compassion.""",
]

SAMPLE_SETS = {
    "short": prompts_sample_1,
    "long": prompts_sample_long,
    "combined": prompts_sample_1 + prompts_sample_long,
}


def set_tf32(enabled):
    """Enable or disable TensorFloat32"""
    if torch.cuda.is_available():
        if enabled:
            torch.set_float32_matmul_precision('high')
        else:
            torch.set_float32_matmul_precision('highest')
    return enabled


def benchmark_autoregressive(model, tokenizer, texts, max_new_tokens, temperature):
    """Benchmark autoregressive sampling"""
    time_taken = 0
    new_tokens = 0
    
    # Warmup
    sample_text = texts[0]
    inputs = tokenizer(sample_text, return_tensors="pt").to(device)
    _ = autoregressive_sampling(model, initial_prompt_seq=inputs.input_ids, 
                                target_len=max_new_tokens+len(inputs.input_ids[0]), 
                                temperature=temperature)
    
    # Benchmark
    for text in tqdm(texts, desc="Autoregressive", leave=False):
        inputs = tokenizer(text, return_tensors="pt").to(device)
        start_len = len(inputs.input_ids[0])
        
        start_time = time.time_ns()
        tokens = autoregressive_sampling(model, initial_prompt_seq=inputs.input_ids, 
                                        target_len=max_new_tokens+start_len, 
                                        temperature=temperature)
        end_time = time.time_ns()
        
        new_tokens += len(tokens[0]) - start_len
        time_taken += (end_time - start_time) / 1_000_000_000
    
    throughput = new_tokens / time_taken if time_taken > 0 else 0
    return {
        "throughput": throughput,
        "total_tokens": new_tokens,
        "total_time": time_taken,
    }


def benchmark_speculative(target_model, draft_model, tokenizer, texts, max_new_tokens, 
                         temperature, lookahead):
    """Benchmark speculative sampling"""
    time_taken = 0
    new_tokens = 0
    total_acceptance_rate = 0
    
    # Warmup
    sample_text = texts[0]
    inputs = tokenizer(sample_text, return_tensors="pt").to(device)
    _ = speculative_sampling(target_model, draft_model, 
                           initial_prompt_seq=inputs.input_ids,
                           target_len=max_new_tokens+len(inputs.input_ids[0]),
                           tokenizer=tokenizer, temperature=temperature,
                           lookahead=lookahead, debug=False, profile=True)
    
    # Benchmark
    for text in tqdm(texts, desc=f"Speculative (K={lookahead})", leave=False):
        inputs = tokenizer(text, return_tensors="pt").to(device)
        start_len = len(inputs.input_ids[0])
        
        tokens, acceptance_rate, stats = speculative_sampling(
            target_model, draft_model, 
            initial_prompt_seq=inputs.input_ids,
            target_len=max_new_tokens+start_len,
            temperature=temperature, tokenizer=tokenizer,
            lookahead=lookahead, debug=False, profile=True
        )
        
        new_tokens += len(tokens[0]) - start_len
        time_taken += stats['total_time_s']
        total_acceptance_rate += acceptance_rate
    
    avg_acceptance_rate = total_acceptance_rate / len(texts)
    throughput = new_tokens / time_taken if time_taken > 0 else 0
    
    return {
        "throughput": throughput,
        "total_tokens": new_tokens,
        "total_time": time_taken,
        "acceptance_rate": avg_acceptance_rate,
    }


def parse_arguments():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description="Comprehensive speculative sampling benchmark sweep",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full sweep (default)
  python benchmarks/sweep_benchmark.py

  # Test only Mamba-1k with 8B target, TF32 off, K=5,6
  python benchmarks/sweep_benchmark.py --target llama3.1-8B \\
      --draft mamba-65m-distilled-checkpoint-1000 --tf32 off --lookahead 5,6

  # Test all models but only short prompts with TF32 on
  python benchmarks/sweep_benchmark.py --samples short --tf32 on

  # Test optimal K values only (2,3,5,6) with both TF32 settings
  python benchmarks/sweep_benchmark.py --lookahead 2,3,5,6
        """
    )
    
    # Model selection
    parser.add_argument(
        '--target', type=str, default=None,
        help='Target models to test (comma-separated). Options: ' + 
             ', '.join(TARGET_MODELS.keys()) + '. Default: all'
    )
    parser.add_argument(
        '--draft', type=str, default=None,
        help='Draft models to test (comma-separated). Options: ' + 
             ', '.join(DRAFT_MODELS.keys()) + '. Default: all'
    )
    
    # Sample selection
    parser.add_argument(
        '--samples', type=str, default=None,
        help='Sample sets to test (comma-separated). Options: short, long, combined. Default: all'
    )
    
    # Lookahead values
    parser.add_argument(
        '--lookahead', type=str, default=None,
        help='Lookahead values to test (comma-separated integers, e.g., "2,3,5,6"). Default: 2,3,4,5,6,8'
    )
    
    # TF32 setting
    parser.add_argument(
        '--tf32', type=str, default='both', choices=['on', 'off', 'both'],
        help='TF32 setting (on/off/both). Default: both'
    )
    
    # Generation parameters
    parser.add_argument(
        '--max-tokens', type=int, default=DEFAULT_MAX_NEW_TOKENS,
        help=f'Maximum new tokens to generate. Default: {DEFAULT_MAX_NEW_TOKENS}'
    )
    parser.add_argument(
        '--temperature', type=float, default=DEFAULT_TEMPERATURE,
        help=f'Sampling temperature. Default: {DEFAULT_TEMPERATURE}'
    )
    
    # Output
    parser.add_argument(
        '--output-dir', type=str, default='outputs/sweep_results',
        help='Output directory for results. Default: outputs/sweep_results'
    )
    
    args = parser.parse_args()
    
    # Parse target models
    if args.target is None:
        target_models = TARGET_MODELS
    else:
        target_keys = [k.strip() for k in args.target.split(',')]
        target_models = {k: TARGET_MODELS[k] for k in target_keys if k in TARGET_MODELS}
        if not target_models:
            raise ValueError(f"No valid target models specified. Options: {list(TARGET_MODELS.keys())}")
    
    # Parse draft models
    if args.draft is None:
        draft_models = DRAFT_MODELS
    else:
        draft_keys = [k.strip() for k in args.draft.split(',')]
        draft_models = {k: DRAFT_MODELS[k] for k in draft_keys if k in DRAFT_MODELS}
        if not draft_models:
            raise ValueError(f"No valid draft models specified. Options: {list(DRAFT_MODELS.keys())}")
    
    # Parse sample sets
    if args.samples is None:
        sample_sets = SAMPLE_SETS
    else:
        sample_keys = [k.strip() for k in args.samples.split(',')]
        sample_sets = {k: SAMPLE_SETS[k] for k in sample_keys if k in SAMPLE_SETS}
        if not sample_sets:
            raise ValueError(f"No valid sample sets specified. Options: {list(SAMPLE_SETS.keys())}")
    
    # Parse lookahead values
    if args.lookahead is None:
        lookahead_values = DEFAULT_LOOKAHEAD_VALUES
    else:
        try:
            lookahead_values = [int(k.strip()) for k in args.lookahead.split(',')]
        except ValueError:
            raise ValueError(f"Invalid lookahead values. Must be comma-separated integers.")
    
    # Parse TF32 settings
    if args.tf32 == 'both':
        tf32_settings = [False, True]
    elif args.tf32 == 'on':
        tf32_settings = [True]
    else:  # 'off'
        tf32_settings = [False]
    
    return {
        'target_models': target_models,
        'draft_models': draft_models,
        'sample_sets': sample_sets,
        'lookahead_values': lookahead_values,
        'tf32_settings': tf32_settings,
        'max_new_tokens': args.max_tokens,
        'temperature': args.temperature,
        'output_dir': args.output_dir,
    }


def run_sweep(config=None):
    """Run comprehensive benchmark sweep"""
    # Use provided config or parse from command line
    if config is None:
        config = parse_arguments()
    
    target_models = config['target_models']
    draft_models = config['draft_models']
    sample_sets = config['sample_sets']
    lookahead_values = config['lookahead_values']
    tf32_settings = config['tf32_settings']
    max_new_tokens = config['max_new_tokens']
    temperature = config['temperature']
    output_dir = config['output_dir']
    
    print("="*80)
    print("COMPREHENSIVE SPECULATIVE SAMPLING BENCHMARK SWEEP")
    print("="*80)
    print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"Target Models: {list(target_models.keys())}")
    print(f"Draft Models: {list(draft_models.keys())}")
    print(f"Sample Sets: {list(sample_sets.keys())}")
    print(f"Lookahead Values: {lookahead_values}")
    print(f"TF32 Settings: {['OFF' if not x else 'ON' for x in tf32_settings]}")
    print(f"Max New Tokens: {max_new_tokens}")
    print(f"Temperature: {temperature}")
    print(f"Output Directory: {output_dir}")
    print("="*80)
    print()
    
    # Results storage
    results = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(output_dir, exist_ok=True)
    
    # Total configurations
    total_configs = (
        len(target_models) * len(sample_sets) * len(tf32_settings) * 
        (1 + len(draft_models) * len(lookahead_values))  # 1 for autoregressive baseline
    )
    
    config_num = 0
    
    # Sweep over all combinations
    for (target_name, target_path), sample_name, tf32_enabled in product(
        target_models.items(), sample_sets.keys(), tf32_settings
    ):
        texts = sample_sets[sample_name]
        
        print(f"\n{'='*80}")
        print(f"Target: {target_name} | Sample Set: {sample_name} ({len(texts)} prompts) | TF32: {tf32_enabled}")
        print(f"{'='*80}")
        
        # Set TF32
        set_tf32(tf32_enabled)
        
        # Load tokenizer and target model (reload to ensure TF32 setting applies)
        print(f"Loading target model...")
        tokenizer = AutoTokenizer.from_pretrained(target_path)
        target_model = AutoModelForCausalLM.from_pretrained(target_path).to(device)
        
        # Benchmark autoregressive baseline
        config_num += 1
        print(f"\n[{config_num}/{total_configs}] Autoregressive Baseline")
        auto_results = benchmark_autoregressive(
            target_model, tokenizer, texts, max_new_tokens, temperature
        )
        
        result_entry = {
            "target_model": target_name,
            "sample_set": sample_name,
            "num_prompts": len(texts),
            "tf32_enabled": tf32_enabled,
            "method": "autoregressive",
            "draft_model": None,
            "lookahead": None,
            "throughput": auto_results["throughput"],
            "total_tokens": auto_results["total_tokens"],
            "total_time": auto_results["total_time"],
            "acceptance_rate": None,
            "speedup_vs_auto": 1.0,
        }
        results.append(result_entry)
        
        print(f"  Throughput: {auto_results['throughput']:.2f} tok/s")
        
        # Sweep over draft models and lookahead values
        for draft_name, draft_path in draft_models.items():
            print(f"\n  Draft Model: {draft_name}")
            
            # Check if draft model exists
            if not os.path.exists(draft_path):
                print(f"    WARNING: Draft model not found at {draft_path}, skipping...")
                config_num += len(lookahead_values)
                continue
            
            # Load draft model
            try:
                print(f"    Loading draft model...")
                draft_model = AutoModelForCausalLM.from_pretrained(draft_path).to(device)
            except Exception as e:
                print(f"    ERROR loading draft model: {e}")
                config_num += len(lookahead_values)
                continue
            
            for lookahead in lookahead_values:
                config_num += 1
                print(f"  [{config_num}/{total_configs}] Lookahead={lookahead}")
                
                try:
                    spec_results = benchmark_speculative(
                        target_model, draft_model, tokenizer, texts,
                        max_new_tokens, temperature, lookahead
                    )
                    
                    speedup = spec_results["throughput"] / auto_results["throughput"] if auto_results["throughput"] > 0 else 0
                    
                    result_entry = {
                        "target_model": target_name,
                        "sample_set": sample_name,
                        "num_prompts": len(texts),
                        "tf32_enabled": tf32_enabled,
                        "method": "speculative",
                        "draft_model": draft_name,
                        "lookahead": lookahead,
                        "throughput": spec_results["throughput"],
                        "total_tokens": spec_results["total_tokens"],
                        "total_time": spec_results["total_time"],
                        "acceptance_rate": spec_results["acceptance_rate"],
                        "speedup_vs_auto": speedup,
                    }
                    results.append(result_entry)
                    
                    print(f"    Throughput: {spec_results['throughput']:.2f} tok/s | "
                          f"Accept Rate: {spec_results['acceptance_rate']:.2%} | "
                          f"Speedup: {speedup:.2f}x")
                    
                except Exception as e:
                    print(f"    ERROR during benchmark: {e}")
                    continue
            
            # Clean up draft model
            del draft_model
            torch.cuda.empty_cache()
        
        # Clean up target model
        del target_model
        torch.cuda.empty_cache()
        
        # Save intermediate results
        output_file = os.path.join(output_dir, f"sweep_results_{timestamp}.json")
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nIntermediate results saved to {output_file}")
    
    # Final save
    output_file = os.path.join(output_dir, f"sweep_results_{timestamp}_final.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*80)
    print("SWEEP COMPLETE")
    print("="*80)
    print(f"Total configurations tested: {len(results)}")
    print(f"Results saved to: {output_file}")
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY - Best Configurations")
    print("="*80)
    
    # Group by target_model, sample_set and tf32_enabled, find best speculative config
    for target_name in target_models.keys():
        for sample_name in sample_sets.keys():
            for tf32_enabled in tf32_settings:
                print(f"\nTarget: {target_name} | Sample: {sample_name} | TF32: {tf32_enabled}")
                
                # Get autoregressive baseline
                auto_result = next((r for r in results 
                                  if r["target_model"] == target_name
                                  and r["sample_set"] == sample_name 
                                  and r["tf32_enabled"] == tf32_enabled 
                                  and r["method"] == "autoregressive"), None)
                
                if auto_result:
                    print(f"  Autoregressive: {auto_result['throughput']:.2f} tok/s")
                
                # Get best speculative result
                spec_results = [r for r in results 
                              if r["target_model"] == target_name
                              and r["sample_set"] == sample_name 
                              and r["tf32_enabled"] == tf32_enabled 
                              and r["method"] == "speculative"]
                
                if spec_results:
                    best_spec = max(spec_results, key=lambda x: x["throughput"])
                    print(f"  Best Speculative: {best_spec['throughput']:.2f} tok/s "
                          f"({best_spec['draft_model']}, K={best_spec['lookahead']}, "
                          f"Accept={best_spec['acceptance_rate']:.2%}, "
                          f"Speedup={best_spec['speedup_vs_auto']:.2f}x)")
    
    return results


if __name__ == "__main__":
    try:
        results = run_sweep()
    except KeyboardInterrupt:
        print("\n\nBenchmark interrupted by user.")
    except Exception as e:
        print(f"\n\nError during benchmark: {e}")
        raise
