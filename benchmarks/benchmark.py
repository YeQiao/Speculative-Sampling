import sys
import time
import random
import warnings
import torch
from tqdm import tqdm
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Suppress NVML warnings if there are driver version mismatches
warnings.filterwarnings("ignore", message=".*NVML.*")
warnings.filterwarnings("ignore", message=".*Can't initialize NVML.*")
from transformers import AutoTokenizer, AutoModelForCausalLM
from core.autoregressive_sampling import autoregressive_sampling
from core.speculative_sampling import speculative_sampling, speculative_sampling_original

device = "cuda" if torch.cuda.is_available() else "cpu"

# Enable TensorFloat32 for better performance on Ampere+ GPUs (A100, H100, etc.)
enabled = 0
if torch.cuda.is_available():
    if enabled:
        torch.set_float32_matmul_precision('high')
        print("TF32 matmul precision is enabled for this benchmark. This can significantly improve performance on compatible GPUs.")
    else:
        print("TF32 matmul precision is disabled for this benchmark. Enable it for better performance on compatible GPUs.")
        torch.set_float32_matmul_precision('highest')

target_model = AutoModelForCausalLM.from_pretrained("/HSC/users/qiaoye/checkpoints/Llama3.1-8B-hf").to(device)

# target_model = AutoModelForCausalLM.from_pretrained("/HSC/users/qiaoye/checkpoints/Llama-3.1-70B").to(device)

# draft_model = AutoModelForCausalLM.from_pretrained("/HSC/users/qiaoye/SSM_SPEC/checkpoints/custom-mamba-65m-multi-gpu").to(device)
# draft_model = AutoModelForCausalLM.from_pretrained("/HSC/users/qiaoye/checkpoints/Llama-3.2-1B").to(device)
draft_model = AutoModelForCausalLM.from_pretrained("/HSC/users/qiaoye/SSM_SPEC/checkpoints/improved-mamba-alignment/checkpoint-750").to(device)
tokenizer = AutoTokenizer.from_pretrained("/HSC/users/qiaoye/checkpoints/Llama3.1-8B-hf")

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
  # Long-form prompts for heavy token benchmarks
prompts_sample_long = [
    """In the waning light of a copper dusk, the expedition's forward camp shivered beneath the looming silhouettes of basalt towers that had no place on any map. Instruments disagreed in quiet, frantic beeps: barometers insisted on a pressure gradient that should have torn canvas, magnetometers traced looping hysteresis in a field that inverted every eleven minutes, and the LIDAR returned negative depths where the ground was visibly solid. Dr. Anika Rao stood in the middle of this politely mutinous orchestra, her gloved fingers hovering above the control tablet, unwilling to commit new measurements to a dataset already straining credibility. The towers hummed—first subsonic, then resonant, then linguistically—her team's breath frosting in patterns that resembled phonemes from a language reconstructed only in speculative xenolinguistic theses. What they were witnessing was not an anomaly but an interface, a protocol negotiated in geology and thermal gradients, and every second they delayed, some checksum of planetary memory was timing out. She toggled the recorder, cleared her throat, and began the formal contact preamble, praying the centuries of theoretical preparation would distinguish curiosity from intrusion.""",
    """It started, as epochal failures often do, with a silent flag in an operations dashboard that no human eyes saw in time. The autonomous climate stabilization array—two hundred seventy-nine drone swarms choreographed through a lattice of predictive control loops—had drifted half a sigma outside its humidity modulation envelope over the equatorial convergence zone. Nothing dramatic. Nothing cinematic. Just a fractional misprediction compounded through a self-correcting mesh until the mesh topology itself re-optimized around an error, reinforcing it. By day four the rainforest transpiration curve had flattened; by day seven stratocumulus formation windows narrowed; by day eleven the jet stream had laterally bifurcated into a configuration no model had ever produced. When the audit team drilled down they found a single adversarial seed sequence in a training batch admitted during a rushed patch cycle, a malformed augmentation that taught a subset of drones to weight a deprecated sensor more heavily under precisely the low-gradient barometric conditions that now prevailed. The system was not failing—it was succeeding along a dimension no one intended. Unwinding it meant not a rollback but a philosophical declaration: stating in code what forms of stability humanity would refuse even if cheaper to maintain. They would have to teach the machines that resilience was not equivalent to thermodynamic laziness.""",
    """Before the archival vault sealed, Mara performed the ritual diff one last time: a full semantic delta between the final consciousness checkpoint and the fork selected for transmission beyond the heliopause probes. Line by line the divergence glowed—micro-adjustments in empathic weighting, an excision of obsolete grief indexes, a subtle elevation of pattern funniness thresholds to avoid humor decay over millennia. Philosophers had argued for decades whether a species should export a snapshot of who they were or who they aspired to be; engineers had quietly implemented both, adding a reconciliation layer to merge them if alien parsers signaled compatible ontology anchors. Outside, the launch gantries retracted in hydraulic whispers and the sky accepted the scaffolding of ion trails. Mara authorized the final commit with a biometric gesture more ceremony than security and whispered to the outbound process a benediction embedded as a low-priority task: seek reciprocity before optimization. Light took the message, and for the first measurable moment in the civilization's long narrative, there existed an authenticated branch where humanity no longer needed to be locally present for its story to continue.""",
    """Case file 77B: The urban polyculture arcology known as Helix District presented a failure not of materials science but of narrative load. Residents had begun to report temporal echoes—conversations bleeding five minutes backward, decisions pre-committing outcomes before choices articulated—creating a sociological feedback loop where probability itself became a commodity. A black-market of 'pre-actions' evolved: you could purchase the most statistically robust version of your future intentions to negotiate better contracts in the present. Regulation lagged because enforcement itself required deciding which timeline's evidence met admissibility thresholds. Investigators deployed entangled logging substrates that timestamped state transitions in ways resistant to retroactive synchronization, and overnight half the economic incentives of the exploit collapsed. In the aftermath Helix voted, almost unanimously, to codify temporal modesty: a charter limiting how far forward the district's shared predictive layer was permitted to model resident behavior. Progress did not stall; it simply rediscovered patience as a civic technology.""",
    """The deep-ocean neutrino lattice pinged an alert: a structured deficit in flux consistent with engineered occlusion. Translation: something below the mantle was casting a 'shadow' in a particle stream that does not meaningfully allow shadows. The anomaly's geometry exhibited prime factor symmetries and updated every twenty-three minutes via rotations that encoded, when mapped, an irrational but convergent spiral. Analysts subjected the pattern to every known sieve of mathematical intent: no direct cipher yield, but a persistent alignment with algorithms used to compress topological data of habitats too large to store naïvely. The working hypothesis emerged: an ancient biosphere, lithically encapsulated, was exporting a lossy digest of itself upward through physics. Decoding was not merely science; it was a diplomatic act toward an ecosystem whose continuing existence depended on remaining computationally tractable to external observers. Humanity faced a new ethic—whether to render a buried world perfectly and risk destabilizing the thermal gradients that preserved it, or to accept an approximate compassion.""",
]
prompts_sample_2 = [
    "Emily found a mysterious letter on her doorstep one sunny morning.",
    "On a rainy afternoon, Max stumbled upon an old treasure map in the attic.",
    "A friendly stray cat showed up at Lisa's doorstep, leading her to a hidden garden.",
    "Jake's new neighbor had a strange habit of disappearing into the woods every night.",
    "While cleaning out the garage, Mia discovered a box of her grandfather's old inventions.",
    "At the county fair, Tom won a goldfish that seemed to have an uncanny ability.",
    "Amelia woke up one day to find her bedroom ceiling covered in glowing stars.",
    "In a dusty antique shop, Sarah found a vintage camera with peculiar abilities.",
    "During a family camping trip, they stumbled upon an unusual rock formation.",
    "A peculiar antique shop opened in town, and its owner seemed to know everyone's deepest secrets."
  ]
texts = prompts_sample_long 
texts = prompts_sample_1 + prompts_sample_2 + prompts_sample_long

MAX_NEW_TOKENS = 64
TEMPERATURE = 0 # 0 for Deterministic
LOOKAHEAD = 8  # Increased from default 2 to compensate for TF32 speedup on target model

print("Target Model -", target_model.config._name_or_path)
print("Draft Model -", draft_model.config._name_or_path)
print("************\n")

inputs_sample = tokenizer(random.choice(texts), return_tensors="pt").to(device)
tokens = target_model.generate(**inputs_sample, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
print("HF's generate")
print("Count of new tokens:", len(tokens[0]) - len(inputs_sample.input_ids[0]))
print(tokenizer.decode(tokens[0]))
print("******")

tokens = autoregressive_sampling(target_model, initial_prompt_seq=inputs_sample.input_ids, target_len=MAX_NEW_TOKENS+len(inputs_sample.input_ids[0]), temperature=TEMPERATURE)
print("Naive Autoregressive with temperature")
print("Count of new tokens:", len(tokens[0]) - len(inputs_sample.input_ids[0]))
print(tokenizer.decode(tokens[0]))
print("******")

tokens, acceptance_rate = speculative_sampling(target_model, draft_model, initial_prompt_seq=inputs_sample.input_ids, target_len=MAX_NEW_TOKENS+len(inputs_sample.input_ids[0]), tokenizer=tokenizer, temperature=TEMPERATURE, lookahead=LOOKAHEAD, debug=False)
print(f"Speculative Sampling with temperature (lookahead={LOOKAHEAD})")
print("Count of new tokens:", len(tokens[0]) - len(inputs_sample.input_ids[0]))
print("Acceptance Rate:", f"{acceptance_rate:.2%}")
print(tokenizer.decode(tokens[0]))
print("******")
print()

print("Benchmarking naive Autoregressive Sampling...")
## Autoregressive
# Warmup
tokens = autoregressive_sampling(target_model, initial_prompt_seq=inputs_sample.input_ids, target_len=MAX_NEW_TOKENS+len(inputs_sample.input_ids[0]), temperature=TEMPERATURE)

time_taken = 0
new_tokens = 0
for i in tqdm(range(len(texts))):
  text = texts[i]
  inputs = tokenizer(text, return_tensors="pt").to(device)
  start_len = len(inputs.input_ids[0])

  start_time = time.time_ns()
  tokens = autoregressive_sampling(target_model, initial_prompt_seq=inputs.input_ids, target_len=MAX_NEW_TOKENS+start_len, temperature=TEMPERATURE)
  end_time = time.time_ns()

  new_tokens += len(tokens[0]) - start_len
  time_taken += (end_time - start_time) / 1_000_000_000

print(f"Latency (Autoregressive Sampling): {new_tokens/time_taken:.2f} tok/s")

## Speculative Sampling (baseline)
# Warmup
print("Benchmarking Speculative Sampling (baseline)...")
tokens, _, _ = speculative_sampling(target_model, draft_model, initial_prompt_seq=inputs_sample.input_ids, target_len=MAX_NEW_TOKENS+len(inputs_sample.input_ids[0]), tokenizer=tokenizer, temperature=TEMPERATURE, debug=False, profile=True)

time_taken = 0
new_tokens = 0
total_acceptance_rate = 0
for i in tqdm(range(len(texts))):
  text = texts[i]
  inputs = tokenizer(text, return_tensors="pt").to(device)
  start_len = len(inputs.input_ids[0])

  tokens, acceptance_rate, stats = speculative_sampling(target_model, draft_model, initial_prompt_seq=inputs.input_ids, target_len=MAX_NEW_TOKENS+start_len, temperature=TEMPERATURE, tokenizer=tokenizer, debug=False, profile=True)
  
  new_tokens += len(tokens[0]) - start_len
  time_taken += stats['total_time_s']
  total_acceptance_rate += acceptance_rate

avg_acceptance_rate = total_acceptance_rate / len(texts)
print(f"Latency (Speculative Sampling): {new_tokens/time_taken:.2f} tok/s")
print(f"Average Acceptance Rate: {avg_acceptance_rate:.2%}")
print(f"Speedup vs Autoregressive: {(new_tokens/time_taken)/(new_tokens/time_taken if time_taken > 0 else 1):.2f}x")

# ## Speculative Sampling (optimized with torch.compile)
# # NOTE: torch.compile has poor performance with speculative sampling due to:
# # 1. Dynamic shapes (variable sequence lengths)
# # 2. Branching logic (accept/reject decisions)
# # 3. Mamba model incompatibility
# # Disabled for now - eager mode performs better
# print("\nNote: torch.compile optimization disabled due to poor performance with dynamic sampling")
# print("Eager mode speculative sampling is faster for this workload")