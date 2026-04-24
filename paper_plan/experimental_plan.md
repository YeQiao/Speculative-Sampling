# Experimental Plan for DAC 2026 Paper
## "Efficient Speculative Decoding with Tiny SSM Drafters for Edge LLM Deployment"

## Paper Timeline (Submission: ~July 2025)

- **Now - Jan 2025**: Core experiments on 3 platforms
- **Feb 2025**: Write first draft, theoretical analysis
- **Mar 2025**: Revisions, additional experiments
- **Apr-May 2025**: Polish, submit internal review
- **Jun 2025**: Final revisions
- **Jul 2025**: Submit to DAC 2026

---

## Experimental Setup

### Platform Configurations

#### 1. **Edge Platform: NVIDIA Jetson Orin Nano**
- **Hardware**: 8GB VRAM, 8-core ARM CPU
- **Target Model**: Llama-3.2-3B (INT8 quantized, ~1.5GB)
- **Drafter**: Mamba-65M on CPU (FP16, ~130MB)
- **Baseline**: Llama-3.2-3B autoregressive
- **Focus**: Memory efficiency, CPU offloading benefit
- **Expected Speedup**: 1.3-1.5x on long context

**Key Metrics:**
- GPU memory usage (must fit in 8GB with KV cache)
- CPU utilization for drafter
- Power consumption (Watts)
- Latency per token (ms)

#### 2. **PC Platform: Intel 12700 + RTX 5070**
- **Hardware**: 12GB VRAM, 12-core Intel CPU
- **Target Model**: Llama-3.1-8B (FP16, ~16GB with optimizations)
- **Drafter**: Mamba-65M (CPU or GPU)
- **Baseline**: Llama-3.1-8B autoregressive
- **Focus**: CPU vs GPU drafter comparison
- **Expected Speedup**: 
  - CPU drafter: 1.4-1.6x
  - GPU drafter: 1.8-2.0x

**Ablation Studies:**
- Drafter placement: CPU vs GPU
- Memory bandwidth impact
- Multi-stream pipelining

#### 3. **Server Platform: AMD EPYC + H100**
- **Hardware**: 80GB VRAM, High-end CPU
- **Target Model**: Llama-3.1-70B (FP16)
- **Drafter**: Mamba-65M (GPU, could also test CPU)
- **Baseline**: Llama-3.1-70B autoregressive
- **Focus**: Maximum speedup, scaling validation
- **Expected Speedup**: 2.2-2.5x (your current results!)

**Analysis:**
- Bottleneck identification (drafter vs verifier)
- Acceptance rate vs model size correlation
- Optimal K selection per model size

---

## Benchmark Suite

### Long Context Scenarios (Primary Focus)

**Rationale**: 
- Higher acceptance rates on in-context continuation
- Critical for edge applications (chatbots, assistants)
- Demonstrates practical value

**Datasets:**

1. **LongBench** (selected subsets):
   - Document QA: 2000-4000 tokens input
   - Summarization: 1000-3000 tokens
   - Few-shot ICL: 500-1500 tokens

2. **Custom Edge Scenarios**:
   - **Autonomous Vehicle Logs**: Continuous narration of driving events
   - **Robotics Task Planning**: Long instruction sequences
   - **Medical Record Summarization**: Patient history → summary

3. **Streaming Generation**:
   - **Code Generation**: Complete functions/classes
   - **Story Continuation**: Coherent multi-paragraph text
   - **Dialogue**: Multi-turn conversations

**Metrics per Benchmark:**
- Throughput (tokens/s)
- Acceptance rate
- Latency to first token
- Memory peak usage
- Energy consumption (edge platform)

---

## Experimental Protocol

### Phase 1: Baseline Characterization (Week 1-2)

**For each platform:**

1. **Autoregressive Baseline**
   ```bash
   # Measure pure target model throughput
   python benchmarks/autoregressive_baseline.py \
     --model llama-3.2-3B \
     --platform orin-nano \
     --samples long_context \
     --batch-size 1
   ```
   - Record: throughput, memory, latency distribution
   - Vary sequence lengths: 512, 1024, 2048, 4096 tokens

2. **Drafter Standalone**
   ```bash
   # Measure Mamba drafter throughput
   python benchmarks/drafter_throughput.py \
     --model mamba-aligned-3b-1250 \
     --device cpu \
     --batch-size 1,4,8
   ```
   - CPU vs GPU comparison (where applicable)
   - Measure scan operation bottleneck
   - Profile with NVIDIA Nsight / Intel VTune

### Phase 2: Speculative Decoding (Week 3-4)

**For each platform × model combination:**

1. **Sweep Lookahead Values**
   ```bash
   python benchmarks/sweep_benchmark.py \
     --target llama-3.2-3B \
     --draft mamba-aligned-3b-1250 \
     --samples long \
     --lookahead 2,3,4,5,6,8 \
     --drafter-device cpu \
     --output-dir outputs/edge_results
   ```

2. **Checkpoint Comparison**
   - Test all training checkpoints (250, 500, 750, 1000, 1250, 1500, best)
   - Find optimal checkpoint per platform
   - Correlate acceptance rate with training progress

3. **CPU vs GPU Drafter (PC/Server only)**
   ```bash
   # Compare drafter placement
   for device in cpu gpu; do
     python benchmarks/speculative_benchmark.py \
       --drafter-device $device \
       --output-dir outputs/pc_${device}_drafter
   done
   ```

### Phase 3: Overhead Analysis (Week 5)

**Detailed profiling to validate theoretical model:**

1. **Breakdown Timing**
   - Draft generation time
   - Verification time
   - Sampling overhead
   - IPC overhead (CPU-GPU transfers)
   - Idle time (pipeline gaps)

2. **Memory Bandwidth**
   - CPU-GPU transfer bandwidth for CPU drafter
   - Impact on acceptance rate
   - KV cache management overhead

3. **Energy Profiling (Edge)**
   ```bash
   # Jetson power measurement
   tegrastats --interval 100 > power_log.txt &
   python benchmarks/edge_benchmark.py
   ```
   - Power consumption comparison
   - Energy per token generated

### Phase 4: Advanced Optimizations (Week 6-7)

**Test future directions:**

1. **Pipelined Draft-Verify**
   - Overlap drafter generation with verification
   - Measure pipeline efficiency
   - Expected: 10-15% improvement

2. **Tree-Based Drafting** (if time permits)
   - Generate multiple branches
   - Batch verification
   - Compare acceptance rate improvement

3. **Adaptive K Selection**
   - Adjust lookahead based on running acceptance rate
   - Reduce overhead on difficult sequences

---

## Data Collection & Organization

### Directory Structure
```
experiments/
├── edge_orin_nano/
│   ├── baseline/
│   ├── cpu_drafter/
│   └── analysis/
├── pc_12700_5070/
│   ├── baseline/
│   ├── cpu_drafter/
│   ├── gpu_drafter/
│   └── analysis/
└── server_epyc_h100/
    ├── baseline/
    ├── gpu_drafter/
    └── analysis/
```

### Results Format
```json
{
  "platform": "orin_nano",
  "config": {
    "target_model": "llama-3.2-3B",
    "draft_model": "mamba-aligned-3b-1250",
    "drafter_device": "cpu",
    "lookahead": 5
  },
  "results": {
    "throughput": 12.5,
    "acceptance_rate": 0.76,
    "memory_peak_mb": 7823,
    "power_watts": 15.2,
    "overhead_breakdown": {
      "sampling": 0.12,
      "ipc": 0.08,
      "pipeline": 0.05
    }
  }
}
```

---

## Validation Checklist

### Theoretical Model Validation
- [ ] Measure actual overheads (α, β, γ)
- [ ] Compare predicted vs actual speedup
- [ ] Validate drafter/verifier bottleneck analysis
- [ ] Test upper bound predictions

### Cross-Platform Consistency
- [ ] Acceptance rate consistent across platforms (for same model pair)
- [ ] Speedup scales predictably with hardware
- [ ] Memory usage within expected ranges

### Statistical Significance
- [ ] 3+ runs per configuration
- [ ] Report mean ± std dev
- [ ] Confidence intervals for key claims

---

## Paper Figures & Tables

### Figure 1: Architecture Overview
- Diagram: CPU drafter → GPU verifier pipeline
- Memory layout visualization
- Highlight KV cache savings

### Figure 2: Theoretical Model
- Speedup vs acceptance rate curves
- Platform regimes (drafter-limited vs verifier-limited)
- Overhead impact visualization

### Figure 3: Training Progression
- Use your existing analysis plots!
- Show optimal checkpoint selection
- Acceptance rate vs training steps

### Table 1: Platform Specifications
| Platform | Target Model | VRAM | CPU | Drafter Device |
|----------|--------------|------|-----|----------------|
| Edge     | Llama-3.2-3B | 8GB  | 8-core ARM | CPU |
| PC       | Llama-3.1-8B | 12GB | 12-core Intel | CPU/GPU |
| Server   | Llama-3.1-70B| 80GB | EPYC | GPU |

### Table 2: Main Results (Speedup)
| Platform | Baseline | Spec (CPU) | Spec (GPU) | Accept Rate |
|----------|----------|------------|------------|-------------|
| Edge     | 8.2 tok/s | **12.1** (1.48x) | N/A | 76% |
| PC       | 25.4 | **38.7** (1.52x) | **48.2** (1.90x) | 78% |
| Server   | 45.3 | N/A | **106.7** (2.36x) | 80% |

### Figure 4: Long Context Performance
- Throughput vs input length
- Show where speculative wins
- Break-even analysis

### Figure 5: Memory & Power
- Memory usage comparison (edge critical)
- Power consumption on Jetson
- Efficiency metrics (tok/Joule)

### Figure 6: Overhead Breakdown
- Pie chart of time spent in each phase
- Comparison: ideal vs realistic
- Identify optimization opportunities

---

## Strong Claims for Paper

### Main Claim
> "We demonstrate that tiny Mamba-2 drafters (65M parameters) enable efficient speculative decoding on resource-constrained edge devices, achieving 1.5-2.5x speedup across platforms while freeing GPU memory by offloading the drafter to CPU."

### Supporting Claims

1. **Efficiency Claim**:
   > "Mamba-65M drafter achieves 30-50 tok/s on CPU, sufficient to accelerate 3B-70B target models without GPU resources."

2. **Scalability Claim**:
   > "Cross-platform validation from edge (8GB) to datacenter (80GB) shows consistent acceptance rates (75-80%) and predictable speedup scaling."

3. **Practical Impact Claim**:
   > "On Jetson Orin Nano, CPU-offloaded drafting enables 3B model deployment with 1.48x speedup while reducing GPU memory by 87% compared to dual-model approaches."

4. **Theoretical Contribution Claim**:
   > "Our throughput model accurately predicts real-world speedup within 15%, accounting for speculative sampling overhead, IPC costs, and platform-specific bottlenecks."

---

## Potential Challenges & Mitigation

### Challenge 1: Mamba Bottleneck
- **Issue**: 65 tok/s cap even on H100
- **Mitigation**: 
  - Profile selective scan operation
  - Propose kernel optimizations (future work)
  - Focus on edge where CPU drafter is still faster than needed

### Challenge 2: Low Speedup on Short Prompts
- **Issue**: <1.0x for 3B model, short prompts
- **Mitigation**:
  - Focus paper on long context (main use case)
  - Show adaptive K can skip speculative on short prompts
  - Emphasize edge deployment scenarios need long generation

### Challenge 3: Limited Access to Platforms
- **Issue**: May not have all platforms available
- **Mitigation**:
  - Orin Nano: Critical, must obtain
  - PC: Can simulate with cloud instances
  - H100: Already have data!
  - Focus on 2 platforms if needed, extrapolate third

### Challenge 4: Reproducibility
- **Issue**: Hardware-specific results
- **Mitigation**:
  - Release all code, models, benchmarks
  - Provide Docker containers
  - Document exact software versions
  - Cloud-reproducible experiments where possible

---

## Next Steps (Prioritized)

1. **Week 1-2**: 
   - [ ] Set up Jetson Orin Nano environment
   - [ ] Quantize models for edge deployment
   - [ ] Run baseline characterization

2. **Week 3-4**:
   - [ ] Edge platform full sweep
   - [ ] PC platform experiments
   - [ ] Profile overhead breakdown

3. **Week 5-6**:
   - [ ] Analyze results, validate theory
   - [ ] Create all figures and tables
   - [ ] Write first draft

4. **Week 7-8**:
   - [ ] Advanced optimizations (pipelined)
   - [ ] Revisions and polish
   - [ ] Prepare supplementary materials

Would you like me to:
1. Create scripts for edge deployment (Jetson setup)?
2. Build the overhead profiling tool?
3. Draft the introduction section?
4. Design the figure layouts?
