python generate.py  --method speculative \
--prompt "Emily found a mysterious letter on her doorstep one sunny morning." \
--max_new_tokens 64 \
--target_model /HSC/users/qiaoye/checkpoints/Llama3.1-8B-hf \
--draft_model /HSC/users/qiaoye/SSM_SPEC/checkpoints/custom-mamba-65m-multi-gpu \
--temperature 0.5

python generate.py  --method autoregressive \
                    --prompt "Emily found a mysterious letter on her doorstep one sunny morning." \
                    --max_new_tokens 64 \
                    --target_model /HSC/users/qiaoye/checkpoints/Llama3.1-8B-hf \
                    --temperature 0.5

# sweep for improved mamba
python benchmarks/sweep_benchmark.py \
  --target llama3.1-8B,llama-3.2-3B \
  --draft mamba-improved-250,mamba-improved-500,mamba-improved-750,mamba-improved-1000,mamba-improved-1250,mamba-improved-1500,mamba-improved-best,mamba-improved-final \
  --samples short,long,combined \
  --lookahead 2,3,4,5,6,8 \
  --tf32 off \
  --output-dir outputs/improved_mamba_sweep