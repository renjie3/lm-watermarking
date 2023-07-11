# CUDA_VISIBLE_DEVICES=4 python demo_watermark.py --model_name_or_path facebook/opt-2.7b
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/egr/research-dselab/renjie3/miniconda3/lib

CUDA_VISIBLE_DEVICES=4 TRANSFORMERS_CACHE="/egr/research-dselab/renjie3/renjie/LLM/cache" python run_watermarking.py --model_name facebook/opt-1.3b \
    --dataset_name c4 \
    --dataset_config_name realnewslike \
    --max_new_tokens 200 \
    --min_prompt_tokens 50 \
    --limit_indices 500 \
    --input_truncation_strategy completion_length \
    --input_filtering_strategy prompt_and_completion_length \
    --output_filtering_strategy max_new_tokens \
    --dynamic_seed markov_1 \
    --bl_proportion 0.5 \
    --bl_logit_bias 2.0 \
    --bl_type soft \
    --store_spike_ents True \
    --num_beams 1 \
    --use_sampling True \
    --sampling_temp 0.7 \
    --oracle_model_name facebook/opt-2.7b \
    --run_name example_run \
    --output_dir ./all_runs \
    --no_wandb True