# Script to run the generation, attack, and evaluation steps of the pipeline

# requires some OUTPUT_DIR to be set in the environment
# as well as a path to the hf format LLAMA model
# /egr/research-dselab/renjie3/renjie/LLM/cache

OUTPUT_DIR="/egr/research-dselab/renjie3/renjie/LLM/watermark_LLM/lm-watermarking/watermark_reliability_release/results"

RUN_NAME=opt_1.3b_semantic_json
LLAMA_PATH=No

GENERATION_OUTPUT_DIR="$OUTPUT_DIR"/"$RUN_NAME"

echo "Running generation pipeline with output dir: $GENERATION_OUTPUT_DIR"

# generation_pipeline.py semantic_train_pipeline

CUDA_VISIBLE_DEVICES=3 HF_HOME="/egr/research-dselab/renjie3/renjie/LLM/cache" python generation_pipeline.py \
    --model_name=facebook/opt-1.3b \
    --dataset_name=json_c4 \
    --dataset_config_name=realnewslike \
    --max_new_tokens=200 \
    --min_prompt_tokens=50 \
    --min_generations=500 \
    --input_truncation_strategy=completion_length \
    --input_filtering_strategy=prompt_and_completion_length \
    --output_filtering_strategy=max_new_tokens \
    --seeding_scheme=selfhash \
    --gamma=0.25 \
    --delta=2.0 \
    --run_name="$RUN_NAME"_gen \
    --wandb=False \
    --verbose=True \
    --generation_batch_size=4 \
    --stream_dataset=True \
    --output_dir=$GENERATION_OUTPUT_DIR 

# CUDA_VISIBLE_DEVICES=3 HF_HOME="/egr/research-dselab/renjie3/renjie/LLM/cache" python semantic_train_pipeline.py \
#     --model_name=facebook/opt-1.3b \
#     --dataset_name=json_c4 \
#     --dataset_config_name=realnewslike \
#     --max_new_tokens=200 \
#     --min_prompt_tokens=50 \
#     --min_generations=500 \
#     --input_truncation_strategy=completion_length \
#     --input_filtering_strategy=prompt_and_completion_length \
#     --output_filtering_strategy=max_new_tokens \
#     --seeding_scheme=selfhash \
#     --gamma=0.25 \
#     --delta=2.0 \
#     --run_name="$RUN_NAME"_gen \
#     --wandb=False \
#     --cl_lr=1e-3 \
#     --cl_mlp_feat_dim=2 \
#     --verbose=True \
#     --generation_batch_size=256 \
#     --stream_dataset=True \
#     --output_dir=$GENERATION_OUTPUT_DIR 

# HF_HOME="/egr/research-dselab/renjie3/renjie/LLM/cache" OPENAI_API_KEY='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' python attack_pipeline.py \
#     --attack_method=gpt \
#     --run_name="$RUN_NAME"_gpt_attack \
#     --wandb=False \
#     --input_dir=$GENERATION_OUTPUT_DIR \
#     --verbose=True

# CUDA_VISIBLE_DEVICES=4 HF_HOME="/egr/research-dselab/renjie3/renjie/LLM/cache"  python evaluation_pipeline.py \
#     --evaluation_metrics=all \
#     --run_name="$RUN_NAME"_eval \
#     --wandb=False \
#     --input_dir=$GENERATION_OUTPUT_DIR \
#     --output_dir="$GENERATION_OUTPUT_DIR"_eval \
#     --roc_test_stat=all