# Script to run the generation, attack, and evaluation steps of the pipeline

# requires some OUTPUT_DIR to be set in the environment
# as well as a path to the hf format LLAMA model
# /egr/research-dselab/renjie3/renjie/LLM/cache

OUTPUT_DIR="/egr/research-dselab/renjie3/renjie/LLM/watermark_LLM/lm-watermarking/watermark_reliability_release/results"

RUN_NAME=opt_2.7b_semantic_json_10_use_sample
LLAMA_PATH=No

GENERATION_OUTPUT_DIR="$OUTPUT_DIR"/"$RUN_NAME"

echo "Running generation pipeline with output dir: $GENERATION_OUTPUT_DIR"

# generation_pipeline.py semantic_train_pipeline

# CUDA_VISIBLE_DEVICES=4 HF_HOME="/egr/research-dselab/renjie3/renjie/LLM/cache" python generation_pipeline.py \
#     --model_name facebook/opt-2.7b \
#     --dataset_name=json_c4 \
#     --dataset_config_name=realnewslike \
#     --max_new_tokens=200 \
#     --min_prompt_tokens=50 \
#     --min_generations=10 \
#     --input_truncation_strategy=completion_length \
#     --input_filtering_strategy=prompt_and_completion_length \
#     --output_filtering_strategy=max_new_tokens \
#     --seeding_scheme=sem \
#     --gamma=0.50 \
#     --delta=2.0 \
#     --run_name="$RUN_NAME"_gen \
#     --wandb=False \
#     --verbose=True \
#     --generation_batch_size=4 \
#     --stream_dataset=True \
#     --load_fp16=False \
#     --num_beams=1 \
#     --use_sampling=True \
#     --cl_mlp_model_path /egr/research-dselab/renjie3/renjie/LLM/watermark_LLM/lm-watermarking/watermark_reliability_release/results/cl_model/cl_model_01_2_2560.pt \
#     --output_dir=$GENERATION_OUTPUT_DIR 

# CUDA_VISIBLE_DEVICES=4 HF_HOME="/egr/research-dselab/renjie3/renjie/LLM/cache" OPENAI_API_KEY=`cat ../openai_key.txt` python attack_pipeline.py \
#     --attack_method=scramble \
#     --run_name="$RUN_NAME"_scramble_attack \
#     --wandb=False \
#     --input_dir=$GENERATION_OUTPUT_DIR \
#     --verbose=True \
#     --overwrite_output_file=True

CUDA_VISIBLE_DEVICES=7 HF_HOME="/egr/research-dselab/renjie3/renjie/LLM/cache"  python evaluation_pipeline.py \
    --evaluation_metrics=z_score \
    --run_name="$RUN_NAME"_eval \
    --wandb=False \
    --input_dir=$GENERATION_OUTPUT_DIR \
    --output_dir="$GENERATION_OUTPUT_DIR"_eval \
    --roc_test_stat=all \
    --cl_mlp_model_path /egr/research-dselab/renjie3/renjie/LLM/watermark_LLM/lm-watermarking/watermark_reliability_release/results/cl_model/cl_model_01_2_2560.pt \
    --overwrite_output_file=True \
    --return_green_token_mask=False \
    --compute_scores_at_T=False
    # --seeding_scheme=sem