from utils.io import read_jsonlines, read_json
import numpy as np
import math

# opt_2.7b_semantic_json_10_use_sample_eval 7b_json_10_eval
gen_table_path = "/egr/research-dselab/renjie3/renjie/LLM/watermark_LLM/lm-watermarking/watermark_reliability_release/results/opt_2.7b_json_10_scramble_eval/gen_table_w_metrics.jsonl"

gen_table_lst = [ex for ex in read_jsonlines(gen_table_path)]

# print(gen_table_lst)

# for item in gen_table_lst:
#     print(type(item))

ave_keys = [
    "w_wm_output_green_fraction", 
    "w_wm_output_z_score", 
    "w_wm_output_attacked_green_fraction", 
    "w_wm_output_attacked_z_score"
    ]

for key in ave_keys:
    v = []
    for item in gen_table_lst:
        if not math.isnan(item[key]):
            v.append(item[key])
        # print(item[key])
    print(key, np.mean(v))

# print(gen_table_lst[0].keys())
