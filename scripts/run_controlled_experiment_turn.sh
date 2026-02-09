#!/bin/bash

set -e 
set -x

unset  http_proxy  https_proxy  all_proxy

python -m "controlled_experiemnt.dialog_turn_run" \
    --bench_data_path "./data/bench_data/cases_counted_difficulty.json" \
    --save_dir "./controlled_experiemnt/results/dialog_turn" \
    --doctor_model_names "qwen2.5-7b-instruct" "gpt-4o-2024-11-20" \
    --doctor_model_url "qwen2.5-7b-base-url" "gpt-4o-base-url" \
    --doctor_model_key "qwen2.5-7b-api-key" "gpt-4o-api-key"\
    --doctor_slow_think_mode "Off" \
    --patient_model_name "gpt-4o-2024-11-20" \
    --patient_model_url "patient-model-gpt-4o-base-url" \
    --patient_model_key "patient-model-gpt-4o-api-key" \
    --patient_slow_think_mode "On" \
    --judge_model_name "gpt-4o-2024-11-20" \
    --judge_model_url "judge-model-gpt-4o-base-url" \
    --judge_model_key "judge-model-gpt-4o-api-key" \
    --num_patient "50" \
    --num_dialog_max_turn "3" \
    --num_thread "32" \
    --judge_record_prompt_mode "one-shot" \
    --run_dialog "True" \
    --run_eval "True"
 