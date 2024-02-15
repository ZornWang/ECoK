 CUDA_VISIBLE_DEVICES=0 python nlg_eval.py \
    --hypothesis_path="data/choose/gpt4_deberta_top1picks.json" \
    --references_path="data/raw/json/test.json" \
    --results_path="data/eval/gpt4_nlg_results.json"