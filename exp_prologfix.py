from utils_exp import (
    setup_ddp, cleanup_ddp,
    get_model_tokenizer, download_dataset, 
    get_data, distribute_dataset, print_distribution_stats,
    get_results
)
from evaluation import summarize_all_methods
from constants import (
    TRAIN_JSON_URL, TRAIN_JSON_FILE,
    DEV_JSON_URL, DEV_JSON_FILE,
    PROLOG_PROMPT, PROLOG_PROMPT_NO_FEW_SHOT, PROLOG_PROMPT_1, COT_PROMPT
)
from grpo import (
    correctness_reward_func_prolog, letter_reward_func_prolog, prolog_format_reward_func, prolog_error_reward_func, 
    ao_correctness_reward_func, ao_letter_reward_func, ao_format_reward_func,
    cot_correctness_reward_func, cot_letter_reward_func, cot_format_reward_func
)

import os

local_rank = int(os.environ.get("LOCAL_RANK", 0))
os.environ["CUDA_VISIBLE_DEVICES"] = str(local_rank)

# Prompting the base model (no fine-tuning) with default parameters using Prolog few-shot prompt.

# Experiment 1: Few-shot Vs Not

def main():
    setup_ddp()

    # Load the model and tokenizer
    #model_name = "meta-llama/Llama-3.2-3B-Instruct"
    #model_name = "meta-llama/meta-Llama-3.1-8B-Instruct"
    model_name = "Qwen/Qwen2.5-Coder-7B-Instruct"
    model, tokenizer = get_model_tokenizer(model_name)

    # Get and prepare dataset
    download_dataset(TRAIN_JSON_URL, TRAIN_JSON_FILE)
    download_dataset(DEV_JSON_URL, DEV_JSON_FILE)
    train_json_file = "prolog_train_data.json" # Change to name corresponding to the dataset you want to use
    test_json_file = "prolog_test_data.json"
    train_json_dataset = get_data(train_json_file, PROLOG_PROMPT, TRAIN_JSON_FILE)
    dev_json_dataset = get_data(test_json_file, PROLOG_PROMPT, DEV_JSON_FILE)

    # Do 80-20 split of train and test datasets with equal distribution of question types
    train_dataset, test_dataset = distribute_dataset(train_json_dataset, dev_json_dataset, get_subset=False)
    print_distribution_stats(train_dataset, test_dataset)

    # Run the evaluation for methods specified and print the results
    methods = ["prolog_fix"] # Options: "direct", "cot", "prolog", "prolog_fix", "prolog_retry", "prolog_backup"
    train_datasets = {"prolog_fix": train_dataset}	
    test_datasets = {"prolog_fix": test_dataset}
    base_methods = {"prolog_fix": "prolog"}
    answer_types = ["prolog_fix_answer"]
    experiment_name = "exp_prologfix"
    rewards = {"prolog": [correctness_reward_func_prolog, letter_reward_func_prolog, prolog_format_reward_func, prolog_error_reward_func]}
    results = get_results(model, tokenizer, model_name, train_datasets, test_datasets, methods, base_methods, experiment_name, backup=None, rewards=rewards, grpo=False)
    summarize_all_methods(results, answer_types)  

    cleanup_ddp()

    return

if __name__ == "__main__":
    main()