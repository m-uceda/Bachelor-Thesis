import os
import json
import requests
import re
import random
import gc
import wandb
import signal
import traceback
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Any

# Environment variables
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["VLLM_USE_V1"] = '0'

import torch
import torch.distributed as dist
from unsloth import FastLanguageModel, is_bfloat16_supported
from vllm import SamplingParams
from pyswip import Prolog, Atom
from huggingface_hub import login
from trl import GRPOConfig, GRPOTrainer
from copy import deepcopy

# Authentication and setup
login(token="valid_token") # Replace with valid Hugging Face token
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

# Monkey patch to fix: _get_train_sampler(self, dataset)
def patched_get_train_sampler(self, dataset):
    return None  # Default behavior

# ========================================================================================
# DDP UTILITIES
# ========================================================================================

def setup_ddp():
    dist.init_process_group(backend="nccl")

def cleanup_ddp():
    dist.destroy_process_group()

# ========================================================================================
# MODEL CONFIGURATIONS
# ========================================================================================

MODEL_CONFIGS = {
    "meta-llama/Llama-3.2-3B-Instruct": {
        "max_seq_length": 4024,
        "lora_rank": 64, 
        "load_in_4bit": False, # False for LoRA 16bit
        "fast_inference": True, # Enable vLLM fast inference
        "gpu_memory_utilization": 0.6
    },
    "meta-llama/meta-Llama-3.1-8B-Instruct": {
        "max_seq_length": 4024,
        "lora_rank": 32, 
        "load_in_4bit": True,
        "fast_inference": True,
        "gpu_memory_utilization": 0.6
    }
}

TRAINING_CONFIGS = {
    "meta-llama/Llama-3.2-3B-Instruct": {                                     
        "learning_rate": 5e-6, 
        "weight_decay": 0.1, 
        "warmup_ratio": 0.1,
        "lr_scheduler_type": "cosine", 
        "optim": "adamw_8bit", 
        "logging_steps": 1,
        "per_device_train_batch_size": 4, 
        "gradient_accumulation_steps": 4,
        "num_generations": 4, 
        "max_steps": 500, 
        "save_steps": 250, 
        "max_grad_norm": 1.0,
        "report_to": "wandb",
        "output_dir": "outputs",
    },
    "meta-llama/meta-Llama-3.1-8B-Instruct": {
        "learning_rate": 5e-6, 
        "adam_beta1": 0.9, 
        "adam_beta2": 0.99,
        "weight_decay": 0.1, 
        "warmup_ratio": 0.1, 
        "lr_scheduler_type": "cosine",
        "optim": "paged_adamw_8bit", 
        "logging_steps": 1, 
        "per_device_train_batch_size": 4,
        "gradient_accumulation_steps": 1, 
        "num_generations": 4, 
        "max_steps": 250,
        "save_steps": 250, 
        "max_grad_norm": 0.1,
        "report_to": "wandb",
        "output_dir": "outputs",
    }
}

EXAMPLES_ERRORS = """
- The logic is not consistent with the problem statement. The objects in the Solution list must be arranged in a fixed order based on the clues provided. For example, if they are being arranged by age, the newest must be at the beginning of the list and the oldest at the end in a consistent way. Moreover, the names stated in the cues must be exactly the same as the ones used throughout the code.
- The Prolog code is not syntactically correct. For example, in Prolog, syntax such as 'nth1(Len-1, S, X)' is incorrect. Instead, it should be 'Pos is Len-1, nth1(Pos, S, X)' to correctly refer to the second from the right ('Len-2' third from the right, 'Len-3' fourth, etc.).
"""


# MODEL AND TOKENIZER UTILITIES

def get_model_tokenizer(model_name: str):
    config = MODEL_CONFIGS.get(model_name)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_name, 
        max_seq_length = config["max_seq_length"], 
        load_in_4bit = config["load_in_4bit"],
        fast_inference = config["fast_inference"],
        max_lora_rank = config["lora_rank"],
        gpu_memory_utilization = config["gpu_memory_utilization"],
        enforce_eager=True,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r = config["lora_rank"],
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj", 
            "gate_proj", "up_proj", "down_proj"
        ], # Remove QKVO if out of memory
        lora_alpha = config["lora_rank"], 
        use_gradient_checkpointing = "unsloth", 
        random_state=3407,
    )


    return model, tokenizer

def cleanup_model(model=None, tokenizer=None):
    if model: del model
    if tokenizer: del tokenizer   
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats()

# ========================================================================================
# DATASET UTILITIES
# ========================================================================================

def download_dataset(dataset_url: str, dataset_file: str) -> str:
    if not os.path.exists(dataset_file):
        r = requests.get(dataset_url)
        r.raise_for_status()
        with open(dataset_file, "w") as f:
            f.write(r.text)
        return f"Dataset downloaded and saved as '{dataset_file}'."
    return f"Dataset file '{dataset_file}' already exists."

def get_data(file_name: str, prompt: list[dict], dataset_file: str):
    if os.path.exists(file_name):
        with open(file_name, "r") as f:
            return json.load(f)
    else:
        return load_dataset_questions(dataset_file, prompt, save_to=file_name)

def load_dataset_questions(dataset_file: str, prompt: list, save_to: str = None) -> list:
    with open(dataset_file, "r") as f:
        data = json.load(f)

    processed_data = []
    for x in data:
        context, options = x.get("context", ""), x.get("options", [])
        parts = ["Q: " + context] if context else []
        if options:
            parts.append("Options:\n" + "\n".join(options))

        sample_prompt = "\n".join(parts)
        messages = prompt.copy()
        sample_prompt_formated = {"role": "user", "content": sample_prompt}
        messages.append(sample_prompt_formated)

        processed_data.append({
            'context': context,
            'prompt': messages,
            'sample_prompt': sample_prompt_formated,
            'answer': x.get("answer"),
            'n_options': len(options),
            'id': x.get("id")
        })

    if save_to:
        with open(save_to, "w") as f:
            json.dump(processed_data, f, indent=2)

    return processed_data

def distribute_dataset(train_json_dataset: list, 
    dev_json_dataset: list, 
    train_ratio: float = 0.8, 
    random_seed: int = 42,
    problem_type_ratios: dict = None,
    get_subset: bool = False
    ) -> tuple:

    random.seed(random_seed)
    all_samples = train_json_dataset + dev_json_dataset

    # Group by problem type
    samples_by_type = defaultdict(list)
    for sample in all_samples:
        samples_by_type[sample["n_options"]].append(sample)

    if get_subset:
        samples_by_type = {k: v[:170] for k, v in samples_by_type.items() if k in [3, 5, 7]}

    # Shuffle and split
    train_dataset, test_dataset = [], []
    for problem_type, samples in samples_by_type.items():
        random.shuffle(samples)
        ratio = problem_type_ratios.get(problem_type, train_ratio) if problem_type_ratios else train_ratio
        split_idx = int(len(samples) * ratio)
        train_dataset.extend(samples[:split_idx])
        test_dataset.extend(samples[split_idx:])

    return train_dataset, test_dataset

def print_distribution_stats(train_dataset: list, test_dataset: list):
    train_counts = defaultdict(int)
    test_counts = defaultdict(int)

    for sample in train_dataset:
        train_counts[sample["n_options"]] += 1
    for sample in test_dataset:
        test_counts[sample["n_options"]] += 1

    print("Dataset distribution:")
    print("-" * 40)
    print(f"{'Problem Type':<15}{'Train':<10}{'Test':<10}{'Total':<10}{'Train %':<10}")
    print("-" * 40)

    for problem_type in sorted(set(train_counts.keys()) | set(test_counts.keys())):
        train_count, test_count = train_counts[problem_type], test_counts[problem_type]
        total = train_count + test_count
        train_percentage = train_count / total * 100 if total > 0 else 0
        print(f"{f'{problem_type}-option':<15}{train_count:<10}{test_count:<10}{total:<10}{train_percentage:.1f}%")

    train_total, test_total = sum(train_counts.values()), sum(test_counts.values())
    overall_total = train_total + test_total
    overall_train_percentage = train_total / overall_total * 100 if overall_total > 0 else 0
    print("-" * 40)
    print(f"{'Overall':<15}{train_total:<10}{test_total:<10}{overall_total:<10}{overall_train_percentage:.1f}%")

# OTHER UTILS

def safe_getattr(obj, attr):
    try:
        val = getattr(obj, attr)
        if callable(val):
            return "[callable omitted]"
        return val
    except Exception as e:
        return f"[error: {e}]"

# MAIN EXPERIMENT RUNNER

def get_results(model, 
    tokenizer, 
    model_name: str, 
    train_datasets: dict, 
    test_datasets: dict, 
    methods: list, 
    base_methods: dict, 
    experiment_name: str, 
    backup: dict = None,
    rewards: dict = None, 
    grpo: bool = False
    ) -> list:

    # Create directories
    os.makedirs("outputs_" + experiment_name, exist_ok=True)
    lora_dir = "lora_adapters_" + experiment_name
    os.makedirs(lora_dir, exist_ok=True)

    results = []
    for i, method in enumerate(methods):
        # Set temperature based on method
        temperature_map = {
            "prolog_temp_0.6": 0.6,
            "prolog_temp_1.0": 1.0,
            "prolog_temp_1.3": 1.3
        }
        temperature = temperature_map.get(method, 0.0)

        sampling_params = SamplingParams(temperature=temperature, top_p=0.95, max_tokens=4024)

        if i == 0:
            retrain = True
        else:
            retrain = False

        print(f"Processing method: {method}")

        if grpo:
            train_dataset, test_dataset = train_datasets[method], test_datasets[method]
            rewards_method = rewards[method]
        else:
            test_dataset = test_datasets[method]

        base_method = base_methods[method]
        lora_path = os.path.join(lora_dir, f"grpo_saved_lora")

        # Fine-tuning if needed
        if retrain and grpo:
            print(f"Fine-tuning model for method: {method}")
            max_prompt_length = 1024
            max_seq_length = 4024

            config = TRAINING_CONFIGS[model_name].copy()
            config.pop("unsloth_num_chunks", None)
            config.pop("n_chunks", None)

            training_args = GRPOConfig(
                max_prompt_length=max_prompt_length,
                max_completion_length=max_seq_length - max_prompt_length,
                **config
            )

            trainer = GRPOTrainer(
                model=model, 
                processing_class=tokenizer, 
                reward_funcs=rewards_method,
                args=training_args, 
                train_dataset=train_dataset
            )

            trainer._get_train_sampler = patched_get_train_sampler.__get__(trainer, type(trainer))

            print(f"\n\n--------------------------- Training args: {training_args}\n\n")
            trainer.train()

            print(f"Saving LoRA adapter to {lora_path}")
            model.save_pretrained(lora_path)

        elif not retrain and grpo:
            if os.path.exists(lora_path):
                print(f"Loading previously trained model from {lora_path}")
                model = model.from_pretrained(lora_path)
            else:
                raise FileNotFoundError(f"LoRA adapter not found at {lora_path}")

        # Generate results
        method_results = get_method_results(test_dataset, method, base_method, model, 
                                          tokenizer, sampling_params, grpo, lora_path, backup)

        # Merge results
        if not results:
            results = method_results
        else:
            for j, result in enumerate(results):
                if base_method == "prolog":
                    if method == "prolog_retry":
                        result.update({
                            f"n_retries": method_results[j].get(f"n_retries"),
                            f"intermediate_generations": method_results[j].get(f"intermediate_generations"),
                            f"fixed": method_results[j].get(f"fixed"),
                            f"temperature": method_results[j].get(f"temperature")
                        })
                    elif method == "prolog_backup":
                        result.update({
                            f"used_backup": method_results[j].get(f"used_backup"),
                            f"intermediate_generations": method_results[j].get(f"intermediate_generations")
                        })
                    elif method == "prolog_fix":
                        result.update({
                            f"intermediate_generations": method_results[j].get(f"intermediate_generations"),
                            f"fixed": method_results[j].get(f"fixed"),
                        })
                    result.update({
                        f"{method}_output": method_results[j].get(f"{method}_output", ""),
                        f"{method}_code": method_results[j].get(f"{method}_code", ""),
                        f"{method}_answer": method_results[j].get(f"{method}_answer", "")
                    })
                else:
                    result.update({
                        f"{method}_output": method_results[j][f"{method}_output"],
                        f"{method}_answer": method_results[j][f"{method}_answer"]
                    })

        # Add random answers
        for j, result in enumerate(results):
            result["random_answer"] = chr(97 + random.randint(0, test_dataset[j]["n_options"]-1)).upper()

        cleanup_model(model, tokenizer)
        print(f"Completed method: {method}")


    write_results_to_file.config_info = {
        "model_name": model_name,
        "grpo": grpo,
        "sampling_params": {k: safe_getattr(sampling_params, k) for k in dir(sampling_params) if not k.startswith('_')},
        "training_config": vars(training_args) if 'training_args' in locals() else {},
        "model_config": model.config.to_dict() if hasattr(model, "config") else {},
        "tokenizer_config": tokenizer.init_kwargs,
    }
    write_results_to_file(results, methods, experiment_name)
    return results

# METHOD RESULT PROCESSING

def get_method_results(test_dataset: list, 
    method: str, 
    base_method: str, 
    model, 
    tokenizer, 
    sampling_params, 
    grpo: bool = False, 
    lora_path: str = None,
    backup: dict = None
    ) -> list:

    method_results = []

    for j, sample in enumerate(test_dataset):
        print(f"Processing sample {sample['id']} (iteration {j}) for method {method}")

        try:
            # Generate based on method
            if base_method == "cot":
                output, result = generate_cot(sample, model, tokenizer, sampling_params, grpo, lora_path)
            elif base_method == "prolog":
                try:
                    if method == "prolog_retry":
                        output, code, result, n_retries, intermediates, fixed, temperature = generate_prolog_retry(sample, model, tokenizer, sampling_params)
                        print(f"Prolog retry output: {output}, code: {code}, result: {result}")
                    elif method == "prolog_fix":
                        output, code, result, intermediates, fixed = generate_prolog_fix(sample, model, tokenizer, sampling_params)
                        print(f"Prolog fix output: {output}, code: {code}, result: {result}")
                    elif method == "prolog_backup":
                        backup_sample = backup[j]
                        output, code, result, intermediates, used_backup = generate_prolog_backup(sample, model, tokenizer, sampling_params, backup_sample)
                        print(f"Prolog backup output: {output}, code: {code}, result: {result}")
                    else:
                        output, code, result = generate_prolog(sample, model, tokenizer, sampling_params, grpo, lora_path)
                        print(f"Prolog output: {result}")
                except Exception as e:
                    print(f"generate_prolog failed for sample {sample['id']}: {e}")
                    traceback.print_exc()
                    raise 

            elif base_method == "direct":    
                output, result = generate_direct_answer(sample, model, tokenizer, sampling_params)                    


            # Build result dict
            result_dict = {
                "question_index": sample["id"], "question": sample["sample_prompt"]["content"],
                "expected_answer": sample["answer"], f"{method}_output": output,
                f"{method}_answer": result, "options": sample["n_options"],
                "prompt": sample.get("prompt"), "tokenized_prompt": tokenizer.apply_chat_template(sample["prompt"], tokenize=False, add_generation_prompt=True)
            }            

            if base_method == "prolog":
                result_dict[f"{method}_code"] = code
                if method == "prolog_retry":
                    result_dict["intermediate_generations"] = intermediates
                    result_dict["fixed"] = fixed
                    result_dict["temperature"] = temperature
                    result_dict["n_retries"] = n_retries
                elif method == "prolog_backup":
                    result_dict["used_backup"] = used_backup
                    result_dict["intermediate_generations"] = intermediates
                elif method == "prolog_fix":
                    result_dict["intermediate_generations"] = intermediates
                    result_dict["fixed"] = fixed


            method_results.append(result_dict)
            print(f"Completed sample {sample['id']} (iteration {j}) for method {method}")

        except Exception as e:
            print(f"Error processing sample {sample['id']} (iteration {j}) for method {method}: {e}")
            traceback.print_exc()
            method_results.append({
                "question_index": j, "question": sample["sample_prompt"]["content"],
                "expected_answer": sample["answer"], f"{method}_output": f"Error: {str(e)}",
                f"{method}_answer": "Error processing sample", "options": sample["n_options"]
            })

    return method_results

def write_results_to_file(results: list, methods: list, experiment_name: str):
    if not os.path.exists("results_text_files"):
        os.makedirs("results_text_files")

    file_name = f"results_text_files/results_{experiment_name}.txt"
    with open(file_name, "w", encoding='utf-8') as f:
        f.write(f"TEST RESULTS {experiment_name.upper()}\n{'='*80}\n\n")

        for result in results:
            f.write(f"SAMPLE {result['question_index']}\n{'-'*80}\n")

            if 'prompt' in result:
                f.write(f"PROMPT MESSAGE STRUCTURE (role/content pairs):\n")
                for msg in result['prompt']:
                    f.write(f"[{msg['role']}] {msg['content']}\n")
                f.write("\n")

            # Tokenized text (if available)
            if 'tokenized_prompt' in result:
                f.write(f"TOKENIZED PROMPT:\n{result['tokenized_prompt']}\n\n")

            #f.write(f"Question (raw prompt text):\n{result['question']}\n\n")
            f.write(f"Expected Answer: {result['expected_answer']}\n")
            f.write(f"Number of Options: {result.get('options', '?')}\n\n")

            # Generation details per method
            for method in methods:
                f.write(f"--- METHOD: {method.upper()} ---\n")

                if f"{method}_output" in result:
                    f.write(f"Model Output:\n{result[f'{method}_output']}\n")
                    f.write(f"{method.upper()} Answer: {result[f'{method}_answer']}\n\n")

                if f"{method}_code" in result:
                    f.write(f"{method.upper()} Generated Code:\n{result[f'{method}_code']}\n")

            if f"intermediate_generations" in result:
                f.write("Intermediate Generations / Retries:\n")
                for idx, g in enumerate(result[f"intermediate_generations"]):
                    f.write(f"Retry {idx + 1}:\n{g}\n\n")

            if "n_retries" in result:
                f.write(f"Number of Retries: {result['n_retries']}\n")

            if "fixed" in result:
                f.write(f"Fixed: {result['fixed']}\n")

            if "used_backup" in result:
                f.write(f"Used backup: {result['used_backup']}\n")

            if "temperature" in result:
                f.write(f"Temperature: {result['temperature']}\n")

            f.write(f"Random Answer: {result.get('random_answer', '?')}\n")
            f.write(f"{'='*80}\n\n")

        # Add final block with config info
        f.write("\nEXPERIMENT CONFIGURATION\n")
        f.write(f"{'-'*80}\n")
        if hasattr(write_results_to_file, "config_info"):
            config = write_results_to_file.config_info
            for key, val in config.items():
                f.write(f"{key}: {val}\n")
        else:
            f.write("No configuration provided.\n")

    print(f"Results saved to {file_name}")

# GENERATION METHODS

def generate(model, tokenizer, prompt, sampling_params, use_lora : bool = False, lora_path : str = None):
    text = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
    if use_lora and lora_path and os.path.exists(lora_path):
        print(f"Using LoRA from {lora_path} for generation")
        return model.fast_generate(text, sampling_params=sampling_params, 
                                lora_request=model.load_lora(lora_path))[0].outputs[0].text
    else:
        print("Generating without LoRA")
        return model.fast_generate(text, sampling_params=sampling_params,
                                lora_request=None)[0].outputs[0].text

def generate_prolog(sample: dict, model, tokenizer, sampling_params, use_lora: bool = False, lora_path: str = None) -> tuple:
    os.makedirs("prolog_code", exist_ok=True)

    output = generate(model, tokenizer, sample["prompt"], sampling_params, use_lora, lora_path)

    prolog_code, warn = extract_prolog(output)
    if prolog_code:
        prolog_filename = "prolog_code/overwritten_code.pl"
        with open(prolog_filename, "w") as pl_file:
            pl_file.write(prolog_code)
        prolog_result = prolog_answer(code_file=prolog_filename)
    else:
        prolog_result = f"No valid Prolog code generated. {warn}"

    return output, prolog_code, prolog_result

def generate_prolog_backup(sample: dict, model, tokenizer, sampling_params, backup_sample) -> tuple:
    used_backup = False
    intermediates = []
    output, prolog_code, prolog_result = "", "", ""

    try:
        output, prolog_code, prolog_result = generate_prolog(sample, model, tokenizer, sampling_params)
        intermediates.append(str(output))

        if prolog_result in "ABCDEFG":
            print(f"RESULT {prolog_result} IS {'CORRECT' if prolog_result == sample['answer'] else 'INCORRECT'}")
            return output, prolog_code, prolog_result, intermediates, used_backup

        print(f"RESULT '{prolog_result}' IS NOT A LETTER")
        print(f"OUTPUT BEFORE BACKUP:\n{output}\n\nCODE BEFORE BACKUP:\n {prolog_code}")

    except Exception as e:
        print(f"An error occurred during execution: {str(e)}")
        import traceback
        traceback.print_exc()

    # Use cot as bakcup
    output, result = generate_cot(backup_sample, model, tokenizer, sampling_params)
    used_backup = True    
    print(f"New response:\n{output}\nNew result:\n{result}")

    if result in "ABCDEFG":
        print(f"NEW RESULT {result} IS VALID")
        print("and correct" if result == sample["answer"] else "but incorrect")
    else:
        print(f"NEW RESULT {result} IS NOT VALID")
    
    intermediates.append(str(output))

    return output, prolog_code, result, intermediates, used_backup

def generate_prolog_fix(sample: dict, model, tokenizer, sampling_params) -> tuple:
    intermediates = []
    fixed = 0

    try:
        output, prolog_code, prolog_result = generate_prolog(sample, model, tokenizer, sampling_params)

        if prolog_result in "ABCDEFG":
            print(f"RESULT {prolog_result} IS {'CORRECT' if prolog_result == sample['answer'] else 'INCORRECT'}")
            return output, prolog_code, prolog_result, intermediates, fixed

        print(f"RESULT '{prolog_result}' IS NOT A LETTER")
        print(f"OUTPUT BEFORE RETRY:\n{output}\n\nCODE BEFORE RETRY:\n {prolog_code}")


    except Exception as e:
        print(f"An error occurred during execution: {str(e)}")
        import traceback
        traceback.print_exc()

    # Retry loop
    new_output, new_result, new_code = fix_prolog(prolog_result, prolog_code, sample, model, tokenizer, sampling_params)    
    intermediates.append(str(new_output))

    if new_result in "ABCDEFG":
        print(f"NEW RESULT {new_result} IS VALID")
        if new_result == sample["answer"]:
            fixed = 2
            print("and correct")
        else:
            print("but incorrect")
            fixed = 1
    else:
        fixed = 1
        print(f"NEW RESULT {new_result} IS NOT VALID")
    
    
    return new_output, new_code, new_result, intermediates, fixed

def generate_prolog_retry(sample: dict, model, tokenizer, sampling_params) -> tuple:
    i = 0
    intermediates = []
    fixed = False

    local_sampling_params = deepcopy(sampling_params)
    temperature = 0.0
    local_sampling_params.temperature = temperature

    try:
        output, prolog_code, prolog_result = generate_prolog(sample, model, tokenizer, sampling_params)
        intermediates.append(str(output))

        if prolog_result in "ABCDEFG":
            print(f"RESULT {prolog_result} IS {'CORRECT' if (prolog_result == sample['answer']) else 'INCORRECT'}")
            return output, prolog_code, prolog_result, i, intermediates, fixed, temperature

        print(f"RESULT '{prolog_result}' IS NOT A LETTER")
        print(f"OUTPUT BEFORE RETRY:\n{output}\n\nCODE BEFORE RETRY:\n {prolog_code}")

    except Exception as e:
        print(f"An error occurred during execution: {str(e)}")
        import traceback
        traceback.print_exc()

    # Retry loop

    while i < 10:
        temperature += 0.1
        local_sampling_params.temperature = temperature
        print(f"\nRetrying '{i}' with temperature {temperature}...")
        new_output, new_code, new_result = generate_prolog(sample, model, tokenizer, local_sampling_params)

        print(f"REGENERATED {i} New response:\n{new_output}\nNew result:\n{new_result}")
        # inside retry loop:
        intermediates.append(str(new_output))

        if new_result in "ABCDEFG":
            print(f"NEW RESULT {new_result} IS VALID (at retry number {i})")
            if new_result == sample["answer"]:
                fixed = True
                print("and correct")
            else:
              print("but incorrect")

            return new_output, new_code, new_result, i+1, intermediates, fixed, temperature
        else:
            print(f"NEW RESULT {new_result} IS NOT VALID")
        i += 1

    return new_output, new_code, new_result, i, intermediates, fixed, temperature

def fix_prolog(error_txt: str, wrong_output: str, sample: dict, model, tokenizer, sampling_params) -> tuple:
    os.makedirs("fix_prolog_code", exist_ok=True)

    sample_prompt = sample["sample_prompt"]["content"]
    prompt_parts = [
        f'Logical deduction problem with a context and multiple choice options: """{sample_prompt}"""',
        f'Prolog program that contains some mistake(s):\n"{wrong_output}"\n',
        f"Solver's error message:\n\"{error_txt}\"\n",
        "Examples of the most common errors:\n", EXAMPLES_ERRORS,
        '\nGenerate a correct version of the Prolog code:\n'
    ]

    error_prompt = [
        {"role": "system", "content": "You are a Prolog program fixer. Given a logical deduction problem with a context and multiple choice options, there is a Prolog program that contains some mistake(s). Your job is to use the solver's error message as feedback and examples of the most common errors to generate a correct version of the code. Generate the solution as a valid Prolog program starting with 'A:\n' followed by `solve(Answer) :-` and having the program enclosed in triple backticks labeled `prolog`."},
        {"role": "user", "content": "\n".join(prompt_parts)}
    ]

    output = generate(model, tokenizer, error_prompt, sampling_params)
    prolog_code, warn = extract_prolog(output)

    if prolog_code:
        prolog_filename = f"fix_prolog_code/{sample['id']}_code.pl"
        with open(prolog_filename, "w") as pl_file:
            pl_file.write(prolog_code)
        print(f"Saved Prolog code to {prolog_filename}")
        prolog_result = prolog_answer(code_file=prolog_filename)
    else:
        prolog_result = f"No valid Prolog code generated. {warn}"

    return output, prolog_result, prolog_code

def generate_cot(sample: dict, model, tokenizer, sampling_params, use_lora: bool = False, lora_path: str = None) -> tuple:
    output = generate(model, tokenizer, sample["prompt"], sampling_params, use_lora, lora_path)
    answer = extract_letter_from_cot(output)
    return output, answer

def generate_direct_answer(sample: dict, model, tokenizer, sampling_params) -> tuple:
    string = sample['sample_prompt']['content']
    direct_prompt = [
        {"role": "system", "content": "You are a logical deduction assistant. Please provide the letter of the correct answer (e.g., A, B, C)."},
        {"role": "user", "content": f"{string}\n\n Type 'Answer: ' followed by the letter of the correct answer."}
    ]

    output = generate(model, tokenizer, direct_prompt, sampling_params)
    return output, extract_letter_from_cot(output)

# Text extraction utilities
def extract_letter_from_cot(cot_text: str) -> str:
    """Extract answer letter from cot response"""
    patterns = [
        r'(?:answer|the answer)[\s]*(?:is|:)[\s]*[\(\[]?([A-G])[\)\]]?',
        r'(?:So|Thus|Therefore|Hence)[\s]*(?:the answer is|the correct answer is)[\s]*[\(\[]?([A-G])[\)\]]?',
        r'(?:So|Thus|Therefore|Hence)[\s]*[\(\[]?([A-G])[\)\]]?',
        r'(?:Option|option|answer|Letter|letter)[\s]*[\(\[]?([A-G])[\)\]]?',
        r'[\(\[]?([A-G])[\)\]]?[\s]*is correct'
    ]

    for pattern in patterns:
        match = re.search(pattern, cot_text, re.IGNORECASE)
        if match:
            return match.group(1).upper()
    return "No answer extracted from output"

def extract_prolog(response_text: str) -> str:
    """Extract Prolog code from response."""
    warn = ""
    if "```prolog" in response_text:
        try:
            prolog_code = response_text.split("```prolog")[1].split("```")[0].strip()
        except IndexError:
            warn = "[WARN] Detected ```prolog but failed to extract code block."
            print(warn)
            return "", None
    else:
        warn = "[WARN] No ```prolog code block found in output."
        print(warn)
        return "", None

    # Find starting point (solve function)
    solve_match = re.search(r"solve\s*\(\s*[A-Za-z_]+\s*\)\s*:-", prolog_code)
    if not solve_match:
        warn = "[WARN] No 'solve(...) :-' clause found in Prolog code."
        print(warn)
    else:
        prolog_code = prolog_code[solve_match.start():]

    # Check choose_option count
    choose_option_count = len(re.findall(r"choose_option\s*\(", prolog_code))
    if choose_option_count < 4:
        print(f"[WARN] Only {choose_option_count} 'choose_option' calls found (expected ≥4).")

    return prolog_code, warn

# Prolog execution utilities
class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException()

def prolog_answer(code_file: str = None, code: str = None) -> str:
    """Run Prolog program and get answer"""
    prolog = Prolog()

    try:
        if not code_file:
            code_file = "prolog_code.pl"
            with open(code_file, "w") as pl_file:
                pl_file.write(code)

        prolog.consult(code_file)

        query = "solve(Answer)"
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(60)  # 5 second timeout (changed to 60)

        solutions = prolog.query(query, maxresult=1)
        result = next(solutions, None)
        signal.alarm(0)

        if result is None or "Answer" not in result:
            return "Error (invalid answer)"

        answer = result["Answer"]
        if isinstance(answer, str):
            return answer.upper()
        elif isinstance(answer, Atom):
            return str(answer).upper()
        else:
            return str(answer)

    except TimeoutException:
        return "Error (timeout)"
    except Exception as e:
        return f"Error: {str(e)}"
    finally:
        try:
            solutions.close()
        except:
            pass