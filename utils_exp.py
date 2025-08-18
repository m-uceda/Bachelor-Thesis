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
from pathlib import Path
from openai import OpenAI, RateLimitError 
import time

# Read API key from private text file
with open("key.txt", "r") as f:
    api_key = f.read().strip()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
)

from pyswip import Prolog, Atom
from trl import GRPOConfig, GRPOTrainer
from copy import deepcopy
import subprocess
import threading
import sys

# ========================================================================================
# MODEL CONFIGURATIONS
# ========================================================================================

EXAMPLES_ERRORS = """
- The logic is not consistent with the problem statement. The objects in the Solution list must be arranged in a fixed order based on the clues provided. For example, if they are being arranged by age, the newest must be at the beginning of the list and the oldest at the end in a consistent way. Moreover, the names stated in the cues must be exactly the same as the ones used throughout the code.
- The Prolog code is not syntactically correct. For example, in Prolog, syntax such as 'nth1(Len-1, S, X)' is incorrect. Instead, it should be 'Pos is Len-1, nth1(Pos, S, X)' to correctly refer to the second from the right ('Len-2' third from the right, 'Len-3' fourth, etc.).
"""


# MODEL AND TOKENIZER UTILITIES

def get_model_tokenizer(model_name: str):
    return None, None

# ========================================================================================
# DATASET UTILITIES
# ========================================================================================

def download_dataset(dataset_url: str, dataset_file: str) -> str:
    if not Path(dataset_file).exists():
        r = requests.get(dataset_url)
        r.raise_for_status()
        with open(dataset_file, "w", encoding="utf-8") as f:
            f.write(r.text)
        return f"Dataset downloaded and saved as '{dataset_file}'."
    return f"Dataset file '{dataset_file}' already exists."

def get_data(file_name: str, prompt: list[dict], dataset_file: str):
    if Path(file_name).exists():
        with open(file_name, "r", encoding="utf-8") as f:
            return json.load(f)
    else:
        return load_dataset_questions(dataset_file, prompt, save_to=file_name)

def load_dataset_questions(dataset_file: str, prompt: list, save_to: str = None) -> list:
    with open(dataset_file, "r", encoding="utf-8") as f:
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
        with open(save_to, "w", encoding="utf-8") as f:
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
        samples_by_type = {k: v[:1] for k, v in samples_by_type.items() if k in [3, 5, 7]}

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
    
class TimeoutException(Exception):
    pass

def prolog_answer(code_file: Path = None, code: str = None) -> str:
    """Run Prolog program and get answer with timeout limit (60s)."""
    final_file_path: Path

    if code_file:
        final_file_path = code_file
    elif code:
        prolog_code_dir = Path("prolog_code")
        prolog_code_dir.mkdir(parents=True, exist_ok=True)
        final_file_path = prolog_code_dir / "overwritten_code.pl"
        with open(final_file_path, "w", encoding="utf-8") as pl_file:
            pl_file.write(code)
    else:
        return "Error: No Prolog code or file provided."

    prolog = Prolog() 
    
    # Normalize path for Prolog consult command
    # Use forward slashes for Prolog paths
    path_for_prolog_query = str(final_file_path.resolve()).replace("\\", "/") 

    consult_solutions = None # Initialize consult_solutions

    try:
        consult_command = f"consult('{path_for_prolog_query}')"
        print(f"Attempting to consult with: {consult_command}") # Debugging line
        
        try:
            # Wrap consult query in try-finally to ensure it's closed
            consult_solutions = prolog.query(consult_command, catcherrors=True)
            list(consult_solutions) # Exhaust the generator
        finally:
            if consult_solutions is not None:
                consult_solutions.close() # Always close the consult query
                consult_solutions = None # Set to None after closing for safety

        query = "solve(Answer)"
        
        result_container = {}
        exception_container = {}

        def run_prolog_query_threaded():
            solve_solutions = None # Initialize solve_solutions for the finally block
            try:
                solve_solutions = prolog.query(query, maxresult=1)
                result_container['result'] = next(solve_solutions, None)
            except Exception as e:
                exception_container['exception'] = e
            finally:
                if solve_solutions is not None:
                    solve_solutions.close() # Always close the solve query
        
        # Start the query in a separate thread
        query_thread = threading.Thread(target=run_prolog_query_threaded)
        query_thread.start()
        
        # Wait for the thread to complete, with a timeout
        query_thread.join(timeout=60) # 60 seconds timeout

        if query_thread.is_alive():
            # If the thread is still alive, it means it timed out
            raise TimeoutException("Prolog query timed out.")
        
        if 'exception' in exception_container:
            raise exception_container['exception']

        result = result_container.get('result')

        if result is None or "Answer" not in result:
            return "Error (invalid answer)"

        answer = result["Answer"]
        if isinstance(answer, str):
            return answer.upper()
        elif isinstance(answer, Atom):
            return str(answer).upper()
        else:
            return str(answer)

    except TimeoutException as e:
        return f"Error (timeout): {e}"
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"Error: {type(e).__name__}: {str(e)}"
    finally:
        pass

# MAIN EXPERIMENT RUNNER

def get_results(model,
    tokenizer,
    model_name: str,
    train_datasets: dict,
    test_datasets: dict,
    methods: list,
    base_methods: dict,
    experiment_name: str,
    rewards: dict = None,
    grpo: bool = False
    ) -> list:

    # Create directories
    """os.makedirs("outputs_" + experiment_name, exist_ok=True)
    lora_dir = "lora_adapters_" + experiment_name
    os.makedirs(lora_dir, exist_ok=True)"""

    results = []
    for i, method in enumerate(methods):
        # Set temperature based on method
        temperature_map = {
            "prolog_temp_0.6": 0.6,
            "prolog_temp_1.0": 1.0,
            "prolog_temp_1.3": 1.3
        }
        temperature = temperature_map.get(method, 0.0)

        sampling_params = None

        print(f"Processing method: {method}")


        test_dataset = test_datasets[method]
        base_method = base_methods[method]

        # Generate results
        lora_path = ""
        method_results = get_method_results(test_dataset, method, base_method, model, 
                                          tokenizer, sampling_params, grpo, lora_path)

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

        #cleanup_model(model, tokenizer)
        print(f"Completed method: {method}")


    write_results_to_file.config_info = {
        "model_name": model_name,
    }
    write_results_to_file(results, methods, experiment_name)
    return results

# METHOD RESULT PROCESSING

def get_method_results(test_dataset: list,
    method: str,
    base_method: str,
    model,
    tokenizer,
    temperature,
    grpo: bool = False,
    lora_path: str = None
    ) -> list:

    method_results = []

    for j, sample in enumerate(test_dataset):
        print(f"Processing sample {sample['id']} (iteration {j}) for method {method}")

        try:
            # Generate based on method
            if base_method == "cot":
                output, result = generate_cot(sample, model, tokenizer, temperature, grpo, lora_path)
            elif base_method == "prolog":
                try:
                    if method == "prolog_retry":
                        output, code, result, n_retries, intermediates, fixed, temperature = generate_prolog_retry(sample, model, tokenizer, temperature)
                        print(f"Prolog retry output: {output}, code: {code}, result: {result}")
                    elif method == "prolog_fix":
                        output, code, result, intermediates, fixed = generate_prolog_fix(sample, model, tokenizer, temperature)
                        print(f"Prolog fix output: {output}, code: {code}, result: {result}")
                    else:
                        output, code, result = generate_prolog(sample, model, tokenizer, temperature, grpo, lora_path)
                        print(f"Prolog output: {result}")
                except Exception as e:
                    print(f"generate_prolog failed for sample {sample['id']}: {e}")
                    traceback.print_exc()
                    raise 

            elif base_method == "direct":    
                output, result = generate_direct_answer(sample, model, tokenizer, temperature)                    


            # Build result dict
            result_dict = {
                "question_index": sample["id"], "question": sample["sample_prompt"]["content"],
                "expected_answer": sample["answer"], f"{method}_output": output,
                f"{method}_answer": result, "options": sample["n_options"],
            }            

            if base_method == "prolog":
                result_dict[f"{method}_code"] = code
                if method == "prolog_retry":
                    result_dict["intermediate_generations"] = intermediates
                    result_dict["fixed"] = fixed
                    result_dict["temperature"] = temperature
                    result_dict["n_retries"] = n_retries
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
    results_dir = Path("results_text_files")
    results_dir.mkdir(parents=True, exist_ok=True)

    file_name = results_dir / f"results_{experiment_name}.txt"
    with open(file_name, "w", encoding='utf-8') as f:
        f.write(f"TEST RESULTS {experiment_name.upper()}\n{'='*80}\n\n")

        for result in results:
            f.write(f"SAMPLE {result['question_index']}\n{'-'*80}\n")



            if 'prompt' in result:
                f.write(f"PROMPT MESSAGE STRUCTURE (role/content pairs):\n")
                for msg in result['prompt']:
                    f.write(f"[{msg['role']}] {msg['content']}\n")
                f.write("\n")


            if 'tokenized_prompt' in result:
                f.write(f"TOKENIZED PROMPT:\n{result['tokenized_prompt']}\n\n")


            f.write(f"Expected Answer: {result['expected_answer']}\n")
            f.write(f"Number of Options: {result.get('options', '?')}\n\n")


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

            if "temperature" in result:
                f.write(f"Temperature: {result['temperature']}\n")

            f.write(f"Random Answer: {result.get('random_answer', '?')}\n")
            f.write(f"{'='*80}\n\n")


        f.write("\nEXPERIMENT CONFIGURATION\n")
        f.write(f"{'-'*80}\n")
        if hasattr(write_results_to_file, "config_info"):
            config = write_results_to_file.config_info
            for key, val in config.items():
                f.write(f"{key}: {val}\n")
        else:
            f.write("No configuration metadata provided.\n")

    print(f"Results saved to {file_name}")

# GENERATION METHODS

def generate(model, tokenizer, prompt, temperature, use_lora: bool = False, lora_path: str = None, max_retries: int = 5, initial_delay: int = 5):
    """
    Generates text using the specified model with a retry mechanism for rate limits.

    Args:
        model (str): The model to use for generation.
        tokenizer: The tokenizer associated with the model (though not directly used in the API call here).
        prompt (list): A list of message dictionaries.
        temperature (float): The sampling temperature.
        use_lora (bool): Whether to use LoRA (not directly used in the API call here).
        lora_path (str): Path to the LoRA model (not directly used in the API call here).
        max_retries (int): Maximum number of times to retry on a rate limit error.
        initial_delay (int): Initial delay in seconds before retrying. This delay will increase exponentially.

    Returns:
        str: The generated text content.
    """
    retries = 0
    delay = initial_delay

    while retries < max_retries:
        try:
            completion = client.chat.completions.create(
                extra_headers={},
                extra_body={},
                #model="meta-llama/llama-3.2-3b-instruct:free",
                #model="meta-llama/llama-3.1-8b-instruct:free",
                model="meta-llama/llama-3.1-405b-instruct:free",
                messages=prompt,
                temperature=temperature
            )
            return completion.choices[0].message.content
        except RateLimitError as e:
            retries += 1
            print(f"Rate limit exceeded. Retrying in {delay} seconds... (Attempt {retries}/{max_retries})")
            time.sleep(delay)
            delay *= 2 # Exponential backoff
        except Exception as e:
            # Catch other potential errors that are not rate limit related
            print(f"An unexpected error occurred: {e}")
            raise # Re-raise the exception if it's not a rate limit error
    
    print(f"Failed to generate after {max_retries} retries due to rate limits.")
    return None

def generate_prolog(sample: dict, model, tokenizer, temperature, use_lora: bool = False, lora_path: str = None) -> tuple:
    prolog_code_dir = Path("prolog_code")
    prolog_code_dir.mkdir(parents=True, exist_ok=True)

    output = generate(model, tokenizer, sample["prompt"], temperature, use_lora, lora_path)

    prolog_code, warn = extract_prolog(output)
    if prolog_code:
        prolog_filename = prolog_code_dir / "overwritten_code.pl"
        with open(prolog_filename, "w", encoding="utf-8") as pl_file:
            pl_file.write(prolog_code)

        prolog_result = prolog_answer(code_file=prolog_filename)
    else:
        prolog_result = f"No valid Prolog code generated. {warn}"

    return output, prolog_code, prolog_result

def generate_prolog_fix(sample: dict, model, tokenizer, temperature) -> tuple:
    intermediates = []
    fixed = 0

    try:
        output, prolog_code, prolog_result = generate_prolog(sample, model, tokenizer, temperature)

        if prolog_result in "ABCDEFG":
            print(f"RESULT {prolog_result} IS {'CORRECT' if prolog_result == sample['answer'] else 'INCORRECT'}")
            return output, prolog_code, prolog_result, intermediates, fixed

        print(f"RESULT '{prolog_result}' IS NOT A LETTER")
        print(f"OUTPUT BEFORE RETRY:\n{output}\n\nCODE BEFORE RETRY:\n {prolog_code}")


    except Exception as e:
        print(f"An error occurred during execution: {str(e)}")
        import traceback
        traceback.print_exc()

    new_output, new_code, new_result = fix_prolog(prolog_result, prolog_code, sample, model, tokenizer, temperature)

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

def generate_prolog_retry(sample: dict, model, tokenizer, temperature) -> tuple:
    i = 0
    intermediates = []
    fixed = False

    temperature = 0.0

    try:
        output, prolog_code, prolog_result = generate_prolog(sample, model, tokenizer, temperature)
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

    while i < 10:
        temperature += 0.1
        print(f"\nRetrying '{i}' with temperature {temperature}...")
        new_output, new_code, new_result = generate_prolog(sample, model, tokenizer, temperature)

        print(f"REGENERATED {i} New response:\n{new_output}\nNew result:\n{new_result}")

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

def fix_prolog(error_txt: str, wrong_output: str, sample: dict, model, tokenizer, temperature) -> tuple:
    fix_prolog_code_dir = Path("fix_prolog_code")
    fix_prolog_code_dir.mkdir(parents=True, exist_ok=True)

    sample_prompt = sample["sample_prompt"]["content"]
    prompt_parts = [
        f'Logical deduction problem with a context and multiple choice options: """{sample_prompt}"""',
        f'Prolog program that contains some mistake(s):\n"{wrong_output}"\n',
        f"Solver's error message:\n\"{error_txt}\"\n",
        "Examples of the most common errors:\n", EXAMPLES_ERRORS,
        '\nGenerate a correct version of the Prolog code:\n'
    ]

    error_prompt = [
        {"role": "system", "content": "You are a Prolog program fixer. Given a logical deduction problem with a context and multiple choice options, there is a Prolog program that contains some mistake(s). Your job is to use the solver's error message as feedback and examples of the most common errors to generate a correct version of the code. Generate the solution as a valid Prolog program starting with 'A:\n' followed by solve(Answer) :- and having the program enclosed in triple backticks labeled prolog."},
        {"role": "user", "content": "\n".join(prompt_parts)}
    ]

    output = generate(model, tokenizer, error_prompt, temperature)
    prolog_code, warn = extract_prolog(output)

    if prolog_code:
        prolog_filename = fix_prolog_code_dir / f"{sample['id']}_code.pl"
        with open(prolog_filename, "w", encoding="utf-8") as pl_file:
            pl_file.write(prolog_code)
        print(f"Saved Prolog code to {prolog_filename}")
        prolog_result = prolog_answer(code_file=prolog_filename)
    else:
        prolog_result = f"No valid Prolog code generated. {warn}"

    return output, prolog_result, prolog_code

def generate_cot(sample: dict, model, tokenizer, temperature, use_lora: bool = False, lora_path: str = None) -> tuple:
    output = generate(model, tokenizer, sample["prompt"], temperature, use_lora, lora_path)
    answer = extract_letter_from_cot(output)
    return output, answer

def generate_direct_answer(sample: dict, model, tokenizer, temperature) -> tuple:
    string = sample['sample_prompt']['content']
    direct_prompt = [
        {"role": "system", "content": "You are a logical deduction assistant. Please provide the letter of the correct answer (e.g., A, B, C)."},
        {"role": "user", "content": f"{string}\n\n Type 'Answer: ' followed by the letter of the correct answer."}
    ]

    output = generate(model, tokenizer, direct_prompt, temperature)
    return output, extract_letter_from_cot(output)

# Text extraction utilities
def extract_letter_from_cot(cot_text: str) -> str:
    """Extract answer letter from chain-of-thought response"""
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
    """Extract Prolog code from LM response, with logging for failures."""
    warn = ""
    match = re.search(r'```prolog\s*\n*(.*?)\n*\s*```', response_text, re.DOTALL)
    if match:
        prolog_code = match.group(1).strip()
    else:
        warn = "[WARN] No prolog code block (```prolog...```) found in output."
        print(warn)
        return "", None

    solve_match = re.search(r"solve\s*\(\s*[A-Za-z_]+\s*\)\s*:-", prolog_code)
    if not solve_match:
        warn = "[WARN] No 'solve(...) :-' clause found in Prolog code."
        print(warn)
    else:
        prolog_code = prolog_code[solve_match.start():]


    choose_option_count = len(re.findall(r"choose_option\s*\(", prolog_code))
    if choose_option_count < 4:
        print(f"[WARN] Only {choose_option_count} 'choose_option' calls found (expected ≥4).")

    return prolog_code, warn

# Prolog execution utilities
class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException()