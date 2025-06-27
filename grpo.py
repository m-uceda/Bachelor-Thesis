import os, re, json
from pyswip import Prolog
from datetime import datetime

os.environ["TRITON_CACHE"] = "/scratch/s5112583/.triton"

# Set up logging
LOG_DIR = "reward_logs"
os.makedirs(LOG_DIR, exist_ok=True)

# Create a timestamped log file
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE = os.path.join(LOG_DIR, f"reward_logs_{timestamp}.txt")
JSON_LOG_FILE = os.path.join(LOG_DIR, f"reward_logs_{timestamp}.json")

def log_reward(func_name, prompt=None, completion=None, answer=None, extracted=None, reward=None):
    """
    Log reward function results to a text file
    """
    with open(LOG_FILE, "a") as f:
        f.write(f"{'='*80}\n")
        f.write(f"REWARD FUNCTION: {func_name}\n")
        f.write(f"TIMESTAMP: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        if prompt is not None:
            f.write(f"PROMPT:\n{prompt}\n\n")
        
        if completion is not None:
            f.write(f"COMPLETION:\n{completion}\n\n")
        
        if answer is not None:
            f.write(f"EXPECTED ANSWER: {answer}\n")
        
        if extracted is not None:
            f.write(f"EXTRACTED ANSWER: {extracted}\n")
        
        if reward is not None:
            f.write(f"REWARD: {reward}\n")
        
        f.write(f"{'='*80}\n\n")

def log_reward_json(records):
    """
    Log reward function results to a JSON file
    """
    try:
        if os.path.exists(JSON_LOG_FILE):
            with open(JSON_LOG_FILE, "r") as f:
                existing_records = json.load(f)
        else:
            existing_records = []
        
        existing_records.extend(records)
        
        with open(JSON_LOG_FILE, "w") as f:
            json.dump(existing_records, f, indent=2)
    except Exception as e:
        with open(os.path.join(LOG_DIR, "json_error.txt"), "a") as f:
            f.write(f"Error writing JSON log: {str(e)}\n")

# Reward functions

def ao_correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    """
    Returns 2.0 if the extracted letter matches the correct answer.
    """
    responses = [completion[0]['content'] for completion in completions]
    extracted = [extract_ao_letter(r) for r in responses]
    rewards = [2.0 if r == a else 0.0 for r, a in zip(extracted, answer)]
    
    # Log all rewards
    log_records = []
    for i, (p, r, a, e, rew) in enumerate(zip(prompts, responses, answer, extracted, rewards)):
        prompt_content = p[-1]['content'] if p and len(p) > 0 else "N/A"
        log_reward(
            "ao_correctness_reward_func",
            prompt=prompt_content,
            completion=r,
            answer=a,
            extracted=e,
            reward=rew
        )
        log_records.append({
            "function": "ao_correctness_reward_func",
            "prompt": prompt_content,
            "completion": r,
            "answer": a,
            "extracted": e,
            "reward": rew,
            "timestamp": datetime.now().isoformat()
        })
    
    # Log to JSON
    log_reward_json(log_records)
    
    return rewards

def ao_letter_reward_func(completions, **kwargs) -> list[float]:
    """
    Returns 0.5 if a valid letter (A–G) can be extracted from the output.
    """
    responses = [completion[0]['content'] for completion in completions]
    extracted = [extract_ao_letter(r) for r in responses]
    rewards = [0.5 if e else 0.0 for e in extracted]
    
    # Log all rewards
    log_records = []
    for i, (r, e, rew) in enumerate(zip(responses, extracted, rewards)):
        log_reward(
            "ao_letter_reward_func",
            completion=r,
            extracted=e,
            reward=rew
        )
        log_records.append({
            "function": "ao_letter_reward_func",
            "completion": r,
            "extracted": e,
            "reward": rew,
            "timestamp": datetime.now().isoformat()
        })
    
    # Log to JSON
    log_reward_json(log_records)
    
    return rewards

def ao_format_reward_func(completions, **kwargs) -> list[float]:
    """
    Returns 0.5 if the format is clean: exactly 'A: X' or similar.
    """
    responses = [completion[0]['content'].strip() for completion in completions]
    matches = [re.fullmatch(r"A:\s*[A-G]", r) for r in responses]
    rewards = [0.5 if m else 0.0 for m in matches]
    
    # Log all rewards
    log_records = []
    for i, (r, m, rew) in enumerate(zip(responses, matches, rewards)):
        log_reward(
            "ao_format_reward_func",
            completion=r,
            extracted="Format match" if m else "No format match",
            reward=rew
        )
        log_records.append({
            "function": "ao_format_reward_func",
            "completion": r,
            "match": bool(m),
            "reward": rew,
            "timestamp": datetime.now().isoformat()
        })
    
    # Log to JSON
    log_reward_json(log_records)
    
    return rewards

def cot_correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    """
    Reward is 2.0 if the extracted letter from CoT matches the correct answer.
    """
    responses = [completion[0]['content'] for completion in completions]
    extracted = [extract_letter_from_cot(r) for r in responses]
    rewards = [2.0 if e == a else 0.0 for e, a in zip(extracted, answer)]
    
    # Log all rewards
    log_records = []
    for i, (p, r, a, e, rew) in enumerate(zip(prompts, responses, answer, extracted, rewards)):
        prompt_content = p[-1]['content'] if p and len(p) > 0 else "N/A"
        log_reward(
            "cot_correctness_reward_func",
            prompt=prompt_content,
            completion=r,
            answer=a,
            extracted=e,
            reward=rew
        )
        log_records.append({
            "function": "cot_correctness_reward_func",
            "prompt": prompt_content,
            "completion": r,
            "answer": a,
            "extracted": e,
            "reward": rew,
            "timestamp": datetime.now().isoformat()
        })
    
    # Log to JSON
    log_reward_json(log_records)
    
    return rewards

def cot_letter_reward_func(completions, **kwargs) -> list[float]:
    """
    Reward is 0.5 if a valid letter (A–G) is found in the CoT response.
    """
    responses = [completion[0]['content'] for completion in completions]
    extracted = [extract_letter_from_cot(r) for r in responses]
    rewards = [0.5 if e else 0.0 for e in extracted]
    
    # Log all rewards
    log_records = []
    for i, (r, e, rew) in enumerate(zip(responses, extracted, rewards)):
        log_reward(
            "cot_letter_reward_func",
            completion=r,
            extracted=e,
            reward=rew
        )
        log_records.append({
            "function": "cot_letter_reward_func",
            "completion": r,
            "extracted": e,
            "reward": rew,
            "timestamp": datetime.now().isoformat()
        })
    
    # Log to JSON
    log_reward_json(log_records)
    
    return rewards

def cot_format_reward_func(completions, **kwargs) -> list[float]:
    """
    Reward:
    += 0.25 if Reasoning starts with "Let's think step by step."
    += 0.25 if Answer block ends with "Answer: [A-G]"
    """
    responses = [completion[0]['content'].strip() for completion in completions]

    rewards = []
    log_records = []

    for response in responses:
        reasoning_match = re.search(r"<reasoning>\s*(.*?)\s*</reasoning>", response, re.DOTALL)
        answer_match = re.search(r"<answer>\s*(.*?)\s*</answer>", response, re.DOTALL)

        reasoning_text = reasoning_match.group(1).strip() if reasoning_match else ""
        answer_text = answer_match.group(1).strip() if answer_match else ""

        reasoning_ok = reasoning_text.startswith("Let's think step by step.")
        answer_ok = re.match(r"Answer:\s*[A-G]$", answer_text.strip())

        reward = 0
        if reasoning_ok:
            reward += 0.25  
        if answer_ok:
            reward += 0.25
        rewards.append(reward)

        # Logging
        format_info = f"Reasoning OK: {'✓' if reasoning_ok else '✗'}, Answer OK: {'✓' if answer_ok else '✗'}"
        log_reward(
            "cot_format_reward_func",
            completion=response,
            extracted=format_info,
            reward=reward
        )
        log_records.append({
            "function": "cot_format_reward_func",
            "completion": response,
            "reasoning_start_ok": reasoning_ok,
            "answer_format_ok": bool(answer_ok),
            "reward": reward,
            "timestamp": datetime.now().isoformat()
        })

    log_reward_json(log_records)
    return rewards

def correctness_reward_func_prolog(prompts, completions, answer, **kwargs) -> list[float]:
    """
    Reward function that returns 2.0 if extracted prolog answer is correct.
    Also prints out the question, true answer, model response, and extracted response.
    """
    responses = [completion[0]['content'] for completion in completions]
    extracted_responses = []
    rewards = []  # Initialize rewards list
    
    log_records = []
    for i, (p, r, a) in enumerate(zip(prompts, responses, answer)):
        q = p[-1]['content'] if p and len(p) > 0 else "N/A"
        
        try:
            prolog_code = extract_prolog(r)
            
            # Log the extracted Prolog code
            with open(os.path.join(LOG_DIR, f"prolog_code_{timestamp}_{i}.pl"), "w") as f:
                f.write(prolog_code if prolog_code else "# No valid Prolog code extracted")
            
            if prolog_code:
                prolog_result = prolog_answer(code=prolog_code)
            else:
                prolog_result = "No valid Prolog code generated"
                
        except Exception as e:
            # Handle any errors in Prolog processing
            prolog_result = f"Error: {str(e)}"
            print(f"Error processing Prolog for completion {i}: {e}")
        
        extracted_responses.append(prolog_result)
        reward = 2.0 if prolog_result == a else 0.0
        rewards.append(reward)  # Always append a reward
        
        log_reward(
            "correctness_reward_func_prolog",
            prompt=q,
            completion=r,
            answer=a,
            extracted=prolog_result,
            reward=reward
        )
        
        log_records.append({
            "function": "correctness_reward_func_prolog",
            "prompt": q,
            "completion": r,
            "answer": a,
            "extracted": prolog_result,
            "prolog_code": prolog_code if 'prolog_code' in locals() else None,
            "reward": reward,
            "timestamp": datetime.now().isoformat()
        })
    
    # Log to JSON
    log_reward_json(log_records)
    
    return rewards


def letter_reward_func_prolog(completions, **kwargs) -> list[float]:
    """
    Reward function that returns 0.5 if extracted answer is a capital letter from A to G.
    """
    responses = [completion[0]['content'] for completion in completions]
    rewards = []  # Initialize rewards list
    
    log_records = []
    for i, r in enumerate(responses):
        try:
            prolog_code = extract_prolog(r)
            
            if prolog_code:
                prolog_result = prolog_answer(code=prolog_code)
                # Handle case where prolog_result might be a string or error
                if isinstance(prolog_result, str) and len(prolog_result) > 0:
                    prolog_result = prolog_result[0] if prolog_result else "None"
                else:
                    prolog_result = "None"
            else:
                prolog_result = "No valid Prolog code generated"
                
        except Exception as e:
            prolog_result = f"Error: {str(e)}"
            print(f"Error processing Prolog for completion {i}: {e}")
        
        is_valid_letter = len(str(prolog_result)) == 1 and str(prolog_result) in "ABCDEFG"
        reward = 0.5 if is_valid_letter else 0.0
        rewards.append(reward)  # Always append a reward
        
        log_reward(
            "letter_reward_func_prolog",
            completion=r,
            extracted=prolog_result,
            reward=reward
        )
        
        log_records.append({
            "function": "letter_reward_func_prolog",
            "completion": r,
            "extracted": prolog_result,
            "is_valid_letter": is_valid_letter,
            "reward": reward,
            "timestamp": datetime.now().isoformat()
        })
    
    # Log to JSON
    log_reward_json(log_records)
    
    return rewards

import re
from datetime import datetime

def prolog_format_reward_func(completions, **kwargs) -> list[float]:
    """
    Awards 0.125 points each for:
    - Uses ```prolog code block
    - Contains solve(Answer) :-
    - Includes at least 4 choose_option calls
    Total reward: 0.0 to 1.0
    """
    rewards = []
    log_records = []

    for completion in completions:
        response = completion[0]["content"].strip()
        enclosed_in_triple_backticks = "```prolog" in response and "```" in response.split("```prolog")[-1]

        prolog_code = extract_prolog(response)
        if prolog_code:  # Only process if we have valid prolog code
            has_solve = bool(re.search(r"solve\s*\(\s*Answer\s*\)\s*:-", prolog_code))
            choose_option_count = len(re.findall(r"choose_option\s*\(", prolog_code))
            has_enough_choose_options = choose_option_count >= 4
        else:
            has_solve = False
            choose_option_count = 0
            has_enough_choose_options = False

        reward = 0.0
        if enclosed_in_triple_backticks:
            reward += 0.125
        if has_solve:
            reward += 0.125
        if has_enough_choose_options:
            reward += 0.125

        rewards.append(reward)  # Make sure we always append a reward

        format_info = (
            f"Enclosed in ```prolog: {'✓' if enclosed_in_triple_backticks else '✗'}, "
            f"Has solve: {'✓' if has_solve else '✗'}, "
            f"Choose options: {choose_option_count}"
        )

        log_reward(
            "prolog_format_reward_func",
            completion=response,
            extracted=format_info,
            reward=reward
        )

        log_records.append({
            "function": "prolog_format_reward_func",
            "completion": response,
            "enclosed_in_triple_backticks": enclosed_in_triple_backticks,
            "has_solve": has_solve,
            "choose_option_count": choose_option_count,
            "reward": reward,
            "timestamp": datetime.now().isoformat()
        })

    log_reward_json(log_records)
    return rewards

def prolog_error_reward_func(completions, **kwargs) -> list[float]:
    """
    Penalizes completions that include invalid Prolog patterns like nth1(Len-1, ...) instead of computing Pos first.
    Returns 0.5 if no unsafe nth1 usage is found; 0.0 otherwise.
    """
    rewards = []
    invalid_pattern = re.compile(r"nth1\s*\(\s*[^,\s()]+\s*[-+*/]\s*[^,\s()]+\s*,")  # Detects arithmetic inside nth1 first argument
    
    log_records = []
    for i, completion in enumerate(completions):
        response = completion[0]["content"]
        prolog_code = extract_prolog(response)
        
        has_invalid_pattern = False
        reward = 0.5
        invalid_matches_found = []

        if prolog_code:
            try:
                if invalid_pattern.search(prolog_code):
                    has_invalid_pattern = True
                    invalid_matches_found = [m.group(0) for m in invalid_pattern.finditer(prolog_code)]
            except TypeError:
                print(f"TypeError during regex in prolog_error_reward_func. Extracted code: {prolog_code}")

        reward_value = 0.0 if has_invalid_pattern else 0.5
        rewards.append(reward_value)
        
        log_reward(
            "prolog_error_reward_func",
            completion=response,
            extracted=f"Invalid nth1 pattern: {'✓' if has_invalid_pattern else '✗'}",
            reward=reward
        )
        
        log_records.append({
            "function": "prolog_error_reward_func",
            "completion": response,
            "has_invalid_pattern": has_invalid_pattern,
            "reward": reward,
            "timestamp": datetime.now().isoformat(),
            "invalid_matches": [m.group(0) for m in invalid_pattern.finditer(prolog_code)] if has_invalid_pattern else []
        })
    
    # Log to JSON
    log_reward_json(log_records)
    return rewards

def extract_prolog(response_text):
    """
    Extract the Prolog code from the language model's response:
    1. Strip any content before "Solve(Answer)"
    2. Strip any content after finding at least 3 occurrences of "choose_option(" followed by an empty line
    
    Args:
        response_text (str): The full text response from the language model.
        
    Returns:
        str: The extracted Prolog code.
    """
    # First check if code is in a code block
    if "```prolog" in response_text:
        prolog_code = response_text.split("```prolog")[1].split("```")[0].strip()
    else:
        prolog_code = response_text.strip()
    
    # Find the starting point (Solve function)
    solve_match = re.search(r"solve\s*\(\s*[A-Za-z_]+\s*\)\s*:-", prolog_code)
    if solve_match:
        # Keep only the code starting from the solve function
        prolog_code = prolog_code[solve_match.start():]
    
    # Find occurrences of "choose_option("
    occurrences = 0
    current_idx = 0
    
    while True:
        # Find the next occurrence of "choose_option("
        next_idx = prolog_code.find("choose_option(", current_idx)
        
        # If we can't find it, just return what we have
        if next_idx == -1:
            return prolog_code
            
        occurrences += 1
        current_idx = next_idx + len("choose_option(")
        
        # Once we've found at least 3 occurrences, start checking for empty lines
        if occurrences >= 3:
            lines = prolog_code[current_idx:].split('\n')
            
            # Look for the first empty line
            for i, line in enumerate(lines):
                if line.strip() == '':
                    # Found an empty line, return everything up to and including this line
                    end_position = current_idx + sum(len(lines[j]) + 1 for j in range(i))
                    return prolog_code[:end_position]
            
            # No empty line found, return all the prolog code
            return prolog_code
    
    return prolog_code

def prolog_answer(code_file: str = None, code: str = None) -> str:
    """
    Run the Prolog program and get the answer.
    
    Args:
        code_file (str): Path to the file containing the Prolog code.
        
    Returns:
        str: Result from the Prolog query, or error information if execution fails.
    """
    # Initialize Prolog
    prolog = Prolog()
    
    try:
        # Consult the Prolog file
        if not code_file:
            code_file = "prolog_code.pl"
            with open(code_file, "w") as pl_file:
                pl_file.write(code)

        prolog.consult(code_file)

        # Query the Prolog database
        query = "solve(Answer)"
        result = list(prolog.query(query))

        # Print the result
        #if result:
            #print(f"Query '{query}' succeeded: {result}")
        #else:
            #print(f"Query '{query}' failed.")

    except Exception as e:
        #print(f"Prolog execution error: {e}")
        result = [{"Answer": "error", "error_message": str(e)}]

    if result and isinstance(result, list) and "Answer" in result[0]:
        if result[0]["Answer"] == "error":
            # Keep the error message
            extracted_answer = result[0].get("error_message", "Unknown error")
        else:
            # Extract just the letter
            extracted_answer = str((result[0]["Answer"])).upper()
    else:
        extracted_answer = "No answer found"

    return extracted_answer

def extract_ao_letter(output: str) -> str | None:
    """
    Extracts the answer letter (A-G) from a string using simple regex patterns.
    Handles cases like 'A: C', 'C)', or just 'C'.
    """
    output = output.strip()
    match = re.search(r'\b([A-G])\)', output) or re.search(r'\b([A-G])\b', output)
    return match.group(1) if match else None

def extract_letter_from_cot(cot_text):
    """
    Extract the answer letter from a chain-of-thought response.
    
    Args:
        cot_text (str): The chain-of-thought text containing reasoning and an answer.
        
    Returns:
        str: The extracted letter or None if not found.
    """
    # Look for patterns like "the answer is (B)" or "answer: B" or "So B is correct"
    patterns = [
        r'(?:A|answer|the answer)[\s]*(?:is|:)[\s]*[\(\[]?([A-G])[\)\]]?',  # answer is (B) or answer: B or A: B
        r'(?:So|Thus|Therefore|Hence)[\s]*(?:the answer is|the correct answer is)[\s]*[\(\[]?([A-G])[\)\]]?',  # So the answer is B
        r'(?:So|Thus|Therefore|Hence)[\s]*[\(\[]?([A-G])[\)\]]?', # So (B)
        r'(?:Option|option|answer|Letter|letter)[\s]*[\(\[]?([A-G])[\)\]]?', # Option B
        r'[\(\[]?([A-G])[\)\]]?[\s]*is correct' # (B) is correct
    ]
    
    for pattern in patterns:
        match = re.search(pattern, cot_text, re.IGNORECASE)
        if match:
            return match.group(1) 

    return None