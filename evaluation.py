from collections import Counter

def compute_accuracy(results, method_key, problem_type=None):
    """
    Compute accuracy for a given method. Can be filtered by problem type.
    
    Args:
        results (list): List of result dictionaries (one per sample).
        method_key (str): The key to compare with the correct answer (e.g., 'AO_answer').
        problem_type (int): If specified, only analyze problems with this many answer options.
        
    Returns:
        float: Accuracy.
    """
    if problem_type:
        filtered_results = [r for r in results if r.get("options") == problem_type]
    else:
        filtered_results = results
        
    total = len(filtered_results)
    correct = sum(1 for r in filtered_results if r.get(method_key) == r.get("expected_answer"))
    return 100.0 * correct / total if total > 0 else 0.0

def answer_distribution(results, method_key, problem_type=None):
    """
    Count how often each answer appears for a method, grouping Prolog-related errors.

    Args:
        results (list): List of result dictionaries.
        method_key (str): The key for the method's answer.
        problem_type (int): If specified, only analyze problems with this many answer options.

    Returns:
        dict: Counter of answer distribution.
    """
    if problem_type:
        filtered_results = [r for r in results if r.get("options") == problem_type]
    else:
        filtered_results = results

    # Process and group answers
    processed_answers = []
    for r in filtered_results:
        answer = r.get(method_key)
        if answer:
            if answer.startswith("Error: Caused by:"):
                processed_answers.append("Prolog Error")
            else:
                processed_answers.append(answer)

    counter = Counter(processed_answers)
    sorted_counter = dict(sorted(counter.items(), key=lambda item: item[1], reverse=True))

    return sorted_counter


def count_problem_types(results):
    """
    Count how many problems of each type (3, 5, or 7 answer options) are in the dataset.
    
    Args:
        results (list): List of result dictionaries.
        
    Returns:
        dict: Count of each problem type.
    """
    problem_types = [r.get("options") for r in results if "options" in r]
    return dict(Counter(problem_types))

def summarize_all_methods(results, answer_types):
    """
    Print accuracy and distribution for each method,
    broken down by problem type.
    
    Args:
        results (list): List of result dictionaries.
    """
    
    problem_types = [3, 5, 7]
    
    # Count problems by type
    problem_counts = count_problem_types(results)
    print("Dataset composition:")
    for pt in problem_types:
        count = problem_counts.get(pt, 0)
        print(f"  {pt}-option problems: {count} ({100.0 * count / len(results):.1f}%)")
    
    # Overall performance for each method
    print("\n=== OVERALL PERFORMANCE ===")
    for answer_type in answer_types:
        acc = compute_accuracy(results, answer_type)
        dist = answer_distribution(results, answer_type)
        
        print(f"\n--- {answer_type} ---")
        print(f"Accuracy: {acc:.2f}%")
        print("Answer distribution:", dist)
    
    # Performance by problem type
    print("\n=== PERFORMANCE BY PROBLEM TYPE ===")
    for problem_type in problem_types:
        print(f"\n## {problem_type}-option problems ##")
        
        for answer_type in answer_types:
            acc = compute_accuracy(results, answer_type, problem_type)
            dist = answer_distribution(results, answer_type, problem_type)
            
            print(f"\n--- {answer_type} ---")
            print(f"Accuracy: {acc:.2f}%")
            print("Answer distribution:", dist)
    
    print("\n=== RETRY ANALYSIS ===")

    retry_stats = analyze_retries(results, "prolog_retry_answer")
    print(f"Retried cases: {retry_stats['retried_cases_count']}")
    print(f"Retry success rate: {retry_stats['retry_success_rate']:.2f}%")
    print(f"Average retries for success: {retry_stats['average_retries_for_success']:.2f}")

    print("\n=== FIX ANALYSIS ===")

    fix_stats = analyze_fix(results)
    print(f"Fixed cases: {fix_stats['fixed_cases_count']}")
    print(f"Fix success rate: {fix_stats['fix_success_rate']:.2f}%")

    print("\n=== BACKUP ANALYSIS ===")

    backup_stats = analyze_backup(results, "prolog_backup_answer")
    print(f"Used-backup cases: {backup_stats['backup_cases_count']}")
    print(f"Backup success rate: {backup_stats['backup_success_rate']:.2f}%")
    
    print("\n=== VALID-ONLY (NO ERROR) PERFORMANCE ===")
    for answer_type in answer_types:
        valid_results = [
            r for r in results
            if r.get(answer_type) is not None and r.get(answer_type) in "ABCDEFG"
        ]

        acc = compute_accuracy(valid_results, answer_type)
        dist = answer_distribution(valid_results, answer_type)

        print(f"\n--- {answer_type} (valid only) ---")
        print(f"Accuracy: {acc:.2f}%")
        print("Answer distribution:", dist)

def analyze_retries(results, method_key):
    """
    Analyze how often retries lead to correct answers and the average number of retries when successful.

    Args:
        results (list): List of result dictionaries.
        method_key (str): The key to compare with the correct answer.

    Returns:
        dict: Dictionary with retry success rate and average retries for correct cases.
    """
    retried_samples = [r for r in results if r.get("n_retries", 0) > 0]
    if not retried_samples:
        return {
            "retry_success_rate": 0.0,
            "average_retries_for_success": 0.0,
            "retried_cases_count": 0
        }

    correct_retries = [r for r in retried_samples if r.get(method_key) == r.get("expected_answer")]
    retry_success_rate = 100.0 * len(correct_retries) / len(retried_samples)

    if correct_retries:
        avg_retries = sum(r["n_retries"] for r in correct_retries) / len(correct_retries)
    else:
        avg_retries = 0.0

    return {
        "retry_success_rate": retry_success_rate,
        "average_retries_for_success": avg_retries,
        "retried_cases_count": len(retried_samples)
    }

def analyze_fix(results):
    """
    Analyze how often fix lead to correct answers.

    Args:
        results (list): List of result dictionaries.
        method_key (str): The key to compare with the correct answer.

    Returns:
        dict: Dictionary with fix count and success rate.
    """
    fixed_samples = [r for r in results if r.get("fixed", 0) > 0]
    correct_fixes = [r for r in results if r.get("fixed", 0) == 2]
    if not fixed_samples:
        return {
            "fix_success_rate": 0.0,
            "fixed_cases_count": 0
        }

    fix_success_rate = 100.0 * len(correct_fixes) / len(fixed_samples)

    return {
        "fix_success_rate": fix_success_rate,
        "fixed_cases_count": len(fixed_samples)
    }

def analyze_backup(results, method_key):
    """
    Analyze how often backup lead to correct answers.

    Args:
        results (list): List of result dictionaries.
        method_key (str): The key to compare with the correct answer.

    Returns:
        dict: Dictionary with backup success rate.
    """
    backup_samples = [r for r in results if r.get("used_backup", False) == True]

    if not backup_samples:
        return {
            "backup_success_rate": 0.0,
            "backup_cases_count": 0
        }

    correct_backups = [r for r in backup_samples if r.get(method_key) == r.get("expected_answer")]
    backup_success_rate = 100.0 * len(correct_backups) / len(backup_samples)

    return {
        "backup_success_rate": backup_success_rate,
        "backup_cases_count": len(backup_samples)
    }

