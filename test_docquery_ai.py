import json
import os
from your_main_project import DocQueryAI  # Import your main project code

def load_test_cases(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)['test_cases']

def compare_outputs(expected, actual):
    # Simple string comparison; consider using more advanced similarity metrics
    return expected.strip().lower() == actual.strip().lower()

def run_tests(test_cases, ai_model):
    results = []
    for case in test_cases:
        pdf_path = case['pdf_path']
        query = case['query']
        expected_output = case['expected_output']
        
        # Generate actual output using the AI model
        actual_output = ai_model.query_pdf(pdf_path, query)
        
        # Compare outputs and store the result
        pass_fail = compare_outputs(expected_output, actual_output)
        results.append({
            "Test Case ID": case['id'],
            "Query": query,
            "Expected Output": expected_output,
            "Actual Output": actual_output,
            "Pass/Fail": "Pass" if pass_fail else "Fail"
        })
    
    return results

def print_test_results(results):
    for result in results:
        print(f"Test Case ID: {result['Test Case ID']}")
        print(f"Query: {result['Query']}")
        print(f"Expected Output: {result['Expected Output']}")
        print(f"Actual Output: {result['Actual Output']}")
        print(f"Result: {result['Pass/Fail']}")
        print("-" * 50)

if __name__ == "__main__":
    test_cases = load_test_cases('data/test_cases.json')
    ai_model = DocQueryAI()  # Initialize your AI model class from the main project

    results = run_tests(test_cases, ai_model)
    print_test_results(results)
    
    # Optional: Write results to a report file
    with open('data/test_report.json', 'w') as f:
        json.dump(results, f, indent=4)
