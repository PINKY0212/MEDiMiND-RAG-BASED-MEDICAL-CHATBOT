import json
import os
import sys
from datetime import datetime

# Add the project root directory to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.append(project_root)

from src.utils.eval_metrics.rouge_metrics import RougeMetrics
from src.utils.eval_metrics.evaluation_data_collector import EvaluationDataCollector

def load_evaluation_data(file_path: str) -> dict:
    """Load evaluation data from JSON file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def calculate_and_save_metrics(eval_data: dict, output_dir: str = "evaluation_results"):
    """Calculate ROUGE metrics and save results"""
    # Initialize metrics calculator
    metrics = RougeMetrics()
    
    # Calculate ROUGE scores
    scores = metrics.calculate_rouge_scores(
        predictions=[eval_data['rag_output']],
        references=[eval_data['reference_answer']]
    )
    
    # Create results dictionary
    results = {
        'timestamp': datetime.now().isoformat(),
        'user_query': eval_data['user_query'],
        'rag_output': eval_data['rag_output'],
        'reference_answer': eval_data['reference_answer'],
        'rouge_scores': scores,
        'metadata': eval_data.get('metadata', {})
    }
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = os.path.join(output_dir, f'evaluation_results_{timestamp}.json')
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    
    return output_file

def main():
    # Path to evaluation data
    eval_file = os.path.join(os.path.dirname(__file__), "demo_evaluation.json")
    
    try:
        # Load evaluation data
        print("Loading evaluation data...")
        eval_data = load_evaluation_data(eval_file)
        
        # Calculate and save metrics
        print("Calculating ROUGE metrics...")
        output_file = calculate_and_save_metrics(eval_data)
        
        print(f"\nEvaluation completed successfully!")
        print(f"Results saved to: {output_file}")
        
        # Print ROUGE scores
        metrics = RougeMetrics()
        scores = metrics.calculate_rouge_scores(
            predictions=[eval_data['rag_output']],
            references=[eval_data['reference_answer']]
        )
        
        print("\nROUGE Scores:")
        print("ROUGE-1:")
        print(f"  Precision: {scores['rouge1']['precision']:.4f}")
        print(f"  Recall: {scores['rouge1']['recall']:.4f}")
        print(f"  F-measure: {scores['rouge1']['fmeasure']:.4f}")
        
        print("\nROUGE-L:")
        print(f"  Precision: {scores['rougeL']['precision']:.4f}")
        print(f"  Recall: {scores['rougeL']['recall']:.4f}")
        print(f"  F-measure: {scores['rougeL']['fmeasure']:.4f}")
        
    except Exception as e:
        print(f"Error during evaluation: {str(e)}")

if __name__ == "__main__":
    main() 