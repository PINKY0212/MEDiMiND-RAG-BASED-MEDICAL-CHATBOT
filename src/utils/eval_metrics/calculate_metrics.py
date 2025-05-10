import os
from typing import Dict, List
from datetime import datetime
from .evaluation_data_collector import EvaluationDataCollector
from .rouge_metrics import RougeMetrics
import json

def calculate_conversation_metrics(log_dir: str = "conversation_logs", output_dir: str = "evaluation_results") -> tuple[Dict, Dict]:
    """
    Calculate ROUGE metrics for all conversations in the specified log directory.
    
    Args:
        log_dir: Directory containing conversation log files
        output_dir: Directory to save evaluation results
        
    Returns:
        Tuple of (detailed_results, summary)
    """
    print("\n=== Starting Evaluation Process ===")
    print(f"Reading logs from: {log_dir}")
    print(f"Output directory: {output_dir}")
    
    # Initialize components
    collector = EvaluationDataCollector(log_dir=log_dir)
    rouge = RougeMetrics()
    
    # Get conversation logs
    logs = collector.get_conversation_logs()
    print(f"\nFound {len(logs)} conversation logs")
    
    results = []
    total_rouge1_scores = {"precision": 0, "recall": 0, "fmeasure": 0}
    total_rougeL_scores = {"precision": 0, "recall": 0, "fmeasure": 0}
    
    # Process each log
    for i, log in enumerate(logs, 1):
        print(f"\n=== Processing Conversation {i}/{len(logs)} ===")
        print(f"Log file: {log.get('timestamp', 'unknown')}")
        
        # Extract conversation data
        user_query = log['user_query']
        bot_response = log['bot_response']
        print("\n--- Conversation Data ---")
        print(f"User Query: {user_query}")
        print(f"Bot Response: {bot_response}")
        
        # Generate reference response
        print("\n--- Generating Reference Response ---")
        reference_response = collector.generate_reference_response(user_query)
        print(f"Reference Response: {reference_response}")
        
        # Calculate ROUGE scores
        print("\n--- Calculating ROUGE Scores ---")
        scores = rouge.calculate_rouge_scores(bot_response, reference_response)
        rouge1_scores = scores['rouge1']
        rougeL_scores = scores['rougeL']
        
        print("ROUGE-1 Scores:")
        print(f"  Precision: {rouge1_scores['precision']:.4f}")
        print(f"  Recall: {rouge1_scores['recall']:.4f}")
        print(f"  F-measure: {rouge1_scores['fmeasure']:.4f}")
        
        print("ROUGE-L Scores:")
        print(f"  Precision: {rougeL_scores['precision']:.4f}")
        print(f"  Recall: {rougeL_scores['recall']:.4f}")
        print(f"  F-measure: {rougeL_scores['fmeasure']:.4f}")
        
        # Store results
        result = {
            "timestamp": datetime.now().isoformat(),
            "conversation_timestamp": log.get('timestamp', 'unknown'),
            "user_query": user_query,
            "bot_response": bot_response,
            "reference_response": reference_response,
            "rouge1_scores": rouge1_scores,
            "rougeL_scores": rougeL_scores
        }
        results.append(result)
        
        # Accumulate scores for averaging
        for metric in ["precision", "recall", "fmeasure"]:
            total_rouge1_scores[metric] += rouge1_scores[metric]
            total_rougeL_scores[metric] += rougeL_scores[metric]
    
    # Calculate average scores
    num_conversations = len(logs)
    print(f"\n=== Evaluation Complete ===")
    print(f"Total conversations evaluated: {num_conversations}")
    
    avg_rouge1 = {
        metric: score / num_conversations 
        for metric, score in total_rouge1_scores.items()
    }
    avg_rougeL = {
        metric: score / num_conversations 
        for metric, score in total_rougeL_scores.items()
    }
    
    # Create summary
    summary = {
        "timestamp": datetime.now().isoformat(),
        "total_conversations": num_conversations,
        "average_rouge1": avg_rouge1,
        "average_rougeL": avg_rougeL
    }
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    detailed_results_file = os.path.join(output_dir, f"detailed_results_{timestamp}.json")
    summary_file = os.path.join(output_dir, f"summary_{timestamp}.json")
    
    print(f"\n--- Saving Results ---")
    print(f"Detailed results saved to: {detailed_results_file}")
    print(f"Summary saved to: {summary_file}")
    
    with open(detailed_results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    return results, summary

def print_metrics_summary(summary: Dict) -> None:
    """
    Print a formatted summary of the evaluation metrics.
    
    Args:
        summary: Dictionary containing evaluation summary
    """
    print("\n=== Evaluation Metrics Summary ===")
    print(f"Total Conversations: {summary['total_conversations']}")
    
    print("\nAverage ROUGE-1 Scores:")
    print(f"  Precision: {summary['average_rouge1']['precision']:.4f}")
    print(f"  Recall: {summary['average_rouge1']['recall']:.4f}")
    print(f"  F-measure: {summary['average_rouge1']['fmeasure']:.4f}")
    
    print("\nAverage ROUGE-L Scores:")
    print(f"  Precision: {summary['average_rougeL']['precision']:.4f}")
    print(f"  Recall: {summary['average_rougeL']['recall']:.4f}")
    print(f"  F-measure: {summary['average_rougeL']['fmeasure']:.4f}")

if __name__ == "__main__":
    print("\n=== Starting Metrics Calculation ===")
    results, summary = calculate_conversation_metrics()
    print_metrics_summary(summary) 