from rouge_score import rouge_scorer
from typing import List, Dict, Union

class RougeMetrics:
    def __init__(self):
        """
        Initialize the ROUGE metrics scorer with ROUGE-1 and ROUGE-L
        """
        self.scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)

    def calculate_rouge_scores(self, 
                             predictions: Union[str, List[str]], 
                             references: Union[str, List[str]]) -> Dict[str, Dict[str, float]]:
        """
        Calculate ROUGE-1 and ROUGE-L scores between predictions and references.
        
        Args:
            predictions: Either a single prediction string or a list of prediction strings
            references: Either a single reference string or a list of reference strings
            
        Returns:
            Dictionary containing ROUGE-1 and ROUGE-L scores with precision, recall, and fmeasure
        """
        # Convert single strings to lists for consistent processing
        if isinstance(predictions, str):
            predictions = [predictions]
        if isinstance(references, str):
            references = [references]
            
        # Ensure equal length
        if len(predictions) != len(references):
            raise ValueError("Number of predictions and references must be equal")
            
        # Calculate scores for each pair
        rouge1_scores = []
        rougeL_scores = []
        
        for pred, ref in zip(predictions, references):
            # Note: rouge_scorer.score takes (reference, prediction) as arguments
            scores = self.scorer.score(pred, ref)  # Changed order: prediction first, then reference
            rouge1_scores.append(scores['rouge1'])
            rougeL_scores.append(scores['rougeL'])
            
        # Calculate average scores
        avg_rouge1 = {
            'precision': sum(s.precision for s in rouge1_scores) / len(rouge1_scores),
            'recall': sum(s.recall for s in rouge1_scores) / len(rouge1_scores),
            'fmeasure': sum(s.fmeasure for s in rouge1_scores) / len(rouge1_scores)
        }
        
        avg_rougeL = {
            'precision': sum(s.precision for s in rougeL_scores) / len(rougeL_scores),
            'recall': sum(s.recall for s in rougeL_scores) / len(rougeL_scores),
            'fmeasure': sum(s.fmeasure for s in rougeL_scores) / len(rougeL_scores)
        }
        
        return {
            'rouge1': avg_rouge1,
            'rougeL': avg_rougeL
        }

# Example usage
if __name__ == "__main__":
    # Initialize the metrics
    metrics = RougeMetrics()
    
    # Example predictions and references
    predictions = [
        "The quick brown fox jumps over the lazy dog",
        "A fast brown fox leaps over a sleepy dog"
    ]
    references = [
        "The quick brown fox jumps over the lazy dog",
        "The quick brown fox jumps over the lazy dog"
    ]
    
    # Calculate scores
    scores = metrics.calculate_rouge_scores(predictions, references)
    
    # Print results
    print("ROUGE-1 Scores:")
    print(f"Precision: {scores['rouge1']['precision']:.4f}")
    print(f"Recall: {scores['rouge1']['recall']:.4f}")
    print(f"F-measure: {scores['rouge1']['fmeasure']:.4f}")
    
    print("\nROUGE-L Scores:")
    print(f"Precision: {scores['rougeL']['precision']:.4f}")
    print(f"Recall: {scores['rougeL']['recall']:.4f}")
    print(f"F-measure: {scores['rougeL']['fmeasure']:.4f}") 