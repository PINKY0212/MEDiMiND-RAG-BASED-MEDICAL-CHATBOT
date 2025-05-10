import os
import json
from typing import List, Dict, Optional
from datetime import datetime
from langchain_ollama import OllamaLLM
from langchain import hub
from src.config import OLLAMA_CONFIG
from .rouge_metrics import RougeMetrics

class EvaluationDataCollector:
    def __init__(self, log_dir: str = "conversation_logs"):
        """
        Initialize the evaluation data collector.
        
        Args:
            log_dir (str): Directory containing conversation logs
        """
        self.log_dir = log_dir
        self.rouge_metrics = RougeMetrics()
        
        # Initialize Ollama model
        self.llm = OllamaLLM(
            model=OLLAMA_CONFIG["model"],
            temperature=OLLAMA_CONFIG["temperature"],
            base_url=OLLAMA_CONFIG["base_url"]
        )
        
        # Load the standard RAG prompt template
        self.prompt_template = hub.pull("rlm/rag-prompt", api_url="https://api.hub.langchain.com")
    
    def get_conversation_logs(self) -> List[Dict]:
        """
        Get all conversation logs from the log directory.
        
        Returns:
            List[Dict]: List of conversation logs
        """
        logs = []
        for filename in os.listdir(self.log_dir):
            if filename.endswith('.json'):
                filepath = os.path.join(self.log_dir, filename)
                with open(filepath, 'r', encoding='utf-8') as f:
                    log_data = json.load(f)
                    logs.append(log_data)
        return logs
    
    def generate_reference_response(self, user_query: str, context: Optional[str] = None) -> str:
        """
        Generate a reference response using the local Ollama model with standard RAG prompt.
        
        Args:
            user_query (str): The user's query
            context (str, optional): Additional context for the response
            
        Returns:
            str: Generated reference response
        """
        try:
            # Format the prompt using the standard RAG template
            prompt = self.prompt_template.format(
                question=user_query,
                context=context if context else "No additional context provided."
            )
            
            # Generate response using the local model
            response = self.llm.invoke(prompt)
            return response.strip()
        except Exception as e:
            print(f"Error generating reference response: {str(e)}")
            return None
    
    def evaluate_conversation(self, conversation_log: Dict) -> Dict:
        """
        Evaluate a conversation using ROUGE metrics.
        
        Args:
            conversation_log (Dict): Conversation log data
            
        Returns:
            Dict: Evaluation results including ROUGE scores
        """
        user_query = conversation_log.get('user_query')
        bot_response = conversation_log.get('bot_response')
        
        if not user_query or not bot_response:
            return None
            
        # Generate reference response
        reference_response = self.generate_reference_response(user_query)
        
        if not reference_response:
            return None
            
        # Calculate ROUGE scores
        scores = self.rouge_metrics.calculate_rouge_scores(bot_response, reference_response)
        
        evaluation_results = {
            'timestamp': datetime.now().isoformat(),
            'user_query': user_query,
            'bot_response': bot_response,
            'reference_response': reference_response,
            'rouge_scores': scores
        }
        
        return evaluation_results
    
    def evaluate_all_conversations(self) -> List[Dict]:
        """
        Evaluate all conversations in the log directory.
        
        Returns:
            List[Dict]: List of evaluation results for each conversation
        """
        conversation_logs = self.get_conversation_logs()
        evaluation_results = []
        
        for log in conversation_logs:
            result = self.evaluate_conversation(log)
            if result:
                evaluation_results.append(result)
                
        return evaluation_results
    
    def save_evaluation_results(self, results: List[Dict], output_dir: str = "evaluation_results"):
        """
        Save evaluation results to a JSON file.
        
        Args:
            results (List[Dict]): List of evaluation results
            output_dir (str): Directory to save evaluation results
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'evaluation_results_{timestamp}.json'
        filepath = os.path.join(output_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
            
        return filepath

# Example usage
if __name__ == "__main__":
    # Initialize the collector
    collector = EvaluationDataCollector()
    
    # Evaluate all conversations
    results = collector.evaluate_all_conversations()
    
    # Save results
    output_file = collector.save_evaluation_results(results)
    print(f"Evaluation results saved to: {output_file}") 