import json
import os
import signal
import sys
from datetime import datetime
from typing import Dict, List

class ConversationLogger:
    def __init__(self, log_dir: str = "conversation_logs"):
        """
        Initialize the conversation logger.
        
        Args:
            log_dir (str): Directory to store conversation logs
        """
        self.log_dir = log_dir
        self.current_conversation = []
        self.conversation_id = None
        
        # Create log directory if it doesn't exist
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        # Start a new conversation by default
        self.start_new_conversation()
    
    def start_new_conversation(self, user_id: str = None):
        """
        Start a new conversation.
        
        Args:
            user_id (str, optional): User identifier
        """
        self.current_conversation = []
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        user_identifier = user_id if user_id else 'anonymous'
        self.conversation_id = f"conv_{timestamp}_{user_identifier}"
    
    def log_interaction(self, user_message: str, bot_response: str, metadata: Dict = None):
        """
        Log a single interaction in the conversation.
        
        Args:
            user_message (str): User's message
            bot_response (str): Bot's response
            metadata (Dict, optional): Additional metadata about the interaction
        """
        # Ensure we have a conversation ID
        if not self.conversation_id:
            self.start_new_conversation()
            
        interaction = {
            "timestamp": datetime.now().isoformat(),
            "user_message": user_message,
            "bot_response": bot_response
        }
        
        if metadata:
            interaction["metadata"] = metadata
            
        self.current_conversation.append(interaction)
        # Save after each interaction to prevent data loss
        self.save_conversation()
    
    def save_conversation(self, evaluation_data: Dict = None):
        """
        Save the current conversation to a JSON file.
        
        Args:
            evaluation_data (Dict, optional): Additional evaluation data to include
        """
        if not self.current_conversation:
            return
            
        # Ensure we have a conversation ID
        if not self.conversation_id:
            self.start_new_conversation()
            
        conversation_data = {
            "conversation_id": self.conversation_id,
            "start_time": self.current_conversation[0]["timestamp"],
            "end_time": self.current_conversation[-1]["timestamp"],
            "interactions": self.current_conversation
        }
        
        if evaluation_data:
            conversation_data["evaluation_data"] = evaluation_data
        
        # Create filename with timestamp and conversation ID
        filename = f"{self.conversation_id}.json"
        filepath = os.path.join(self.log_dir, filename)
        
        # Save to JSON file
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(conversation_data, f, indent=4, ensure_ascii=False)
        
        return filepath
    
    def get_conversation_history(self) -> List[Dict]:
        """
        Get the current conversation history.
        
        Returns:
            List[Dict]: List of conversation interactions
        """
        return self.current_conversation
    
    def log_conversation(self, 
                        user_query: str, 
                        bot_response: str, 
                        index_name: str,
                        vector_stores: Dict,
                        history: List[Dict] = None,
                        evaluation_data: Dict = None,
                        reference_answers: List[str] = None):
        """
        Log a conversation interaction with all necessary details for evaluation.
        
        Args:
            user_query (str): The user's query
            bot_response (str): The bot's response
            index_name (str): The index used for the response
            vector_stores (Dict): Dictionary of vector stores
            history (List[Dict]): Conversation history
            evaluation_data (Dict): Any additional evaluation data
            reference_answers (List[str]): List of reference answers for evaluation
        """
        # Create conversation entry
        conversation = {
            'timestamp': datetime.now().isoformat(),
            'user_query': user_query,
            'bot_response': bot_response,
            'index_used': index_name,
            'vector_stores_available': list(vector_stores.keys()) if vector_stores else [],
            'conversation_history': history if history else [],
            'evaluation_data': {
                **(evaluation_data if evaluation_data else {}),
                'reference_answers': reference_answers if reference_answers else []
            }
        }
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'conversation_{timestamp}.json'
        filepath = os.path.join(self.log_dir, filename)
        
        # Save to file
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(conversation, f, indent=4, ensure_ascii=False)
        
        return filepath 