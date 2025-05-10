from flask import render_template, jsonify, request
import json
from urllib.parse import unquote
from src.utils import analyze_medical_query, get_relevant_content
from src.config import CATEGORIES
from src.utils.conversation_logger import ConversationLogger
import time
# Initialize conversation logger
conversation_logger = ConversationLogger()

def setup_routes(app, models):
    """Setup Flask routes with the initialized models"""
    
    @app.route('/')
    def index():
        return render_template('chat.html')
    
    @app.route('/analyze', methods=['POST'])
    def analyze():
        """Endpoint to analyze query and get index name for loading state"""
        msg = request.form.get('msg', '')
        history = request.form.get('history', '[]')
        
        # Parse history
        try:
            parsed_history = json.loads(history) if isinstance(history, str) else history
        except json.JSONDecodeError:
            parsed_history = []
            
        is_medical, index_name, is_asthma = analyze_medical_query(msg, parsed_history)
        return jsonify({
            "index_name": index_name,
            "is_medical": is_medical,
            "is_asthma": is_asthma
        })
    
    @app.route('/get', methods=['POST'])
    def chat():
        msg = request.form['msg']
        history = request.form.get('history', '[]')
        index_name = request.form.get('index_name', 'hybrid')  # Default to hybrid if not provided
        
        # Parse history
        try:
            parsed_history = json.loads(history) if isinstance(history, str) else history
        except json.JSONDecodeError:
            parsed_history = []
        
        # Get response from the appropriate vector store with conversation history
        # print(f"---------------------Start of Vector Store Response--------------------------------")
        start_time = time.time()
        response = get_relevant_content(msg, index_name, models['vector_stores'], parsed_history)
        end_time = time.time()
        print(f"Total time taken for Vector Store Response: {end_time - start_time}")
        # print(f"---------------------End of Vector Store Response--------------------------------\n\n")
        
        # Update history
        current_history = parsed_history + [{"user": msg, "assistant": response}]
        print(f"Response: {response}")
        print(f"-----------------------------------------------------------")
        # print(f"Current history: {current_history}")

        # Log the conversation
        conversation_logger.log_interaction(
            user_message=msg,
            bot_response=response,
            metadata={
                "index_name": index_name,
                "response_time": end_time - start_time,
                "vector_stores": list(models['vector_stores'].keys())
            }
        )
        
        return jsonify({
            "response": response,
            "history": current_history,
            "index_name": index_name
        })
