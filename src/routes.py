from flask import render_template, jsonify, request
import json
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
    
    @app.route('/get', methods=['GET','POST'])
    def chat():
        msg = request.form['msg']
        history = request.form.get('history', '[]')
        
        # Parse history
        try:
            parsed_history = json.loads(history) if isinstance(history, str) else history
        except json.JSONDecodeError:
            parsed_history = []
        

        # Analyze query to determine category
        is_medical, index_name = analyze_medical_query(msg)
        print(f"Index Name from LLM\n Index Name: {index_name}, is_medical: {is_medical}")
        
        # Return initial response with index name for loading message
        if request.form.get('loading', 'false') == 'true':
            return jsonify({
                "loading": True,
                "index_name": index_name
            })
        print(f"---------------------Start of Vector Store Response--------------------------------")
        start_time = time.time()
        # Get response from the appropriate vector store with conversation history
        response = get_relevant_content(msg, index_name, models['vector_stores'], parsed_history)
        end_time = time.time()
        print(f"Total time taken: {end_time - start_time}")
        print(f"---------------------End of Vector Store Response--------------------------------\n\n")
        
        # Update history
        current_history = parsed_history + [{"user": msg, "assistant": response}]
        print(f"---------------------Start of Final Response--------------------------------")
        print(f"Response: {response}")
        print(f"Current history: {current_history}")
        print(f"---------------------End of Final Response--------------------------------\n\n")
        return jsonify({
            "response": response,
            "history": current_history,
            "index_name": index_name
        })
