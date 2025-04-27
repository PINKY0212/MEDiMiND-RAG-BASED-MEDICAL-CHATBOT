import warnings
warnings.filterwarnings("ignore", message=".*Torch was not compiled with flash attention.*")

from typing import Dict, List, Optional
import re
from langchain_ollama import OllamaLLM
from src.config import OLLAMA_CONFIG
from src.utils.conversation_logger import ConversationLogger
import time
# Initialize conversation logger
conversation_logger = ConversationLogger()

def analyze_medical_query(query: str) -> tuple[bool, str]:
    """
    Use LLM to analyze if the query is medical and determine which index to use.
    
    Args:
        query (str): The user's query
        
    Returns:
        tuple[bool, str]: (is_medical, index_name)
            is_medical: True if content is medical, False otherwise
            index_name: The index to use ('research_papers', 'guidelines', 'drug_info', 'hybrid', or 'non_medical')
    """
    # Initialize LLM
    llm = OllamaLLM(
        model=OLLAMA_CONFIG['model'],
        temperature=OLLAMA_CONFIG['temperature'],
        base_url=OLLAMA_CONFIG['base_url']
    )
    
    # Create prompt for medical content analysis
    prompt = f"""
    Analyze the following query and determine:
    1. If it is medical in nature
    2. If medical, which category it belongs to:
       - research_papers: For research, studies, clinical trials, scientific literature
       - guidelines: For medical guidelines, protocols, standards, procedures
       - drug_info: For drug-related information, medications, prescriptions, side effects
       - hybrid: For medical queries that don't fit clearly into the above categories
    
    Query: {query}
    
    Respond in this exact format:
    MEDICAL: YES/NO
    CATEGORY: [research_papers/guidelines/drug_info/hybrid/non_medical]
    """
    print(f"User Query: {query}\n")
    start_time = time.time()
    
    # Get response from LLM
    response = llm.invoke(prompt).strip()
    end_time = time.time()
    print(f"Time for Medical or Non-Medical Category: {end_time - start_time}")
    # Parse the response
    is_medical = "MEDICAL: YES" in response.upper()
    index_name = "non_medical"
    
    if is_medical:
        # Extract category from response
        category_line = [line for line in response.split('\n') if "CATEGORY:" in line.upper()]
        if category_line:
            # Extract just the category name without any additional description
            category_text = category_line[0].split("CATEGORY:")[1].strip().lower()
            # Take only the first word (the category name) and remove any parentheses or additional text
            index_name = category_text.split()[0].split('(')[0]
    
    return is_medical, index_name

def get_relevant_content(query: str, index_name: str, vector_stores: Dict, history: List[Dict] = None) -> str:
    """
    Get relevant content from vector stores based on the query.
    
    Args:
        query (str): User's query
        index_name (str): Name of the index to use
        vector_stores (Dict): Dictionary of vector stores
        history (List[Dict]): Conversation history
        
    Returns:
        str: Generated response
    """
    try:
        print(f"Query for Vector Store: {query}\n\n")
        # If query is non-medical, return appropriate message
        if index_name == 'non_medical':
            response = "I'm sorry, but I can only assist with medical-related queries. Please ask a question about health, medicine, or medical conditions."
            conversation_logger.log_conversation(query, response, index_name, vector_stores, history)
            return response
        
        # Check if vector stores are available
        if not vector_stores:
            raise Exception("System not properly initialized")
            
        # Check if the specific vector store exists
        if index_name not in vector_stores:
            raise Exception(f"{index_name} database not available")
        
        print(f"Time before vector store called: {time.time()}")
            
        # Get the appropriate vector store
        vector_store = vector_stores[index_name]
        
        # Get relevant documents
        docs = vector_store.similarity_search(query, k=3)
        print(f"--------------Start of Docs from Vector Store-------------------------------")
        print(docs)
        print(f"--------------End of Docs from Vector Store---------------------------------")

        print(f"Time after vector store called: {time.time()}")
        
        # If no documents found
        if not docs:
            raise Exception(f"No relevant information found in {index_name} database")
        
        # Initialize LLM
        llm = OllamaLLM(
            model=OLLAMA_CONFIG['model'],
            temperature=OLLAMA_CONFIG['temperature'],
            base_url=OLLAMA_CONFIG['base_url']
        )
        
        # Prepare conversation history text
        history_text = ""
        if history and len(history) > 0:
            history_text = "\nPrevious Conversation:\n"
            for turn in history[-3:]:  # Only use last 3 turns for context
                history_text += f"User: {turn.get('user', '')}\n"
                history_text += f"Assistant: {turn.get('assistant', '')}\n"
        
        # Create prompt for processing the documents
        prompt = f"""
        Based on the following medical documents and conversation history, please provide a clear and concise answer to the user's query.
        Make sure to:
        1. Consider the conversation history for context
        2. Focus on the most relevant information
        3. Present the information in a clear, organized manner
        4. Use simple language that a non-medical person can understand
        5. If there are multiple sources, combine them coherently
        6. If there are conflicting information, mention this
        7. If the information is incomplete, mention this
        8. Write the answer in 100 words or less.
        
        {history_text}
        
        Current User Query: {query}
        
        Relevant Documents:
        {chr(10).join([doc.page_content for doc in docs])}
        
        Please provide a comprehensive but concise answer to the user's query, taking into account the conversation history if relevant.
        """
        print("----------------------------------------------Start of LLM Output----------------------------------------------")
        print(f"LLM Final- Time before for processing the raw data: {time.time()}")
        # Get processed response from LLM
        response = llm.invoke(prompt)
        print("LLM Generated Response: ", response)
        print(f"LLM Final- Time after for processing the raw data: {time.time()}")
        print("----------------------------------------------End of LLM Output----------------------------------------------\n\n")
        
        
        return response
    except Exception as e:
        print(f"Error in get_relevant_content: {str(e)}")
        raise e
    









    # Generate reference answers based on the documents
        # reference_answers = []
        # for doc in docs:
        #     # Create a reference answer from each document
        #     ref_prompt = f"""
        #     Based on the following medical document, provide a concise and accurate answer to the query: {query}
            
        #     Document:
        #     {doc.page_content}
            
        #     Please provide a clear and accurate answer that could be used as a reference for evaluation.
        #     """
        #     ref_answer = llm.invoke(ref_prompt)
        #     reference_answers.append(ref_answer)
        
        # # Log the conversation with reference answers
        # conversation_logger.log_conversation(
        #     query, 
        #     response, 
        #     index_name, 
        #     vector_stores, 
        #     history,
        #     reference_answers=reference_answers
        # )