import warnings
warnings.filterwarnings("ignore", message=".*Torch was not compiled with flash attention.*")

from typing import Dict, List, Optional
import re
from langchain_ollama import OllamaLLM
from src.config import OLLAMA_CONFIG
from src.utils.conversation_logger import ConversationLogger
import time
from datetime import datetime
# Initialize conversation logger
conversation_logger = ConversationLogger()

def analyze_medical_query(query: str, history: List[Dict] = None) -> tuple[bool, str, bool]:
    """
    Use LLM to analyze if the query is medical and determine which index to use.
    
    Args:
        query (str): The user's query
        history (List[Dict], optional): Conversation history for context
        
    Returns:
        tuple[bool, str, bool]: (is_medical, index_name, is_asthma)
            is_medical: True if content is medical, False otherwise
            index_name: The index to use ('research_papers', 'guidelines', 'drug_info', 'hybrid', or 'non_medical')
            is_asthma: True if content is asthma, False otherwise
    """
    
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
        for turn in history[-1:]:  # Only use last turn for context
            if isinstance(turn, dict):
                user_msg = turn.get('user', '')
                assistant_msg = turn.get('assistant', '')
                if user_msg and assistant_msg:
                    history_text += f"User: {user_msg}\n"
                    history_text += f"Assistant: {assistant_msg}\n"
    
    # print(f"History Text after preparation in Analyze_Medical_Query: {history_text}\n\n")

    # Create prompt for medical content analysis
    prompt = f"""
    Analyze the following query and conversation history to determine:
    1. If it is medical in nature
    2. If medical, which category it belongs to:
       - research_papers: For research, studies, clinical trials, scientific literature
       - guidelines: For medical guidelines, protocols, standards, procedures
       - drug_info: For drug-related information, medications, prescriptions, side effects
       - hybrid: For medical queries that don't fit clearly into the above categories
    
    Important: If the current query is a follow-up question (like asking about symptoms after asking about a condition), 
    use the context from the previous conversation to determine if it's related to asthma or other medical conditions.
    
    Conversation History: {history_text}
    
    Current Query: {query}
    
    Respond in this exact format:
    MEDICAL: YES/NO (give me the reason for YES/NO)
    ASTHMA: YES/NO (give me the reason for YES/NO)
    CATEGORY: [research_papers/guidelines/drug_info/hybrid/non_medical] (give me the reason for the category)
    """
    # print(f"user's query: {query}")
    # print(f"History hereeeeeeeeeeeeeeeee : {history_text}")
    # print(f"Analyzing query with time: {datetime.now().strftime('%d-%m-%y %H:%M:%S')}")
    start_time = time.time()
    
    # Get response from LLM
    response = llm.invoke(prompt).strip()
    end_time = time.time()
    print(f"Total time taken for query analysis: {end_time - start_time}")
    
    # Parse the response
    is_medical = "MEDICAL: YES" in response.upper()
    is_asthma = "ASTHMA: YES" in response.upper()
    index_name = "non_medical"

    print(f"\nUser's query: {query}\n")
    print(f"Response from LLM after analysis: \n{response}")
    print(f"History in Analyze_Medical_Query: {history_text}\n\n")
    
    if is_medical:
        if is_asthma:
            # For asthma queries, use specific medical indices
            category_line = [line for line in response.split('\n') if "CATEGORY:" in line.upper()]
            if category_line:
                # Extract just the category name without any additional description
                category_text = category_line[0].split("CATEGORY:")[1].strip().lower()
                # Take only the first word (the category name) and remove any parentheses or additional text
                category = category_text.split()[0].split('(')[0]
                # Only allow specific medical indices for asthma
                # if category in ['research_papers', 'guidelines', 'drug_info']:
                #     index_name = category
                # else:
                #     index_name = 'hybrid'
                index_name = category
        else:
            # For non-asthma medical queries, use hybrid
            index_name = 'hybrid'

        print(f"Response from LLM after analysis: {response}")
        print(f"user's query: {query}")
        print(f"Is Medical: {is_medical}")
        print(f"Index Name: {index_name}")
        print(f"Is Asthma: {is_asthma}")
        print(f"History in Analyze_Medical_Query: {history}\n\n")
    
    return is_medical, index_name, is_asthma

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
            
        # print(f"--------------Start of Docs from Vector Store-------------------------------")
        start_time = time.time()
            
        # Get the appropriate vector store
        vector_store = vector_stores[index_name]
        
        # Get relevant documents
        docs = vector_store.similarity_search(query, k=3)
        
        # If no documents found in primary index, try other indices
        if not docs and index_name == 'hybrid':
            print("No results in hybrid index, trying other indices...")
            for alt_index in ['research_papers', 'guidelines', 'drug_info']:
                if alt_index in vector_stores:
                    print(f"Trying {alt_index} index...")
                    alt_docs = vector_stores[alt_index].similarity_search(query, k=3)
                    if alt_docs:
                        docs = alt_docs
                        index_name = alt_index
                        print(f"Found results in {alt_index} index")
                        break
        
        # print(docs)
        end_time = time.time()
        print(f"Total time taken for Doc Retrieval: {end_time - start_time}")
        # print(f"--------------End of Docs from Vector Store---------------------------------")
        
        # If still no documents found
        if not docs:
            response = "I apologize, but I couldn't find specific information about your query in my knowledge base. Could you please rephrase your question or provide more details?"
            conversation_logger.log_conversation(query, response, index_name, vector_stores, history)
            return response
        
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
                if isinstance(turn, dict) and 'user' in turn and 'assistant' in turn:
                    history_text += f"User: {turn['user']}\n"
                    history_text += f"Assistant: {turn['assistant']}\n"
        
        print(f"History here in Get_Relevant_Content : \n {history_text}")
        
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
        8. Strictly write the answer in 100 words or less.
        9. If the Relevant Documents content is less, give the answer in the same word ratio. But make it sure that it should not be more than 100 words.
        
        Conversation History: {history_text}
        
        Current User Query: {query}
        
        Relevant Documents:
        {chr(10).join([doc.page_content for doc in docs])}
        
        Please provide a comprehensive but concise answer to the user's query strictly within 100 words, taking into account the conversation history if relevant.
        """
        # print("----------------------------------------------Start of LLM Output----------------------------------------------")
        # print(f"LLM Final- Time before for processing the raw data: {time.time()}")
        start_time = time.time()
        # Get processed response from LLM
        response = llm.invoke(prompt)
        end_time = time.time()
        # print("LLM Generated Response: ", response)
        print(f"LLM Final- Total time taken for processing the raw data: {end_time - start_time}")
        # print("----------------------------------------------End of LLM Output----------------------------------------------\n\n")
        
        # Log the conversation
        conversation_logger.log_conversation(query, response, index_name, vector_stores, history)
        
        return response
    except Exception as e:
        print(f"Error in get_relevant_content: {str(e)}")
        error_response = "I apologize, but I encountered an error while processing your query. Please try again or rephrase your question."
        conversation_logger.log_conversation(query, error_response, index_name, vector_stores, history)
        return error_response









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
