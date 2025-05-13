# **MediMind - Medical Chatbot Using RAG**

## **Introduction**
MediMind addresses the critical challenge of misinformation in medical AI by combining the fluency of large language models with the reliability of trusted clinical sources. Motivated by the risks of hallucinations in standalone LLMs, especially in healthcare, we aim to develop a Retrieval-Augmented Generation (RAG) based chatbot that delivers concise, accurate, and context-aware responses for asthma care. By leveraging semantic search, query rewriting, and Mistral 7B, our objective is to build a clinically safe and scalable medical assistant that empowers patients without replacing medical professionals.
We aim to - 
- Minimize hallucination and misinformation in AI responses
- Maintain context-aware dialogue over multiple conversation turns
- Improve patient comprehension using user-friendly language
- Provide a scalable, modular architecture that can be extended to other medical domains

This project is an advanced medical chatbot designed to assist users by answering medical-related queries using a robust retrieval-augmented generation (RAG) framework. The chatbot leverages the **Pinecone vector database** for efficient document retrieval, **Mistralai LLM** from Hugging Face for natural language generation, and **Research Papers, Arcticles, Books, Encyclopedia for Medicine** documents as the primary knowledge base. It provides accurate, contextual, and relevant answers to medical queries in real time.

## **Features**
- **Interactive Chat Interface**: User-friendly chat interface built with HTML, CSS
- **Retrieval-Augmented Generation (RAG)**: Combines the power of document retrieval with generative AI for accurate responses
- **Pinecone Vector Database**: Ensures fast and efficient similarity searches within the knowledge base
- **Mistralai LLM**: A cutting-edge language model from Hugging Face
- **Knowledge Base**: 30+ Research paper and articles, GINA and NHILBI Guidelines and Protocols, Drug Descriptions and side effects, 3 volumes of Gale Encyclopedia of Medicine, A-Z Family Medical Encyclopedia, Medical Books, Asthma-focused dataset - for meta-routing
- **Conversation History**: Maintains conversation logs for context-aware responses
- **Evaluation Framework**: Includes comprehensive evaluation metrics and results
- **Modular Architecture**: Well-organized codebase with clear separation of concerns

## **Project Structure**
```
├── src/                    # Source code directory
├── static/                 # Static assets (CSS, JS, images)
├── templates/             # HTML templates
├── Data/                  # Data directory for medical documents
├── conversation_logs/     # Stored conversation histories
├── evaluation_results/    # Evaluation metrics and results
├── app.py                # Main Flask application
├── template.py           # Template processing utilities
└── requirements.txt      # Project dependencies
```

## **Technology Stack**
- **Backend**:
  - Python (Flask Framework)
  - Pinecone Vector Database
  - Hugging Face's Mistral LLM (In this project Mistral-7B quantized has been used)
  - LangChain for RAG implementation
- **Frontend**:
  - HTML5, CSS3 (Bootstrap Framework)
  - JavaScript (jQuery and AJAX)
- **Environment Configuration**:
  - `.env` file for API keys and sensitive configurations

## **Setup and Installation**

### Prerequisites
Ensure you have the following installed:
- Python 3.12 or later
- pip (Python package manager)
- Pinecone API key
- Hugging Face API key

### Steps
1. **Clone the Repository**
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. **Create Virtual Environment**
   ```bash
   python -m venv .venvmed
   source .venvmed/bin/activate  # On Windows: .venvmed\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set Environment Variables**
   Create a `.env` file in the root directory and add:
   ```bash
   PINECONE_API_KEY=<your-pinecone-api-key>
   HUGGINGFACEHUB_API_TOKEN=<your-huggingface-api-token>
   ```

5. **Initialize the Knowledge Base**
   - Navigate to the `src` folder:
     ```bash
     cd src
     ```
   - Process the medical documents:
     ```bash
     python helper.py
     ```
   - Create and store vector embeddings:
     ```bash
     python store_index.py
     ```

6. **Run the Flask Application**
   ```bash
   python app.py
   ```

## **Usage**
1. Open your web browser and navigate to `http://localhost:8080`
2. Type your medical query in the chat interface
3. The chatbot will process your query and provide a relevant response
4. Conversation history is maintained for context-aware responses (Till 3 epochs)

## **Evaluation**
The project includes a comprehensive evaluation framework that assesses:
- Response accuracy
- Relevance of retrieved information
- Response generation quality
- System performance metrics

Evaluation results are stored in the `evaluation_results/` directory.
Conversation logs are stored in the `conversation_logs/` directory.
