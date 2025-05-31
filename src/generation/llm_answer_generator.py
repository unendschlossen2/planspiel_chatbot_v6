import logging
from typing import List, Dict, Any, Optional
import ollama

logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False

def format_retrieved_context(retrieved_chunks: List[Dict[str, Any]]) -> str:
    if not retrieved_chunks:
        return "No relevant context was retrieved from the handbook."

    context_str = "Retrieved Context Snippets from TOPSIM Handbook:\n"
    for i, chunk in enumerate(retrieved_chunks):
        context_str += f"---\nSnippet {i+1}:\n"
        metadata = chunk.get('metadata', {})
        document_content = chunk.get('document', "N/A")

        source = metadata.get('source_filename', 'Unknown Source')
        header = metadata.get('header_text', 'N/A')
        original_base = metadata.get('original_base_header', 'N/A')

        context_str += f"  Source File: {source}\n"
        context_str += f"  Original Section: {original_base}\n"
        context_str += f"  Specific Header: {header}\n"
        context_str += f"  Content: {document_content}\n"
    context_str += "---\n"
    return context_str

def generate_llm_answer(
        user_query: str,
        retrieved_chunks: List[Dict[str, Any]],
        ollama_model_name: str,
        ollama_host: Optional[str] = None,
        ollama_options: Optional[Dict[str, Any]] = None,
        prompt_template_str: Optional[str] = None
) -> str:

    formatted_context = format_retrieved_context(retrieved_chunks)

    if prompt_template_str:
        current_prompt_template = prompt_template_str
    else:
        current_prompt_template = """You are an expert assistant for the TOPSIM business game, providing answers based *only* on the provided context snippets from the TOPSIM Handbook.
If the context does not contain the information to answer the question, you must clearly state that the information is not found in the provided snippets.
Do not make up information or use external knowledge.
If possible, you can subtly refer to the section from which the information was drawn if it helps clarity.
ALWAYS answer in German, even if the question is in English.

Provided Context from TOPSIM Handbook:
{context}

User's Question:
{query}

Answer:"""

    full_prompt = current_prompt_template.format(context=formatted_context, query=user_query)
    logger.debug(f"Full prompt being sent to Ollama model '{ollama_model_name}':\n{full_prompt}")

    try:
        client_args = {}
        if ollama_host:
            client_args['host'] = ollama_host

        client = ollama.Client(**client_args)

        logger.info(f"Sending request to Ollama model '{ollama_model_name}'...")
        response = client.chat(
            model=ollama_model_name,
            messages=[{'role': 'user', 'content': full_prompt}],
            stream=False,
            options=ollama_options
        )
        llm_answer = response['message']['content'].strip()
        logger.info(f"Received answer from Ollama model '{ollama_model_name}'.")
        return llm_answer

    except ollama.ResponseError as e:
        logger.error(f"Ollama API Response Error for model '{ollama_model_name}': {e.status_code} - {e.error}")
        if e.status_code == 404 or (e.error and "model not found" in e.error.lower()):
            return f"Error: Ollama model '{ollama_model_name}' not found. Please ensure it's pulled via 'ollama pull {ollama_model_name}'. (Details: {e.error})"
        return f"Error: API error from Ollama model '{ollama_model_name}'. (Details: {e.error})"
    except Exception as e:
        logger.error(f"Error interacting with Ollama model '{ollama_model_name}': {e}", exc_info=True)
        if "connection refused" in str(e).lower() or "failed to connect" in str(e).lower() or "Max retries exceeded with url" in str(e): # Added Max retries
            return f"Error: Could not connect to Ollama. Please ensure Ollama is running (often at http://localhost:11434 and the model is served). (Details: {e})"
        return f"Error: Could not generate an answer from Ollama model '{ollama_model_name}'. (Details: {e})"