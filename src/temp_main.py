import numpy as np
from helper.file_loader import *
from helper.load_gpu import *
# from helper.huggingface_login import attempt_huggingface_login
from preprocessing.text_cleanup import *
from processing.chunking import *
from embeddings.embedding_generator import load_embedding_model, embed_chunks
from vector_store.vector_store_manager import create_and_populate_vector_store
from retrieval.retriever import embed_query, query_vector_store
from generation.llm_answer_generator import * # Added import

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def main():
    # --- Parameters for Script Execution ---
    initial_split_level = 3
    max_chars_per_chunk = 1800
    min_chars_per_chunk = 75
    enable_verbose_logging = False # Set True for DEBUG logs from modules & more main script printouts

    embedding_model_id = "infly/inf-retriever-v1-1.5b"

    # --- Ollama Configuration ---
    ollama_llm_model = "gemma3:4b" # Default testing model from your list
    # ollama_llm_model = "mistral:7b-instruct" # Another good option for 12GB VRAM
    # ollama_llm_model = "gemma3:12b-it-q4_K_M" # For 24GB VRAM
    # ollama_llm_model = "mixtral:8x7b-instruct" # Powerful option for 24GB VRAM

    ollama_generation_options = {
        "temperature": 0.3, # Lower for more factual/grounded RAG responses
        "num_predict": 512  # Max number of tokens to generate for the answer
        # Add other Ollama options here if needed, e.g., "top_k", "top_p"
    }

    # --- Vector Store Configuration ---
    db_persist_path = "./db/chroma_vector_db"
    db_collection_name = "topsim_gm_coll_gemma3_test" # Unique name for this setup

    force_rebuild_collection = False # Set False to use existing DB for retrieval testing

    # huggingface_token = "YOUR_TOKEN"
    # if huggingface_token == "YOUR_TOKEN": huggingface_token = None
    # if huggingface_token: attempt_huggingface_login(huggingface_token)

    try:
        processing_device_obj = load_gpu()
        processing_device_str = str(processing_device_obj)
    except RuntimeError as e:
        logging.error(f"GPU loading error: {e}. Defaulting to CPU.")
        processing_device_str = "cpu"
    logging.info(f"Using device: {processing_device_str} for embedding model operations.")

    loaded_embedding_model = None
    try:
        logging.info(f"Loading embedding model '{embedding_model_id}' for the session...")
        loaded_embedding_model = load_embedding_model(embedding_model_id, processing_device_str)
        logging.info("Embedding model loaded successfully for the session.")
    except Exception as e:
        logging.error(f"CRITICAL: Could not load embedding model '{embedding_model_id}'. Cannot proceed. Error: {e}", exc_info=enable_verbose_logging)
        return

    all_files_chunks_with_embeddings = []
    if force_rebuild_collection:
        logging.info("Mode: Full data processing and vector store collection rebuild ENABLED.")
        input_dir = get_input_files_directory()
        file_list = load_markdown_directory(input_dir)

        if not file_list:
            logging.warning(f"No Markdown files found in {input_dir} for collection rebuild.")
        else:
            logging.info(f"Found files: {[file_item['source'] for file_item in file_list]}")
            print("-" * 30)
            for file_item in file_list:
                logging.info(f"Processing file: {file_item['source']} for rebuild...")
                normalized_content = normalize_markdown_whitespace(file_item["content"])
                if not normalized_content.strip():
                    logging.info(f"  File '{file_item['source']}' is empty. Skipping.")
                    print("-" * 30); continue
                logging.info(f"  Normalized content length: {len(normalized_content)}")
                logging.info(f"  Starting chunking for {file_item['source']}...")
                chunks = split_markdown_by_headers(
                    markdown_text=normalized_content, source_filename=file_item["source"],
                    split_level=initial_split_level, max_chars_per_chunk=max_chars_per_chunk,
                    min_chars_per_chunk=min_chars_per_chunk, verbose=enable_verbose_logging
                )
                logging.info(f"  Chunking complete. Generated {len(chunks)} chunks.")
                if chunks:
                    logging.info(f"  Starting embedding generation for {len(chunks)} chunks...")
                    try:
                        chunks_with_embeddings_for_file = embed_chunks(
                            chunks_data=chunks, model_id=embedding_model_id, device=processing_device_str,
                            preloaded_model=loaded_embedding_model, normalize_embeddings=True
                        )
                        logging.info(f"  Embedding complete for {file_item['source']}.")
                        all_files_chunks_with_embeddings.extend(chunks_with_embeddings_for_file)
                        if enable_verbose_logging and chunks_with_embeddings_for_file:
                            logging.info(f"    First {min(1, len(chunks_with_embeddings_for_file))} embedding shapes: {[c.get('embedding', np.array([])).shape for c in chunks_with_embeddings_for_file[:1]]}")
                    except Exception as e:
                        logging.error(f"  Error during embedding for {file_item['source']}: {e}", exc_info=enable_verbose_logging)
                print("-" * 30)
            logging.info(f"File processing done. Total chunks with embeddings: {len(all_files_chunks_with_embeddings)}")
    else:
        logging.info(f"Mode: Skipping data processing. Will attempt to load existing vector store: '{db_collection_name}'.")

    if force_rebuild_collection and not all_files_chunks_with_embeddings:
        logging.warning("Rebuild requested, but no chunks processed. Vector store might be empty if created.")

    db_collection = None
    logging.info(f"Ensuring vector store collection '{db_collection_name}' at '{db_persist_path}'...")
    try:
        data_to_populate = all_files_chunks_with_embeddings if force_rebuild_collection else []
        db_collection = create_and_populate_vector_store(
            chunks_with_embeddings=data_to_populate, db_path=db_persist_path,
            collection_name=db_collection_name, force_rebuild_collection=force_rebuild_collection
        )
        if db_collection:
            logging.info(f"Vector store ready. Collection '{db_collection.name}' contains {db_collection.count()} items.")
        else:
            logging.error(f"Failed to setup/load collection '{db_collection_name}'. Cannot proceed.")
            return
    except Exception as e:
        logging.error(f"Vector store error for '{db_collection_name}': {e}", exc_info=enable_verbose_logging)
        return

    if db_collection and loaded_embedding_model:
        logging.info(f"\n--- TOPSIM RAG Chatbot Ready (Model: {ollama_llm_model}) ---")
        logging.info("Type your query or 'quit' to exit.")

        while True:
            try:
                user_query_text = input("\nQuery: ").strip()
            except KeyboardInterrupt:
                logging.info("\nExiting chatbot due to keyboard interrupt...")
                break
            if not user_query_text:
                continue
            if user_query_text.lower() in ['quit', 'exit']:
                logging.info("Exiting chatbot.")
                break

            top_k_for_retrieval = 3

            logging.info(f"Processing query: '{user_query_text}'...")
            try:
                query_embedding_vector = embed_query(loaded_embedding_model, user_query_text)
                retrieved_docs = query_vector_store(db_collection, query_embedding_vector, top_k_for_retrieval)

                if not retrieved_docs:
                    logging.info("Could not find relevant documents in the handbook for your query.")

                logging.info(f"Generating answer using {ollama_llm_model} based on {len(retrieved_docs)} retrieved chunk(s)...")

                final_answer = generate_llm_answer(
                    user_query_text,
                    retrieved_docs,
                    ollama_model_name=ollama_llm_model,
                    ollama_options=ollama_generation_options
                )

                print(f"\nAssistant: {final_answer}")

            except Exception as e:
                logging.error(f"Error during query processing or LLM generation: {e}", exc_info=enable_verbose_logging)
                print("Assistant: Sorry, I encountered an error trying to process your request.")
    else:
        logging.error("Chatbot cannot start: DB collection or embedding model not available.")

    logging.info("Main script execution finished.")

if __name__ == "__main__":
    main()