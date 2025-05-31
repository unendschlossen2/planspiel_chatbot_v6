# src/embeddings/embedding_generator.py
import logging
from typing import List, Dict, Any, Optional # Added Optional
from sentence_transformers import SentenceTransformer
import numpy as np

logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False

def load_embedding_model(model_id: str, device: str) -> SentenceTransformer:
    logger.info(f"Loading sentence embedding model '{model_id}' onto device '{device}'.")
    try:
        model = SentenceTransformer(model_id, device=device) # Corrected this line from previous potential error
        logger.info(f"Successfully loaded model '{model_id}'.")
        return model
    except Exception as e:
        logger.error(f"Error loading sentence embedding model '{model_id}': {e}")
        raise

def generate_embeddings_for_corpus(
        model: SentenceTransformer,
        text_corpus: List[str],
        batch_size: int = 32,
        show_progress_bar: bool = True,
        normalize_embeddings: bool = True
) -> np.ndarray:
    if not text_corpus:
        logger.warning("Text corpus is empty. Returning empty NumPy array for embeddings.")
        return np.array([])

    logger.info(f"Generating embeddings for {len(text_corpus)} documents (batch_size: {batch_size}, normalize: {normalize_embeddings})...")
    try:
        embeddings = model.encode(
            text_corpus,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            normalize_embeddings=normalize_embeddings,
            convert_to_numpy=True
        )
        logger.info(f"Successfully generated {len(embeddings)} embeddings.")
        return embeddings
    except Exception as e:
        logger.error(f"Error during embedding generation: {e}")
        raise

def embed_chunks(
        chunks_data: List[Dict[str, Any]],
        model_id: str, # Still useful for logging or if preloaded_model is None
        device: str,   # Still useful if preloaded_model is None
        batch_size: int = 32,
        normalize_embeddings: bool = True,
        preloaded_model: Optional[SentenceTransformer] = None # New parameter
) -> List[Dict[str, Any]]:
    if not chunks_data:
        logger.warning("Received an empty list of chunks. No embeddings to generate.")
        return []

    model_to_use: SentenceTransformer
    if preloaded_model:
        logger.info(f"Using pre-loaded embedding model for {len(chunks_data)} chunks.")
        model_to_use = preloaded_model
    else:
        logger.info(f"No pre-loaded model provided, loading model '{model_id}' for {len(chunks_data)} chunks.")
        model_to_use = load_embedding_model(model_id=model_id, device=device)

    texts_to_embed_map = []
    for i, chunk in enumerate(chunks_data):
        content = chunk.get("content", "")
        if content.strip():
            texts_to_embed_map.append((i, content))
        else:
            chunk['embedding'] = np.array([])

    if not texts_to_embed_map:
        logger.warning("All chunk contents are empty or whitespace after initial check. No embeddings generated beyond empty arrays.")
        return chunks_data

    original_indices = [item[0] for item in texts_to_embed_map]
    actual_texts_to_embed = [item[1] for item in texts_to_embed_map]

    embeddings_array = generate_embeddings_for_corpus(
        model=model_to_use,
        text_corpus=actual_texts_to_embed,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=normalize_embeddings
    )

    for i, original_idx in enumerate(original_indices):
        if i < len(embeddings_array):
            chunks_data[original_idx]['embedding'] = embeddings_array[i]
        else:
            logger.error(f"Mismatch in embedding count for chunk index {original_idx}. Assigning empty embedding.")
            chunks_data[original_idx]['embedding'] = np.array([])

    logger.info("Finished adding embeddings to chunk data.")
    return chunks_data