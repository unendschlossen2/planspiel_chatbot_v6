import logging
from typing import List, Dict, Any, Optional
import chromadb
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

def embed_query(
        model: SentenceTransformer,
        query_text: str,
        query_prefix: str = "query: ",
        normalize_embeddings: bool = True
) -> np.ndarray:
    if not query_text.strip():
        logger.warning("Received empty query text. Returning zero embedding.")
        # Return a zero vector of the expected dimensionality if known, or raise error
        # For simplicity, let's assume model.encode handles empty string to some default vector
        # or we can get model's embedding dimension: model.get_sentence_embedding_dimension()
        # For now, rely on model.encode's behavior with potentially prefixed empty string.
        pass # Fall through to encode prefixed empty string.

    prefixed_query = f"{query_prefix}{query_text}"
    logger.debug(f"Prefixed query for embedding: '{prefixed_query}'")

    try:
        query_embedding = model.encode(
            prefixed_query,
            normalize_embeddings=normalize_embeddings,
            convert_to_numpy=True
        )
        return query_embedding
    except Exception as e:
        logger.error(f"Error embedding query '{query_text}': {e}", exc_info=True)
        raise

def query_vector_store(
        collection: chromadb.Collection,
        query_embedding: np.ndarray,
        top_k: int,
        filter_metadata: Optional[Dict[str, str]] = None # Optional metadata filter
) -> List[Dict[str, Any]]:
    if query_embedding is None or query_embedding.size == 0:
        logger.warning("Received empty query embedding. Returning no results.")
        return []
    if top_k <= 0:
        logger.warning(f"top_k value {top_k} is not valid. Returning no results.")
        return []

    logger.info(f"Querying collection '{collection.name}' for top {top_k} results.")

    try:
        # ChromaDB's query_embeddings expects a list of lists/arrays
        results = collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=min(top_k, collection.count()), # Don't ask for more than what's in the collection
            include=['documents', 'metadatas', 'distances'],
            where=filter_metadata # Pass the metadata filter if provided
        )
    except Exception as e:
        logger.error(f"Error querying ChromaDB collection '{collection.name}': {e}", exc_info=True)
        return []

    retrieved_chunks = []
    if not results or not results.get('ids') or not results['ids'][0]: # results['ids'] is a list containing one list of IDs for our single query
        logger.info("No results found in ChromaDB for the query.")
        return []

    ids_list = results['ids'][0]
    documents_list = results['documents'][0] if results['documents'] else [None] * len(ids_list)
    metadatas_list = results['metadatas'][0] if results['metadatas'] else [None] * len(ids_list)
    distances_list = results['distances'][0] if results['distances'] else [None] * len(ids_list)

    for i, item_id in enumerate(ids_list):
        retrieved_chunks.append({
            "id": item_id,
            "document": documents_list[i] if documents_list else None,
            "metadata": metadatas_list[i] if metadatas_list else None,
            "distance": distances_list[i] if distances_list else None,
        })

    logger.info(f"Retrieved {len(retrieved_chunks)} chunks from vector store.")
    return retrieved_chunks