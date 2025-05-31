import logging
from typing import List, Dict, Any
import chromadb
import chromadb.errors # For specific errors like DuplicateCollectionError
import numpy as np
import re

logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False

def _sanitize_id_part(part: str) -> str:
    part = str(part)
    part = re.sub(r'\s+', '_', part)
    part = re.sub(r'[^a-zA-Z0-9_.-]+', '', part)
    return part[:60]

def _prepare_data_for_chroma(
        chunks_with_embeddings: List[Dict[str, Any]]
) -> tuple[List[str], List[list | np.ndarray], List[Dict[str, Any]], List[str]]:
    ids_list: List[str] = []
    embeddings_list: List[list | np.ndarray] = []
    metadatas_list: List[Dict[str, Any]] = []
    documents_list: List[str] = []
    processed_ids = set()
    valid_chunks_processed_count = 0

    for i, chunk in enumerate(chunks_with_embeddings):
        embedding_vector = chunk.get('embedding')
        content_text = chunk.get('content', "")

        is_embedding_valid = False
        if isinstance(embedding_vector, np.ndarray) and embedding_vector.size > 0:
            is_embedding_valid = True
        elif isinstance(embedding_vector, list) and len(embedding_vector) > 0:
            is_embedding_valid = True

        if not is_embedding_valid:
            logger.debug(
                f"Chunk (index {i}) from '{chunk.get('source_filename', 'sfn')}' "
                f"titled '{chunk.get('header_text', 'N/A')[:30]}...' has no valid embedding. Skipping."
            )
            continue

        if not content_text.strip() and not chunk.get('header_text', '').strip():
            logger.debug(
                f"Chunk (index {i}) from '{chunk.get('source_filename', 'sfn')}' "
                f"has no content and no header. Skipping."
            )
            continue

        sfn_part = _sanitize_id_part(chunk.get('source_filename', f'sfn_unknown'))
        obh_part = _sanitize_id_part(chunk.get('original_base_header', f'obh_unknown'))
        ht_part = _sanitize_id_part(chunk.get('header_text', f'ht_unknown'))

        base_id = f"{sfn_part}_{obh_part}_{ht_part}_{valid_chunks_processed_count}"
        unique_id = base_id
        id_suffix_counter = 0
        while unique_id in processed_ids:
            id_suffix_counter += 1
            unique_id = f"{base_id}_dup{id_suffix_counter}"

        processed_ids.add(unique_id)
        ids_list.append(unique_id)

        if isinstance(embedding_vector, np.ndarray):
            embeddings_list.append(embedding_vector.tolist())
        else:
            embeddings_list.append(embedding_vector)

        documents_list.append(content_text)

        metadata_item = {
            key: value for key, value in chunk.items()
            if key not in ['content', 'embedding'] and isinstance(value, (str, int, float, bool))
        }
        metadatas_list.append(metadata_item)
        valid_chunks_processed_count +=1

    if valid_chunks_processed_count == 0 and len(chunks_with_embeddings) > 0:
        logger.warning("No valid chunks with embeddings found to prepare for ChromaDB.")
        return [], [], [], []

    return ids_list, embeddings_list, metadatas_list, documents_list

def create_and_populate_vector_store(
        chunks_with_embeddings: List[Dict[str, Any]],
        db_path: str,
        collection_name: str,
        force_rebuild_collection: bool = True
) -> chromadb.Collection | None:

    logger.info(f"Initializing ChromaDB PersistentClient at path: {db_path}")
    try:
        client = chromadb.PersistentClient(path=db_path)
        existing_collections_objects = client.list_collections()
        existing_collection_names = [col.name for col in existing_collections_objects]
        logger.info(f"Available collections in DB at '{db_path}': {existing_collection_names}")
    except Exception as e:
        logger.error(f"Failed to initialize ChromaDB client or list collections at {db_path}: {e}", exc_info=True)
        return None

    collection: chromadb.Collection | None = None

    if force_rebuild_collection:
        logger.info(f"Force_rebuild_collection is True for collection '{collection_name}'. Ensuring a fresh build.")
        if collection_name in existing_collection_names:
            try:
                logger.info(f"Collection '{collection_name}' exists. Deleting it for rebuild.")
                client.delete_collection(name=collection_name)
                logger.info(f"Collection '{collection_name}' deleted successfully.")
            except Exception as e_del:
                logger.error(f"Failed to delete existing collection '{collection_name}': {e_del}. Aborting rebuild to prevent data inconsistency.", exc_info=True)
                return None # If delete fails, we can't guarantee a fresh collection
        else:
            logger.info(f"Collection '{collection_name}' does not exist. No deletion needed.")

        try:
            collection = client.create_collection(name=collection_name, metadata={"hnsw:space": "cosine"})
            logger.info(f"Successfully created new empty collection '{collection_name}'.")

        except Exception as e_create:
            logger.error(f"Generic error during creation of collection '{collection_name}' for rebuild: {e_create}", exc_info=True)
            return None

        if not chunks_with_embeddings and collection is not None:
            logger.warning("Rebuild is True, but no chunks provided to populate. Collection will be empty.")
            return collection

        ids, embeddings, metadatas, documents = _prepare_data_for_chroma(chunks_with_embeddings)
        if not ids:
            logger.warning(f"No valid data prepared from chunks. Collection '{collection_name}' will not be populated in this run.")
            return collection

        try:
            logger.info(f"Adding {len(ids)} items to newly created ChromaDB collection '{collection_name}'...")
            collection.add(ids=ids, embeddings=embeddings, metadatas=metadatas, documents=documents)
            logger.info(f"Successfully added {collection.count()} items to collection '{collection_name}'.")
        except Exception as e_add:
            logger.error(f"Failed to add data to collection '{collection_name}' during rebuild: {e_add}", exc_info=True)
            return collection

    else:
        logger.info(f"Force_rebuild_collection is False. Attempting to load existing collection '{collection_name}'.")
        if collection_name in existing_collection_names:
            try:
                collection = client.get_collection(name=collection_name)
                logger.info(f"Successfully loaded existing collection '{collection_name}' with {collection.count()} items. Data will not be re-added.")
            except Exception as e_get:
                logger.error(f"An error occurred while trying to get existing collection '{collection_name}': {e_get}", exc_info=True)
                return None
        else:
            logger.error(f"Collection '{collection_name}' does not exist. "
                         f"Please run with force_rebuild_collection=True to create and populate it first.")
            return None