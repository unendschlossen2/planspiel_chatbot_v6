from helper.file_loader import *
from helper.load_gpu import *
from preprocessing.text_cleanup import *
from processing.chunking import *
from embeddings.embedding_generator import load_embedding_model, embed_chunks
from vector_store.vector_store_manager import create_and_populate_vector_store
from retrieval.retriever import embed_query, query_vector_store
from generation.llm_answer_generator import *
from retrieval.reranker import load_reranker_model, gap_based_rerank_and_filter

def main():
    initial_split_level = 3
    max_chars_per_chunk = 2100
    min_chars_per_chunk = 75

    embedding_model_id = "infly/inf-retriever-v1-1.5b"

    USE_RERANKER = True
    RERANKER_MODEL_ID = "BAAI/bge-reranker-large" # BAAI/bge-reranker-large # mixedbread-ai/mxbai-rerank-large-v2
    RERANKER_INITIAL_TOP_K = 15

    MIN_ABSOLUTE_RERANK_SCORE_THRESHOLD_RERANKED = 0.001 # NEUER PARAMETER: Schwellenwert für "potenziell relevant"
    MIN_CHUNKS_TO_LLM_RERANKED = 1
    MAX_CHUNKS_TO_LLM_RERANKED = 5
    MIN_CHUNKS_FOR_GAP_DETECTION_RERANKED = 4
    GAP_DETECTION_FACTOR_RERANKED = 0.25
    SMALL_EPSILON_RERANKED = 1e-5

    DEFAULT_RETRIEVAL_TOP_K = 3

    ollama_llm_model = "gemma3:4b"
    ollama_generation_options = {
        "temperature": 0.3,
        "num_predict": 512
    }

    db_persist_path = "./db/chroma_vector_db"
    db_collection_name = "topsim_gm_coll_gemma3_test"
    force_rebuild_collection = False

    try:
        processing_device_obj = load_gpu()
        processing_device_str = str(processing_device_obj)
    except RuntimeError as e:
        print(f"GPU Ladefehler: {e}. Wechsle zu CPU.")
        processing_device_str = "cpu"
    print(f"Verwende Gerät: {processing_device_str} für Modell-Operationen.")

    try:
        print(f"Lade Embedding-Modell '{embedding_model_id}' für die Sitzung...")
        loaded_embedding_model = load_embedding_model(embedding_model_id, processing_device_str)
        print("Embedding-Modell erfolgreich für die Sitzung geladen.")
    except Exception as e:
        print(f"KRITISCH: Konnte Embedding-Modell '{embedding_model_id}' nicht laden. Fortfahren nicht möglich. Fehler: {e}")
        return

    loaded_reranker_model = None
    if USE_RERANKER:
        try:
            print(f"Lade Reranker-Modell '{RERANKER_MODEL_ID}' für die Sitzung...")
            loaded_reranker_model = load_reranker_model(RERANKER_MODEL_ID, device=processing_device_str)
            if loaded_reranker_model is None:
                print(f"WARNUNG: Reranker-Modell konnte nicht geladen werden, Reranking wird deaktiviert.")
                USE_RERANKER = False
            else:
                print("Reranker-Modell erfolgreich für die Sitzung geladen.")
        except Exception as e:
            print(f"WARNUNG: Konnte Reranker-Modell '{RERANKER_MODEL_ID}' nicht laden. Reranking wird deaktiviert. Fehler: {e}")
            USE_RERANKER = False

    # ... (Rest des Datenverarbeitungs- und DB-Aufbau-Codes bleibt gleich) ...
    all_files_chunks_with_embeddings = []
    if force_rebuild_collection:
        print("Modus: Vollständige Datenverarbeitung und Neuaufbau der Vektor-Speicher-Kollektion AKTIVIERT.")
        input_dir = get_input_files_directory()
        file_list = load_markdown_directory(input_dir)

        if not file_list:
            print(f"Keine Markdown-Dateien in {input_dir} für den Neuaufbau der Kollektion gefunden.")
        else:
            print(f"Gefundene Dateien: {[file_item['source'] for file_item in file_list]}")
            print("-" * 30)
            for file_item in file_list:
                print(f"Verarbeite Datei: {file_item['source']} für den Neuaufbau...")
                normalized_content = normalize_markdown_whitespace(file_item["content"])
                if not normalized_content.strip():
                    print(f"  Datei '{file_item['source']}' ist leer. Überspringe.")
                    print("-" * 30); continue
                print(f"  Länge des normalisierten Inhalts: {len(normalized_content)}")
                print(f"  Starte Chunking für {file_item['source']}...")
                chunks = split_markdown_by_headers(
                    markdown_text=normalized_content, source_filename=file_item["source"],
                    split_level=initial_split_level, max_chars_per_chunk=max_chars_per_chunk,
                    min_chars_per_chunk=min_chars_per_chunk
                )
                print(f"  Chunking abgeschlossen. {len(chunks)} Chunks generiert.")
                if chunks:
                    print(f"  Starte Embedding-Generierung für {len(chunks)} Chunks...")
                    try:
                        chunks_with_embeddings_for_file = embed_chunks(
                            chunks_data=chunks, model_id=embedding_model_id, device=processing_device_str,
                            preloaded_model=loaded_embedding_model, normalize_embeddings=True
                        )
                        print(f"  Embedding-Generierung für {file_item['source']} abgeschlossen.")
                        all_files_chunks_with_embeddings.extend(chunks_with_embeddings_for_file)
                    except Exception as e:
                        print(f"  Fehler während der Embedding-Generierung für {file_item['source']}: {e}")
                print("-" * 30)
            print(f"Dateiverarbeitung abgeschlossen. Gesamtanzahl Chunks mit Embeddings: {len(all_files_chunks_with_embeddings)}")
    else:
        print(f"Modus: Datenverarbeitung wird übersprungen. Versuche, existierenden Vektor-Speicher zu laden: '{db_collection_name}'.")

    if force_rebuild_collection and not all_files_chunks_with_embeddings:
        print("Neuaufbau angefordert, aber keine Chunks verarbeitet. Vektor-Speicher könnte leer sein, falls erstellt.")

    print(f"Stelle Vektor-Speicher-Kollektion '{db_collection_name}' unter '{db_persist_path}' sicher...")
    try:
        data_to_populate = all_files_chunks_with_embeddings if force_rebuild_collection else []
        db_collection = create_and_populate_vector_store(
            chunks_with_embeddings=data_to_populate, db_path=db_persist_path,
            collection_name=db_collection_name, force_rebuild_collection=force_rebuild_collection
        )
        if db_collection:
            print(f"Vektor-Speicher bereit. Kollektion '{db_collection.name}' enthält {db_collection.count()} Elemente.")
        else:
            print(f"Fehler beim Einrichten/Laden der Kollektion '{db_collection_name}'. Fortfahren nicht möglich.")
            return
    except Exception as e:
        print(f"Vektor-Speicher-Fehler für '{db_collection_name}': {e}")
        return

    if db_collection and loaded_embedding_model:
        print(f"\n--- TOPSIM RAG Chatbot Bereit (Modell: {ollama_llm_model}) ---")
        if USE_RERANKER and loaded_reranker_model:
            print(f"--- Reranking AKTIVIERT mit Modell: {RERANKER_MODEL_ID} ---")
        else:
            print(f"--- Reranking DEAKTIVIERT ---")
        print("Geben Sie Ihre Anfrage ein oder 'quit' zum Beenden.")

        while True:
            try:
                user_query_text = input("\nAnfrage: ").strip()
            except KeyboardInterrupt:
                print("\nChatbot wird aufgrund einer Tastaturunterbrechung beendet...")
                break
            if not user_query_text:
                continue
            if user_query_text.lower() in ['quit', 'exit', 'beenden']:
                print("Chatbot wird beendet.")
                break

            print(f"Verarbeite Anfrage: '{user_query_text}'...")
            try:
                query_embedding_vector = embed_query(loaded_embedding_model, user_query_text)

                current_retrieval_top_k = RERANKER_INITIAL_TOP_K if USE_RERANKER and loaded_reranker_model else DEFAULT_RETRIEVAL_TOP_K

                print(f"Rufe Top-{current_retrieval_top_k} Dokumente aus dem Vektor-Speicher ab.")
                retrieved_docs_initial = query_vector_store(db_collection, query_embedding_vector, current_retrieval_top_k)

                final_docs_for_llm = []
                if USE_RERANKER and loaded_reranker_model:
                    if retrieved_docs_initial:
                        reranked_and_filtered_docs = gap_based_rerank_and_filter(
                            user_query=user_query_text,
                            initial_retrieved_docs=retrieved_docs_initial,
                            reranker_model=loaded_reranker_model,
                            min_absolute_rerank_score_threshold=MIN_ABSOLUTE_RERANK_SCORE_THRESHOLD_RERANKED, # HIER ÜBERGEBEN
                            min_chunks_to_llm=MIN_CHUNKS_TO_LLM_RERANKED,
                            max_chunks_to_llm=MAX_CHUNKS_TO_LLM_RERANKED,
                            min_chunks_for_gap_detection=MIN_CHUNKS_FOR_GAP_DETECTION_RERANKED,
                            gap_detection_factor=GAP_DETECTION_FACTOR_RERANKED,
                            small_epsilon=SMALL_EPSILON_RERANKED
                        )
                        final_docs_for_llm = reranked_and_filtered_docs
                    else:
                        print("Keine Dokumente vom initialen Retrieval für Reranking erhalten.")
                        final_docs_for_llm = []
                else:
                    final_docs_for_llm = retrieved_docs_initial

                if not final_docs_for_llm:
                    print("Konnte keine relevanten Dokumente im Handbuch für Ihre Anfrage finden (nach ggf. Reranking).")
                    # Optional hier eine Antwort geben, z.B.:
                    # print("\nAssistent: Ich konnte leider keine passenden Informationen zu Ihrer Anfrage finden.")
                    # continue

                print(f"Generiere Antwort mit {ollama_llm_model} basierend auf {len(final_docs_for_llm)} abgerufenen/gefilterten Chunk(s)...")

                llm_answer = generate_llm_answer(
                    user_query_text,
                    final_docs_for_llm,
                    ollama_model_name=ollama_llm_model,
                    ollama_options=ollama_generation_options
                )

                print(f"\nAssistent: {llm_answer}")

            except Exception as e:
                print(f"Fehler während der Anfrageverarbeitung oder LLM-Generierung: {e}")
                print("Assistent: Entschuldigung, bei der Bearbeitung Ihrer Anfrage ist ein Fehler aufgetreten.")
    else:
        print("Chatbot kann nicht gestartet werden: DB-Kollektion oder Embedding-Modell nicht verfügbar.")

    print("Ausführung des Hauptskripts beendet.")

if __name__ == "__main__":
    main()