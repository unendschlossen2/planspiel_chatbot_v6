from typing import List, Dict, Any, Optional
import ollama

def format_retrieved_context(retrieved_chunks: List[Dict[str, Any]]) -> str:
    if not retrieved_chunks:
        return "Kein relevanter Kontext wurde aus dem Handbuch abgerufen."

    context_str = "Abgerufene Kontext-Schnipsel aus dem TOPSIM Handbuch:\n"
    for i, chunk in enumerate(retrieved_chunks):
        context_str += f"---\nSchnipsel {i+1}:\n"
        metadata = chunk.get('metadata', {})
        document_content = chunk.get('document', "N/A")

        source = metadata.get('source_filename', 'Unbekannte Quelle')
        header = metadata.get('header_text', 'N/A')
        original_base = metadata.get('original_base_header', 'N/A')

        context_str += f"  Quelldatei: {source}\n"
        context_str += f"  Ursprünglicher Abschnitt: {original_base}\n"
        context_str += f"  Spezifische Überschrift: {header}\n"
        context_str += f"  Inhalt: {document_content}\n"
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
        current_prompt_template = """Sie sind ein Experte für das Planspiel TOPSIM und geben Antworten, die *ausschließlich* auf den bereitgestellten Kontext-Schnipseln aus dem TOPSIM Handbuch basieren.
Wenn der Kontext die Informationen zur Beantwortung der Frage nicht enthält, müssen Sie klar angeben, dass die Informationen in den bereitgestellten Schnipseln nicht gefunden wurden.
Erfinden Sie keine Informationen und verwenden Sie kein externes Wissen, außer Allgemeinwissen und unabhängige Fakten.
Wenn möglich, können Sie auf den Abschnitt (nicht SCHNIPSEL!) verweisen, aus dem die Informationen stammen, falls dies die Klarheit erhöht.
ANTWORTEN SIE IMMER AUF DEUTSCH, auch wenn die Frage auf Englisch ist.

Bereitgestellter Kontext aus dem TOPSIM Handbuch:
{context}

Benutzerfrage:
{query}

Antwort:"""

    full_prompt = current_prompt_template.format(context=formatted_context, query=user_query)
    print(f"Vollständiger Prompt an Ollama-Modell '{ollama_model_name}':\n{full_prompt}") # Optional for debugging

    try:
        client_args = {}
        if ollama_host:
            client_args['host'] = ollama_host

        client = ollama.Client(**client_args)

        print(f"Sende Anfrage an Ollama-Modell '{ollama_model_name}'...")
        response = client.chat(
            model=ollama_model_name,
            messages=[{'role': 'user', 'content': full_prompt}],
            stream=False,
            options=ollama_options
        )
        llm_answer = response['message']['content'].strip()
        print(f"Antwort von Ollama-Modell '{ollama_model_name}' empfangen.")
        return llm_answer

    except ollama.ResponseError as e:
        error_message = f"Ollama API Antwortfehler für Modell '{ollama_model_name}': {e.status_code} - {e.error}"
        print(error_message)
        if e.status_code == 404 or (e.error and "model not found" in e.error.lower()):
            return f"Fehler: Ollama-Modell '{ollama_model_name}' nicht gefunden. Bitte stellen Sie sicher, dass es über 'ollama pull {ollama_model_name}' heruntergeladen wurde. (Details: {e.error})"
        return f"Fehler: API-Fehler vom Ollama-Modell '{ollama_model_name}'. (Details: {e.error})"
    except Exception as e:
        error_message = f"Fehler bei der Interaktion mit dem Ollama-Modell '{ollama_model_name}': {e}"
        print(error_message)
        if "connection refused" in str(e).lower() or "failed to connect" in str(e).lower() or "Max retries exceeded with url" in str(e):
            return f"Fehler: Konnte keine Verbindung zu Ollama herstellen. Bitte stellen Sie sicher, dass Ollama läuft (oft unter http://localhost:11434 und das Modell bereitgestellt wird). (Details: {e})"
        return f"Fehler: Konnte keine Antwort vom Ollama-Modell '{ollama_model_name}' generieren. (Details: {e})"