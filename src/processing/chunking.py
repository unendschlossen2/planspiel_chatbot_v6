import re
from typing import List, Dict, Any
import logging

DEFAULT_MAX_CHARS_PER_CHUNK = 2000
DEFAULT_MIN_CHARS_PER_CHUNK = 50
CHUNK_HEADER_OVERVIEW_SUFFIX = " (Overview)"
MERGED_HEADER_SEPARATOR = " & "

logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False

def _create_chunk_dict(
        header_text: str,
        header_level: int,
        content: str,
        source_filename: str,
        original_base_header: str
) -> Dict[str, Any]:
    return {
        "header_text": header_text,
        "header_level": header_level,
        "content": content.strip(),
        "source_filename": source_filename,
        "original_base_header": original_base_header
    }

def filter_missing_content_lines(text: str) -> str:
    lines = text.splitlines()
    filtered_lines = [
        line for line in lines
        if not (line.strip().startswith(">") and "fehlt" in line.lower())
    ]
    return "\n".join(filtered_lines)

def _find_all_headers_in_text(markdown_text: str) -> List[Dict[str, Any]]:
    headers = []
    header_regex = re.compile(r"^(#{1,6})\s+(.*)", re.MULTILINE)
    for match in header_regex.finditer(markdown_text):
        level = len(match.group(1))
        title = match.group(2).strip()
        header_line_start_idx = match.start()
        header_line_end_idx = match.end()

        content_start_idx = header_line_end_idx
        if header_line_end_idx < len(markdown_text) and markdown_text[header_line_end_idx] == '\n':
            content_start_idx = header_line_end_idx + 1
        elif header_line_end_idx + 1 < len(markdown_text) and markdown_text[header_line_end_idx:header_line_end_idx+2] == '\r\n':
            content_start_idx = header_line_end_idx + 2

        headers.append({
            "level": level,
            "title": title,
            "header_start_idx": header_line_start_idx,
            "content_start_idx": content_start_idx
        })
    return headers

def _split_by_found_headers(
        content_to_split: str, sub_headers: List[Dict[str, Any]],
        parent_chunk_dict: Dict[str, Any]
) -> List[Dict[str, Any]]:
    newly_created_chunks = []
    parent_header_text = parent_chunk_dict.get("header_text", "")
    parent_header_level = parent_chunk_dict.get("header_level", 0)
    original_base_header = parent_chunk_dict.get("original_base_header", "")
    source_filename = parent_chunk_dict.get("source_filename", "unknown")

    current_pos = 0

    if sub_headers:
        first_sub_header_starts_at = sub_headers[0]['header_start_idx']
        if first_sub_header_starts_at > 0:
            intro_content_raw = content_to_split[current_pos:first_sub_header_starts_at]
            intro_content_filtered = filter_missing_content_lines(intro_content_raw)
            if intro_content_filtered.strip():
                newly_created_chunks.append(_create_chunk_dict(
                    header_text=f"{parent_header_text}{CHUNK_HEADER_OVERVIEW_SUFFIX}",
                    header_level=parent_header_level,
                    content=intro_content_filtered,
                    source_filename=source_filename,
                    original_base_header=original_base_header
                ))

    for i, header_data in enumerate(sub_headers):
        section_content_start_in_parent = header_data['content_start_idx']

        if (i + 1) < len(sub_headers):
            section_content_end_in_parent = sub_headers[i+1]['header_start_idx']
        else:
            section_content_end_in_parent = len(content_to_split)

        raw_section_content = content_to_split[section_content_start_in_parent:section_content_end_in_parent]
        filtered_section_content = filter_missing_content_lines(raw_section_content)

        if filtered_section_content.strip() or header_data['title']:
            newly_created_chunks.append(_create_chunk_dict(
                header_text=header_data['title'],
                header_level=header_data['level'],
                content=filtered_section_content,
                source_filename=source_filename,
                original_base_header=original_base_header
            ))
    return newly_created_chunks


def _handle_oversized_chunk(
        oversized_chunk: Dict[str, Any]
) -> List[Dict[str, Any]]:
    content = oversized_chunk["content"]
    parent_level = oversized_chunk["header_level"]
    logger.info(f"    Handling oversized: '{oversized_chunk['header_text']}' (len {len(content)}), level H{parent_level}.")

    headers_in_content = _find_all_headers_in_text(content)

    sub_headers_to_split_by = []
    start_level_for_sub_split = parent_level + 1 if parent_level > 0 else 1
    for level_to_try in range(start_level_for_sub_split, 7):
        found_headers_at_level = [h for h in headers_in_content if h['level'] == level_to_try]
        if found_headers_at_level:
            sub_headers_to_split_by = found_headers_at_level
            logger.debug(f"        DynamicSplit: Found {len(sub_headers_to_split_by)} H{level_to_try} headers within '{oversized_chunk['header_text']}'.")
            break

    if sub_headers_to_split_by:
        split_chunks = _split_by_found_headers(content, sub_headers_to_split_by, oversized_chunk)
        if split_chunks and not (len(split_chunks) == 1 and split_chunks[0]["content"].strip() == content.strip()):
            logger.debug(f"        DynamicSplit: Successfully split '{oversized_chunk['header_text']}' into {len(split_chunks)} sub-chunks by H{sub_headers_to_split_by[0]['level']}.")
            return split_chunks
        else:
            header_level_log = sub_headers_to_split_by[0].get('level', 'unknown') if sub_headers_to_split_by else 'unknown'
            logger.debug(f"        DynamicSplit: Splitting by H{header_level_log} for '{oversized_chunk['header_text']}' did not yield smaller/multiple distinct chunks. Will not split further by headers.")
            return [oversized_chunk]
    else:
        logger.debug(f"        DynamicSplit: No suitable deeper headers found in '{oversized_chunk['header_text']}'. Will not split further by headers.")
        return [oversized_chunk]

def _initial_split(
        markdown_text: str, source_filename: str, split_level: int
) -> List[Dict[str, Any]]:
    logger.info(f"Initial split for '{source_filename}' by H{split_level} headers.")

    all_doc_headers = _find_all_headers_in_text(markdown_text)
    initial_chunks = []

    target_level_headers = [h for h in all_doc_headers if h['level'] == split_level]

    if not target_level_headers:
        logger.warning(f"No H{split_level} headers found in '{source_filename}'. Treating entire document as one initial chunk.")
        cleaned_full_content = filter_missing_content_lines(markdown_text)
        if cleaned_full_content.strip() or (not markdown_text.strip() and cleaned_full_content == ""):
            initial_chunks.append(_create_chunk_dict(
                header_text=f"{source_filename} (Full Document)",
                header_level=0,
                content=cleaned_full_content,
                source_filename=source_filename,
                original_base_header=f"{source_filename} (Full Document)"
            ))
        return initial_chunks

    for i, header_data in enumerate(target_level_headers):
        chunk_title = header_data['title']
        chunk_level = header_data['level']
        content_start = header_data['content_start_idx']
        content_end = len(markdown_text)

        for next_h_data in all_doc_headers:
            if next_h_data['header_start_idx'] > header_data['header_start_idx'] and next_h_data['level'] <= split_level:
                content_end = next_h_data['header_start_idx']
                break

        raw_chunk_content = markdown_text[content_start:content_end]
        cleaned_content = filter_missing_content_lines(raw_chunk_content)

        if cleaned_content.strip() or chunk_title:
            initial_chunks.append(_create_chunk_dict(
                header_text=chunk_title,
                header_level=chunk_level,
                content=cleaned_content,
                source_filename=source_filename,
                original_base_header=chunk_title
            ))

    logger.info(f"Found {len(initial_chunks)} initial sections from H{split_level} split.")
    return initial_chunks


def split_markdown_by_headers(
        markdown_text: str,
        source_filename: str = "unknown",
        split_level: int = 3,
        max_chars_per_chunk: int = DEFAULT_MAX_CHARS_PER_CHUNK,
        min_chars_per_chunk: int = DEFAULT_MIN_CHARS_PER_CHUNK,
        verbose: bool = False
) -> List[Dict[str, Any]]:
    if verbose:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.WARNING)

    logger.info(f"Starting Markdown chunking for '{source_filename}' (max_chars: {max_chars_per_chunk}, min_chars: {min_chars_per_chunk}, split_level: H{split_level}). Mode: Header splitting only.")

    base_chunks = _initial_split(markdown_text, source_filename, split_level)

    processing_queue = list(base_chunks)
    fully_processed_chunks = []

    MAX_ITERATIONS_OVERSIZED = len(base_chunks) + 20
    iter_count = 0

    while processing_queue and iter_count < MAX_ITERATIONS_OVERSIZED :
        iter_count += 1
        chunk = processing_queue.pop(0)
        current_content_len = len(chunk["content"])

        if current_content_len > max_chars_per_chunk:
            if not chunk["content"].strip() and not chunk["header_text"]:
                logger.debug(f"    Skipping effectively empty oversized chunk that was '{chunk['header_text']}'.")
                continue

            sub_chunks = _handle_oversized_chunk(chunk)

            if len(sub_chunks) == 1 and sub_chunks[0] is chunk:
                logger.error(
                    f"Chunk '{chunk['header_text']}' (length {current_content_len}) "
                    f"exceeds max_chars_per_chunk ({max_chars_per_chunk}) and "
                    f"could not be split further by available headers. This chunk will be OMITTED."
                )
            else:
                processing_queue = sub_chunks + processing_queue

        elif chunk["content"].strip() or chunk["header_text"]:
            fully_processed_chunks.append(chunk)

    if iter_count >= MAX_ITERATIONS_OVERSIZED and processing_queue:
        logger.warning(f"Max processing iterations ({MAX_ITERATIONS_OVERSIZED}) reached for oversized chunks. {len(processing_queue)} chunks remain in queue and will be OMITTED if still oversized.")

    logger.info(f"After oversized processing, {len(fully_processed_chunks)} potential chunks before merging.")

    if not fully_processed_chunks:
        logger.info("No chunks to process for merging.")
        return []

    merged_chunks: List[Dict[str, Any]] = []
    i = 0
    while i < len(fully_processed_chunks):
        current_chunk = fully_processed_chunks[i]

        if not current_chunk["content"].strip() and not current_chunk["header_text"].strip():
            logger.debug(f"    Skipping merge consideration for empty chunk: '{current_chunk['header_text']}'.")
            i += 1
            continue

        current_content_len = len(current_chunk["content"])

        if current_content_len < min_chars_per_chunk and (i + 1) < len(fully_processed_chunks):
            next_chunk = fully_processed_chunks[i+1]

            if not next_chunk["content"].strip() and not next_chunk["header_text"].strip():
                logger.debug(f"    Next chunk for merging is empty ('{next_chunk['header_text']}'), keeping current ('{current_chunk['header_text']}') as is.")
                merged_chunks.append(current_chunk)
                i += 2
                continue

            combined_content_len = current_content_len + len(next_chunk["content"]) + (2 if current_chunk["content"].strip() and next_chunk["content"].strip() else 0)

            if combined_content_len <= max_chars_per_chunk:
                logger.info(f"    Merging undersized '{current_chunk['header_text']}' (len {current_content_len}) with '{next_chunk['header_text']}' (len {len(next_chunk['content'])}).")

                new_content_parts = []
                if current_chunk["content"].strip(): new_content_parts.append(current_chunk["content"].strip())
                if next_chunk["content"].strip(): new_content_parts.append(next_chunk["content"].strip())
                new_content = "\n\n".join(new_content_parts)

                base_cur = current_chunk["original_base_header"]
                base_next = next_chunk["original_base_header"]

                new_header_text = current_chunk["header_text"]
                if base_cur == base_next and base_cur :
                    new_header_text = base_cur
                elif current_chunk["header_text"] != next_chunk["header_text"]:
                    if current_chunk["header_text"].startswith(next_chunk["header_text"] + CHUNK_HEADER_OVERVIEW_SUFFIX):
                        new_header_text = next_chunk["header_text"]
                    elif next_chunk["header_text"].startswith(current_chunk["header_text"] + CHUNK_HEADER_OVERVIEW_SUFFIX):
                        new_header_text = current_chunk["header_text"]
                    else:
                        new_header_text = f"{current_chunk['header_text']}{MERGED_HEADER_SEPARATOR}{next_chunk['header_text']}"

                merged_chunks.append(_create_chunk_dict(
                    header_text=new_header_text,
                    header_level=current_chunk["header_level"],
                    content=new_content,
                    source_filename=current_chunk["source_filename"],
                    original_base_header=base_cur
                ))
                i += 2
            else:
                merged_chunks.append(current_chunk)
                i += 1
        else:
            merged_chunks.append(current_chunk)
            i += 1

    logger.info(f"After merging undersized, {len(merged_chunks)} chunks.")

    final_valid_chunks = [
        chunk for chunk in merged_chunks if chunk["content"].strip() or chunk["header_text"].strip()
    ]
    logger.info(f"Markdown chunking process complete. Returning {len(final_valid_chunks)} chunks. Mode: Header splitting only.")
    return final_valid_chunks