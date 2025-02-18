import json
from pathlib import Path
from typing import List
from docling.chunking import HybridChunker
from docling.document_converter import DocumentConverter
from docling_core.types.doc import DoclingDocument
from dotenv import load_dotenv
from openai import OpenAI
from utils.tokenizer import OpenAITokenizerWrapper
import pprint

load_dotenv()

# Initialize OpenAI client (ensure OPENAI_API_KEY is set in your environment)
client = OpenAI()

# Initialize our custom tokenizer
tokenizer = OpenAITokenizerWrapper()
MAX_TOKENS = 8191  # text-embedding-3-large's maximum context length

# Directory containing the extracted JSON files from PDF extraction
EXTRACTED_DIR = Path("data-pipeline/extracted-pdfs")

def inspect_docling_document(doc):
    # Use dir() to list all attributes and methods
    print("Attributes and methods:")
    pprint.pprint(dir(doc))

    # Use vars() to get the __dict__ attribute of the object
    print("\nObject dictionary:")
    pprint.pprint(vars(doc))

    # If the object has a specific attribute you want to inspect
    if hasattr(doc, 'meta'):
        print("\nMeta attribute:")
        pprint.pprint(doc.meta)

    # Export to a file for further inspection
    with open("docling_document_inspection.txt", "w") as file:
        file.write(pprint.pformat(vars(doc)))

def chunk_markdown_files() -> List:
    """
    Process all JSON files in the extracted directory using HybridChunker.
    Uses chunker.serialize(chunk) for context-enriched text and prints page number information
    for a few chunks to verify that it's captured.

    Returns:
        A list of all chunks generated from all JSON files.
    """
    json_files = list(EXTRACTED_DIR.glob("*.json"))

    if not json_files:
        print(f"No JSON files found in {EXTRACTED_DIR.resolve()}.")
        return []

    chunker = HybridChunker(
        tokenizer=tokenizer,
        max_tokens=MAX_TOKENS,
        merge_peers=True,
    )

    all_chunks = []

    for file_index, json_file in enumerate(json_files):
        print(f"\nProcessing file {file_index + 1}/{len(json_files)}: {json_file.name}")
        # Load the DoclingDocument from JSON by parsing the file then validating with the pydantic API.
        with open(json_file, "r", encoding="utf-8") as f:
            json_str = f.read()
        doc_dict = json.loads(json_str)
        doc = DoclingDocument.model_validate(doc_dict)
        if not doc:
            print(f"Loading document failed for {json_file.name}.")
            continue

        # Commented out to avoid overwhelming output:
        # if file_index == 0:
        #     print("\nFirst DoclingDocument for inspection:")
        #     inspect_docling_document(doc)

        chunk_iter = chunker.chunk(dl_doc=doc)
        chunks = list(chunk_iter)

        non_empty_chunks = []
        empty_chunk_count = 0

        for i, chunk in enumerate(chunks):
            # Use the enriched serialization instead of chunk.text
            serialized_text = chunker.serialize(chunk)
            if not serialized_text.strip():
                empty_chunk_count += 1
                continue

            non_empty_chunks.append(chunk)

            # For the first file, print page numbers for the first 3 chunks only.
            if file_index == 0 and i < 3:
                page_numbers = sorted(
                    set(
                        prov.page_no
                        for item in chunk.meta.doc_items
                        for prov in item.prov
                        if hasattr(prov, "page_no")
                    )
                )
                print(f"Chunk {i}, Text Preview: {repr(serialized_text[:40])}…, Page Numbers: {page_numbers}")

        if empty_chunk_count > 0:
            print(f"Warning: Found {empty_chunk_count} empty chunks in {json_file.name}")

        print(f"Generated {len(non_empty_chunks)} valid chunks from {json_file.name}")
        all_chunks.extend(non_empty_chunks)

    print(f"\nTotal non-empty chunks generated: {len(all_chunks)}")
    return all_chunks

if __name__ == "__main__":
    print("Starting chunking process...")
    chunks = chunk_markdown_files()
    print(f"Chunking complete: {len(chunks)} chunks generated.")
