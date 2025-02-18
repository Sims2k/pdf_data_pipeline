from pathlib import Path
from typing import List
from docling.chunking import HybridChunker
from docling.document_converter import DocumentConverter
from dotenv import load_dotenv
from openai import OpenAI
from utils.tokenizer import OpenAITokenizerWrapper

load_dotenv()

# Initialize OpenAI client (ensure OPENAI_API_KEY is set in your environment)
client = OpenAI()

# Initialize our custom tokenizer
tokenizer = OpenAITokenizerWrapper()
MAX_TOKENS = 8191  # text-embedding-3-large's maximum context length

# Directory containing the extracted Markdown files from PDF extraction
EXTRACTED_DIR = Path("data-pipeline/extracted-pdfs")

def chunk_markdown_files() -> List:
    """
    Process all Markdown files in the extracted directory using HybridChunker.
    Only prints page number information for a few chunks to verify that it's captured.
    
    Returns:
        A list of all chunks generated from all Markdown files.
    """
    markdown_files = list(EXTRACTED_DIR.glob("*.md"))
    
    if not markdown_files:
        print(f"No markdown files found in {EXTRACTED_DIR.resolve()}.")
        return []
    
    chunker = HybridChunker(
        tokenizer=tokenizer,
        max_tokens=MAX_TOKENS,
        merge_peers=True,
    )
    
    all_chunks = []
    
    for file_index, md_file in enumerate(markdown_files):
        print(f"\nProcessing file {file_index + 1}/{len(markdown_files)}: {md_file.name}")
        doc = DocumentConverter().convert(source=str(md_file.resolve())).document
        if not doc:
            print(f"Conversion failed for {md_file.name}.")
            continue
        
        chunk_iter = chunker.chunk(dl_doc=doc)
        chunks = list(chunk_iter)
        
        non_empty_chunks = []
        empty_chunk_count = 0
        
        for i, chunk in enumerate(chunks):
            if not chunk.text.strip():
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
                print(f"Chunk {i} Page Numbers: {page_numbers}")
        
        if empty_chunk_count > 0:
            print(f"Warning: Found {empty_chunk_count} empty chunks in {md_file.name}")
            
        print(f"Generated {len(non_empty_chunks)} valid chunks from {md_file.name}")
        all_chunks.extend(non_empty_chunks)
    
    print(f"\nTotal non-empty chunks generated: {len(all_chunks)}")
    return all_chunks

if __name__ == "__main__":
    print("Starting chunking process...")
    chunks = chunk_markdown_files()
    print(f"Chunking complete: {len(chunks)} chunks generated.")
