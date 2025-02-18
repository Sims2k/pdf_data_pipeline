from pathlib import Path
from typing import List
import pickle
import lancedb
from dotenv import load_dotenv
from lancedb.embeddings import get_registry
from lancedb.pydantic import LanceModel, Vector
from openai import OpenAI
from utils.tokenizer import OpenAITokenizerWrapper

# Load environment variables
load_dotenv()

print("Initializing OpenAI client and tokenizer...")
client = OpenAI()
tokenizer = OpenAITokenizerWrapper()
MAX_TOKENS = 8191  # text-embedding-3-large's maximum context length

print("Starting embedding process...")

# ---------------------------------------------------------------------
# Extraction and Chunking
# ---------------------------------------------------------------------
# Optionally, if you have new PDFs to extract using the enhanced PDF extraction process:
from pdf_extraction import extract_pdf_documents
extract_pdf_documents()  # Uncomment this line to extract new PDF documents

print("Importing chunking functionality...")
from chunking import chunk_markdown_files

# Directory containing the extracted JSON files from PDF extraction
extracted_dir = Path("data-pipeline/extracted-pdfs")
if not list(extracted_dir.glob("*.json")):
    print("No extracted JSON files found in data-pipeline/extracted-pdfs.")
    print("Please run the extraction process (uncomment the extraction code in this file) or add new data.")
    exit(1)

# Use caching to avoid reprocessing chunks
cache_path = Path("data-pipeline/chunks.pkl")
if cache_path.exists():
    print(f"Loading cached chunks from {cache_path}...")
    with open(cache_path, "rb") as f:
        chunks = pickle.load(f)
    print(f"Loaded {len(chunks)} chunks from cache.")
else:
    print("No cached chunks found. Running chunking process...")
    chunks = chunk_markdown_files()
    with open(cache_path, "wb") as f:
        pickle.dump(chunks, f)
    print(f"Generated and saved {len(chunks)} chunks to {cache_path}.")

# ---------------------------------------------------------------------
# Create a LanceDB database and table, embed the chunks.
# ---------------------------------------------------------------------
print("Connecting to LanceDB...")
db = lancedb.connect("data/lancedb")

print("Fetching OpenAI embedding function from registry...")
registry = get_registry()
func = registry.get("openai").create(name="text-embedding-3-large")

# Define a simplified metadata schema.
class ChunkMetadata(LanceModel):
    """
    Metadata for each chunk.
    Fields must be in alphabetical order.
    """
    filename: str | None
    page_numbers: List[int] | None
    title: str | None

# Define the main schema for the chunks.
class Chunks(LanceModel):
    text: str = func.SourceField()
    vector: Vector(func.ndims()) = func.VectorField()  # type: ignore
    metadata: ChunkMetadata

print("Creating (or overwriting) table 'docling' in LanceDB...")
table = db.create_table("docling", schema=Chunks, mode="overwrite")

# ---------------------------------------------------------------------
# Prepare the chunks for insertion into the table.
# ---------------------------------------------------------------------
print("Processing chunks for insertion...")
processed_chunks = []
for i, chunk in enumerate(chunks):
    meta = {
        "filename": chunk.meta.origin.filename,
        "page_numbers": sorted(
            set(
                prov.page_no
                for item in chunk.meta.doc_items
                for prov in item.prov
            )
        ) or None,
        "title": chunk.meta.headings[0] if chunk.meta.headings else None,
    }
    processed_chunks.append({
        "text": chunk.text,
        "metadata": meta,
    })
    # Print preview for the first few chunks
    if i < 3:
        print(f"\nPreview of chunk {i}:")
        print("Text:", chunk.text[:300] + "…")
        print("Metadata:", meta)

# --- Batch insertion to avoid hitting token limits ---
print(f"\nInserting {len(processed_chunks)} chunks into the table in batches...")
batch_size = 100  # Adjust this value as needed
total_batches = (len(processed_chunks) - 1) // batch_size + 1
for i in range(0, len(processed_chunks), batch_size):
    batch = processed_chunks[i : i + batch_size]
    print(f"Inserting batch {i//batch_size + 1} of {total_batches} with {len(batch)} chunks...")
    table.add(batch)
print("Insertion complete.")

# ---------------------------------------------------------------------
# Display the table.
# ---------------------------------------------------------------------
print("Fetching table content as pandas dataframe...")
df = table.to_pandas()

# Instead of printing the entire table, show only the first 5 rows and the table shape.
print("\nLanceDB Table preview (first 5 rows):")
print(df.head())
print(f"\nTable shape: {df.shape}")
print("Total rows in LanceDB table:", table.count_rows())
