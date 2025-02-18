import os
import json
from pathlib import Path
from typing import List
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
from docling.datamodel.base_models import InputFormat
from docling.utils.model_downloader import download_models

# Optionally prefetch the models (run this only once, or use CLI: docling-tools models download)
download_models()

# Specify the path where the models are stored.
# Adjust the artifacts_path below to your preferred directory.
artifacts_path = os.path.expanduser("~/.cache/docling/models")
# Alternatively, if you want to store them in a custom location:
# artifacts_path = "/opt/docling_models"

def extract_pdf_documents(
    pdf_folder: Path = Path("data-pipeline/pdf-data"),
    output_folder: Path = Path("data-pipeline/extracted-pdfs")
) -> List:
    """
    Extract documents from PDFs using PDF pipeline options and save them as JSON files.
    This extraction workflow enables OCR and table structure enrichment, ensuring metadata like
    page numbers are captured in a lossless format.

    Args:
        pdf_folder: Directory containing the source PDF files.
        output_folder: Directory where extracted JSON files will be saved.

    Returns:
        List of document objects extracted by DocumentConverter.
    """
    output_folder.mkdir(parents=True, exist_ok=True)
    pdf_files = list(pdf_folder.glob("*.pdf"))

    if not pdf_files:
        print(f"No PDF files found in '{pdf_folder}' folder.")
        return []

    # Create the pipeline options using the local artifacts_path.
    pipeline_options = PdfPipelineOptions(
        artifacts_path=artifacts_path,
        # do_ocr=True,
        do_table_structure=True,
        # do_code_enrichment=True,          # Enable code enrichment for code snippets
        # do_formula_enrichment=True,       # Enable formula enrichment for mathematical formulas
        # do_picture_classification=True,   # Classify images if present
        # do_picture_description=True,      # Generate descriptive captions for images
        # generate_page_images=True,        # Capture page images for visual context
        # images_scale=0.8,                 # Adjust the scale of generated images
        table_structure_options=dict(
            mode=TableFormerMode.FAST   # Use an accurate mode for table extraction
        ),
        enable_remote_services=False
    )

    # Create the document converter with your custom pipeline options.
    doc_converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )
    documents = []

    for pdf_file in pdf_files:
        print(f"Extracting from: {pdf_file}")
        result = doc_converter.convert(str(pdf_file.resolve()))
        if result.document:
            document = result.document
            # Export the document to a dictionary and then convert to JSON.
            doc_dict = document.export_to_dict()
            json_output = json.dumps(doc_dict)
            documents.append(document)

            # Save the JSON output.
            output_file = output_folder / f"{pdf_file.stem}.json"
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(json_output)

            print("Extraction successful. Preview of extracted content (first 500 chars):")
            print(json_output[:500])
            print("-" * 40 + "\n")
        else:
            print(f"Extraction failed for: {pdf_file}\n")

    print(f"Total extracted documents: {len(documents)}")
    return documents

if __name__ == "__main__":
    extract_pdf_documents()
