import argparse
import os
from pathlib import Path

from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    TesseractCliOcrOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption


def _build_converter() -> DocumentConverter:
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = True
    pipeline_options.do_table_structure = True
    pipeline_options.ocr_options = TesseractCliOcrOptions(
        lang=["rus"], tesseract_cmd="tesseract"
    )

    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options, backend=PyPdfiumDocumentBackend
            )
        }
    )


def _write_output(output_filename: str, markdown_output: str) -> None:
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)
    with open(output_filename, "w", encoding="utf-8") as f:
        f.write(markdown_output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("filepath", type=str)
    args = parser.parse_args()

    filepath = args.filepath
    base_name = str(Path(filepath).with_suffix(""))[9:]

    print(f"Processing {filepath}...")
    doc_converter = _build_converter()
    result = doc_converter.convert(filepath)

    markdown_output = result.document.export_to_markdown()
    output_filename = "data/parsed" + base_name + "_parsed.md"
    _write_output(output_filename, markdown_output)

    print(f"Conversion complete! Markdown saved to '{output_filename}'")


if __name__ == "__main__":
    main()
