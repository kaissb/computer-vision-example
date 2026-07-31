"""
Test 3: PaddleOCR-VL extraction on scanned PDFs.

Validates:
- Good scan invoice: baseline accuracy + table structure
- Bad scan invoice: robustness on degraded input
- Handwriting prescription: baseline for the deferred path

Checks for:
- Reading order on structured documents
- Table cell boundaries and structure
- Repetition / hallucination in output
- Traceability (does output match visual content?)

Input: all PDFs in paddleocr-vl/
Output: JSON + markdown to output/vlm_results/
"""

import json
import os
import glob

import fitz  # PyMuPDF for rasterizing pages to images
import paddle

from paddleocr import PaddleOCRVL

BASE = os.path.dirname(__file__) + "/.."
INPUT_DIR = os.path.join(BASE, "paddleocr-vl")
OUTPUT_DIR = os.path.join(BASE, "output", "vlm_results")
RASTER_DPI = 150  # DPI for rasterizing PDF pages to images for the VLM


def rasterize_page(pdf_path, page_num=0, dpi=RASTER_DPI):
    """Convert a PDF page to a PNG image for PaddleOCR-VL input."""
    doc = fitz.open(pdf_path)
    page = doc[page_num]
    zoom = dpi / 72
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat)
    img_path = os.path.join(OUTPUT_DIR, f"{os.path.basename(pdf_path)}_page{page_num}.png")
    pix.save(img_path)
    doc.close()
    return img_path


def run():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Initializing PaddleOCR-VL (first run downloads model weights)...")
    ocr = PaddleOCRVL(
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_layout_detection=False,
    )

    print(f"GPU memory after load: {paddle.device.cuda.memory_allocated() / 1e9:.2f} GB\n")

    pdfs = sorted(glob.glob(os.path.join(INPUT_DIR, "*.pdf")))

    for pdf_path in pdfs:
        pdf_name = os.path.basename(pdf_path)
        print(f"\n{'='*70}")
        print(f"Processing: {pdf_name}")
        print(f"{'='*70}")

        # Rasterize page to image
        img_path = rasterize_page(pdf_path)
        print(f"  Rasterized to: {img_path}")

        # Run VLM
        print("  Running PaddleOCR-VL...")
        result = ocr.predict(
            input=img_path,
            max_new_tokens=4096,
            repetition_penalty=1.2,
        )

        # Extract results
        result_data = result[0] if isinstance(result, list) else result

        # Get markdown output if available
        markdown_text = ""
        json_data = None
        try:
            markdown_text = result_data.markdown if hasattr(result_data, 'markdown') else str(result_data)
        except Exception as e:
            markdown_text = f"[markdown extraction error: {e}]"

        try:
            json_data = result_data.json if hasattr(result_data, 'json') else None
        except Exception:
            pass

        # Print summary
        print(f"\n  --- Markdown output (first 2000 chars) ---")
        print(f"  {markdown_text[:2000]}")
        if len(markdown_text) > 2000:
            print(f"  ... [{len(markdown_text)} total chars]")

        # Save outputs
        md_path = os.path.join(OUTPUT_DIR, pdf_name.replace('.pdf', '.md'))
        with open(md_path, 'w') as f:
            f.write(markdown_text)
        print(f"\n  Markdown saved to: {md_path}")

        json_path = os.path.join(OUTPUT_DIR, pdf_name.replace('.pdf', '.json'))
        with open(json_path, 'w') as f:
            if json_data:
                json.dump(json_data, f, indent=2, ensure_ascii=False)
            else:
                json.dump({"raw": str(result_data)}, f, indent=2, ensure_ascii=False)
        print(f"  JSON saved to: {json_path}")

        # Hallucination / repetition quick check
        lines = markdown_text.split('\n')
        seen = {}
        duplicates = []
        for line in lines:
            line_clean = line.strip()
            if line_clean and len(line_clean) > 10:
                if line_clean in seen:
                    duplicates.append(line_clean)
                else:
                    seen[line_clean] = True
        if duplicates:
            print(f"\n  ⚠️  Repeated lines detected ({len(duplicates)}):")
            for d in duplicates[:5]:
                print(f"    -> '{d[:80]}'")
        else:
            print(f"\n  ✓ No repeated lines detected")

    print(f"\n\nAll outputs saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    run()
