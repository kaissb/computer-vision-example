"""
Test 2: Triage / routing heuristic validation.

Validates that the routing logic classifies each sample PDF correctly:
- Text layer detection (char threshold)
- Page size classification (standard vs large-format)
- Expected route assignment

Input: all PDFs in paddleocr-vl/ and paddleocr-tiled/
Output: pass/fail summary + detailed per-PDF breakdown
"""

import os
import glob
import fitz  # PyMuPDF

# --- Configurable constants (mirrors what the router will use) ---
TEXT_LAYER_CHAR_THRESHOLD = 100
LARGE_FORMAT_MAX_DIM_PT = 1000  # ~A2 and bigger

# --- Expected routes for known samples ---
EXPECTED = {
    "paddleocr-vl/glasses-invoice-bad-scan.pdf": "vlm",
    "paddleocr-vl/glasses-invoice-good-scan.pdf": "vlm",
    "paddleocr-vl/prescription-handwriting-scanned.pdf": "vlm",
    "paddleocr-tiled/sample-engineering-drawing.pdf": "text_layer",
    "paddleocr-tiled/sample-engineering-drawing-no-text.pdf": "tiled_ocr",
}


def classify_page(page):
    """Classify a single page and return route + metadata."""
    text = page.get_text("text").strip()
    char_count = len(text)
    has_text_layer = char_count > TEXT_LAYER_CHAR_THRESHOLD
    w, h = page.rect.width, page.rect.height
    max_dim = max(w, h)
    is_large_format = max_dim >= LARGE_FORMAT_MAX_DIM_PT

    if has_text_layer:
        route = "text_layer"
    elif is_large_format:
        route = "tiled_ocr"
    else:
        route = "vlm"

    return {
        "route": route,
        "char_count": char_count,
        "has_text_layer": has_text_layer,
        "page_size": [round(w, 2), round(h, 2)],
        "max_dim": round(max_dim, 2),
        "is_large_format": is_large_format,
    }


def run():
    base = os.path.dirname(__file__) + "/.."
    pdfs = sorted(glob.glob(os.path.join(base, "paddleocr-vl/*.pdf")) +
                  glob.glob(os.path.join(base, "paddleocr-tiled/*.pdf")))

    # Normalize to relative paths for matching against EXPECTED
    pdfs_rel = [os.path.relpath(p, base) for p in pdfs]

    print(f"{'PDF':<55} {'Page':>4} {'Chars':>6} {'Size':>12} {'Route':<12} {'Expected':<12} {'PASS'}")
    print("-" * 115)

    all_pass = True
    for pdf_rel, pdf_abs in zip(pdfs_rel, pdfs):
        doc = fitz.open(pdf_abs)
        for i, page in enumerate(doc):
            info = classify_page(page)
            expected = EXPECTED.get(pdf_rel, "?")
            passed = info["route"] == expected
            if not passed:
                all_pass = False
            size_str = f"{info['page_size'][0]:.0f}x{info['page_size'][1]:.0f}"
            mark = "✓" if passed else "✗"
            print(f"{pdf_rel:<55} {i:>4} {info['char_count']:>6} "
                  f"{size_str:>12} {info['route']:<12} {expected:<12} {mark}")
        doc.close()

    print("-" * 115)
    if all_pass:
        print("\n✅ ALL PASSED — routing heuristic classifies every sample correctly.")
    else:
        print("\n❌ FAILURES detected — review mismatches above.")

    return all_pass


if __name__ == "__main__":
    run()
