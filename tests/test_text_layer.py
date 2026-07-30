"""
Test 1: PyMuPDF text-layer extraction.

Validates that PyMuPDF correctly extracts:
- Per-span text with bbox and rotation angle
- Rotated/angled labels (key concern for engineering drawings)
- Completeness (no missing spans)

Input: paddleocr-tiled/sample-engineering-drawing.pdf (has text layer)
Output: JSON to output/text_layer_results.json + human-readable summary
"""

import json
import os
import fitz  # PyMuPDF

INPUT_PDF = os.path.join(
    os.path.dirname(__file__), "..", "paddleocr-tiled", "sample-engineering-drawing.pdf"
)
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "output")

TEXT_LAYER_CHAR_THRESHOLD = 100


def extract_spans(page):
    """Extract every text span with bbox and rotation from a page."""
    spans = []
    blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]
    for block in blocks:
        if block["type"] != 0:  # skip non-text blocks (images)
            continue
        for line in block["lines"]:
            for span in line["spans"]:
                text = span["text"].strip()
                if not text:
                    continue
                bbox = span["bbox"]  # [x0, y0, x1, y1]
                rotation = span.get("dir", (1, 0))  # direction vector
                # Convert direction vector to degrees
                import math
                angle = math.degrees(math.atan2(rotation[1], rotation[0]))
                # Normalize to 0-360
                if angle < 0:
                    angle += 360
                spans.append(
                    {
                        "text": text,
                        "bbox": [round(c, 2) for c in bbox],
                        "rotation_deg": round(angle, 1),
                        "font": span.get("font", ""),
                        "size": round(span.get("size", 0), 1),
                    }
                )
    return spans


def run():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    doc = fitz.open(INPUT_PDF)
    all_pages = []

    for page_num, page in enumerate(doc):
        text = page.get_text("text").strip()
        has_text_layer = len(text) > TEXT_LAYER_CHAR_THRESHOLD
        w, h = page.rect.width, page.rect.height

        print(f"\n{'='*60}")
        print(f"Page {page_num}: {w:.0f}x{h:.0f} pt | {len(text)} chars | "
              f"{'TEXT LAYER' if has_text_layer else 'needs OCR'}")
        print(f"{'='*60}")

        if not has_text_layer:
            print("  [SKIP] No text layer — would route to OCR")
            all_pages.append({
                "page": page_num,
                "route": "ocr",
                "char_count": len(text),
                "spans": [],
            })
            continue

        spans = extract_spans(page)

        # Rotation analysis
        rotation_counts = {}
        for s in spans:
            r = s["rotation_deg"]
            rotation_counts[r] = rotation_counts.get(r, 0) + 1

        print(f"  Total spans: {len(spans)}")
        print(f"  Rotation angles found: {dict(sorted(rotation_counts.items()))}")
        print(f"  Fonts used: {set(s['font'] for s in spans)}")

        # Show spans with non-zero rotation (the interesting ones)
        rotated = [s for s in spans if abs(s["rotation_deg"]) > 0.1 and abs(s["rotation_deg"] - 360) > 0.1]
        if rotated:
            print(f"\n  Rotated spans ({len(rotated)}):")
            for s in rotated:
                print(f"    [{s['rotation_deg']}°] '{s['text'][:60]}'"
                      f"  bbox={s['bbox']}  size={s['size']}")
        else:
            print("\n  No rotated spans found.")

        # Show first 20 spans for a quick sanity check
        print(f"\n  First 20 spans:")
        for s in spans[:20]:
            print(f"    [{s['rotation_deg']}°] '{s['text'][:60]}'  bbox={s['bbox']}")

        all_pages.append({
            "page": page_num,
            "route": "text_layer",
            "char_count": len(text),
            "page_size": [round(w, 2), round(h, 2)],
            "span_count": len(spans),
            "rotation_summary": dict(sorted(rotation_counts.items())),
            "spans": spans,
        })

    doc.close()

    output_path = os.path.join(OUTPUT_DIR, "text_layer_results.json")
    with open(output_path, "w") as f:
        json.dump(all_pages, f, indent=2, ensure_ascii=False)
    print(f"\n\nResults saved to {output_path}")


if __name__ == "__main__":
    run()
