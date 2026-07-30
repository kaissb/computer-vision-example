"""
Test 4: Classic PaddleOCR with tiling on large-format scanned PDF.

Validates:
- Fine print survival at high DPI
- Rotation angle preservation on angled labels
- Boundary artifacts on tile overlaps (de-duplication)
- Ground truth comparison against text layer output from the original

Input: paddleocr-tiled/sample-engineering-drawing-no-text.pdf
Ground truth: output/text_layer_results.json (from test_text_layer.py)
Output: JSON to output/tiled_ocr_results.json + comparison summary
"""

import json
import os
import math
import fitz  # PyMuPDF
from paddleocr import PaddleOCR

BASE = os.path.dirname(__file__) + "/.."
INPUT_PDF = os.path.join(BASE, "paddleocr-tiled", "sample-engineering-drawing-no-text.pdf")
GROUND_TRUTH_PATH = os.path.join(BASE, "output", "text_layer_results.json")
OUTPUT_DIR = os.path.join(BASE, "output")

# --- Configurable constants ---
RASTER_DPI = 300
TILE_PIXELS = 2048          # tile size in pixels (square)
TILE_OVERLAP_PIXELS = 256   # overlap between tiles for de-duplication


def rasterize_page(pdf_path, dpi=RASTER_DPI):
    """Rasterize a PDF page to a PIL Image at high DPI."""
    doc = fitz.open(pdf_path)
    page = doc[0]
    zoom = dpi / 72
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat)
    img_path = os.path.join(OUTPUT_DIR, "_tiled_page_raster.png")
    pix.save(img_path)
    doc.close()
    return img_path, pix.width, pix.height


def generate_tiles(img_w, img_h, tile_size=TILE_PIXELS, overlap=TILE_OVERLAP_PIXELS):
    """Generate tile coordinates (x0, y0, x1, y1) covering the image with overlap."""
    tiles = []
    step = tile_size - overlap
    y = 0
    while y < img_h:
        x = 0
        while x < img_w:
            x1 = min(x + tile_size, img_w)
            y1 = min(y + tile_size, img_h)
            tiles.append((x, y, x1, y1))
            if x1 >= img_w:
                break
            x += step
        if y1 >= img_h:
            break
        y += step
    return tiles


def run():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- Rasterize ---
    print(f"Rasterizing {INPUT_PDF} at {RASTER_DPI} DPI...")
    img_path, img_w, img_h = rasterize_page(INPUT_PDF)
    print(f"  Image size: {img_w}x{img_h} pixels")

    # --- Generate tiles ---
    tiles = generate_tiles(img_w, img_h)
    print(f"  Tiles: {len(tiles)} (size={TILE_PIXELS}px, overlap={TILE_OVERLAP_PIXELS}px)")
    for i, (x0, y0, x1, y1) in enumerate(tiles):
        print(f"    tile {i}: ({x0},{y0}) -> ({x1},{y1})  [{x1-x0}x{y1-y0}]")

    # --- Initialize PaddleOCR (classic) ---
    print("\nInitializing PaddleOCR (classic detection + recognition)...")
    ocr = PaddleOCR(use_textline_orientation=True, lang='en')
    print("Model ready.\n")

    # --- Run OCR on each tile ---
    from PIL import Image
    full_img = Image.open(img_path)

    all_boxes = []
    for i, (x0, y0, x1, y1) in enumerate(tiles):
        print(f"\n  Processing tile {i}/{len(tiles)-1} ({x0},{y0})->({x1},{y1})...")
        tile_img = full_img.crop((x0, y0, x1, y1))
        tile_path = os.path.join(OUTPUT_DIR, f"_tile_{i}.png")
        tile_img.save(tile_path)

        result = ocr.predict(input=tile_path)
        result_data = result[0] if isinstance(result, list) else result

        # Parse results — PaddleOCR returns json with rec_texts, dt_polys, rec_scores
        try:
            json_data = result_data.json
            texts = json_data.get("rec_texts", [])
            polys = json_data.get("dt_polys", [])
            scores = json_data.get("rec_scores", [])
        except Exception:
            texts, polys, scores = [], [], []

        print(f"    Found {len(texts)} text regions")

        for j, (text, poly, score) in enumerate(zip(texts, polys, scores)):
            # poly is [[x0,y0],[x1,y1],[x2,y2],[x3,y3]] relative to tile
            # Convert to absolute image coordinates
            abs_poly = [[px + x0, py + y0] for px, py in poly]

            # Compute bbox from poly
            xs = [p[0] for p in abs_poly]
            ys = [p[1] for p in abs_poly]
            bbox = [min(xs), min(ys), max(xs), max(ys)]

            # Estimate rotation from poly (angle of top edge)
            dx = abs_poly[1][0] - abs_poly[0][0]
            dy = abs_poly[1][1] - abs_poly[0][1]
            angle = math.degrees(math.atan2(dy, dx))

            all_boxes.append({
                "text": text,
                "bbox_abs": [round(c, 1) for c in bbox],
                "poly_abs": [[round(c, 1) for c in p] for p in abs_poly],
                "rotation_deg": round(angle, 1),
                "confidence": round(score, 3) if isinstance(score, (int, float)) else None,
                "tile": i,
            })

        os.remove(tile_path)  # cleanup

    # --- De-duplicate overlapping detections ---
    print(f"\n\nTotal detections before dedup: {len(all_boxes)}")

    def boxes_overlap(b1, b2, threshold=0.5):
        """Check if two boxes have IoU > threshold."""
        ix0 = max(b1[0], b2[0])
        iy0 = max(b1[1], b2[1])
        ix1 = min(b1[2], b2[2])
        iy1 = min(b1[3], b2[3])
        if ix0 >= ix1 or iy0 >= iy1:
            return False
        inter = (ix1 - ix0) * (iy1 - iy0)
        area1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
        area2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
        union = area1 + area2 - inter
        return inter / union > threshold if union > 0 else False

    def texts_similar(t1, t2):
        """Check if two text strings are similar enough to be duplicates."""
        t1, t2 = t1.strip().lower(), t2.strip().lower()
        if t1 == t2:
            return True
        # Allow minor OCR differences
        if abs(len(t1) - len(t2)) <= 2 and (t1 in t2 or t2 in t1):
            return True
        return False

    # Simple greedy dedup: sort by confidence, remove later overlapping duplicates
    all_boxes.sort(key=lambda b: b.get("confidence", 0) or 0, reverse=True)
    kept = []
    removed = 0
    for box in all_boxes:
        is_dup = False
        for k in kept:
            if boxes_overlap(box["bbox_abs"], k["bbox_abs"]) and texts_similar(box["text"], k["text"]):
                is_dup = True
                break
        if is_dup:
            removed += 1
        else:
            kept.append(box)

    print(f"Duplicates removed: {removed}")
    print(f"Final detections after dedup: {len(kept)}")

    # --- Rotation analysis ---
    rotation_counts = {}
    for b in kept:
        r = b["rotation_deg"]
        # Normalize
        r_norm = round(r) % 360
        rotation_counts[r_norm] = rotation_counts.get(r_norm, 0) + 1
    print(f"\nRotation angles found: {dict(sorted(rotation_counts.items()))}")

    rotated = [b for b in kept if abs(b["rotation_deg"]) > 1 and abs(b["rotation_deg"] - 360) > 1]
    if rotated:
        print(f"Rotated detections ({len(rotated)}):")
        for b in rotated:
            print(f"  [{b['rotation_deg']}°] '{b['text'][:60]}'  conf={b['confidence']}")
    else:
        print("No rotated detections found.")

    # --- Print all detections ---
    print(f"\nAll detections ({len(kept)}):")
    for b in kept:
        print(f"  [{b['rotation_deg']:>6.1f}°] conf={b['confidence']}  '{b['text'][:60]}'")

    # --- Compare against ground truth ---
    print(f"\n{'='*70}")
    print("Ground truth comparison (vs text layer extraction)")
    print(f"{'='*70}")

    gt_texts = set()
    if os.path.exists(GROUND_TRUTH_PATH):
        with open(GROUND_TRUTH_PATH) as f:
            gt_data = json.load(f)
        for page in gt_data:
            for span in page.get("spans", []):
                gt_texts.add(span["text"].strip().lower())
        print(f"Ground truth spans: {len(gt_texts)}")
    else:
        print(f"⚠️  No ground truth file at {GROUND_TRUTH_PATH}")
        print("  Run test_text_layer.py first to generate it.")

    if gt_texts:
        ocr_texts = set(b["text"].strip().lower() for b in kept)
        exact_matches = gt_texts & ocr_texts
        gt_only = gt_texts - ocr_texts
        ocr_only = ocr_texts - gt_texts

        print(f"\n  Exact text matches: {len(exact_matches)}/{len(gt_texts)} ({100*len(exact_matches)/len(gt_texts):.1f}%)")
        print(f"  Ground truth only (missed by OCR): {len(gt_only)}")
        print(f"  OCR only (not in ground truth): {len(ocr_only)}")

        if gt_only:
            print(f"\n  Missed by OCR (first 20):")
            for t in sorted(gt_only)[:20]:
                print(f"    - '{t[:60]}'")
        if ocr_only:
            print(f"\n  OCR-only detections (first 20, possible hallucinations or extras):")
            for t in sorted(ocr_only)[:20]:
                print(f"    + '{t[:60]}'")

    # --- Save results ---
    output = {
        "input": INPUT_PDF,
        "raster_dpi": RASTER_DPI,
        "image_size": [img_w, img_h],
        "tile_size": TILE_PIXELS,
        "tile_overlap": TILE_OVERLAP_PIXELS,
        "tile_count": len(tiles),
        "detections_before_dedup": len(all_boxes),
        "duplicates_removed": removed,
        "detections_after_dedup": len(kept),
        "rotation_summary": dict(sorted(rotation_counts.items())),
        "boxes": kept,
    }
    output_path = os.path.join(OUTPUT_DIR, "tiled_ocr_results.json")
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    run()
