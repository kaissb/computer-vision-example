# PDF Extraction POC — Handoff

**Status:** research done, environment being set up, router script not yet written.
**Goal:** extract content from non-conventional PDFs — custom page sizes, text in multiple orientations (vertical + horizontal), embedded diagrams/shapes, occasional handwritten annotations.

---

## 1. Decision Summary

**Do not use a single model.** Route each page to the cheapest extractor that can handle it.

| Priority | Page type | Extractor | Why |
|---|---|---|---|
| 1 | Any page with a real text layer | **PyMuPDF** (`fitz`) direct extraction | Free, instant, exact. Returns per-span position **and rotation angle**, so rotated/margin labels come out correctly — something OCR struggles with. |
| 2 | Raster document-style pages (columns, headings, inline figures) | **PaddleOCR-VL** | Single-pass VLM: reads the whole page and emits structured markdown/JSON. Leads OmniDocBench (~96%). Handles mixed vertical/horizontal text notably better than DeepSeek-OCR. |
| 3 | Dense large-format sheets (fine print, angled labels) | **Classic PaddleOCR, tiled at high DPI** | A whole-page VLM downsamples oversized pages and destroys small text. Classic detection outputs rotated/quad boxes, and tiling preserves fine detail. |

### Why PaddleOCR-VL over the alternatives
- ~0.9B params, SOTA on OmniDocBench — best accuracy-per-VRAM in the open-weight field.
- Materially better on **vertically written text** than DeepSeek-OCR (independent evaluation), which is the key differentiator for this corpus.
- Strong on header/footer regions.
- Small enough to run comfortably on a consumer laptop GPU.

### Deprioritized
- **Handwriting** — possible in the corpus but not a priority. If it becomes one, fall back to **olmOCR** or **Qwen-VL** class models for those pages only (higher compute, outputs need validation for hallucination).

### Key insight
Large-format engineering/CAD-style sheets are frequently **born-digital exports with real vector text**, including the rotated labels. Always check for a text layer before sending anything to OCR — step 1 of the router will handle a meaningful share of pages at zero cost.

---

## 2. Hardware

Target machine: laptop, NVIDIA RTX GPU, i7, 16 GB RAM. **Sufficient.**

- PaddleOCR-VL needs **~2 GB VRAM at FP16** (~0.5 GB at INT4).
- Requires **compute capability ≥ 7.0** → RTX 20-series or newer.
- Driver must support **CUDA 12.6** (the default Paddle GPU build).
- CPU/RAM are not a bottleneck; PyMuPDF and classic PaddleOCR are light.
- If VRAM is ever tight (≤4 GB), there is a **llama.cpp path** for the VLM component (added March 2026).

Verify before installing:
```bash
nvidia-smi   # note VRAM and that the driver supports CUDA 12.6+
```

---

## 3. Environment Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# Triage + rasterizing (CPU, small)
pip install pymupdf pillow

# PaddleOCR-VL — CUDA 12.6 build.
# For other CUDA versions or CPU, see the PaddlePaddle install page.
pip install paddlepaddle-gpu==3.2.1 \
  -i https://www.paddlepaddle.org.cn/packages/stable/cu126/
pip install -U "paddleocr[doc-parser]>=3.4.0"
```

**Note:** stick with the default Paddle/transformers engine for now. vLLM and SGLang backends have known `transformers` version conflicts and cannot share an environment with the Transformers engine. They only matter for throughput — irrelevant while validating correctness.

---

## 4. Triage Check (written, needs running)

Establishes which pages are born-digital vs. which actually need OCR.

```python
import fitz  # PyMuPDF

doc = fitz.open("sample.pdf")
for i, page in enumerate(doc):
    text = page.get_text("text")
    # Heuristic: enough characters = born-digital, skip OCR entirely.
    has_text_layer = len(text.strip()) > 100
    print(f"page {i}: {'TEXT LAYER' if has_text_layer else 'needs OCR'} "
          f"({len(text.strip())} chars)")
```

Smoke-test the VLM on a page that needs OCR:
```bash
paddleocr doc_parser -i sample.pdf
```

---

## 5. Next Step — Build the Router

Single, well-commented, maintainable module that runs against a folder of test PDFs.

**Required behaviour:**
1. Open each PDF page-by-page with PyMuPDF.
2. **Classify** the page:
   - has usable text layer → extract directly (capture text, bbox, **rotation angle**)
   - no text layer, normal page size → PaddleOCR-VL
   - no text layer, oversized page → rasterize at high DPI, tile with overlap, run classic PaddleOCR per tile, merge with de-duplication on the overlaps
3. **Emit** a consistent structured record per page regardless of route, e.g.:
   ```json
   {
     "page": 0,
     "route": "text_layer | vlm | tiled_ocr",
     "blocks": [{"text": "...", "bbox": [x0,y0,x1,y1], "rotation": 0}],
     "confidence": null
   }
   ```
   Uniform schema matters — downstream consumers shouldn't care which extractor ran.
4. Make the page-size threshold, DPI, tile size/overlap, and text-layer character threshold **configurable constants at the top of the file**, since they'll need tuning against real samples.

---

## 6. Validation Checklist

Benchmark scores in this space are largely **vendor self-reported and not independently reproduced** — treat them as a shortlist signal only. Validate on real pages against:

- [ ] **Reading order** on multi-column pages
- [ ] **Rotated and margin labels** — do they survive, and is the angle preserved?
- [ ] **Fine print on large-format sheets** — where does it get dropped? (signal to tune tiling/DPI)
- [ ] **Tables** — cell boundaries and structure, a known weak spot
- [ ] **Repetition / hallucination** in VLM output — a reported failure mode
- [ ] **Traceability** — can a reviewer trace an uncertain value back to its location on the page?

Expect dense large-format sheet bodies to need **human verification** regardless of model.

---

## 7. Open Risks

- Handwriting quality is the weakest area across all OSS models; deferred by decision, not solved.
- Complex tables and graphics remain hard for compact VLMs.
- Tiling introduces boundary artifacts — de-duplication logic on overlaps needs care.