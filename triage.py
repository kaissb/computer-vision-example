import fitz  # PyMuPDF

doc = fitz.open("sample.pdf")
for i, page in enumerate(doc):
    text = page.get_text("text")
    # Heuristic: enough characters = born-digital, skip OCR entirely.
    has_text_layer = len(text.strip()) > 100
    print(f"page {i}: {'TEXT LAYER' if has_text_layer else 'needs OCR'} "
          f"({len(text.strip())} chars)")