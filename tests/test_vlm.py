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

# Let PaddlePaddle manage GPU memory dynamically (no pre-allocation limit)

import fitz  # PyMuPDF for rasterizing pages to images

# Patch _convert_state_dict_dtype_and_shape to cast tensors on CPU instead of GPU.
# The model params are float32 but weights are bfloat16; casting on GPU triples VRAM.
import numpy as np
import paddle
import paddlex.inference.models.common.transformers.transformers.model_utils as _model_utils

_orig_convert = _model_utils._convert_state_dict_dtype_and_shape

def _patched_convert(state_dict, model_to_load, convert_from_hf):
    def is_0d_or_1d(tensor):
        return len(tensor.shape) == 0 or list(tensor.shape) == [1]

    if convert_from_hf:
        model_state_dict = model_to_load.get_hf_state_dict()
    else:
        model_state_dict = model_to_load.state_dict()
    for key, value in model_state_dict.items():
        if key in list(state_dict.keys()):
            if isinstance(state_dict[key], np.ndarray):
                raise ValueError(
                    "convert_state_dict_dtype expected paddle.Tensor not numpy.ndarray"
                )
            if (
                state_dict[key].is_floating_point()
                and state_dict[key].dtype != value.dtype
            ):
                # Cast on CPU to avoid doubling GPU memory
                tensor = state_dict.pop(key)
                tensor = tensor.cpu()
                state_dict[key] = paddle.cast(tensor, value.dtype)
            if is_0d_or_1d(value) and is_0d_or_1d(state_dict[key]):
                if list(value.shape) != list(state_dict[key].shape):
                    state_dict[key] = paddle.reshape(state_dict.pop(key), value.shape)

_model_utils._convert_state_dict_dtype_and_shape = _patched_convert

# Patch set_hf_state_dict to load weights one at a time onto GPU.
# Original moves ALL tensors to GPU before set_state_dict, doubling VRAM.
# This version sets each parameter individually and frees the source after.
import paddlex.inference.models.doc_vlm.modeling.paddleocr_vl._paddleocr_vl as _vlm_module

_orig_set_hf = _vlm_module.PaddleOCRVLForConditionalGeneration.set_hf_state_dict

def _patched_set_hf(self, state_dict, *args, **kwargs):
    def _split_attention_weights(weight=None, bias=None):
        if weight is not None:
            split_size = weight.shape[1] // 3
            q_weight = weight[:, :split_size]
            k_weight = weight[:, split_size : 2 * split_size]
            v_weight = weight[:, 2 * split_size :]
            return q_weight, k_weight, v_weight
        elif bias is not None:
            split_size = bias.shape[0] // 3
            q_bias = bias[:split_size]
            k_bias = bias[split_size : 2 * split_size]
            v_bias = bias[2 * split_size :]
            return q_bias, k_bias, v_bias

    def _convert_state_dict(old_state_dict):
        new_state_dict = {}
        for key, value in old_state_dict.items():
            if "head.attention.in_proj" in key:
                if key.endswith("weight"):
                    q_w, k_w, v_w = _split_attention_weights(weight=value)
                    new_state_dict[
                        key.replace("in_proj_weight", "q_proj.weight")
                    ] = q_w
                    new_state_dict[
                        key.replace("in_proj_weight", "k_proj.weight")
                    ] = k_w
                    new_state_dict[
                        key.replace("in_proj_weight", "v_proj.weight")
                    ] = v_w
                elif key.endswith("bias"):
                    q_b, k_b, v_b = _split_attention_weights(bias=value)
                    new_state_dict[key.replace("in_proj_bias", "q_proj.bias")] = q_b
                    new_state_dict[key.replace("in_proj_bias", "k_proj.bias")] = k_b
                    new_state_dict[key.replace("in_proj_bias", "v_proj.bias")] = v_b
                else:
                    raise ValueError(f"Unexpected key: {key}")
            else:
                new_state_dict[key] = value

        for key in list(new_state_dict.keys()):
            if key.startswith("model."):
                if "mlp.gate_proj." in key:
                    gate_proj = new_state_dict.pop(key)
                    up_proj = new_state_dict.pop(
                        key.replace("gate_proj", "up_proj")
                    )
                    new_state_dict[key.replace("gate_proj", "up_gate_proj")] = (
                        paddle.concat([gate_proj, up_proj], axis=-1)
                    )

                if "self_attn.q_proj" in key:
                    q_proj = new_state_dict.pop(key)
                    k_proj = new_state_dict.pop(key.replace("q_proj", "k_proj"))
                    v_proj = new_state_dict.pop(key.replace("q_proj", "v_proj"))
                    new_state_dict[key.replace("q_proj", "qkv_proj")] = (
                        paddle.concat([q_proj, k_proj, v_proj], axis=-1)
                    )

        return new_state_dict

    state_dict = _convert_state_dict(state_dict)
    std_state_dict = self.state_dict()

    # Load weights one at a time: move to GPU, set value, free source
    for key in list(std_state_dict.keys()):
        v1 = std_state_dict[key]
        cpu_tensor = state_dict.pop(key)
        gpu_tensor = cpu_tensor.to(v1.place)
        v1.set_value(gpu_tensor)
        del cpu_tensor, gpu_tensor

    return None

_vlm_module.PaddleOCRVLForConditionalGeneration.set_hf_state_dict = _patched_set_hf

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

    # Convert model to bfloat16 to halve VRAM (model loads in float32)
    import gc
    inner = ocr.paddlex_pipeline._pipeline
    model = inner.vl_rec_model.infer

    # Debug: check actual dtype and memory
    sample_param = model.parameters()[0]
    print(f"Model param dtype: {sample_param.dtype}, place: {sample_param.place}")
    print(f"GPU memory after load: {paddle.device.cuda.memory_allocated() / 1e9:.2f} GB")

    # If model is in float32, try to convert to bfloat16
    if str(sample_param.dtype) == "paddle.float32":
        print("Model is float32, converting to bfloat16...")
        for name, param in model.named_parameters():
            if param.is_floating_point():
                np_val = param.numpy()
                bf16_val = paddle.to_tensor(np_val).astype(paddle.bfloat16)
                # Need to recreate parameter with correct dtype
                new_param = paddle.create_parameter(
                    shape=param.shape,
                    dtype=paddle.bfloat16,
                    default_initializer=paddle.nn.initializer.Assign(bf16_val),
                )
                # Replace in model
                parent = model
                parts = name.split(".")
                for p in parts[:-1]:
                    idx = int(p) if p.isdigit() else p
                    parent = parent[idx] if isinstance(idx, int) else getattr(parent, idx)
                last = parts[-1]
                setattr(parent, last, new_param)
                del np_val, bf16_val, new_param
        gc.collect()
        paddle.device.cuda.empty_cache()
        sample_param = model.parameters()[0]
        print(f"After convert - dtype: {sample_param.dtype}")
    else:
        print("Model already in half precision")

    gc.collect()
    paddle.device.cuda.empty_cache()
    print(f"GPU memory ready: {paddle.device.cuda.memory_allocated() / 1e9:.2f} GB\n")

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
        result = ocr.predict(input=img_path)

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
