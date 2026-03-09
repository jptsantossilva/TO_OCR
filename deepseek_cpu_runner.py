from __future__ import annotations

import argparse
import json
import os
import tempfile

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from PIL import Image
import cv2
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


def build_prompt(mode: str, expected_text: str) -> str:
    if mode == "industrial_ocv" and expected_text:
        return (
            "<image>\n"
            "OCR this image. Return only the single printed lot/code line above the barcode. "
            f"Expected format similar to: {expected_text}. No explanation."
        )
    if mode == "industrial_ocv":
        return "<image>\nOCR this image. Return only the single printed line above the barcode. No explanation."
    return "<image>\nFree OCR."


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--image-path", required=True)
    parser.add_argument("--mode", required=True)
    parser.add_argument("--expected-text", default="")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id,
        trust_remote_code=True,
        revision=args.revision,
    )
    model = AutoModel.from_pretrained(
        args.model_id,
        trust_remote_code=True,
        use_safetensors=True,
        revision=args.revision,
    )
    model = model.eval().to("cpu")

    image_bgr = cv2.imread(args.image_path)
    if image_bgr is None:
        raise RuntimeError(f"Could not read image: {args.image_path}")

    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb)
    prompt = build_prompt(args.mode, args.expected_text)

    with tempfile.TemporaryDirectory(prefix="deepseek_cpu_runner_") as temp_dir:
        input_path = os.path.join(temp_dir, "input.png")
        pil_image.save(input_path)
        with torch.inference_mode():
            try:
                result = model.infer(
                    tokenizer,
                    prompt=prompt,
                    image_file=input_path,
                    output_path=temp_dir,
                    base_size=640 if args.mode == "industrial_ocv" else 1024,
                    image_size=640,
                    crop_mode=args.mode == "industrial_ocv",
                    save_results=False,
                    test_compress=True,
                    eval_mode=True,
                )
            except TypeError:
                result = model.infer(
                    tokenizer,
                    prompt=prompt,
                    image=pil_image,
                    image_file=input_path,
                    output_path=temp_dir,
                    base_size=640 if args.mode == "industrial_ocv" else 1024,
                    image_size=640,
                    crop_mode=args.mode == "industrial_ocv",
                    save_results=False,
                    test_compress=True,
                    eval_mode=True,
                )

    if isinstance(result, dict):
        text = result.get("text") or result.get("output") or result.get("response") or ""
    else:
        text = str(result)

    print(json.dumps({"text": text}))


if __name__ == "__main__":
    main()
