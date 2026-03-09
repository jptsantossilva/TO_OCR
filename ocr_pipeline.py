from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
import json
import os
import subprocess
import re
import sys
from typing import Any

import cv2
import numpy as np


@dataclass
class OCRResult:
    source: str
    backend: str
    mode: str
    recognized_text: str
    raw_text: str
    matched_tokens: list[str]
    regex_match: bool
    score: float
    expected_text: str
    expected_match: bool
    similarity: float
    barcode_found: bool
    barcode_value: str
    roi_bbox: tuple[int, int, int, int] | None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        if self.roi_bbox is not None:
            data["roi_bbox"] = ",".join(str(value) for value in self.roi_bbox)
        return data


class OCRPipeline:
    def __init__(
        self,
        languages: list[str] | None = None,
        required_tokens: list[str] | None = None,
        regex_pattern: str = "",
        expected_text: str = "",
        backend: str = "easyocr",
        mode: str = "ocr",
        deepseek_device: str = "cpu",
        easyocr_device: str = "auto",
    ) -> None:
        self.languages = languages or ["en", "pt"]
        self.required_tokens = [token.strip() for token in (required_tokens or []) if token.strip()]
        self.regex_pattern = regex_pattern.strip()
        self.expected_text = self._normalize_text(expected_text.strip()) if expected_text.strip() else ""
        self.backend = backend.strip().lower() or "easyocr"
        self.mode = mode.strip().lower() or "ocr"
        self.deepseek_device = deepseek_device.strip().lower() or "cpu"
        self.easyocr_device = easyocr_device.strip().lower() or "auto"
        self._reader = None
        self._allowlist = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ. -/"

    def _load_reader(self):
        if self._reader is not None:
            return self._reader

        if self.backend != "easyocr":
            return None

        try:
            import easyocr
        except ImportError as exc:
            raise RuntimeError(
                "A dependência 'easyocr' não está instalada. Instale as dependências do projeto."
            ) from exc

        self._reader = _build_easyocr_reader(tuple(self.languages), self._resolve_easyocr_gpu())
        return self._reader

    def _resolve_easyocr_gpu(self) -> bool:
        if self.easyocr_device == "cuda":
            return True
        if self.easyocr_device == "cpu":
            return False

        # Fallback opcional por variável de ambiente:
        # EASYOCR_GPU=true|false
        raw = os.environ.get("EASYOCR_GPU", "").strip().lower()
        if raw in {"1", "true", "yes", "on"}:
            return True
        if raw in {"0", "false", "no", "off"}:
            return False

        try:
            import torch
        except ImportError:
            return False
        return bool(torch.cuda.is_available())

    def analyze(self, image_bgr: np.ndarray, source: str = "image") -> OCRResult:
        barcode_value: str | None = None
        barcode_bbox: tuple[int, int, int, int] | None = None
        if self.mode == "industrial_ocv":
            barcode_value, barcode_bbox = self._decode_barcode(image_bgr)
            roi, roi_bbox = self._extract_single_line_roi(image_bgr, barcode_bbox)
        else:
            roi, roi_bbox = image_bgr, None

        recognized_lines = self._read_text(roi)

        if not recognized_lines:
            recognized_lines = self._read_text(image_bgr)
            roi_bbox = None

        raw_text = self._normalize_text(" ".join(recognized_lines).strip())
        recognized_text = self._correct_to_expected(raw_text, self.expected_text) if self.expected_text else raw_text
        matched_tokens = self._find_required_tokens(recognized_text)
        regex_match = self._matches_regex(recognized_text)
        similarity = self._similarity(recognized_text, self.expected_text) if self.expected_text else 0.0
        expected_match = bool(self.expected_text) and similarity >= 0.92
        score = self._score_result(matched_tokens, regex_match, similarity)

        return OCRResult(
            source=source,
            backend=self.backend,
            mode=self.mode,
            recognized_text=recognized_text,
            raw_text=raw_text,
            matched_tokens=matched_tokens,
            regex_match=regex_match,
            score=score,
            expected_text=self.expected_text,
            expected_match=expected_match,
            similarity=similarity,
            barcode_found=barcode_bbox is not None,
            barcode_value=barcode_value or "",
            roi_bbox=roi_bbox,
        )

    def build_debug_views(self, image_bgr: np.ndarray) -> list[tuple[str, np.ndarray]]:
        barcode_value: str | None = None
        barcode_bbox: tuple[int, int, int, int] | None = None
        if self.mode == "industrial_ocv":
            barcode_value, barcode_bbox = self._decode_barcode(image_bgr)
            roi, _ = self._extract_single_line_roi(image_bgr, barcode_bbox)
        else:
            roi = image_bgr

        views: list[tuple[str, np.ndarray]] = [("Original", image_bgr)]
        if barcode_bbox is not None:
            annotated = image_bgr.copy()
            x, y, w, h = barcode_bbox
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (35, 120, 255), 2)
            label = barcode_value or "Barcode region"
            cv2.putText(
                annotated,
                label,
                (x, max(24, y - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (35, 120, 255),
                2,
                cv2.LINE_AA,
            )
            views.append(("Barcode region", annotated))

        base_label = "ROI OCR" if self.mode == "industrial_ocv" else "Input OCR"
        views.append((base_label, roi))

        for label, variant in self._preprocess_for_ocr(roi):
            views.append((label, self._ensure_bgr(variant)))

        return views

    def evaluate_debug_filters(self, image_bgr: np.ndarray, source: str = "debug") -> list[dict[str, Any]]:
        barcode_value: str | None = None
        barcode_bbox: tuple[int, int, int, int] | None = None
        if self.mode == "industrial_ocv":
            barcode_value, barcode_bbox = self._decode_barcode(image_bgr)
            roi, roi_bbox = self._extract_single_line_roi(image_bgr, barcode_bbox)
        else:
            roi, roi_bbox = image_bgr, None

        base_label = "ROI OCR" if self.mode == "industrial_ocv" else "Input OCR"
        candidates: list[tuple[str, np.ndarray, tuple[int, int, int, int] | None]] = [
            (base_label, roi, roi_bbox)
        ]
        for label, variant in self._preprocess_for_ocr(roi):
            candidates.append((label, self._ensure_bgr(variant), roi_bbox))

        evaluations: list[dict[str, Any]] = []
        for label, candidate_image, candidate_bbox in candidates:
            try:
                is_preprocessed = label != base_label
                recognized_lines = self._read_text(
                    candidate_image,
                    already_preprocessed=is_preprocessed,
                )
                raw_text = self._normalize_text(" ".join(recognized_lines).strip())
                recognized_text = self._correct_to_expected(raw_text, self.expected_text) if self.expected_text else raw_text
                matched_tokens = self._find_required_tokens(recognized_text)
                regex_match = self._matches_regex(recognized_text)
                similarity = self._similarity(recognized_text, self.expected_text) if self.expected_text else 0.0
                expected_match = bool(self.expected_text) and similarity >= 0.92
                score = self._score_result(matched_tokens, regex_match, similarity)
                error = ""
            except Exception as exc:
                raw_text = ""
                recognized_text = ""
                matched_tokens = []
                regex_match = False
                similarity = 0.0
                expected_match = False
                score = 0.0
                error = str(exc)

            evaluations.append(
                {
                    "label": label,
                    "image": candidate_image,
                    "recognized_text": recognized_text,
                    "raw_text": raw_text,
                    "matched_tokens": matched_tokens,
                    "regex_match": regex_match,
                    "score": score,
                    "similarity": similarity,
                    "expected_match": expected_match,
                    "barcode_value": barcode_value or "",
                    "roi_bbox": candidate_bbox,
                    "source": source,
                    "error": error,
                }
            )

        return evaluations

    def annotate(self, image_bgr: np.ndarray, result: OCRResult) -> np.ndarray:
        annotated = image_bgr.copy()

        if result.roi_bbox is not None:
            x, y, w, h = result.roi_bbox
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (45, 198, 83), 2)
            cv2.putText(
                annotated,
                "ROI OCR",
                (x, max(20, y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (45, 198, 83),
                2,
                cv2.LINE_AA,
            )

        overlay_lines = [
            f"Score: {result.score:.2f}",
            f"Barcode: {result.barcode_value or 'n/a'}",
            (result.recognized_text[:60] + "...") if len(result.recognized_text) > 60 else result.recognized_text,
        ]

        panel_height = 28 * len(overlay_lines) + 16
        cv2.rectangle(annotated, (10, 10), (min(annotated.shape[1] - 10, 760), 10 + panel_height), (0, 0, 0), -1)
        for index, line in enumerate(overlay_lines):
            cv2.putText(
                annotated,
                line,
                (18, 35 + (index * 28)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        return annotated

    def _decode_barcode(self, image_bgr: np.ndarray) -> tuple[str | None, tuple[int, int, int, int] | None]:
        try:
            from pyzbar.pyzbar import decode
        except ImportError:
            decode = None

        if decode is not None:
            decoded_objects = decode(image_bgr)
            if decoded_objects:
                decoded = max(decoded_objects, key=lambda item: item.rect.width * item.rect.height)
                rect = decoded.rect
                return decoded.data.decode("utf-8", errors="ignore"), (rect.left, rect.top, rect.width, rect.height)

        return None, self._detect_barcode_region(image_bgr)

    def _detect_barcode_region(self, image_bgr: np.ndarray) -> tuple[int, int, int, int] | None:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        grad_x = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
        grad_y = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=-1)
        gradient = cv2.convertScaleAbs(cv2.subtract(grad_x, grad_y))
        gradient = cv2.blur(gradient, (9, 9))

        _, thresh = cv2.threshold(gradient, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 7))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        closed = cv2.erode(closed, None, iterations=2)
        closed = cv2.dilate(closed, None, iterations=4)

        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates: list[tuple[int, int, int, int]] = []
        image_area = image_bgr.shape[0] * image_bgr.shape[1]

        image_center_x = image_bgr.shape[1] / 2.0
        image_center_y = image_bgr.shape[0] * 0.68

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            aspect_ratio = w / max(float(h), 1.0)
            if area < image_area * 0.015:
                continue
            if aspect_ratio < 1.8:
                continue
            candidates.append((x, y, w, h))

        if not candidates:
            return None

        def candidate_score(bbox: tuple[int, int, int, int]) -> float:
            x, y, w, h = bbox
            cx = x + (w / 2.0)
            cy = y + (h / 2.0)
            area_score = w * h
            center_penalty = abs(cx - image_center_x) * 1.25 + abs(cy - image_center_y) * 1.8
            return area_score - center_penalty

        return max(candidates, key=candidate_score)

    def _extract_text_roi(
        self,
        image_bgr: np.ndarray,
        barcode_bbox: tuple[int, int, int, int] | None,
    ) -> tuple[np.ndarray, tuple[int, int, int, int] | None]:
        height, width = image_bgr.shape[:2]

        if barcode_bbox is None:
            return image_bgr, None

        x, y, w, h = barcode_bbox
        left = max(0, x - int(w * 0.06))
        right = min(width, x + w + int(w * 0.06))
        top = max(0, y - int(h * 0.95))
        bottom = max(0, y - int(h * 0.05))

        if bottom <= top or right <= left:
            return image_bgr, None

        roi = image_bgr[top:bottom, left:right]
        refined_roi, refined_bbox = self._refine_text_band(roi, (left, top, right - left, bottom - top))
        return refined_roi, refined_bbox

    def _extract_single_line_roi(
        self,
        image_bgr: np.ndarray,
        barcode_bbox: tuple[int, int, int, int] | None,
    ) -> tuple[np.ndarray, tuple[int, int, int, int] | None]:
        height, width = image_bgr.shape[:2]
        if barcode_bbox is None:
            fallback = self._resize_line_roi(image_bgr)
            return fallback, None

        x, y, w, h = barcode_bbox
        left = max(0, x - int(w * 0.02))
        right = min(width, x + w + int(w * 0.02))
        top = max(0, y - int(h * 0.82))
        bottom = max(0, y - int(h * 0.20))
        if bottom <= top or right <= left:
            fallback = self._resize_line_roi(image_bgr)
            return fallback, None

        roi = image_bgr[top:bottom, left:right]
        refined_roi, refined_bbox = self._refine_single_line_band(roi, (left, top, right - left, bottom - top))
        return self._resize_line_roi(refined_roi), refined_bbox

    def _refine_single_line_band(
        self,
        roi_bgr: np.ndarray,
        roi_bbox: tuple[int, int, int, int],
    ) -> tuple[np.ndarray, tuple[int, int, int, int]]:
        left, top, width, height = roi_bbox
        gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
        enlarged = cv2.resize(gray, None, fx=4.0, fy=4.0, interpolation=cv2.INTER_CUBIC)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(enlarged)
        blackhat = cv2.morphologyEx(
            enhanced,
            cv2.MORPH_BLACKHAT,
            cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5)),
        )
        _, binary = cv2.threshold(blackhat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        row_energy = binary.mean(axis=1)
        row_threshold = max(row_energy.mean() * 1.15, 8.0)
        strong_rows = np.where(row_energy >= row_threshold)[0]
        if len(strong_rows) == 0:
            return roi_bgr, roi_bbox

        band_top = max(0, int(strong_rows[0] / 4.0) - 4)
        band_bottom = min(height, int(strong_rows[-1] / 4.0) + 4)
        if band_bottom - band_top < 10:
            return roi_bgr, roi_bbox

        refined = roi_bgr[band_top:band_bottom, :]
        return refined, (left, top + band_top, width, band_bottom - band_top)

    def _resize_line_roi(self, roi_bgr: np.ndarray) -> np.ndarray:
        height, width = roi_bgr.shape[:2]
        if height == 0 or width == 0:
            return roi_bgr
        target_height = 96 if self.mode == "industrial_ocv" else height
        scale = target_height / float(height)
        target_width = max(int(width * scale), 32)
        return cv2.resize(roi_bgr, (target_width, target_height), interpolation=cv2.INTER_CUBIC)

    def _refine_text_band(
        self,
        roi_bgr: np.ndarray,
        roi_bbox: tuple[int, int, int, int],
    ) -> tuple[np.ndarray, tuple[int, int, int, int]]:
        left, top, width, height = roi_bbox
        gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
        blackhat = cv2.morphologyEx(
            gray,
            cv2.MORPH_BLACKHAT,
            cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7)),
        )
        _, binary = cv2.threshold(blackhat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        row_strength = binary.mean(axis=1)
        strong_rows = np.where(row_strength > row_strength.mean() * 1.25)[0]

        if len(strong_rows) == 0:
            return roi_bgr, roi_bbox

        band_top = max(0, int(strong_rows[0] / 3.0) - 6)
        band_bottom = min(height, int(strong_rows[-1] / 3.0) + 6)

        if band_bottom - band_top < 12:
            return roi_bgr, roi_bbox

        refined = roi_bgr[band_top:band_bottom, :]
        return refined, (left, top + band_top, width, band_bottom - band_top)

    def _read_text(self, image_bgr: np.ndarray, already_preprocessed: bool = False) -> list[str]:
        if self.backend == "deepseek_ocr":
            return self._read_text_deepseek(image_bgr)
        if self.backend == "paddleocr":
            return self._read_text_paddleocr(image_bgr, already_preprocessed=already_preprocessed)
        if self.backend == "tesseract":
            return self._read_text_tesseract(image_bgr, already_preprocessed=already_preprocessed)

        reader = self._load_reader()
        best_lines: list[str] = []
        best_score = -1.0

        variants = [("Input", image_bgr)] if already_preprocessed else self._preprocess_for_ocr(image_bgr)
        for variant_index, (_, variant) in enumerate(variants):
            raw_lines = reader.readtext(
                variant,
                detail=1,
                paragraph=False,
                allowlist=self._allowlist,
                decoder="greedy",
                text_threshold=0.5,
                low_text=0.2,
                link_threshold=0.2,
            )

            candidate_lines: list[str] = []
            confidence_sum = 0.0
            for bbox, text, confidence in raw_lines:
                normalized = self._normalize_text(text.strip())
                if confidence < 0.15 or len(normalized) < 2:
                    continue

                if self._looks_like_barcode_digits(normalized):
                    continue

                line_score = float(confidence)
                line_score += self._score_line_pattern(normalized)
                line_score += self._score_line_position(bbox, variant.shape[0])
                if self.expected_text:
                    line_score += self._similarity(normalized, self.expected_text) * 2.5

                if line_score < 0.55:
                    continue

                candidate_lines.append(normalized)
                confidence_sum += float(confidence)

            if not candidate_lines:
                continue

            candidate_text = self._select_best_text(candidate_lines)
            candidate_score = confidence_sum
            candidate_score += len(self._find_required_tokens(candidate_text)) * 0.8
            if self._matches_regex(candidate_text):
                candidate_score += 1.2
            if self.expected_text:
                candidate_score += self._similarity(candidate_text, self.expected_text) * 2.0
            if variant_index == 0:
                candidate_score += 0.25
            if re.search(r"\b\d{2}\s*\.\s*\d{4}\b", candidate_text):
                candidate_score += 0.8
            if re.search(r"\b[A-Z0-9]{4,}\b", candidate_text):
                candidate_score += 0.4

            if candidate_score > best_score:
                best_score = candidate_score
                best_lines = candidate_lines

        return best_lines

    def _read_text_tesseract(self, image_bgr: np.ndarray, already_preprocessed: bool = False) -> list[str]:
        try:
            import pytesseract
        except ImportError as exc:
            raise RuntimeError(
                "A dependência 'pytesseract' não está instalada. Instale as dependências do projeto."
            ) from exc

        variants = [("Input", image_bgr)] if already_preprocessed else self._preprocess_for_ocr(image_bgr)

        if self.mode == "industrial_ocv":
            text = self._read_text_tesseract_single_line(variants)
            return [text] if text else []

        best_text = ""
        best_score = -1.0

        psm_values = [7, 13] if self.mode == "industrial_ocv" else [6, 7]
        oem_values = [1]
        for variant_index, (_, variant) in enumerate(variants):
            for psm in psm_values:
                for oem in oem_values:
                    whitelist = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ.-/"
                    config = (
                        f"--oem {oem} --psm {psm} "
                        f"-c tessedit_char_whitelist={whitelist} "
                    )
                    text = pytesseract.image_to_string(variant, config=config)
                    normalized = self._normalize_text(text)
                    if not normalized:
                        continue

                    score = self._score_line_pattern(normalized)
                    if self.expected_text:
                        score += self._similarity(normalized, self.expected_text) * (
                            3.2 if self.mode == "industrial_ocv" else 2.0
                        )
                        score += self._weighted_alignment_score(normalized, self.expected_text) * 2.4
                    if self._matches_regex(normalized):
                        score += 1.2
                    if variant_index == 0:
                        score += 0.35
                    if len(normalized) > len(self.expected_text) + 4 and self.expected_text:
                        score -= 0.6
                    if self.mode == "industrial_ocv":
                        if len(normalized) < 6:
                            score -= 0.8
                        if re.search(r"[^0-9A-Z\.\s/-]", normalized):
                            score -= 0.6

                    if score > best_score:
                        best_score = score
                        best_text = normalized

        return [best_text] if best_text else []

    def _read_text_tesseract_single_line(self, variants: list[tuple[str, np.ndarray]]) -> str:
        import pytesseract

        best_text = ""
        best_score = -1.0

        for variant_index, (_, variant) in enumerate(variants):
            left_image, right_image = self._split_line_regions(variant)
            left_text = self._run_tesseract_segment(
                pytesseract,
                left_image,
                whitelist="0123456789. ",
                psm_values=[7, 13],
            )
            right_text = self._run_tesseract_segment(
                pytesseract,
                right_image,
                whitelist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
                psm_values=[7, 13],
            )

            combined = self._normalize_text(" ".join(part for part in [left_text, right_text] if part).strip())
            if not combined:
                continue

            score = self._score_line_pattern(combined)
            if left_text:
                score += 0.8
            if right_text:
                score += 1.2
            if self.expected_text:
                score += self._similarity(combined, self.expected_text) * 3.0
                score += self._weighted_alignment_score(combined, self.expected_text) * 2.8
            if re.search(r"\b\d{2}\s*\.\s*\d{4}\b", combined):
                score += 1.0
            if re.search(r"\bL\d{3}[A-Z0-9]\b", combined):
                score += 1.2
            if variant_index == 0:
                score += 0.35

            if score > best_score:
                best_score = score
                best_text = combined

        return best_text

    def _split_line_regions(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        height, width = gray.shape[:2]
        split_x = int(width * 0.58)
        split_x = min(max(split_x, int(width * 0.45)), int(width * 0.72))

        left = gray[:, :split_x]
        right = gray[:, split_x:]

        left = self._pad_image(left, pad_x=12, pad_y=8)
        right = self._pad_image(right, pad_x=12, pad_y=8)
        return left, right

    def _pad_image(self, image: np.ndarray, pad_x: int = 8, pad_y: int = 6) -> np.ndarray:
        border_value = 255 if image.ndim == 2 else (255, 255, 255)
        return cv2.copyMakeBorder(
            image,
            pad_y,
            pad_y,
            pad_x,
            pad_x,
            cv2.BORDER_CONSTANT,
            value=border_value,
        )

    def _run_tesseract_segment(
        self,
        pytesseract_module: Any,
        image: np.ndarray,
        whitelist: str,
        psm_values: list[int],
    ) -> str:
        best_text = ""
        best_score = -1.0

        for psm in psm_values:
            config = (
                f"--oem 1 --psm {psm} "
                f"-c tessedit_char_whitelist={whitelist}"
            )
            try:
                text = pytesseract_module.image_to_string(image, config=config)
            except Exception:
                continue

            normalized = self._normalize_text(text)
            if not normalized:
                continue

            score = len(normalized)
            if re.search(r"\d", normalized):
                score += 0.4
            if re.search(r"[A-Z]", normalized):
                score += 0.4
            if score > best_score:
                best_score = score
                best_text = normalized

        return best_text

    def _read_text_deepseek(self, image_bgr: np.ndarray) -> list[str]:
        if self.deepseek_device == "cpu":
            return self._read_text_deepseek_cpu_subprocess(image_bgr)

        try:
            from PIL import Image
            import torch
            from transformers import AutoModel, AutoTokenizer
            import tempfile
        except ImportError as exc:
            raise RuntimeError(
                "O backend DeepSeek-OCR requer dependências opcionais: torch, transformers e pillow."
            ) from exc

        model_id = os.environ.get("DEEPSEEK_OCR_MODEL_ID", "deepseek-ai/DeepSeek-OCR")
        revision = _resolve_deepseek_revision(self.deepseek_device)
        tokenizer, model, device, dtype = _load_deepseek_ocr_model(model_id, revision, self.deepseek_device)

        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb)
        prompt = self._deepseek_prompt()

        with tempfile.TemporaryDirectory(prefix="deepseek_ocr_") as temp_dir:
            temp_path = os.path.join(temp_dir, "input.png")
            pil_image.save(temp_path)
            try:
                with torch.inference_mode():
                    result = model.infer(
                        tokenizer,
                        prompt=prompt,
                        image_file=temp_path,
                        output_path=temp_dir,
                        base_size=640 if self.mode == "industrial_ocv" else 1024,
                        image_size=640,
                        crop_mode=self.mode == "industrial_ocv",
                        save_results=False,
                        test_compress=True,
                        eval_mode=True,
                    )
            except TypeError:
                with torch.inference_mode():
                    result = model.infer(
                        tokenizer,
                        prompt=prompt,
                        image=pil_image,
                        image_file=temp_path,
                        output_path=temp_dir,
                        base_size=640 if self.mode == "industrial_ocv" else 1024,
                        image_size=640,
                        crop_mode=self.mode == "industrial_ocv",
                        save_results=False,
                        test_compress=True,
                        eval_mode=True,
                    )

        if isinstance(result, dict):
            text = result.get("text") or result.get("output") or result.get("response") or ""
        else:
            text = str(result)

        normalized = self._normalize_text(self._extract_deepseek_candidate(text))
        return [normalized] if normalized else []

    def _read_text_paddleocr(self, image_bgr: np.ndarray, already_preprocessed: bool = False) -> list[str]:
        try:
            from paddleocr import PaddleOCR
        except ImportError as exc:
            raise RuntimeError(
                "O backend PaddleOCR requer dependencias opcionais. "
                "Instale com: python -m pip install paddleocr paddlepaddle"
            ) from exc

        reader = _build_paddleocr_reader()
        variants = [("Input", image_bgr)] if already_preprocessed else self._preprocess_for_ocr(image_bgr)

        best_text = ""
        best_score = -1.0
        for variant_index, (_, variant) in enumerate(variants):
            variant_bgr = self._ensure_bgr(variant)
            text = self._paddle_predict_text(reader, variant_bgr)
            normalized = self._normalize_text(text)
            if not normalized:
                continue

            score = self._score_line_pattern(normalized)
            if self.expected_text:
                score += self._similarity(normalized, self.expected_text) * 2.8
                score += self._weighted_alignment_score(normalized, self.expected_text) * 2.2
            if self._matches_regex(normalized):
                score += 1.0
            if variant_index == 0:
                score += 0.2

            if score > best_score:
                best_score = score
                best_text = normalized

        return [best_text] if best_text else []

    def _paddle_predict_text(self, reader: Any, image_bgr: np.ndarray) -> str:
        result = None
        errors: list[str] = []
        if hasattr(reader, "predict"):
            try:
                result = reader.predict(input=image_bgr)
            except TypeError:
                try:
                    result = reader.predict(image_bgr)
                except Exception as exc:
                    errors.append(str(exc))
            except Exception as exc:
                errors.append(str(exc))
        elif hasattr(reader, "ocr"):
            try:
                result = reader.ocr(image_bgr, cls=False)
            except TypeError:
                try:
                    result = reader.ocr(image_bgr)
                except Exception as exc:
                    errors.append(str(exc))
            except Exception as exc:
                errors.append(str(exc))

        if result is None and hasattr(reader, "ocr"):
            try:
                result = reader.ocr(image_bgr)
            except Exception as exc:
                errors.append(str(exc))

        if result is None:
            if errors:
                message = errors[-1]
                if "ConvertPirAttribute2RuntimeAttribute" in message or "pir::ArrayAttribute" in message:
                    raise RuntimeError(
                        "PaddleOCR/PaddlePaddle incompatível com o runtime atual (erro PIR/oneDNN). "
                        "Tente: python -m pip install -U paddlepaddle==3.2.0 paddleocr==3.3.3"
                    )
                raise RuntimeError(message)
            return ""

        texts = self._extract_texts_recursive(result)
        if not texts:
            return ""
        return " ".join(texts)

    def _extract_texts_recursive(self, obj: Any) -> list[str]:
        texts: list[str] = []
        if obj is None:
            return texts

        if isinstance(obj, str):
            cleaned = obj.strip()
            if cleaned:
                texts.append(cleaned)
            return texts

        if isinstance(obj, dict):
            for key, value in obj.items():
                lowered = str(key).lower()
                if lowered in {"rec_text", "text", "texts", "rec_texts"}:
                    if isinstance(value, list):
                        for item in value:
                            texts.extend(self._extract_texts_recursive(item))
                    else:
                        texts.extend(self._extract_texts_recursive(value))
                else:
                    texts.extend(self._extract_texts_recursive(value))
            return texts

        if isinstance(obj, (list, tuple)):
            # Legacy format: [[box, (text, conf)], ...]
            if len(obj) == 2 and isinstance(obj[1], (list, tuple)) and len(obj[1]) >= 1 and isinstance(obj[1][0], str):
                texts.extend(self._extract_texts_recursive(obj[1][0]))
                return texts
            for item in obj:
                texts.extend(self._extract_texts_recursive(item))
            return texts

        if hasattr(obj, "res"):
            texts.extend(self._extract_texts_recursive(getattr(obj, "res")))
            return texts

        if hasattr(obj, "__dict__"):
            texts.extend(self._extract_texts_recursive(vars(obj)))
            return texts

        return texts

    def _read_text_deepseek_cpu_subprocess(self, image_bgr: np.ndarray) -> list[str]:
        import tempfile

        model_id = os.environ.get("DEEPSEEK_OCR_MODEL_ID", "deepseek-ai/DeepSeek-OCR")
        revision = _resolve_deepseek_revision("cpu")
        runner_path = os.path.join(os.path.dirname(__file__), "deepseek_cpu_runner.py")

        with tempfile.TemporaryDirectory(prefix="deepseek_cpu_call_") as temp_dir:
            image_path = os.path.join(temp_dir, "input.png")
            cv2.imwrite(image_path, image_bgr)
            command = [
                sys.executable,
                runner_path,
                "--model-id",
                model_id,
                "--image-path",
                image_path,
                "--revision",
                revision,
                "--mode",
                self.mode,
                "--expected-text",
                self.expected_text,
            ]
            try:
                completed = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    check=False,
                    cwd=os.path.dirname(__file__),
                    env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
                    timeout=300,
                )
            except subprocess.TimeoutExpired as exc:
                raise RuntimeError(
                    "DeepSeek-OCR demorou mais de 300s. "
                    "Em CPU isso pode acontecer; reduza resolução/filtros ou use outro backend."
                ) from exc

            if completed.returncode != 0:
                error_text = completed.stderr.strip() or completed.stdout.strip() or "DeepSeek CPU runner failed."
                raise RuntimeError(error_text)

            try:
                payload = json.loads(completed.stdout.strip())
            except json.JSONDecodeError as exc:
                raise RuntimeError(completed.stdout.strip() or "Invalid DeepSeek CPU runner output.") from exc

        text = str(payload.get("text", ""))
        normalized = self._normalize_text(self._extract_deepseek_candidate(text))
        return [normalized] if normalized else []

    def _preprocess_for_ocr(self, image_bgr: np.ndarray) -> list[tuple[str, np.ndarray]]:
        if self.mode == "industrial_ocv":
            return self._preprocess_single_line_for_ocr(image_bgr)

        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        scale = 3.0 if self.mode == "industrial_ocv" else 2.2
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        gray = cv2.fastNlMeansDenoising(gray, None, 12, 7, 21)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        clahe_gray = clahe.apply(gray)

        _, otsu = cv2.threshold(clahe_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        adaptive = cv2.adaptiveThreshold(
            clahe_gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            5,
        )
        blackhat = cv2.morphologyEx(
            clahe_gray,
            cv2.MORPH_BLACKHAT,
            cv2.getStructuringElement(cv2.MORPH_RECT, (17, 5)),
        )
        _, blackhat_binary = cv2.threshold(blackhat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        variants: list[tuple[str, np.ndarray]] = [
            ("CLAHE grayscale", clahe_gray),
            ("ROI + Otsu threshold", otsu),
            ("ROI + Blackhat threshold", blackhat_binary),
        ]
        if self.mode == "industrial_ocv":
            variants.append(("ROI + Adaptive threshold", adaptive))
        normalized_variants: list[tuple[str, np.ndarray]] = []
        for label, variant in variants:
            if np.mean(variant == 255) < 0.45:
                variant = cv2.bitwise_not(variant)
            normalized_variants.append((label, variant))

        return normalized_variants

    def _preprocess_single_line_for_ocr(self, image_bgr: np.ndarray) -> list[tuple[str, np.ndarray]]:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        clahe_gray = clahe.apply(gray)

        _, otsu = cv2.threshold(clahe_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        blackhat = cv2.morphologyEx(
            clahe_gray,
            cv2.MORPH_BLACKHAT,
            cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5)),
        )
        _, blackhat_binary = cv2.threshold(blackhat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        adaptive = cv2.adaptiveThreshold(
            clahe_gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            21,
            3,
        )

        variants: list[tuple[str, np.ndarray]] = [
            ("CLAHE grayscale", clahe_gray),
            ("ROI + Otsu threshold", otsu),
            ("ROI + Blackhat threshold", blackhat_binary),
            ("ROI + Adaptive threshold", adaptive),
        ]
        output: list[tuple[str, np.ndarray]] = []
        for label, variant in variants:
            if np.mean(variant == 255) < 0.45:
                variant = cv2.bitwise_not(variant)
            output.append((label, variant))
        return output

    def _ensure_bgr(self, image: np.ndarray) -> np.ndarray:
        if image.ndim == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        return image

    def _find_required_tokens(self, text: str) -> list[str]:
        normalized_text = text.lower()
        return [token for token in self.required_tokens if token.lower() in normalized_text]

    def _matches_regex(self, text: str) -> bool:
        if not self.regex_pattern:
            return False

        try:
            return re.search(self.regex_pattern, text) is not None
        except re.error:
            return False

    def _score_result(self, matched_tokens: list[str], regex_match: bool, similarity: float) -> float:
        if not self.required_tokens and not self.regex_pattern and not self.expected_text:
            return 1.0

        score = 0.0
        components = 0
        if self.required_tokens:
            score += len(matched_tokens) / len(self.required_tokens)
            components += 1
        if self.regex_pattern:
            score += 1.0 if regex_match else 0.0
            components += 1
        if self.expected_text:
            score += similarity
            components += 1

        if components == 0:
            return 1.0

        return round(min(score / components, 1.0), 2)

    def _correct_to_expected(self, recognized_text: str, expected_text: str) -> str:
        if not recognized_text or not expected_text:
            return recognized_text

        recognized = self._normalize_text(recognized_text)
        expected = self._normalize_text(expected_text)

        if self._similarity(recognized, expected) >= 0.92:
            return expected

        if self.mode == "industrial_ocv":
            return self._force_align_to_expected(recognized, expected)

        best_candidate = recognized
        best_score = self._weighted_alignment_score(recognized, expected)

        trimmed = recognized
        for _ in range(4):
            trimmed = re.sub(r"\s+[A-Z0-9]{1,3}$", "", trimmed).strip()
            if not trimmed:
                break
            score = self._weighted_alignment_score(trimmed, expected)
            if score > best_score:
                best_candidate = trimmed
                best_score = score

        corrected_chars: list[str] = []
        recognized_index = 0
        for expected_char in expected:
            while recognized_index < len(best_candidate) and best_candidate[recognized_index] == " " and expected_char != " ":
                recognized_index += 1

            if recognized_index >= len(best_candidate):
                corrected_chars.append(expected_char)
                continue

            recognized_char = best_candidate[recognized_index]
            if recognized_char == expected_char:
                corrected_chars.append(expected_char)
                recognized_index += 1
                continue

            if self._is_confusable(recognized_char, expected_char):
                corrected_chars.append(expected_char)
                recognized_index += 1
                continue

            if expected_char == " ":
                corrected_chars.append(expected_char)
                continue

            corrected_chars.append(recognized_char)
            recognized_index += 1

        corrected = self._normalize_text("".join(corrected_chars))
        return expected if self._weighted_alignment_score(corrected, expected) >= 0.9 else corrected

    def _force_align_to_expected(self, recognized: str, expected: str) -> str:
        compact_recognized = re.sub(r"\s+", "", recognized)
        compact_expected = re.sub(r"\s+", "", expected)
        output: list[str] = []
        recognized_index = 0

        for expected_char in compact_expected:
            while recognized_index < len(compact_recognized):
                current = compact_recognized[recognized_index]
                if current == expected_char or self._is_confusable(current, expected_char):
                    output.append(expected_char)
                    recognized_index += 1
                    break
                if recognized_index + 1 < len(compact_recognized):
                    next_char = compact_recognized[recognized_index + 1]
                    if next_char == expected_char or self._is_confusable(next_char, expected_char):
                        recognized_index += 2
                        output.append(expected_char)
                        break
                recognized_index += 1
            else:
                output.append(expected_char)

        rebuilt = "".join(output)
        if len(rebuilt) == len(compact_expected):
            if "." in expected:
                rebuilt = rebuilt[:2] + " . " + rebuilt[2:]
            if len(compact_expected) > 6:
                rebuilt = rebuilt[:8] + " " + rebuilt[8:]
        rebuilt = self._normalize_text(rebuilt)
        return expected if self._weighted_alignment_score(rebuilt, expected) >= 0.82 else rebuilt

    def _deepseek_prompt(self) -> str:
        if self.mode == "industrial_ocv" and self.expected_text:
            return (
                "<image>\n"
                "OCR this image. Return only the single printed lot/code line above the barcode. "
                f"Expected format similar to: {self.expected_text}. No explanation."
            )
        if self.mode == "industrial_ocv":
            return "<image>\nOCR this image. Return only the single printed line above the barcode. No explanation."
        return "<image>\nFree OCR."

    def _extract_deepseek_candidate(self, text: str) -> str:
        candidates = [self._normalize_text(line) for line in text.splitlines() if line.strip()]
        if not candidates:
            return text
        return self._select_best_text(candidates)

    def _normalize_text(self, text: str) -> str:
        text = text.upper()
        text = text.replace("|", "1")
        text = text.replace("LSO", "L30")
        text = text.replace("LS", "L3")
        text = re.sub(r"\s+", " ", text).strip()
        text = re.sub(r"(\d)\.(\d)", r"\1 . \2", text)
        text = re.sub(r"\s*\.\s*", " . ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _select_best_text(self, candidate_lines: list[str]) -> str:
        if not candidate_lines:
            return ""

        if self.expected_text:
            return max(candidate_lines, key=lambda line: self._similarity(line, self.expected_text))

        return max(candidate_lines, key=self._score_line_pattern)

    def _looks_like_barcode_digits(self, text: str) -> bool:
        compact = re.sub(r"\s+", "", text)
        return compact.isdigit() and len(compact) >= 8

    def _score_line_pattern(self, text: str) -> float:
        score = 0.0
        if re.search(r"\b\d{2}\s*\.\s*\d{4}\b", text):
            score += 1.2
        if re.search(r"\b[A-Z]\d{3}[A-Z0-9]{1,3}\b", text):
            score += 1.0
        if re.search(r"[A-Z]", text):
            score += 0.25
        if re.search(r"\d", text):
            score += 0.25
        if len(text) > 24:
            score -= 0.5
        if self._looks_like_barcode_digits(text):
            score -= 1.0
        return score

    def _score_line_position(self, bbox: Any, image_height: int) -> float:
        if not bbox:
            return 0.0

        ys = [point[1] for point in bbox]
        center_y = sum(ys) / max(len(ys), 1)
        normalized = center_y / max(float(image_height), 1.0)
        return max(0.0, 0.45 - normalized)

    def _similarity(self, left: str, right: str) -> float:
        if not left or not right:
            return 0.0

        distance = self._levenshtein(left, right)
        baseline = max(len(left), len(right), 1)
        return round(max(0.0, 1.0 - (distance / baseline)), 2)

    def _weighted_alignment_score(self, recognized: str, expected: str) -> float:
        if not recognized or not expected:
            return 0.0

        matches = 0.0
        max_len = max(len(recognized), len(expected), 1)
        for left_char, right_char in zip(recognized, expected):
            if left_char == right_char:
                matches += 1.0
            elif self._is_confusable(left_char, right_char):
                matches += 0.85

        length_penalty = abs(len(recognized) - len(expected)) * 0.08
        return max(0.0, min(1.0, (matches / max_len) - length_penalty))

    def _is_confusable(self, left_char: str, right_char: str) -> bool:
        confusion_sets = (
            {"0", "O", "Q"},
            {"1", "I", "L"},
            {"2", "Z"},
            {"5", "S"},
            {"6", "G"},
            {"8", "B"},
        )
        for confusion_set in confusion_sets:
            if left_char in confusion_set and right_char in confusion_set:
                return True
        return False

    def _levenshtein(self, left: str, right: str) -> int:
        if left == right:
            return 0
        if not left:
            return len(right)
        if not right:
            return len(left)

        previous = list(range(len(right) + 1))
        for i, left_char in enumerate(left, start=1):
            current = [i]
            for j, right_char in enumerate(right, start=1):
                insertions = previous[j] + 1
                deletions = current[j - 1] + 1
                substitutions = previous[j - 1] + (left_char != right_char)
                current.append(min(insertions, deletions, substitutions))
            previous = current

        return previous[-1]


@lru_cache(maxsize=8)
def _build_easyocr_reader(languages: tuple[str, ...], use_gpu: bool):
    import easyocr

    return easyocr.Reader(list(languages), gpu=use_gpu)


@lru_cache(maxsize=1)
def _build_paddleocr_reader():
    import inspect
    os.environ.setdefault("FLAGS_use_mkldnn", "0")
    os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
    from paddleocr import PaddleOCR

    signature = inspect.signature(PaddleOCR.__init__)
    accepted = set(signature.parameters.keys())

    kwargs: dict[str, Any] = {"lang": "en"}
    if "use_doc_orientation_classify" in accepted:
        kwargs["use_doc_orientation_classify"] = False
    if "use_doc_unwarping" in accepted:
        kwargs["use_doc_unwarping"] = False
    if "use_textline_orientation" in accepted:
        kwargs["use_textline_orientation"] = False
    if "use_angle_cls" in accepted:
        kwargs["use_angle_cls"] = False

    if "device" in accepted:
        kwargs["device"] = "cpu"
    elif "use_gpu" in accepted:
        kwargs["use_gpu"] = False

    try:
        return PaddleOCR(**kwargs)
    except Exception:
        # Fallback para compatibilidade com APIs mais antigas/novas.
        fallback_kwargs = {"lang": "en"}
        if "use_angle_cls" in accepted:
            fallback_kwargs["use_angle_cls"] = False
        if "device" in accepted:
            fallback_kwargs["device"] = "cpu"
        elif "use_gpu" in accepted:
            fallback_kwargs["use_gpu"] = False
        try:
            return PaddleOCR(**fallback_kwargs)
        except Exception as exc:
            message = str(exc)
            if "ConvertPirAttribute2RuntimeAttribute" in message or "pir::ArrayAttribute" in message:
                raise RuntimeError(
                    "PaddleOCR/PaddlePaddle incompatível com o runtime atual (erro PIR/oneDNN). "
                    "Tente: python -m pip install -U paddlepaddle==3.2.0 paddleocr==3.3.3"
                ) from exc
            raise


@lru_cache(maxsize=4)
def _load_deepseek_ocr_model(model_id: str, revision: str, requested_device: str):
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:
        message = str(exc)
        if "LlamaFlashAttention2" in message:
            raise RuntimeError(
                "DeepSeek-OCR falhou por incompatibilidade de transformers. "
                "Instale as versões testadas: torch==2.6.0, transformers==4.46.3 e tokenizers==0.20.3."
            ) from exc
        raise

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
        revision=revision,
    )

    if requested_device == "cuda":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = "cpu"
    if device == "cuda":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        dtype = torch.float32

    try:
        model = AutoModel.from_pretrained(
            model_id,
            trust_remote_code=True,
            use_safetensors=True,
            revision=revision,
        )
    except ImportError as exc:
        message = str(exc)
        if "LlamaFlashAttention2" in message:
            raise RuntimeError(
                "DeepSeek-OCR falhou por incompatibilidade de transformers. "
                "Instale as versões testadas: torch==2.6.0, transformers==4.46.3 e tokenizers==0.20.3."
            ) from exc
        raise
    model = model.eval().to(device)
    try:
        model = model.to(dtype)
    except (TypeError, RuntimeError):
        pass

    return tokenizer, model, device, dtype


def _resolve_deepseek_revision(requested_device: str) -> str:
    revision = os.environ.get("DEEPSEEK_OCR_REVISION")
    if revision:
        return revision
    # CPU execution still needs the infer() fix from this revision.
    if requested_device == "cpu":
        return "refs/pr/6"
    return "main"
