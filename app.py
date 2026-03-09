from __future__ import annotations

import json
from pathlib import Path
import shutil
import time
from typing import Any

import av
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from streamlit_webrtc import VideoProcessorBase, WebRtcMode, webrtc_streamer

from ocr_pipeline import OCRPipeline, OCRResult


st.set_page_config(page_title="Flexible Packaging OCR", page_icon="🔎", layout="wide")

IMAGE_LIBRARY_DIR = Path("image_library")
FILTER_SETTINGS_PATH = IMAGE_LIBRARY_DIR / "filter_settings.json"
DEFAULT_FILTER_SETTINGS = {
    "exposure": 1.0,
    "saturation": 1.0,
    "contrast": 1.0,
    "brightness": 0.0,
    "sharpness": 0.0,
    "bw_threshold": 127,
    "grayscale": False,
}


def build_pipeline() -> OCRPipeline:
    return OCRPipeline(
        languages=["en", "pt"],
        required_tokens=st.session_state.get("required_tokens", []),
        regex_pattern=st.session_state.get("regex_pattern", ""),
        expected_text=st.session_state.get("expected_text", ""),
        backend=st.session_state.get("ocr_backend", "easyocr"),
        mode=st.session_state.get("ocr_mode", "ocr"),
        deepseek_device=st.session_state.get("deepseek_device", "cpu"),
        easyocr_device=st.session_state.get("easyocr_device", "auto"),
    )


def build_pipeline_for(backend: str, mode: str) -> OCRPipeline:
    return OCRPipeline(
        languages=["en", "pt"],
        required_tokens=st.session_state.get("required_tokens", []),
        regex_pattern=st.session_state.get("regex_pattern", ""),
        expected_text=st.session_state.get("expected_text", ""),
        backend=backend,
        mode=mode,
        deepseek_device=st.session_state.get("deepseek_device", "cpu"),
        easyocr_device=st.session_state.get("easyocr_device", "auto"),
    )


def pil_to_bgr(image: Image.Image) -> np.ndarray:
    rgb = np.array(image.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def ensure_library_paths() -> None:
    IMAGE_LIBRARY_DIR.mkdir(exist_ok=True)


def load_filter_settings() -> dict[str, Any]:
    ensure_library_paths()
    if not FILTER_SETTINGS_PATH.exists():
        return DEFAULT_FILTER_SETTINGS.copy()

    try:
        data = json.loads(FILTER_SETTINGS_PATH.read_text())
    except (json.JSONDecodeError, OSError):
        return DEFAULT_FILTER_SETTINGS.copy()

    return {
        "exposure": float(data.get("exposure", 1.0)),
        "saturation": float(data.get("saturation", 1.0)),
        "contrast": float(data.get("contrast", 1.0)),
        "brightness": float(data.get("brightness", 0.0)),
        "sharpness": float(data.get("sharpness", 0.0)),
        "bw_threshold": int(data.get("bw_threshold", 127)),
        "grayscale": bool(data.get("grayscale", False)),
    }


def save_filter_settings(settings: dict[str, Any]) -> None:
    ensure_library_paths()
    FILTER_SETTINGS_PATH.write_text(json.dumps(settings, indent=2))


def sync_filter_widget_state(settings: dict[str, Any]) -> None:
    st.session_state["filter_exposure"] = float(settings["exposure"])
    st.session_state["filter_saturation"] = float(settings["saturation"])
    st.session_state["filter_contrast"] = float(settings["contrast"])
    st.session_state["filter_brightness"] = float(settings["brightness"])
    st.session_state["filter_sharpness"] = float(settings["sharpness"])
    st.session_state["filter_bw_threshold"] = int(settings["bw_threshold"])
    st.session_state["filter_grayscale"] = bool(settings["grayscale"])


def queue_filter_reset(settings: dict[str, Any]) -> None:
    st.session_state["filter_reset_pending"] = settings.copy()


def apply_filter_settings(image_bgr: np.ndarray, settings: dict[str, Any]) -> np.ndarray:
    image = image_bgr.astype(np.float32) / 255.0
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * float(settings["saturation"]), 0.0, 1.0)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * float(settings["exposure"]), 0.0, 1.0)
    filtered = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    contrast = float(settings["contrast"])
    filtered = np.clip((filtered - 0.5) * contrast + 0.5, 0.0, 1.0)
    filtered = np.clip(filtered + float(settings["brightness"]), 0.0, 1.0)
    filtered = (filtered * 255.0).astype(np.uint8)

    sharpness = float(settings["sharpness"])
    if sharpness > 0.0:
        gaussian = cv2.GaussianBlur(filtered, (0, 0), sigmaX=1.2)
        filtered = cv2.addWeighted(filtered, 1.0 + sharpness, gaussian, -sharpness, 0)

    if settings["grayscale"]:
        gray = cv2.cvtColor(filtered, cv2.COLOR_BGR2GRAY)
        threshold_value = int(settings["bw_threshold"])
        _, binary = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)
        filtered = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

    return filtered


def save_uploaded_images(uploaded_files: list[Any]) -> int:
    ensure_library_paths()
    saved = 0
    for uploaded_file in uploaded_files:
        target = IMAGE_LIBRARY_DIR / uploaded_file.name
        target.write_bytes(uploaded_file.getbuffer())
        saved += 1
    return saved


def list_library_images() -> list[Path]:
    ensure_library_paths()
    return sorted(
        path for path in IMAGE_LIBRARY_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    )


def run_single_analysis(image_bgr: np.ndarray, source: str, backend: str, mode: str) -> tuple[OCRResult | None, float, str | None]:
    pipeline = build_pipeline_for(backend, mode)
    started_at = time.perf_counter()
    try:
        result = pipeline.analyze(image_bgr, source=source)
    except Exception as exc:
        elapsed_ms = (time.perf_counter() - started_at) * 1000.0
        return None, elapsed_ms, str(exc)

    elapsed_ms = (time.perf_counter() - started_at) * 1000.0
    return result, elapsed_ms, None


def run_single_analysis_with_status(
    image_bgr: np.ndarray,
    source: str,
    backend: str,
    mode: str,
    container: Any,
) -> tuple[OCRResult | None, float, str | None]:
    with container:
        if backend == "deepseek_ocr":
            with st.status(f"{backend}: preparação", expanded=True) as status:
                status.write("1/4 Preparar pipeline e filtros.")
                status.update(label=f"{backend}: inferência", state="running")
                status.write("2/4 Carregar modelo/tokenizer (pode demorar na primeira execução).")
                status.write("3/4 Executar inferência OCR.")
                result, elapsed_ms, error = run_single_analysis(
                    image_bgr=image_bgr,
                    source=source,
                    backend=backend,
                    mode=mode,
                )
                if error:
                    status.update(label=f"{backend}: falhou", state="error")
                    status.write(f"4/4 Erro: {error}")
                else:
                    status.update(label=f"{backend}: concluído", state="complete")
                    status.write("4/4 Resultado recebido.")
                return result, elapsed_ms, error

        with st.spinner(f"A processar com {backend} ({mode})..."):
            return run_single_analysis(
                image_bgr=image_bgr,
                source=source,
                backend=backend,
                mode=mode,
            )


class LiveOCRProcessor(VideoProcessorBase):
    def __init__(self) -> None:
        self.pipeline = OCRPipeline(
            languages=["en", "pt"],
            required_tokens=st.session_state.get("required_tokens", []),
            regex_pattern=st.session_state.get("regex_pattern", ""),
            expected_text=st.session_state.get("expected_text", ""),
            backend=st.session_state.get("ocr_backend", "easyocr"),
            mode=st.session_state.get("ocr_mode", "ocr"),
            deepseek_device=st.session_state.get("deepseek_device", "cpu"),
            easyocr_device=st.session_state.get("easyocr_device", "auto"),
        )
        self.frame_index = 0
        self.frame_stride = max(int(st.session_state.get("frame_stride", 8)), 1)
        self.last_result: dict[str, Any] | None = None

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        image = frame.to_ndarray(format="bgr24")
        self.frame_index += 1

        if self.frame_index % self.frame_stride == 0:
            result = self.pipeline.analyze(image, source="webcam")
            self.last_result = result.to_dict()
            image = self.pipeline.annotate(image, result)
        elif self.last_result:
            pseudo_result = OCRResult(
                source="webcam-preview",
                backend=str(self.last_result.get("backend", "")),
                mode=str(self.last_result.get("mode", "")),
                recognized_text="",
                raw_text="",
                matched_tokens=[],
                regex_match=bool(self.last_result.get("regex_match", False)),
                score=0.0,
                expected_text=str(self.last_result.get("expected_text", "")),
                expected_match=bool(self.last_result.get("expected_match", False)),
                similarity=float(self.last_result.get("similarity", 0.0)),
                barcode_found=bool(self.last_result.get("barcode_found", False)),
                barcode_value="",
                roi_bbox=None,
            )
            pseudo_result.recognized_text = str(self.last_result.get("recognized_text", ""))
            pseudo_result.raw_text = str(self.last_result.get("raw_text", ""))
            pseudo_result.barcode_value = str(self.last_result.get("barcode_value", ""))
            pseudo_result.score = float(self.last_result.get("score", 0.0))
            bbox = self.last_result.get("roi_bbox")
            if bbox:
                pseudo_result.roi_bbox = tuple(int(value) for value in str(bbox).split(","))
            image = self.pipeline.annotate(image, pseudo_result)

        return av.VideoFrame.from_ndarray(image, format="bgr24")


def render_sidebar() -> None:
    st.sidebar.header("Validation")
    tokens = st.sidebar.text_input(
        "Palavras obrigatórias",
        # value="LOT,EXP,L302R",
        help="Separadas por vírgula. A app marca a imagem como mais fiável quando encontra estes tokens.",
    )
    regex_pattern = st.sidebar.text_input(
        "Regex opcional",
        # value=r"\b\d{2}\s*\.\s*\d{4}\s+[A-Z0-9]{4,}\b",
        help="Exemplo para validar strings como 10 . 2030 L302R.",
    )
    ocr_backend = st.sidebar.selectbox(
        "Backend OCR",
        options=["tesseract", "easyocr", "paddleocr", "deepseek_ocr"],
        index=0,
        help="Tesseract tende a ser melhor para linha única restrita; EasyOCR, PaddleOCR e DeepSeek-OCR ficam disponíveis para comparação.",
    )
    ocr_mode = st.sidebar.selectbox(
        "Modo OCR",
        options=["ocr"],
        index=0,
        help="Modo OCR geral na imagem completa, sem dependência de ROI fixa acima do barcode.",
    )
    if ocr_backend == "deepseek_ocr":
        deepseek_device = st.sidebar.selectbox(
            "DeepSeek device",
            options=["cpu", "cuda"],
            index=0,
            help="Por defeito usa CPU para evitar erros de memória na GPU. Use CUDA apenas se tiver VRAM suficiente.",
        )
    else:
        deepseek_device = "cpu"
    if ocr_backend == "easyocr":
        easyocr_device = st.sidebar.selectbox(
            "EasyOCR device",
            options=["auto", "cpu", "cuda"],
            index=0,
            help="auto usa CUDA se disponível no PyTorch; use cpu/cuda para forçar explicitamente.",
        )
    else:
        easyocr_device = st.session_state.get("easyocr_device", "auto")
    expected_text = st.sidebar.text_input(
        "Texto esperado",
        # value="10 . 2030 L302R",
        help="Modo de verificação: compara o OCR com a string esperada.",
    )
    frame_stride = st.sidebar.slider(
        "Analisar 1 em cada N frames (webcam)",
        min_value=1,
        max_value=20,
        value=8,
        help="Aumente este valor para aliviar CPU no modo webcam.",
    )

    st.session_state["required_tokens"] = [token.strip() for token in tokens.split(",") if token.strip()]
    st.session_state["regex_pattern"] = regex_pattern.strip()
    st.session_state["ocr_backend"] = ocr_backend
    st.session_state["ocr_mode"] = ocr_mode
    st.session_state["deepseek_device"] = deepseek_device
    st.session_state["easyocr_device"] = easyocr_device
    st.session_state["expected_text"] = expected_text.strip()
    st.session_state["frame_stride"] = frame_stride

    if ocr_backend == "deepseek_ocr":
        st.sidebar.warning(
            f"DeepSeek-OCR é opcional e experimental aqui. Device atual: {deepseek_device}. Em CPU pode ser lento; em CUDA pode falhar por falta de VRAM."
        )
    if ocr_backend == "easyocr":
        st.sidebar.caption(f"EasyOCR device atual: {easyocr_device}")
    # if ocr_backend == "paddleocr":
    #     st.sidebar.info(
    #         "PaddleOCR é opcional. Se faltar no ambiente, instale com: python -m pip install -r requirements-paddle.txt"
    #     )


def render_batch_tab(pipeline: OCRPipeline) -> None:
    compare_all = st.checkbox(
        "Comparar todos os backends",
        value=False,
        help="Executa tesseract, easyocr, paddleocr e deepseek_ocr na mesma imagem e mostra os resultados em lista.",
    )
    show_debug_views = st.checkbox(
        "Mostrar filtros de diagnóstico",
        value=False,
        help="Mostra a ROI e as variantes filtradas usadas pelo OCR para perceber qual funciona melhor.",
    )
    uploaded_files = st.file_uploader(
        "Carregar imagens",
        type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"],
        accept_multiple_files=True,
    )

    if not uploaded_files:
        st.info("Carregue um conjunto de imagens para processar em batch.")
        return

    rows = []
    for uploaded_file in uploaded_files:
        image = Image.open(uploaded_file)
        image_bgr = pil_to_bgr(image)
        if compare_all:
            comparison_targets = [
                ("tesseract", "ocr"),
                ("easyocr", st.session_state.get("ocr_mode", "ocr")),
                ("paddleocr", st.session_state.get("ocr_mode", "ocr")),
                ("deepseek_ocr", "ocr"),
            ]
            with st.container(border=True):
                st.image(image, caption=uploaded_file.name, width="stretch")
                st.markdown("**Comparação de backends**")
                for backend, mode in comparison_targets:
                    with st.container(border=True):
                        col_header, col_body = st.columns([0.28, 0.72])

                        with col_header:
                            st.markdown(f"**{backend}**")
                            st.caption(mode)

                        with col_body:
                            status_placeholder = st.empty()
                            result, elapsed_ms, error = run_single_analysis_with_status(
                                image_bgr=image_bgr,
                                source=uploaded_file.name,
                                backend=backend,
                                mode=mode,
                                container=status_placeholder.container(),
                            )
                            status_placeholder.empty()
                            st.caption(f"Tempo: {elapsed_ms:.0f} ms")
                            if error:
                                st.error(error)
                                rows.append(
                                    {
                                        "source": uploaded_file.name,
                                        "backend": backend,
                                        "mode": mode,
                                        "elapsed_ms": round(elapsed_ms, 2),
                                        "error": error,
                                    }
                                )
                                continue

                            assert result is not None
                            st.write(f"**Final:** {result.recognized_text or 'Sem texto'}")
                            st.write(f"**Bruto:** {result.raw_text or 'Sem texto'}")
                            st.write(f"**Score:** {result.score:.2f}")
                            st.write(f"**Similaridade:** {result.similarity:.2f}")
                            st.write(f"**Match:** {'Sim' if result.expected_match else 'Não'}")

                            row = result.to_dict()
                            row["elapsed_ms"] = round(elapsed_ms, 2)
                            row["error"] = ""
                            rows.append(row)
        else:
            with st.container(border=True):
                col1, col2 = st.columns([1.1, 1])
                with col1:
                    status_placeholder = st.empty()
                    result, elapsed_ms, error = run_single_analysis_with_status(
                        image_bgr=image_bgr,
                        source=uploaded_file.name,
                        backend=st.session_state.get("ocr_backend", "tesseract"),
                        mode=st.session_state.get("ocr_mode", "ocr"),
                        container=status_placeholder.container(),
                    )
                    status_placeholder.empty()
                    if result is not None:
                        annotated = pipeline.annotate(image_bgr, result)
                        st.image(bgr_to_rgb(annotated), caption=uploaded_file.name, width="stretch")
                    else:
                        st.image(image, caption=uploaded_file.name, width="stretch")
                with col2:
                    st.write(f"**Tempo:** {elapsed_ms:.0f} ms")
                    if error:
                        st.error(error)
                        rows.append(
                            {
                                "source": uploaded_file.name,
                                "backend": st.session_state.get("ocr_backend", "tesseract"),
                                "mode": st.session_state.get("ocr_mode", "ocr"),
                                "elapsed_ms": round(elapsed_ms, 2),
                                "error": error,
                            }
                        )
                    else:
                        assert result is not None
                        st.metric("Score", result.score)
                        st.write(f"**Texto OCR final:** {result.recognized_text or 'Sem texto válido'}")
                        st.write(f"**Texto OCR bruto:** {result.raw_text or 'Sem texto bruto'}")
                        st.write(f"**Backend:** {result.backend}")
                        st.write(f"**Modo:** {result.mode}")
                        st.write(f"**Barcode:** {result.barcode_value or 'Não detetado'}")
                        st.write(f"**Tokens encontrados:** {', '.join(result.matched_tokens) or 'Nenhum'}")
                        st.write(f"**Regex válido:** {'Sim' if result.regex_match else 'Não'}")
                        st.write(f"**Texto esperado:** {result.expected_text or 'Não definido'}")
                        st.write(f"**Similaridade:** {result.similarity:.2f}")
                        st.write(f"**Match esperado:** {'Sim' if result.expected_match else 'Não'}")

                        row = result.to_dict()
                        row["elapsed_ms"] = round(elapsed_ms, 2)
                        row["error"] = ""
                        rows.append(row)

            if show_debug_views:
                debug_pipeline = build_pipeline_for(
                    backend=st.session_state.get("ocr_backend", "tesseract"),
                    mode=st.session_state.get("ocr_mode", "ocr"),
                )
                debug_views = debug_pipeline.build_debug_views(image_bgr)
                filter_evaluations = debug_pipeline.evaluate_debug_filters(
                    image_bgr,
                    source=uploaded_file.name,
                )
                if filter_evaluations:
                    valid_filters = [item for item in filter_evaluations if not item.get("error")]
                    best_filter = max(
                        valid_filters or filter_evaluations,
                        key=lambda item: (item["similarity"], item["score"]),
                    )
                    st.success(
                        f"Melhor filtro: {best_filter['label']} | "
                        f"similaridade={best_filter['similarity']:.2f} | "
                        f"score={best_filter['score']:.2f} | "
                        f"texto={best_filter['recognized_text'] or 'Sem texto'}"
                    )
                    st.caption("Os filtros mostrados abaixo sao aplicados antes da inferencia OCR.")
                st.markdown("**Diagnóstico de filtros**")
                debug_columns = st.columns(min(3, len(debug_views)))
                for index, (label, debug_image) in enumerate(debug_views):
                    with debug_columns[index % len(debug_columns)]:
                        st.image(bgr_to_rgb(debug_image), caption=label, width="stretch")
                if filter_evaluations:
                    filter_rows = []
                    for item in filter_evaluations:
                        filter_rows.append(
                            {
                                "filter": item["label"],
                                "recognized_text": item["recognized_text"],
                                "raw_text": item["raw_text"],
                                "score": item["score"],
                                "similarity": item["similarity"],
                                "expected_match": item["expected_match"],
                                "regex_match": item["regex_match"],
                                "error": item.get("error", ""),
                            }
                        )
                    st.dataframe(pd.DataFrame(filter_rows), width="stretch")

    dataframe = pd.DataFrame(rows)
    st.subheader("Resultados")
    st.dataframe(dataframe, width="stretch")
    st.download_button(
        "Descarregar CSV",
        data=dataframe.to_csv(index=False).encode("utf-8"),
        file_name="ocr_results.csv",
        mime="text/csv",
    )


def render_image_library_tab() -> None:
    ensure_library_paths()
    st.subheader("Image Library")
    uploaded_files = st.file_uploader(
        "Upload para a pasta da biblioteca",
        type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"],
        accept_multiple_files=True,
        key="library_upload",
    )

    container_actions = st.container(horizontal=True)
    if uploaded_files and container_actions.button("Guardar imagens na pasta"):
            saved_count = save_uploaded_images(uploaded_files)
            st.success(f"{saved_count} imagem(ns) guardadas em {IMAGE_LIBRARY_DIR}.")

    st.space()

    st.markdown("**Imagens na Biblioteca**")

    images = list_library_images()
    if not images:
        st.info("A biblioteca está vazia. Carregue imagens para começar.")
        return

    image_rows = []
    for image_path in images:
        stat = image_path.stat()
        image_rows.append(
            {
                "filename": image_path.name,
                "size_kb": round(stat.st_size / 1024.0, 1),
            }
        )
    st.dataframe(pd.DataFrame(image_rows), width="content")
    if st.button("Apagar todas as imagens", type="primary"):
            if IMAGE_LIBRARY_DIR.exists():
                shutil.rmtree(IMAGE_LIBRARY_DIR)
            ensure_library_paths()
            st.success("Biblioteca limpa.")

    st.space()

    selected_path = st.selectbox(
        "Escolher imagem",
        options=images,
        format_func=lambda path: path.name,
    )
    selected_image = Image.open(selected_path)
    selected_bgr = pil_to_bgr(selected_image)

    saved_settings = load_filter_settings()
    pending_reset = st.session_state.pop("filter_reset_pending", None)
    if pending_reset is not None:
        sync_filter_widget_state(pending_reset)
        saved_settings = pending_reset
    filter_keys = [
        "filter_exposure",
        "filter_saturation",
        "filter_contrast",
        "filter_brightness",
        "filter_sharpness",
        "filter_bw_threshold",
        "filter_grayscale",
    ]
    if not all(key in st.session_state for key in filter_keys):
        sync_filter_widget_state(saved_settings)

    st.markdown("**Editor de filtros**")
    slider_col1, slider_col2, slider_col3 = st.columns(3)
    with slider_col1:
        exposure = st.slider("Exposure", 0.4, 2.5, key="filter_exposure", step=0.05)
    with slider_col2:
        saturation = st.slider("Saturation", 0.0, 2.5, key="filter_saturation", step=0.05)
    with slider_col3:
        contrast = st.slider("Contrast", 0.5, 2.5, key="filter_contrast", step=0.05)
    slider_col4, slider_col5, slider_col6 = st.columns(3)
    with slider_col4:
        brightness = st.slider("Brightness", -0.5, 0.5, key="filter_brightness", step=0.02)
    with slider_col5:
        sharpness = st.slider("Sharpness", 0.0, 2.0, key="filter_sharpness", step=0.05)
    with slider_col6:
        bw_threshold = st.slider("Black/White threshold", 0, 255, key="filter_bw_threshold", step=1)
    extra_col = st.columns(1)[0]
    with extra_col:
        grayscale = st.checkbox("Grayscale", key="filter_grayscale")

    current_settings = {
        "exposure": exposure,
        "saturation": saturation,
        "contrast": contrast,
        "brightness": brightness,
        "sharpness": sharpness,
        "bw_threshold": bw_threshold,
        "grayscale": grayscale,
    }
    filtered_bgr = apply_filter_settings(selected_bgr, current_settings)

    preview_col1, preview_col2 = st.columns(2)
    with preview_col1:
        st.image(selected_image, caption=f"Original: {selected_path.name}", width="stretch")
    with preview_col2:
        st.image(bgr_to_rgb(filtered_bgr), caption="Preview com filtros", width="stretch")

    
    container_filter_settings = st.container(horizontal=True)
    if container_filter_settings.button("Guardar settings dos filtros"):
            save_filter_settings(current_settings)
            st.success("Settings guardados.")
    
    if container_filter_settings.button("Reset filtros default"):
            save_filter_settings(DEFAULT_FILTER_SETTINGS)
            queue_filter_reset(DEFAULT_FILTER_SETTINGS)
            st.rerun()

    run_ocr = st.button("Executar OCR nas imagens da pasta")
    
    if not run_ocr:
        return

    pipeline = build_pipeline()
    persisted_settings = load_filter_settings()
    rows = []
    st.markdown("**OCR batch com filtros guardados**")
    for image_path in images:
        image = Image.open(image_path)
        image_bgr = pil_to_bgr(image)
        filtered = apply_filter_settings(image_bgr, persisted_settings)

        status_placeholder = st.empty()
        result, elapsed_ms, error = run_single_analysis_with_status(
            image_bgr=filtered,
            source=image_path.name,
            backend=st.session_state.get("ocr_backend", "tesseract"),
            mode=st.session_state.get("ocr_mode", "ocr"),
            container=status_placeholder.container(),
        )
        status_placeholder.empty()

        with st.container(border=True):
            col1, col2 = st.columns([1.1, 1])
            with col1:
                if result is not None:
                    rendered = pipeline.annotate(filtered, result)
                else:
                    rendered = filtered
                st.image(bgr_to_rgb(rendered), caption=image_path.name, width="stretch")
            with col2:
                st.write(f"**Tempo:** {elapsed_ms:.0f} ms")
                if error:
                    st.error(error)
                    rows.append(
                        {
                            "source": image_path.name,
                            "backend": st.session_state.get("ocr_backend", "tesseract"),
                            "mode": st.session_state.get("ocr_mode", "ocr"),
                            "elapsed_ms": round(elapsed_ms, 2),
                            "error": error,
                        }
                    )
                else:
                    assert result is not None
                    st.write(f"**Texto OCR final:** {result.recognized_text or 'Sem texto'}")
                    st.write(f"**Texto OCR bruto:** {result.raw_text or 'Sem texto'}")
                    st.write(f"**Similarity:** {result.similarity:.2f}")
                    row = result.to_dict()
                    row["elapsed_ms"] = round(elapsed_ms, 2)
                    row["filter_exposure"] = persisted_settings["exposure"]
                    row["filter_saturation"] = persisted_settings["saturation"]
                    row["filter_contrast"] = persisted_settings["contrast"]
                    row["filter_brightness"] = persisted_settings["brightness"]
                    row["filter_sharpness"] = persisted_settings["sharpness"]
                    row["filter_bw_threshold"] = persisted_settings["bw_threshold"]
                    row["filter_grayscale"] = persisted_settings["grayscale"]
                    row["error"] = ""
                    rows.append(row)

    if rows:
        dataframe = pd.DataFrame(rows)
        st.dataframe(dataframe, width="stretch")
        st.download_button(
            "Descarregar CSV do OCR batch",
            data=dataframe.to_csv(index=False).encode("utf-8"),
            file_name="ocr_batch_filtered.csv",
            mime="text/csv",
        )


def render_webcam_tab() -> None:
    st.write(
        "O modo webcam usa `streamlit-webrtc` e faz OCR periódico nos frames para reduzir custo de CPU."
    )
    ctx = webrtc_streamer(
        key="flexible-packaging-ocr",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=LiveOCRProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

    if ctx.video_processor and ctx.video_processor.last_result:
        st.subheader("Última leitura")
        st.json(ctx.video_processor.last_result)
    else:
        st.info("Inicie a webcam para começar a leitura em tempo real.")


def render_example() -> None:
    st.subheader("Fluxo recomendado")
    st.markdown(
        """
1. Aplicar filtros de pré-processamento à imagem completa.
2. Correr OCR sem dependência de ROI fixa.
3. Validar o resultado com tokens esperados e/ou regex.
4. Repetir nos frames da webcam com amostragem para manter a UI fluida.
"""
    )


def main() -> None:
    ensure_library_paths()
    render_sidebar()
    pipeline = build_pipeline()

    st.title("OCR Text Verification for Flexible Packaging")
    st.write(
        "MVP em Streamlit para ler texto junto ao código de barras em imagens e vídeo em tempo real."
    )

    render_example()
    # batch_tab, library_tab, webcam_tab = st.tabs(["Batch Upload", "Image Library", "Webcam Realtime"])
    library_tab, webcam_tab = st.tabs(["Image Library", "Webcam Realtime"])
    # with batch_tab:
    #     render_batch_tab(pipeline)
    with library_tab:
        render_image_library_tab()
    with webcam_tab:
        render_webcam_tab()


if __name__ == "__main__":
    main()
