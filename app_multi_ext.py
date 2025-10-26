# app_multi_ext.py
# -*- coding: utf-8 -*-
import os, json, csv, gc, torch, gradio as gr
from collections import OrderedDict
from functools import lru_cache
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
from PIL import Image

from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionXLPipeline,
    EulerDiscreteScheduler,
    DDIMScheduler,
    DPMSolverMultistepScheduler,
)
from compel_sdxl_utils import get_compel_for_sdxl

# HF Hub
try:
    from huggingface_hub import hf_hub_download
except Exception:
    hf_hub_download = None

# ONNXRuntime (for WD-EVA02 tagger)
try:
    import onnxruntime as ort
except Exception:
    ort = None

# Transformers (for BLIP2)
try:
    from transformers import Blip2Processor, Blip2ForConditionalGeneration, CLIPTokenizer
except Exception:
    Blip2Processor = None
    Blip2ForConditionalGeneration = None
    CLIPTokenizer = None

try:
    import psutil  # optional, used for memory-aware caching of SD pipelines
except Exception:
    psutil = None

DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Device for img2text (default CPU; use GPU: export IMG2TEXT_DEVICE=cuda)
IMG2TEXT_DEVICE = os.getenv("IMG2TEXT_DEVICE", "cpu").strip().lower()  # "cpu" / "cuda"

# ========================
# 1) Model registry (add/remove as needed)
# ========================
MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {
    # "Yuunagi-SDXL": {
    #     "name": "Yuunagi-SDXL",
    #     "type": "sdxl",
    #     "load": {
    #         "mode": "local",
    #         "path": "/mnt/shared/model/yuunagi-SDXL"  # Put in current directory or fetch via repo
    #     },
    #     "presets": {
    #         "widths":  [512, 648, 720, 768, 896, 1024, 1152, 1232, 1280],
    #         "heights": [512, 648, 720, 768, 896, 1024, 1152, 1232, 1280],
    #         "default_w": 1280,
    #         "default_h": 720,
    #         "steps": 28,
    #         "guidance": 5.5,
    #         "default_scheduler": "euler",
    #     },
    # },
    "illustrious_emberveil": {
        "name": "【illustrious】EmberVeilMix (merge)",
        "type": "sdxl",
        "load": {
            "mode": "singlefile",
            "filename": "IllustriousEmberveilmix_v10.safetensors"  # Put in current directory or fetch via repo
            # "repo": "YourOrg/IllustriousEmberveilmix_v10",
        },
        "presets": {
            "widths":  [512, 648, 720, 768, 896, 1024],
            "heights": [512, 648, 720, 768, 896, 1024],
            "default_w": 1024,
            "default_h": 1024,
            "steps": 28,
            "guidance": 5.5,
            "default_scheduler": "euler",
        }
    },
    # "noobaiXLNAIXL_vPred10Version": {
    #     "name": "noobaiXLNAIXL_vPred10Version",
    #     "type": "sdxl",
    #     "load": {
    #         "mode": "singlefile",
    #         "filename": "noobaiXLNAIXL_vPred10Version.safetensors"  # Put in current directory or fetch via repo
    #     },
    #     "presets": {
    #         "widths":  [512, 648, 720, 768, 896, 1024, 1152, 1232, 1280],
    #         "heights": [512, 648, 720, 768, 896, 1024, 1152, 1232, 1280],
    #         "default_w": 1024,
    #         "default_h": 1024,
    #         "steps": 28,
    #         "guidance": 4.8,
    #         "default_scheduler": "dpmpp2m",
    #     },
    #     "scheduler_config": {
    #         "prediction_type": "v_prediction",
    #     },
    # },
    # "sd15_official": {
    #     "name": "SD15 Official (runwayml/stable-diffusion-v1-5)",
    #     "type": "sd15",
    #     "load": {
    #         "mode": "pretrained",
    #         "repo": "runwayml/stable-diffusion-v1-5"
    #     },
    #     "presets": {
    #         "widths":  [384, 448, 512, 576, 640, 704, 768],
    #         "heights": [384, 448, 512, 576, 640, 704, 768],
    #         "default_w": 512,
    #         "default_h": 512,
    #         "steps": 28,
    #         "guidance": 7.0,
    #     }
    # },
    # To add SD15/official SDXL models, copy an entry in this format
}

# Env var overrides (e.g., inject paths/repos in CI/CD)
def apply_env_overrides(reg: Dict[str, Dict[str, Any]]):
    prefix = 'REG__'
    for k, v in os.environ.items():
        if not k.startswith(prefix): continue
        parts = k[len(prefix):].split('__')
        cur = reg
        for p in parts[:-1]:
            if p not in cur: cur[p] = {}
            cur = cur[p]
        leaf = parts[-1]
        try: cur[leaf] = json.loads(v)
        except Exception: cur[leaf] = v
apply_env_overrides(MODEL_REGISTRY)

# ===============
# 2) SD pipeline cache
# ===============
@dataclass
class PipeCache:
    pipe: Optional[object] = None
    model_key: Optional[str] = None
CACHE = PipeCache()

# Allow keeping recently used pipelines in memory to avoid repeated disk loads.
def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


PIPE_CACHE_LIMIT = max(0, _env_int("SD_PIPE_CACHE_LIMIT", 2))
PIPE_CACHE_MODE = os.getenv("SD_PIPE_CACHE_MODE", "cpu").strip().lower()
if PIPE_CACHE_MODE not in {"cpu", "gpu", "off"}:
    PIPE_CACHE_MODE = "cpu"
if PIPE_CACHE_MODE == "gpu" and not torch.cuda.is_available():
    PIPE_CACHE_MODE = "cpu"
PIPE_CACHE_DEBUG = os.getenv("SD_PIPE_CACHE_DEBUG", "0") in {"1", "true", "True"}
PIPE_CACHE_MIN_FREE_GB = max(0.0, _env_float("SD_PIPE_CACHE_MIN_FREE_GB", 1.5))
PIPE_STASH: "OrderedDict[str, object]" = OrderedDict()

scheduler_map = {
    'euler': EulerDiscreteScheduler,
    'ddim': DDIMScheduler,
    'dpmpp2m': DPMSolverMultistepScheduler,
}

def _release_pipe(pipe: Optional[object], *, move_to_cpu: bool = True):
    if pipe is None:
        return
    if move_to_cpu:
        try:
            pipe.to('cpu', torch_dtype=torch.float32)
        except Exception:
            pass
    # help python drop remaining references promptly
    del pipe
    gc.collect()


def _maybe_evict_for_memory():
    if psutil is None or PIPE_CACHE_MIN_FREE_GB <= 0.0:
        return
    try:
        available_gb = psutil.virtual_memory().available / (1024 ** 3)
    except Exception:
        return
    while PIPE_STASH and available_gb < PIPE_CACHE_MIN_FREE_GB:
        key, old_pipe = PIPE_STASH.popitem(last=False)
        if PIPE_CACHE_DEBUG:
            print(f"[PIPE CACHE] low RAM – evict {key} (available {available_gb:.2f} GiB)")
        _release_pipe(old_pipe, move_to_cpu=(PIPE_CACHE_MODE != "gpu"))
        try:
            available_gb = psutil.virtual_memory().available / (1024 ** 3)
        except Exception:
            break


def _free_pipe():
    if CACHE.pipe is None:
        return

    model_key = CACHE.model_key
    pipe = CACHE.pipe

    if PIPE_CACHE_MODE == "off" or PIPE_CACHE_LIMIT <= 0 or model_key is None:
        _release_pipe(pipe, move_to_cpu=(PIPE_CACHE_MODE != "gpu"))
    else:
        if PIPE_CACHE_MODE == "cpu":
            try:
                pipe.to('cpu', torch_dtype=torch.float32)
            except Exception:
                pass
        if PIPE_CACHE_DEBUG:
            print(f"[PIPE CACHE] stash {model_key} (mode={PIPE_CACHE_MODE})")
        if model_key in PIPE_STASH:
            _release_pipe(PIPE_STASH.pop(model_key), move_to_cpu=(PIPE_CACHE_MODE != "gpu"))
        _maybe_evict_for_memory()
        PIPE_STASH[model_key] = pipe
        PIPE_STASH.move_to_end(model_key, last=True)
        if PIPE_CACHE_DEBUG:
            print(f"[PIPE CACHE] stash set {model_key} (size={len(PIPE_STASH)})")
        while len(PIPE_STASH) > PIPE_CACHE_LIMIT:
            _, old_pipe = PIPE_STASH.popitem(last=False)
            _release_pipe(old_pipe, move_to_cpu=(PIPE_CACHE_MODE != "gpu"))
            if PIPE_CACHE_DEBUG:
                print(f"[PIPE CACHE] evicted oldest pipeline (size={len(PIPE_STASH)})")

    CACHE.pipe = None
    CACHE.model_key = None
    if torch.cuda.is_available() and PIPE_CACHE_MODE != "gpu":
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

def _load_from_cfg(cfg: Dict[str, Any]):
    mtype = cfg['type']
    load = cfg['load']
    mode = load.get('mode', 'local')

    if mtype == 'sd15':
        if mode == 'local':
            p = StableDiffusionPipeline.from_pretrained(load['path'], torch_dtype=DTYPE)
        elif mode == 'pretrained':
            p = StableDiffusionPipeline.from_pretrained(load['repo'], torch_dtype=DTYPE)
        else:
            raise ValueError(f"SD15 unsupported mode: {mode}")
    elif mtype == 'sdxl':
        if mode == 'local':
            p = StableDiffusionXLPipeline.from_pretrained(load['path'], torch_dtype=DTYPE, use_safetensors=True, low_cpu_mem_usage=False)
        elif mode == 'pretrained':
            p = StableDiffusionXLPipeline.from_pretrained(load['repo'], torch_dtype=DTYPE, use_safetensors=True, low_cpu_mem_usage=False)
        elif mode == 'singlefile':
            if hf_hub_download is None or "repo" not in load:
                file_path = load['filename']
            else:
                file_path = hf_hub_download(repo_id=load['repo'], filename=load['filename'])
            p = StableDiffusionXLPipeline.from_single_file(file_path, torch_dtype=DTYPE)
        else:
            raise ValueError(f"SDXL unsupported mode: {mode}")
    else:
        raise ValueError(f"Unknown model type: {mtype}")

    try:
        p = p.to(DEVICE, torch_dtype=DTYPE)
    except Exception:
        p = p.to(DEVICE)

    sched_cfg_overrides = cfg.get('scheduler_config')
    if sched_cfg_overrides:
        base_config = dict(p.scheduler.config)
        base_config.update(sched_cfg_overrides)
        p.scheduler = p.scheduler.__class__.from_config(base_config)

    try: p.enable_xformers_memory_efficient_attention()
    except Exception: pass
    p.enable_vae_slicing(); p.enable_vae_tiling()
    return p

def ensure_pipe(model_key: str):
    if CACHE.pipe is not None and CACHE.model_key == model_key:
        return
    _free_pipe()
    reused = PIPE_STASH.pop(model_key, None)
    if reused is not None:
        if PIPE_CACHE_MODE == "cpu":
            try:
                reused = reused.to(DEVICE, torch_dtype=DTYPE)
            except Exception:
                reused = reused.to(DEVICE)
        pipe = reused
        if PIPE_CACHE_DEBUG:
            print(f"[PIPE CACHE] reuse pipeline {model_key}")
    else:
        cfg = MODEL_REGISTRY[model_key]
        pipe = _load_from_cfg(cfg)
        if PIPE_CACHE_DEBUG:
            print(f"[PIPE CACHE] load pipeline {model_key} from config")

    CACHE.pipe = pipe
    CACHE.model_key = model_key

# ==================================
# 3) WD-EVA02 tagger (ONNXRuntime)
# ==================================
from typing import NamedTuple

@dataclass
class WDTaggerCache:
    ort_session: Optional['ort.InferenceSession'] = None
    # Legacy fields (for backward compatibility)
    labels: List[str] = None
    rating_labels: List[str] = None
    general_labels: List[str] = None
    character_labels: List[str] = None
    # New fields: indices (exact CSV row order)
    idx_rating: List[int] = None
    idx_general: List[int] = None
    idx_character: List[int] = None

WD = WDTaggerCache()


@dataclass
class CamieTaggerCache:
    ort_session: Optional['ort.InferenceSession'] = None
    idx_to_tag: Optional[List[str]] = None
    tag_to_category: Optional[Dict[str, str]] = None
    img_size: int = 512


CAMIE = CamieTaggerCache()

# ==================================
# Danbooru tags (EN->JA) lookup cache
# ==================================
TAGS_JA: Optional[List[Tuple[str, str]]] = None

def _load_tags_ja(csv_path: str = "/mnt/shared/data/all_tags_ja.csv") -> List[Tuple[str, str]]:
    global TAGS_JA
    if TAGS_JA is not None:
        return TAGS_JA
    items: List[Tuple[str, str]] = []
    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                if not row or len(row) < 2:
                    continue
                en = (row[0] or "").strip()
                ja = (row[1] or "").strip()
                if en:
                    items.append((en, ja))
    except Exception as e:
        print(f"[TAGS_JA] failed to load {csv_path}: {e}")
        items = []
    TAGS_JA = items
    return TAGS_JA

def _search_tags_ja(query: str, max_results: int = 50) -> Tuple[List[List[str]], List[str]]:
    tags = _load_tags_ja()
    q = (query or "").strip()
    if not q:
        rows = tags[:max_results]
    else:
        ql = q.lower()
        res: List[Tuple[str, str]] = []
        for en, ja in tags:
            if (ql in en.lower()) or (q in ja):
                res.append((en, ja))
            if len(res) >= max_results:
                break
        rows = res
    table = [[en, ja] for en, ja in rows]
    suggestions = [en for en, _ in rows]
    return table, suggestions


def _danbooru_lang_text(lang_code: str) -> Dict[str, Any]:
    if lang_code == "en":
        return {
            "intro": "## Danbooru Tag Search\n- Search by English or Japanese phrase.\n- Pick tags from the results to add them below.",
            "search_label": "Search (EN/JP)",
            "search_placeholder": "e.g. girl / 女の子 / hair",
            "max_label": "Max results",
            "table_label": "Tag table",
            "table_headers": ["English", "Japanese"],
            "choices_label": "Candidate tags",
            "add_label": "Add",
            "clear_label": "Clear",
            "selected_label": "Selected tags",
        }
    return {
        "intro": "## Danbooru 用語検索\n- 英語/日本語で検索できます。\n- 候補から英語タグを選んで、下のボックスへ追加します。",
        "search_label": "検索（英語/日本語）",
        "search_placeholder": "例: girl / 女の子 / hair",
        "max_label": "最大件数",
        "table_label": "タグ一覧",
        "table_headers": ["英語タグ", "日本語"],
        "choices_label": "候補タグ",
        "add_label": "追加",
        "clear_label": "クリア",
        "selected_label": "選択済みタグ",
    }


def _ensure_wd_eva02():
    """
    Download and load ONNX + labels for SmilingWolf/wd-eva02-large-tagger-v3.
    - Build ORT session (CPU by default; use CUDA if IMG2TEXT_DEVICE=cuda and available)
    - Read CSV preserving row order; build labels and idx_* lists
    - If only legacy fields exist, backfill idx_* accordingly
    """
    if WD.ort_session is not None:
        return

    if hf_hub_download is None:
        raise RuntimeError("huggingface_hub is unavailable; cannot download wd-eva02 model files.")
    if ort is None:
        raise RuntimeError("onnxruntime is not installed; please pip install onnxruntime or onnxruntime-gpu.")

    repo = "SmilingWolf/wd-eva02-large-tagger-v3"

    # onnx
    onnx_candidates = ["model.onnx", "wd-eva02-large-tagger-v3.onnx"]
    model_path = None
    for cand in onnx_candidates:
        try:
            model_path = hf_hub_download(repo_id=repo, filename=cand)
            break
        except Exception:
            pass
    if model_path is None:
        raise RuntimeError("wd-eva02 ONNX model file not found (model.onnx).")

    # csv
    labels_candidates = ["selected_tags.csv", "tags.csv"]
    labels_path = None
    for cand in labels_candidates:
        try:
            labels_path = hf_hub_download(repo_id=repo, filename=cand)
            break
        except Exception:
            pass
    if labels_path is None:
        raise RuntimeError("wd-eva02 label file not found (selected_tags.csv / tags.csv).")

    # providers
    providers = ['CPUExecutionProvider']
    if IMG2TEXT_DEVICE == "cuda" and 'CUDAExecutionProvider' in ort.get_available_providers():
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    WD.ort_session = ort.InferenceSession(model_path, providers=providers)

    # Read CSV (strictly preserve row order)
    all_labels, idx_rating, idx_general, idx_character = [], [], [], []
    rating_names = {"general", "sensitive", "questionable", "explicit"}

    with open(labels_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            name = row.get("name") or row.get("tag")
            cat  = row.get("category") or row.get("type")
            if not name:
                continue
            all_labels.append(name)

            # First detect rating by name for robustness (no category dependence)
            if name in rating_names and i not in idx_rating:
                idx_rating.append(i)
                continue

            # For the rest, use Danbooru categories: 0=general, 4=character
            c = None
            if cat is not None:
                cs = str(cat).strip()
                try:
                    c = int(cs)
                except Exception:
                    try:
                        c = int(float(cs))
                    except Exception:
                        c = None

            if c == 0:
                idx_general.append(i)
            elif c == 4:
                idx_character.append(i)
            else:
                # Ignore other categories (artist=1, copyright=3, meta=5, etc.)
                pass

    WD.labels = all_labels
    WD.idx_rating = idx_rating
    WD.idx_general = idx_general
    WD.idx_character = idx_character

    # Compatibility: if legacy fields exist (rating_labels/general_labels/character_labels), backfill/deduplicate here
    # (If you kept previous backfill code, it's fine; no conflict)

    # Optional: validation + logs
    try:
        print(f"[WD INIT] labels={len(WD.labels)} | "
              f"rating={len(WD.idx_rating)} general={len(WD.idx_general)} character={len(WD.idx_character)}")
        if len(WD.idx_rating) != 4:
            print("[WD INIT] WARNING: rating labels count is not 4; please check whether CSV is the official version.")
    except Exception:
        pass

def _pad_to_square_white(img_rgb: Image.Image) -> Image.Image:
    """Pad on a white canvas so the image becomes square."""
    w, h = img_rgb.size
    if w == h:
        return img_rgb
    m = max(w, h)
    canvas = Image.new("RGB", (m, m), (255, 255, 255))
    canvas.paste(img_rgb, ((m - w) // 2, (m - h) // 2))
    return canvas


def _preprocess_for_wd(img: Image.Image) -> Image.Image:
    # Align with image_tagger: composite transparency on white, pad square, resize to 448×448.
    if img.mode != "RGBA":
        img = img.convert("RGBA")
    canvas = Image.new("RGBA", img.size, (255, 255, 255, 255))
    canvas.alpha_composite(img)
    img = canvas.convert("RGB")

    img = _pad_to_square_white(img)

    target = 448
    if img.size != (target, target):
        img = img.resize((target, target), Image.BICUBIC)
    return img


import numpy as np
def _run_wd_tagger(
    img: Image.Image,
    th_general: float = 0.25,
    th_character: float = 0.75,
    norm: str = "0to255",            # "0to255" (recommended for eva02 onnx) | "0to1" | "minus1to1" | "imagenet"
    force_layout: str = None,         # None | "nhwc" | "nchw"
    topk_preview: int = 10,           # DEBUG: how many to preview
    fallback_topk: int = 30,          # Fallback topK
    debug: bool = False
) -> Tuple[str, List[Tuple[str, float]], List[Tuple[str, float]], List[Tuple[str, float]], List[Tuple[str, float]]]:
    """
    Returns:
        tags_sentence: str (comma-separated)
        results: List[(tag, score)] (descending, general+character)
        rating_list: rating predictions
        general_list: general-only predictions
        character_list: character-only predictions
    """
    _ensure_wd_eva02()
    if img is None:
        return "", [], [], [], []

    # Safety: if idx_* not ready, backfill from legacy fields on the fly
    if (WD.idx_rating is None or WD.idx_general is None or WD.idx_character is None) and WD.labels:
        name_to_indices = {}
        for i, name in enumerate(WD.labels):
            name_to_indices.setdefault(name, []).append(i)
        def to_indices(names: List[str]) -> List[int]:
            idxs = []
            if names:
                for n in names:
                    for i in name_to_indices.get(n, []):
                        idxs.append(i)
            return sorted(set(idxs))
        WD.idx_rating = WD.idx_rating or to_indices(WD.rating_labels or [])
        WD.idx_general = WD.idx_general or to_indices(WD.general_labels or [])
        WD.idx_character = WD.idx_character or to_indices(WD.character_labels or [])

    proc = _preprocess_for_wd(img)  # 448x448 RGB
    arr = np.asarray(proc).astype("float32")  # HWC in 0..255 (RGB)

    # Normalization strategy (match ONNX training; eva02 ONNX typically expects 0..255 raw pixels)
    if norm == "0to255":
        norm_used = "[0,255]"
    elif norm == "0to1":
        arr = arr / 255.0
        norm_used = "[0,1]"
    elif norm == "minus1to1":
        # Map 0..255 to [-1,1]
        arr = arr / 127.5 - 1.0
        norm_used = "[-1,1]"
    elif norm == "imagenet":
        arr = arr / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype="float32")[None, None, :]
        std  = np.array([0.229, 0.224, 0.225], dtype="float32")[None, None, :]
        arr = (arr - mean) / std
        norm_used = "ImageNet"
    else:
        norm_used = f"<custom:{norm}>"

    # Input layout adaptation
    inp = WD.ort_session.get_inputs()[0]
    in_name = inp.name
    shape = inp.shape  # may contain -1

    def is_nchw(s): return len(s) == 4 and (s[1] in (3, "3"))
    def is_nhwc(s): return len(s) == 4 and (s[-1] in (3, "3"))

    # Decide target layout
    if force_layout == "nchw":
        target_layout = "nchw"
        layout_used = "NCHW (forced)"
    elif force_layout == "nhwc":
        target_layout = "nhwc"
        layout_used = "NHWC (forced)"
    else:
        if is_nchw(shape):
            target_layout = "nchw"
            layout_used = "NCHW (auto)"
        else:
            target_layout = "nhwc"
            layout_used = "NHWC (auto)"

    # Channel order: many wd-eva02 NHWC ONNX expect BGR in [0,255]
    channel_order = "RGB"
    if target_layout == "nhwc" and norm == "0to255":
        arr = arr[:, :, ::-1]  # RGB->BGR
        channel_order = "BGR"

    # Apply final layout
    if target_layout == "nchw":
        arr = arr.transpose(2, 0, 1)[None, ...]  # 1x3xHxW
    else:
        arr = arr[None, ...]                    # 1xHxWx3

    out = WD.ort_session.run(None, {in_name: arr})
    raw = out[0][0].astype("float32")   # [num_tags]
    # Some wd-eva02 ONNX exports already include Sigmoid and yield [0,1].
    # Double-applying sigmoid would cap max at ~0.731. Detect and handle.
    rmin, rmax = float(np.min(raw)), float(np.max(raw))
    if 0.0 <= rmin and rmax <= 1.0:
        probs = raw
        activation = "identity (onnx outputs prob)"
    else:
        probs = 1.0 / (1.0 + np.exp(-raw))
        activation = "sigmoid (applied here)"
    num_tags = int(probs.shape[0])

    # Consistency check
    if WD.labels is None or len(WD.labels) != num_tags:
        raise RuntimeError(
            f"[WD ERROR] Label count mismatches output dimension: labels={len(WD.labels) if WD.labels else None}, probs={num_tags}. "
            "Ensure ONNX and selected_tags.csv come from the same repo/version (same snapshot)."
        )

    # Filter general / character respectively
    general_list: List[Tuple[str, float]] = []
    character_list: List[Tuple[str, float]] = []
    rating_list: List[Tuple[str, float]] = []

    for i in WD.idx_general:
        p = float(probs[i])
        if p >= th_general:
            general_list.append((WD.labels[i], p))
    for i in WD.idx_character:
        p = float(probs[i])
        if p >= th_character:
            character_list.append((WD.labels[i], p))
    # Ratings: always report all (no threshold)
    for i in (WD.idx_rating or []):
        rating_list.append((WD.labels[i], float(probs[i])))

    general_list.sort(key=lambda x: x[1], reverse=True)
    character_list.sort(key=lambda x: x[1], reverse=True)
    rating_list.sort(key=lambda x: x[1], reverse=True)

    # Merge + fallback (avoid empty result)
    results = general_list + character_list
    if not results or not general_list:
        ban = set(WD.idx_rating or [])
        all_scored = [(i, float(probs[i])) for i in range(num_tags) if i not in ban]
        all_scored.sort(key=lambda x: x[1], reverse=True)
        top_general_fill = [(WD.labels[i], s) for i, s in all_scored if (WD.idx_general and i in set(WD.idx_general))]
        if not results:
            results = top_general_fill[:fallback_topk] if top_general_fill else [(WD.labels[i], s) for i, s in all_scored[:fallback_topk]]
        else:
            # If only character present, prepend some general tags
            if top_general_fill:
                results = top_general_fill[:min(15, fallback_topk)] + results

    # DEBUG
    if debug:
        try:
            providers = WD.ort_session.get_providers()
        except Exception:
            providers = ["<unknown>"]
        gmax = max((p for _, p in general_list), default=0.0)
        cmax = max((p for _, p in character_list), default=0.0)

        print("[WD DEBUG] ---- wd-eva02 run ----")
        print(f"[WD DEBUG] providers={providers}")
        print(f"[WD DEBUG] input_shape_decl={shape} | layout={layout_used} | norm={norm_used} | channels={channel_order}")
        print(f"[WD DEBUG] output_activation={activation} | raw_range=[{rmin:.3f},{rmax:.3f}]")
        print(f"[WD DEBUG] thresholds: general={th_general:.3f}, character={CAMIE_THRESHOLD_MAP.get('character', 0.35):.3f}")
        print(f"[WD DEBUG] counts: general={len(general_list)}, character={len(character_list)}, total={num_tags}")
        print(f"[WD DEBUG] max: general={gmax:.3f}, character={cmax:.3f}")

        ban = set(WD.idx_rating or [])
        all_scored_prev = [(WD.labels[i], float(probs[i])) for i in range(num_tags) if i not in ban]
        all_scored_prev.sort(key=lambda x: x[1], reverse=True)
        preview = ", ".join([f"{t}:{s:.2f}" for t, s in all_scored_prev[:10]])
        print(f"[WD DEBUG] top10 (all but rating): {preview}")

        if len(general_list) == 0:
            print("[WD DEBUG] general is empty: try lowering threshold (e.g., general=0.10), switch norm='imagenet', or ensure CSV/ONNX versions match.")

    # Build sentence
    tags_sentence = ", ".join([t.replace("_", " ") for t, _ in results])
    return tags_sentence, results, rating_list, general_list, character_list


# ===============================
# 3b) Camie tagger (ONNXRuntime)
# ===============================
CAMIE_REPO_DEFAULT = "Camais03/camie-tagger-v2"
CAMIE_MODEL_FILENAME = "camie-tagger-v2.onnx"
CAMIE_META_FILENAME = "camie-tagger-v2-metadata.json"
CAMIE_THRESHOLD_MAP = {
    "default": 0.50,
    "character": 0.70,
    "copyright": 0.50,
    "artist": 0.50,
    "meta": 0.50,
    "year": 0.50,
    "rating": 0.50,
}
CAMIE_TOPK_MAP = {
    "default": 32,
    "character": 40,
    "copyright": 24,
    "artist": 24,
    "meta": 16,
    "year": 8,
    "rating": 4,
}


def _ensure_camie_tagger():
    if CAMIE.ort_session is not None and CAMIE.idx_to_tag is not None and CAMIE.tag_to_category is not None:
        return
    if ort is None:
        raise RuntimeError("onnxruntime not available; install onnxruntime or onnxruntime-gpu to use camie-tagger")

    repo = os.getenv("CAMIE_TAGGER_REPO", CAMIE_REPO_DEFAULT)
    model_path = os.getenv("CAMIE_TAGGER_MODEL_PATH")
    meta_path = os.getenv("CAMIE_TAGGER_META_PATH")

    if not model_path:
        if hf_hub_download is None:
            raise RuntimeError("huggingface_hub not available; set CAMIE_TAGGER_MODEL_PATH to local ONNX file")
        model_path = hf_hub_download(repo_id=repo, filename=CAMIE_MODEL_FILENAME)
    if not meta_path:
        if hf_hub_download is None:
            raise RuntimeError("huggingface_hub not available; set CAMIE_TAGGER_META_PATH to local metadata JSON")
        meta_path = hf_hub_download(repo_id=repo, filename=CAMIE_META_FILENAME)

    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
    except Exception as exc:
        raise RuntimeError(f"Failed to load camie metadata: {exc}") from exc

    ds = meta.get("dataset_info", {})
    tag_map = ds.get("tag_mapping", {})
    idx_to_tag_map = tag_map.get("idx_to_tag", {}) or {}
    tag_to_category = tag_map.get("tag_to_category", {}) or {}
    total_tags = int(ds.get("total_tags", len(idx_to_tag_map)))
    idx_to_tag: List[str] = [""] * total_tags
    for k, v in idx_to_tag_map.items():
        try:
            i = int(k)
        except ValueError:
            continue
        if 0 <= i < total_tags:
            idx_to_tag[i] = v

    CAMIE.idx_to_tag = idx_to_tag
    CAMIE.tag_to_category = tag_to_category
    CAMIE.img_size = int(meta.get("model_info", {}).get("img_size", 512))

    providers = []
    if torch.cuda.is_available():
        providers.append("CUDAExecutionProvider")
    providers.append("CPUExecutionProvider")

    sess_options = ort.SessionOptions()
    try:
        CAMIE.ort_session = ort.InferenceSession(model_path, sess_options, providers=providers)
    except Exception as exc:
        # Retry CPU-only if CUDA fails
        if providers and providers[0] == "CUDAExecutionProvider":
            try:
                CAMIE.ort_session = ort.InferenceSession(model_path, sess_options, providers=["CPUExecutionProvider"])
            except Exception as exc_cpu:
                raise RuntimeError(f"Failed to initialize camie ONNX session (CPU fallback also failed): {exc_cpu}") from exc_cpu
        else:
            raise RuntimeError(f"Failed to initialize camie ONNX session: {exc}") from exc


CAMIE_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
CAMIE_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _camie_preprocess(img: Image.Image, target: int) -> np.ndarray:
    if img.mode not in ("RGB",):
        img = img.convert("RGB")

    w, h = img.size
    if w == 0 or h == 0:
        canvas = Image.new("RGB", (target, target), (124, 116, 104))
        arr = np.asarray(canvas).astype(np.float32) / 255.0
    else:
        aspect = w / h
        if aspect >= 1.0:
            new_w = target
            new_h = max(1, int(round(target / aspect)))
        else:
            new_h = target
            new_w = max(1, int(round(target * aspect)))

        try:
            resample = Image.Resampling.LANCZOS  # Pillow >=9
        except AttributeError:
            resample = Image.LANCZOS

        resized = img.resize((new_w, new_h), resample)
        pad_color = (124, 116, 104)
        canvas = Image.new("RGB", (target, target), pad_color)
        offset = ((target - new_w) // 2, (target - new_h) // 2)
        canvas.paste(resized, offset)
        arr = np.asarray(canvas).astype(np.float32) / 255.0

    arr = (arr - CAMIE_IMAGENET_MEAN) / CAMIE_IMAGENET_STD
    arr = np.transpose(arr, (2, 0, 1)).astype(np.float32)  # CHW
    return arr


def _run_camie_tagger(
    img: Image.Image,
    threshold_overrides: Optional[Dict[str, float]] = None,
) -> Dict[str, List[Tuple[str, float]]]:
    if ort is None:
        print("[CAMIE] onnxruntime unavailable; skipping camie tagger")
        return {}
    try:
        _ensure_camie_tagger()
    except Exception as exc:
        print(f"[CAMIE] initialization failed: {exc}")
        return {}
    if CAMIE.ort_session is None or CAMIE.idx_to_tag is None or CAMIE.tag_to_category is None:
        return {}
    if img is None:
        return {}

    arr = _camie_preprocess(img, CAMIE.img_size)
    inputs = CAMIE.ort_session.get_inputs()
    if not inputs:
        print("[CAMIE] ONNX session has no inputs; skipping")
        return {}
    input_name = inputs[0].name
    try:
        raw_outputs = CAMIE.ort_session.run(None, {input_name: arr[None, ...]})
    except Exception as exc:
        print(f"[CAMIE] inference failed: {exc}")
        return {}

    if not raw_outputs:
        return {}
    logits = raw_outputs[1] if len(raw_outputs) >= 2 else raw_outputs[0]
    probs = 1.0 / (1.0 + np.exp(-logits))
    if probs.ndim == 2:
        probs = probs[0]
    probs = np.asarray(probs, dtype=np.float32)

    tag_map = CAMIE.tag_to_category or {}
    idx_to_tag = CAMIE.idx_to_tag or []
    limit = min(len(idx_to_tag), probs.shape[0])
    buckets: Dict[str, List[Tuple[str, float]]] = {}
    for i in range(limit):
        tag = idx_to_tag[i]
        if not tag:
            continue
        score = float(probs[i])
        cat = tag_map.get(tag, "general")
        buckets.setdefault(cat, []).append((tag, score))

    filtered: Dict[str, List[Tuple[str, float]]] = {}
    thr_map = CAMIE_THRESHOLD_MAP.copy()
    overrides = threshold_overrides or {}
    if overrides:
        thr_map.update(overrides)
    default_thr = thr_map.get("default", 0.45)
    default_topk = CAMIE_TOPK_MAP.get("default", 32)
    for cat, items in buckets.items():
        items.sort(key=lambda x: x[1], reverse=True)
        thr = thr_map.get(cat, default_thr)
        topk = CAMIE_TOPK_MAP.get(cat, default_topk)
        selected = [(tag, score) for tag, score in items if score >= thr]
        if not selected:
            if cat in overrides:
                selected = []
            else:
                selected = items[:max(1, topk)]
        filtered[cat] = selected[:topk]

    return filtered


def _combine_tagger_outputs(
    wd_general: List[Tuple[str, float]],
    wd_rating: List[Tuple[str, float]],
    wd_character: List[Tuple[str, float]],
    camie_buckets: Dict[str, List[Tuple[str, float]]],
    *,
    max_general: int = 50,
    sentence_cap: int = 80,
) -> Tuple[str, Dict[str, str]]:
    sections: Dict[str, str] = {}

    def prettify(tag: str) -> str:
        return tag.replace("_", " ")

    def add_section(key: str, title: str, source: str, items: List[Tuple[str, float]]):
        count = len(items)
        header = f"><font size=\"4\" color=\"red\"> **{title}** ({count} tag{'s' if count != 1 else ''} · {source})</font><br>"
        if items:
            body = ">" + ",".join(
                f" {tag} ({score * 100:.0f}%)" for tag, score in items
            )
        else:
            body = ">_**No tags above threshold.**_"
        sections[key] = "\n".join([header, body]) + "\n***"

    add_section("general", "General", "wd-eva02", wd_general[:max_general])
    add_section("rating", "Rating", "wd-eva02", wd_rating)

    def _resolve_cat(cat: str) -> List[Tuple[str, float]]:
        items = camie_buckets.get(cat, [])
        if cat == "character" and not items:
            return wd_character[:max_general]
        return items

    for cat in ["character", "copyright", "artist", "meta", "year"]:
        items = _resolve_cat(cat)
        source = "camie" # if camie_buckets.get(cat) else "wd-eva02"
        add_section(cat, cat.title(), source, items)

    sentence_tags: List[str] = []
    sentence_tags.extend([t for t, _ in wd_general[:max_general]])
    for cat in ["character", "copyright", "artist", "meta", "year"]:
        for t, _ in _resolve_cat(cat)[:max_general]:
            sentence_tags.append(t)

    sentence = ", ".join(prettify(t) for t in sentence_tags[:sentence_cap])

    return sentence, sections


TAG_SECTION_ORDER = [
    ("general", "General"),
    ("rating", "Rating"),
    ("character", "Character"),
    ("copyright", "Copyright"),
    ("artist", "Artist"),
    ("meta", "Meta"),
    ("year", "Year"),
]


def _reset_tag_updates():
    updates = [gr.update(visible=False)]
    updates.extend(gr.update(value="", visible=False) for _ in TAG_SECTION_ORDER)
    return updates


def _apply_tag_updates(sections_md: Optional[Dict[str, str]]):
    has_sections = bool(sections_md)
    updates = [gr.update(visible=has_sections)]
    for key, _ in TAG_SECTION_ORDER:
        text = (sections_md or {}).get(key, "")
        updates.append(gr.update(value=text, visible=bool(text.strip())))
    return updates


# ============================
# 4) BLIP2 (natural language caption)
# ============================
@dataclass
class BLIP2Cache:
    model: Optional[Blip2ForConditionalGeneration] = None
    processor: Optional[Blip2Processor] = None
B2 = BLIP2Cache()

def _ensure_blip2():
    if B2.model is not None and B2.processor is not None:
        return
    if Blip2Processor is None or Blip2ForConditionalGeneration is None:
        raise RuntimeError("transformers not installed; please pip install transformers accelerate")

    repo = "Salesforce/blip2-opt-2.7b"
    B2.processor = Blip2Processor.from_pretrained(repo)
    if IMG2TEXT_DEVICE == "cpu":
        dtype = torch.float32
        B2.model = Blip2ForConditionalGeneration.from_pretrained(repo, torch_dtype=dtype, device_map=None)
        B2.model = B2.model.to("cpu")
    else:
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        B2.model = Blip2ForConditionalGeneration.from_pretrained(repo, torch_dtype=dtype, device_map="auto" if torch.cuda.is_available() else None)
        if torch.cuda.is_available():
            B2.model = B2.model.to("cuda")

@torch.inference_mode()
def blip2_caption(img: Image.Image, prompt: Optional[str] = None, max_new_tokens: int = 40) -> str:
    _ensure_blip2()
    if img.mode != "RGB":
        img = img.convert("RGB")
    inputs = B2.processor(images=img, text=prompt, return_tensors="pt").to(B2.model.device)
    generated_ids = B2.model.generate(**inputs, max_new_tokens=max_new_tokens)
    text = B2.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    return text

# =================
# 5) SD inference functions
# =================
def generate(model_key: str, prompt: str, neg: Optional[str], steps: int, guidance: float,
             width: int, height: int, scheduler: str, seed: Optional[str]):
    ensure_pipe(model_key)

    Sched = scheduler_map.get(scheduler, EulerDiscreteScheduler)
    CACHE.pipe.scheduler = Sched.from_config(CACHE.pipe.scheduler.config)

    cfg = MODEL_REGISTRY.get(model_key, {})
    sched_cfg_overrides = cfg.get('scheduler_config')
    if sched_cfg_overrides:
        base_config = dict(CACHE.pipe.scheduler.config)
        base_config.update(sched_cfg_overrides)
        CACHE.pipe.scheduler = CACHE.pipe.scheduler.__class__.from_config(base_config)

    g = None
    if seed and str(seed).strip() != '':
        g = torch.Generator(device=DEVICE if torch.cuda.is_available() else "cpu").manual_seed(int(seed))

    pipe_cfg_type = cfg.get('type')
    if pipe_cfg_type == 'sdxl' and isinstance(CACHE.pipe, StableDiffusionXLPipeline):
        exec_device = getattr(CACHE.pipe, "_execution_device", CACHE.pipe.text_encoder.device)
        compel_obj, empty_conditioning = get_compel_for_sdxl(
            [CACHE.pipe.tokenizer, getattr(CACHE.pipe, "tokenizer_2", None)],
            [CACHE.pipe.text_encoder, getattr(CACHE.pipe, "text_encoder_2", None)],
            device=exec_device,
        )
        text_dtype = CACHE.pipe.text_encoder.dtype
        pooled_dtype = getattr(CACHE.pipe.text_encoder_2, "dtype", text_dtype)

        pos_list = [prompt]
        neg_text = neg if (neg is not None) else ""
        neg_list = [neg_text]

        with torch.no_grad():
            prompt_embeds, pooled_prompt_embeds = compel_obj(pos_list)
            negative_prompt_embeds, negative_pooled_prompt_embeds = compel_obj(neg_list)

        prompt_embeds, negative_prompt_embeds = compel_obj.pad_conditioning_tensors_to_same_length(
            [prompt_embeds, negative_prompt_embeds], precomputed_padding=empty_conditioning
        )

        prompt_embeds = prompt_embeds.to(device=exec_device, dtype=text_dtype)
        negative_prompt_embeds = negative_prompt_embeds.to(device=exec_device, dtype=text_dtype)
        pooled_prompt_embeds = pooled_prompt_embeds.to(device=exec_device, dtype=pooled_dtype)
        negative_pooled_prompt_embeds = negative_pooled_prompt_embeds.to(device=exec_device, dtype=pooled_dtype)

        image = CACHE.pipe(
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            num_inference_steps=int(steps),
            guidance_scale=float(guidance),
            width=int(width),
            height=int(height),
            generator=g,
        ).images[0]
    else:
        image = CACHE.pipe(
            prompt=prompt,
            negative_prompt=(neg or None),
            num_inference_steps=int(steps),
            guidance_scale=float(guidance),
            width=int(width),
            height=int(height),
            generator=g,
        ).images[0]
    return image

# text-to-image + img2text
def generate_then_interrogate(model_name: str, prompt: str, neg: str, steps: int, guidance: float,
                              width: int, height: int, scheduler: str, seed: str,
                              th_general: float,
                              camie_general_thr: float = CAMIE_THRESHOLD_MAP.get("default", 0.45),
                              camie_character_thr: float = CAMIE_THRESHOLD_MAP.get("character", 0.35),
                              blip_prompt: str = "", max_tokens: int = 40):
    model_key = key_by_name.get(model_name, DEFAULT_KEY)
    img = generate(model_key, prompt, neg, steps, guidance, width, height, scheduler, seed)

    # Run taggers (WD for general/rating, Camie for others)
    wd_sentence, _, wd_rating, wd_general, wd_character = _run_wd_tagger(
        img, th_general=th_general, th_character=CAMIE_THRESHOLD_MAP.get("character", 0.35)
    )
    camie_buckets = _run_camie_tagger(
        img,
        threshold_overrides={
            "character": camie_character_thr,
            "artist": camie_general_thr,
            "copyright": camie_general_thr,
            "meta": camie_general_thr,
            "year": camie_general_thr,
        },
    )
    tag_sentence, tag_sections = _combine_tagger_outputs(wd_general, wd_rating, wd_character, camie_buckets)

    # Run BLIP2
    nl = blip2_caption(img, prompt=blip_prompt if blip_prompt.strip() else None, max_new_tokens=max_tokens)

    return img, (tag_sentence or wd_sentence), nl, tag_sections

# Standalone img2text
def img2text_handle(img: Image.Image, th_general: float,
                    camie_general_thr: float = CAMIE_THRESHOLD_MAP.get("default", 0.45),
                    camie_character_thr: float = CAMIE_THRESHOLD_MAP.get("character", 0.35),
                    blip_prompt: str = "", max_tokens: int = 40):
    if img is None:
        return "", "", {}
    wd_sentence, _, wd_rating, wd_general, wd_character = _run_wd_tagger(
        img, th_general=th_general, th_character=CAMIE_THRESHOLD_MAP.get("character", 0.35)
    )
    camie_buckets = _run_camie_tagger(
        img,
        threshold_overrides={
            "character": camie_character_thr,
            "artist": camie_general_thr,
            "copyright": camie_general_thr,
            "meta": camie_general_thr,
            "year": camie_general_thr,
        },
    )
    tag_sentence, tag_sections = _combine_tagger_outputs(wd_general, wd_rating, wd_character, camie_buckets)
    nl = blip2_caption(img, prompt=blip_prompt if blip_prompt.strip() else None, max_new_tokens=max_tokens)
    return (tag_sentence or wd_sentence), nl, tag_sections

# =========================
# 6) Token counting helpers
# =========================
SD15_TOKEN_LIMIT = 77
SDXL_TOKEN_LIMIT = 77


@lru_cache(maxsize=1)
def _get_sd15_tokenizer():
    if CLIPTokenizer is None:
        raise RuntimeError("transformers not installed; please pip install transformers")
    tok = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    tok.model_max_length = 100000
    return tok


@lru_cache(maxsize=1)
def _get_sdxl_tokenizer_primary():
    if CLIPTokenizer is None:
        raise RuntimeError("transformers not installed; please pip install transformers")
    tok = CLIPTokenizer.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", subfolder="tokenizer")
    tok.model_max_length = 100000
    return tok


@lru_cache(maxsize=1)
def _get_sdxl_tokenizer_secondary():
    if CLIPTokenizer is None:
        raise RuntimeError("transformers not installed; please pip install transformers")
    tok = CLIPTokenizer.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", subfolder="tokenizer_2")
    tok.model_max_length = 100000
    return tok


def _extract_ids(tokenizer, text: str) -> List[int]:
    encoded = tokenizer(
        text,
        add_special_tokens=True,
        padding=False,
        truncation=False,
        return_attention_mask=False,
    )["input_ids"]
    if encoded and isinstance(encoded[0], list):
        return encoded[0]
    return encoded


LANG_CHOICES = {
    "日本語": "ja",
    "English": "en",
}


TOKEN_LIMIT_INFO_MD = {
    "en": (
        "### Text Encoder Basics\n"
        "- SD1.5 uses a single CLIP text encoder and only reads the first 77 tokens. Anything beyond that is truncated.\n"
        "- SDXL feeds the same prompt into two CLIP text encoders. In this app, prompts longer than 77 tokens are automatically chunked into multiple 77-token blocks so no text is lost.\n"
        "- The [BOS] (beginning of sentence) and [EOS] (end of sentence) tokens take up 2 tokens per block."
    ),
    "ja": (
        "### テキストエンコーダの基本\n"
        "- SD1.5はCLIPテキストエンコーダを1つだけ使い、先頭から**77トークン**までしか読みません。これを超えた部分は切り捨てられます。\n"
        "- SDXLは同じプロンプトを2種類のテキストエンコーダに流します。このアプリでは77トークンを超えると自動的に複数ブロック（各77トークン）に分割して全てのテキストを処理します。\n"
        "- 各ブロックごとに[BOS]・[EOS]で2トークンを消費します。"
    ),
}


def _resolve_lang(lang_choice: str) -> str:
    return LANG_CHOICES.get(lang_choice, "en")


def token_count_handle(text: str, lang_choice: str) -> str:
    lang = _resolve_lang(lang_choice)
    if CLIPTokenizer is None:
        return (
            "transformers/CLIPTokenizer unavailable. Install transformers to enable token counting."
            if lang == "en"
            else "transformers/CLIPTokenizerが見つかりません。token countingを使うにはtransformersをインストールしてください。"
        )

    prompt = (text or "").strip()
    if not prompt:
        return "Please enter text to analyze." if lang == "en" else "テキストを入力してください。"

    try:
        sd15_ids = _extract_ids(_get_sd15_tokenizer(), prompt)
    except Exception as exc:
        return (
            f"Failed to load SD1.5 tokenizer: {exc}"
            if lang == "en"
            else f"SD1.5トークナイザの読み込みに失敗しました: {exc}"
        )

    try:
        sdxl_ids_primary = _extract_ids(_get_sdxl_tokenizer_primary(), prompt)
        sdxl_ids_secondary = _extract_ids(_get_sdxl_tokenizer_secondary(), prompt)
    except Exception as exc:
        return (
            f"Failed to load SDXL tokenizers: {exc}"
            if lang == "en"
            else f"SDXLトークナイザの読み込みに失敗しました: {exc}"
        )

    def _format_line(label_en: str, label_ja: str, length: int, limit: int, chunked: bool = False) -> str:
        blocks = max(1, (length + limit - 1) // limit)
        effective = min(length, limit)
        usable = max(effective - 2, 0)
        if chunked:
            if lang == "en":
                status = f"auto-chunked into {blocks} block(s); all tokens are kept."
                return f"- {label_en}: **{length}** tokens (block size {limit}, usable {usable} per block) – {status}"
            status = f"自動的に{blocks}ブロックへ分割され、全トークンが処理されます。"
            return f"- {label_ja}: トークン数{length}（ブロック長{limit}・各ブロック有効{usable}） – {status}"

        over = max(0, length - limit)
        if over > 0:
            status_en = f"truncates {over} token(s)"
            status_ja = f"超過分{over}トークンが切り捨てられます"
        else:
            status_en = "fits within the limit"
            status_ja = "制限内に収まります"
        if lang == "en":
            return f"- {label_en}: **{length}** tokens (usable {usable}/{limit}) – {status_en}."
        return f"- {label_ja}: トークン数{length}（有効{usable}/{limit}） – {status_ja}。"

    header = "### Token counts" if lang == "en" else "### トークン数"
    lines = [header]
    lines.append(_format_line("SD1.5 CLIP", "SD1.5のCLIP", len(sd15_ids), SD15_TOKEN_LIMIT))
    lines.append(
        _format_line(
            "SDXL text encoder 1",
            "SDXLテキストエンコーダ1",
            len(sdxl_ids_primary),
            SDXL_TOKEN_LIMIT,
            chunked=True,
        )
    )
    lines.append(
        _format_line(
            "SDXL text encoder 2",
            "SDXLテキストエンコーダ2",
            len(sdxl_ids_secondary),
            SDXL_TOKEN_LIMIT,
            chunked=True,
        )
    )

    return "\n".join(lines)

# ============ UI: 6) Components & wiring ============
model_keys = list(MODEL_REGISTRY.keys())
model_names = [MODEL_REGISTRY[k]['name'] for k in model_keys]
key_by_name = {MODEL_REGISTRY[k]['name']: k for k in model_keys}

DEFAULT_KEY = model_keys[0]
DEFAULT_NAME = MODEL_REGISTRY[DEFAULT_KEY]['name']

def on_model_change(model_name: str):
    k = key_by_name.get(model_name, DEFAULT_KEY)
    cfg = MODEL_REGISTRY[k]
    p = cfg['presets']
    return (
        gr.update(choices=p['widths'], value=p['default_w']),
        gr.update(choices=p['heights'], value=p['default_h']),
        gr.update(value=p['guidance']),
        gr.update(value=p['steps']),
        gr.update(value=p.get('default_scheduler', 'euler')),
    )

with gr.Blocks(title='Stable Diffusion Toolkit') as demo:
    gr.Markdown(
        "# Stable Diffusion Toolkit\n"
        f"- Text-to-image device: **{DEVICE}** · Image-to-text device: **{IMG2TEXT_DEVICE}**\n"
        "- Generate images, auto-tag them, or run BLIP2 captions in one place.\n"
        "- Use other tabs for image description, Danbooru tag lookup, and prompt token counts."
    )

    with gr.Tabs():
        with gr.Tab("Generate Image"):
            with gr.Row():
                model_sel_name = gr.Dropdown(model_names, value=DEFAULT_NAME, label='Model')
            with gr.Row():
                prompt = gr.Textbox(label='Prompt (what to generate)', value='masterpiece, best quality, 1girl, looking at viewer')
                neg = gr.Textbox(label='Negative (what to avoid)', value='nsfw, lowres, blurry, watermark')
            with gr.Row():
                steps = gr.Slider(5, 100, value=MODEL_REGISTRY[DEFAULT_KEY]['presets']['steps'], step=1, label='Steps (more = better, slower)')
                guidance = gr.Slider(0.5, 20.0, value=MODEL_REGISTRY[DEFAULT_KEY]['presets']['guidance'], step=0.1, label='Guidance (higher = follow prompt)')
            with gr.Row():
                width = gr.Dropdown(choices=MODEL_REGISTRY[DEFAULT_KEY]['presets']['widths'], value=MODEL_REGISTRY[DEFAULT_KEY]['presets']['default_w'], label='Width (px)')
                height = gr.Dropdown(choices=MODEL_REGISTRY[DEFAULT_KEY]['presets']['heights'], value=MODEL_REGISTRY[DEFAULT_KEY]['presets']['default_h'], label='Height (px)')
            with gr.Row():
                scheduler = gr.Dropdown(
                    choices=['euler','ddim','dpmpp2m'],
                    value=MODEL_REGISTRY[DEFAULT_KEY]['presets'].get('default_scheduler', 'euler'),
                    label='Sampler (scheduler)'
                )
                seed = gr.Textbox(label='Seed (leave blank for random)', value='')

            with gr.Accordion("Auto description settings", open=False):
                th_general = gr.Slider(0.0, 1.0, 0.35, 0.01, label="WD tagger threshold (general)")
                with gr.Row():
                    camie_general_thr = gr.Slider(0.05, 1.0, 0.492, 0.01, label="Camie tagger threshold (copyright/artist/meta/year)")
                    camie_character_thr = gr.Slider(0.05, 1.0, 0.75, 0.01, label="Camie tagger threshold (character)")
                with gr.Row():
                    blip_prompt = gr.Textbox(label="Caption prompt (optional)", value="")
                    max_tokens = gr.Slider(8, 120, 40, 1, label="Caption length (max tokens)")

            btn = gr.Button('Generate Image')
            out_img = gr.Image(label='Generated image', type='pil', interactive=False, sources=[])
            btn_interrogate = gr.Button('Describe Image')
            out_wd_sentence = gr.Textbox(label="Tag summary", lines=6, show_copy_button=True)
            out_blip = gr.Textbox(label="Caption", lines=6, show_copy_button=True)
            with gr.Group(visible=False) as tag_sections_group:
                gr.Markdown("### Tag Details")
                tag_section_components: Dict[str, gr.Markdown] = {}
                for key, title in TAG_SECTION_ORDER:
                    tag_section_components[key] = gr.Markdown(label=title, value="", show_label=False)
                tag_section_component_list = [tag_section_components[key] for key, _ in TAG_SECTION_ORDER]

            model_sel_name.change(on_model_change, inputs=[model_sel_name], outputs=[width, height, guidance, steps, scheduler])

            # 1) Generate image only
            def _on_generate(model_name, prm, neg_prompt, steps_v, guidance_v, width_v, height_v, scheduler_v, seed_v):
                img = generate(key_by_name.get(model_name, DEFAULT_KEY), prm, neg_prompt, steps_v, guidance_v, width_v, height_v, scheduler_v, seed_v)
                return (img, *_reset_tag_updates())

            btn.click(
                _on_generate,
                inputs=[model_sel_name, prompt, neg, steps, guidance, width, height, scheduler, seed],
                outputs=[out_img, tag_sections_group, *tag_section_component_list],
            )

            # 2) Manually run img2text on the image above when needed
            def _run_interrogate(img_in, *params):
                tags, caption, sections_md = img2text_handle(img_in, *params)
                updates = _apply_tag_updates(sections_md) if img_in else _reset_tag_updates()
                return (tags, caption, *updates)

            btn_interrogate.click(
                _run_interrogate,
                inputs=[out_img, th_general, camie_general_thr, camie_character_thr, blip_prompt, max_tokens],
                outputs=[out_wd_sentence, out_blip, tag_sections_group, *tag_section_component_list],
            )

        with gr.Tab("Describe Image"):
            in_img = gr.Image(label="Image", type="pil")
            th_general2 = gr.Slider(0.0, 1.0, 0.35, 0.01, label="WD tagger threshold (general)")
            with gr.Row():
                camie_general_thr2 = gr.Slider(0.05, 1.0, 0.492, 0.01, label="Camie tagger threshold (copyright/artist/meta/year)")
                camie_character_thr2 = gr.Slider(0.05, 1.0, 0.75, 0.01, label="Camie tagger threshold (character)")
            with gr.Row():
                blip_prompt2 = gr.Textbox(label="Caption prompt (optional)", value="")
                max_tokens2 = gr.Slider(8, 120, 40, 1, label="Caption length (max tokens)")

            btn2 = gr.Button("Describe Image")
            out_wd_sentence2 = gr.Textbox(label="Tag summary", lines=6, show_copy_button=True)
            out_blip2 = gr.Textbox(label="Caption", lines=6, show_copy_button=True)
            with gr.Group(visible=False) as tag_sections_group2:
                gr.Markdown("### Tag Details")
                tag_section_components2: Dict[str, gr.Markdown] = {}
                for key, title in TAG_SECTION_ORDER:
                    tag_section_components2[key] = gr.Markdown(label=title, value="", show_label=False)
                tag_section_component_list2 = [tag_section_components2[key] for key, _ in TAG_SECTION_ORDER]

            btn2.click(
                _run_interrogate,
                inputs=[in_img, th_general2, camie_general_thr2, camie_character_thr2, blip_prompt2, max_tokens2],
                outputs=[out_wd_sentence2, out_blip2, tag_sections_group2, *tag_section_component_list2]
            )

        # Danbooru tag lookup (JA-friendly/EN)
        with gr.Tab("Danbooru Tag Search"):
            danbooru_lang = gr.Radio(
                choices=list(LANG_CHOICES.keys()),
                value="日本語",
                label="Language / 言語",
            )
            _initial_texts = _danbooru_lang_text(_resolve_lang("日本語"))
            danbooru_intro = gr.Markdown(_initial_texts["intro"])
            with gr.Row():
                tag_query = gr.Textbox(label=_initial_texts["search_label"], placeholder=_initial_texts["search_placeholder"], lines=1)
                tag_max = gr.Slider(5, 200, value=50, step=5, label=_initial_texts["max_label"])
            with gr.Row():
                tag_table = gr.Dataframe(label=_initial_texts["table_label"], headers=_initial_texts["table_headers"], interactive=False, row_count=5, col_count=2)
            tag_choices = gr.CheckboxGroup(choices=[], label=_initial_texts["choices_label"])
            with gr.Row():
                tag_add = gr.Button(_initial_texts["add_label"])
                tag_clear = gr.Button(_initial_texts["clear_label"])
            tag_prompt_box = gr.Textbox(label=_initial_texts["selected_label"], show_copy_button=True)

            def _on_tag_query(q, m):
                table, sugg = _search_tags_ja(q, int(m))
                # update table and reset checkbox selections when choices change
                return gr.update(value=table), gr.update(choices=sugg, value=[])

            def _on_tag_add(sel: List[str], cur: str):
                # current box tokens (pretty form, may contain spaces)
                cur_list = [x.strip() for x in (cur or "").replace("，", ",").split(",") if x.strip()]
                # canonical set for dedup (underscored lowercase)
                exist_canon = set()
                for x in cur_list:
                    canon = x.replace(" ", "_").lower()
                    if canon:
                        exist_canon.add(canon)
                added_pretty: List[str] = []
                for s in (sel or []):
                    if not s:
                        continue
                    canon = s.lower()  # selected tags are underscored
                    if canon not in exist_canon:
                        added_pretty.append(s.replace("_", " "))
                        exist_canon.add(canon)
                out_list = cur_list + added_pretty
                out = ", ".join(out_list) if out_list else ""
                # also clear selections after adding
                return out, gr.update(value=[])

            def _on_tag_clear():
                # clear prompt box and selections
                return "", gr.update(value=[])

            tag_query.change(_on_tag_query, inputs=[tag_query, tag_max], outputs=[tag_table, tag_choices])
            tag_max.change(_on_tag_query, inputs=[tag_query, tag_max], outputs=[tag_table, tag_choices])
            tag_add.click(_on_tag_add, inputs=[tag_choices, tag_prompt_box], outputs=[tag_prompt_box, tag_choices])
            tag_clear.click(_on_tag_clear, inputs=[], outputs=[tag_prompt_box, tag_choices])

            def _on_danbooru_lang_change(choice: str):
                lang = _resolve_lang(choice)
                texts = _danbooru_lang_text(lang)
                return (
                    gr.update(value=texts["intro"]),
                    gr.update(label=texts["search_label"], placeholder=texts["search_placeholder"]),
                    gr.update(label=texts["max_label"]),
                    gr.update(label=texts["table_label"], headers=texts["table_headers"]),
                    gr.update(label=texts["choices_label"]),
                    gr.update(value=texts["add_label"]),
                    gr.update(value=texts["clear_label"]),
                    gr.update(label=texts["selected_label"]),
                )

            danbooru_lang.change(
                _on_danbooru_lang_change,
                inputs=[danbooru_lang],
                outputs=[danbooru_intro, tag_query, tag_max, tag_table, tag_choices, tag_add, tag_clear, tag_prompt_box],
            )

        with gr.Tab("Token Counter"):
            token_lang = gr.Radio(
                choices=list(LANG_CHOICES.keys()),
                value="日本語",
                label="Language / 言語",
            )
            token_info = gr.Markdown(TOKEN_LIMIT_INFO_MD["ja"])
            token_input = gr.Textbox(
                label="入力テキスト",
                lines=6,
                placeholder="プロンプトまたはネガティブプロンプトを入力してください…",
            )
            token_button = gr.Button("トークン数を計算")
            token_output = gr.Markdown()

            token_button.click(token_count_handle, inputs=[token_input, token_lang], outputs=[token_output])
            token_input.submit(token_count_handle, inputs=[token_input, token_lang], outputs=[token_output])

            def _on_token_lang_change(lang_choice: str, current_text: str):
                lang = _resolve_lang(lang_choice)
                info = TOKEN_LIMIT_INFO_MD.get(lang, TOKEN_LIMIT_INFO_MD["en"])
                textbox_update = gr.update(
                    label="Input text" if lang == "en" else "入力テキスト",
                    placeholder="Type your prompt or negative prompt here..." if lang == "en" else "プロンプトまたはネガティブプロンプトを入力してください…",
                )
                button_update = gr.update(value="Count tokens" if lang == "en" else "トークン数を計算")
                output = token_count_handle(current_text, lang_choice) if (current_text or "").strip() else ""
                return gr.update(value=info), textbox_update, button_update, output

            token_lang.change(
                _on_token_lang_change,
                inputs=[token_lang, token_input],
                outputs=[token_info, token_input, token_button, token_output],
            )

if __name__ == '__main__':
    # Allow lazy loading (download/load on first use), or warm up:
    try:
        _ensure_wd_eva02()
    except Exception as e:
        print(f"[WD-EVA02] Preload failed (will delay to first call): {e}")
    try:
        _ensure_blip2()
    except Exception as e:
        print(f"[BLIP2] Preload failed (will delay to first call): {e}")
    try:
        _ensure_camie_tagger()
    except Exception as e:
        print(f"[CAMIE] Preload failed (will delay to first call): {e}")

    launch_kwargs = dict(server_name='0.0.0.0', server_port=7860, share=False)
    favicon_candidate = os.getenv("GRADIO_FAVICON_PATH", "")
    if favicon_candidate and os.path.exists(favicon_candidate):
        launch_kwargs["favicon_path"] = favicon_candidate
    else:
        default_icon = os.path.join(os.path.dirname(__file__), "favicon.png")
        if os.path.exists(default_icon):
            launch_kwargs["favicon_path"] = default_icon

    demo.queue(max_size=32).launch(**launch_kwargs)
