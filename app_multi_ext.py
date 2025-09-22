# app_multi_ext.py
# -*- coding: utf-8 -*-
import os, json, csv, html, torch, gradio as gr
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

DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Device for img2text (default CPU; use GPU: export IMG2TEXT_DEVICE=cuda)
IMG2TEXT_DEVICE = os.getenv("IMG2TEXT_DEVICE", "cpu").strip().lower()  # "cpu" / "cuda"

# ========================
# 1) Model registry (add/remove as needed)
# ========================
MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {
    "illustrious_emberveil": {
        "name": "【illustrious】EmberVeilMix (merge)",
        "type": "sdxl",
        "load": {
            "mode": "singlefile",
            "filename": "IllustriousEmberveilmix_v10.safetensors"  # Put in current directory or fetch via repo
            # "repo": "YourOrg/IllustriousEmberveilmix_v10",
        },
        "presets": {
            "widths":  [512, 640, 768, 896, 1024, 1152, 1232, 1280],
            "heights": [512, 640, 768, 896, 1024, 1152, 1232, 1280],
            "default_w": 1024,
            "default_h": 1024,
            "steps": 28,
            "guidance": 5.5,
        }
    },
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

scheduler_map = {
    'euler': EulerDiscreteScheduler,
    'ddim': DDIMScheduler,
    'dpmpp2m': DPMSolverMultistepScheduler,
}

def _free_pipe():
    if CACHE.pipe is not None:
        try: CACHE.pipe.to('cpu')
        except Exception: pass
        del CACHE.pipe
        CACHE.pipe = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

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

    p = p.to(DEVICE)
    try: p.enable_xformers_memory_efficient_attention()
    except Exception: pass
    p.enable_vae_slicing(); p.enable_vae_tiling()
    return p

def ensure_pipe(model_key: str):
    if CACHE.pipe is not None and CACHE.model_key == model_key:
        return
    _free_pipe()
    cfg = MODEL_REGISTRY[model_key]
    CACHE.pipe = _load_from_cfg(cfg)
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

def _preprocess_for_wd(img: Image.Image) -> Image.Image:
    # Target 448×448; resize with aspect ratio, then center-crop to avoid distortion
    if img.mode != "RGB":
        img = img.convert("RGB")
    w, h = img.size
    target = 448
    scale = target / min(w, h)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    img = img.resize((new_w, new_h), Image.BICUBIC)
    # Center crop
    left = (new_w - target) // 2
    top = (new_h - target) // 2
    img = img.crop((left, top, left + target, top + target))
    return img


import numpy as np
def _run_wd_tagger(
    img: Image.Image,
    th_general: float = 0.25,
    th_character: float = 0.75,
    *,
    norm: str = "0to255",            # "0to255" (recommended for eva02 onnx) | "0to1" | "minus1to1" | "imagenet"
    force_layout: str = None,         # None | "nhwc" | "nchw"
    topk_preview: int = 10,           # DEBUG: how many to preview
    fallback_topk: int = 30,          # Fallback topK
    debug: bool = False
) -> Tuple[str, List[Tuple[str, float]], List[Tuple[str, float]]]:
    """
    Returns:
        tags_sentence: str (comma-separated)
        results: List[(tag, score)] (descending)
    """
    _ensure_wd_eva02()
    if img is None:
        return "", [], []

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
        print(f"[WD DEBUG] thresholds: general={th_general:.3f}, character={th_character:.3f}")
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
    return tags_sentence, results, rating_list

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

    g = None
    if seed and str(seed).strip() != '':
        g = torch.Generator(device=DEVICE if torch.cuda.is_available() else "cpu").manual_seed(int(seed))

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
                              th_general: float, th_character: float,
                              blip_prompt: str, max_tokens: int):
    model_key = key_by_name.get(model_name, DEFAULT_KEY)
    img = generate(model_key, prompt, neg, steps, guidance, width, height, scheduler, seed)

    # Run WD-EVA02
    wd_sentence, wd_pairs, wd_rating = _run_wd_tagger(img, th_general=th_general, th_character=th_character)
    # Run BLIP2
    nl = blip2_caption(img, prompt=blip_prompt if blip_prompt.strip() else None, max_new_tokens=max_tokens)

    # Show rating + top tags as Markdown tables
    r_tbl = ""
    if wd_rating:
        r_tbl = "### rating\n| tag | score |\n|---|---|\n" + "\n".join([f"| {t} | {s:.3f} |" for t, s in wd_rating]) + "\n\n"
    topk = min(30, len(wd_pairs))
    md_table = r_tbl + "### general/character\n| tag | score |\n|---|---|\n" + "\n".join([f"| {t} | {s:.3f} |" for t, s in wd_pairs[:topk]])

    return img, wd_sentence, nl, md_table

# Standalone img2text
def img2text_handle(img: Image.Image, th_general: float, th_character: float,
                    blip_prompt: str, max_tokens: int):
    if img is None:
        return "", "", ""
    wd_sentence, wd_pairs, wd_rating = _run_wd_tagger(img, th_general=th_general, th_character=th_character)
    nl = blip2_caption(img, prompt=blip_prompt if blip_prompt.strip() else None, max_new_tokens=max_tokens)
    topk = min(30, len(wd_pairs))
    r_tbl = ""
    if wd_rating:
        r_tbl = "### rating\n| tag | score |\n|---|---|\n" + "\n".join([f"| {t} | {s:.3f} |" for t, s in wd_rating]) + "\n\n"
    md_table = r_tbl + "### general/character\n| tag | score |\n|---|---|\n" + "\n".join([f"| {t} | {s:.3f} |" for t, s in wd_pairs[:topk]])
    return wd_sentence, nl, md_table

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
        "- SD1.5 uses a single CLIP text encoder and only reads the first 77 tokens (≈75 prompt tokens). Anything beyond that is truncated.\n"
        "- SDXL feeds the same prompt into two CLIP text encoders, but each still keeps only the first 77 tokens, so extra tokens past 77 are dropped."
    ),
    "ja": (
        "### テキストエンコーダの基本\n"
        "- SD1.5はCLIPテキストエンコーダを1つだけ使い、先頭から77トークン（実質約75語）までしか読みません。これを超えた部分は切り捨てられます。\n"
        "- SDXLは同じプロンプトを2種類のテキストエンコーダに流しますが、どちらも77トークンまでしか処理しないため、超過分は同様に切り捨てられます。"
    ),
}


TOKENIZER_SPECS = {
    "sd15": {
        "label_en": "SD 1.5",
        "label_ja": "SD1.5",
        "ui_choice": "SD 1.5",
        "limit": SD15_TOKEN_LIMIT,
        "getter": _get_sd15_tokenizer,
    },
    "sdxl_primary": {
        "label_en": "SDXL (Primary)",
        "label_ja": "SDXL（第1エンコーダ）",
        "ui_choice": "SDXL (Primary)",
        "limit": SDXL_TOKEN_LIMIT,
        "getter": _get_sdxl_tokenizer_primary,
    },
    "sdxl_secondary": {
        "label_en": "SDXL (Secondary)",
        "label_ja": "SDXL（第2エンコーダ）",
        "ui_choice": "SDXL (Secondary)",
        "limit": SDXL_TOKEN_LIMIT,
        "getter": _get_sdxl_tokenizer_secondary,
    },
}

MODEL_KEY_ORDER = ["sd15", "sdxl_primary", "sdxl_secondary"]
MODEL_RADIO_CHOICES = [TOKENIZER_SPECS[k]["ui_choice"] for k in MODEL_KEY_ORDER]
MODEL_CHOICE_TO_KEY = {TOKENIZER_SPECS[k]["ui_choice"]: k for k in MODEL_KEY_ORDER}

DISPLAY_MODE_RADIO_CHOICES = ["Text", "Token IDs"]
DISPLAY_MODE_TO_KEY = {"Text": "text", "Token IDs": "ids"}
DISPLAY_MODE_LABELS = {
    "text": {"en": "Text", "ja": "テキスト"},
    "ids": {"en": "Token IDs", "ja": "トークンID"},
}

TOKEN_COLOR_PALETTE = [
    "#FDE1D3",
    "#FFE4F2",
    "#E3F7FF",
    "#E7F8F0",
    "#FDF2D7",
    "#EDEBFF",
    "#F6E8FF",
    "#DDF0FF",
]

SPECIAL_TOKEN_DISPLAY = {
    "<|startoftext|>": "⟨start⟩",
    "<|endoftext|>": "⟨end⟩",
    "<s>": "⟨start⟩",
    "</s>": "⟨end⟩",
    "[CLS]": "⟨CLS⟩",
    "[SEP]": "⟨SEP⟩",
}

TOKEN_COUNTER_EMPTY_SUMMARY = {
    "en": "Enter text to count tokens.",
    "ja": "テキストを入力するとトークン数を計算します。",
}

TOKEN_COUNTER_EXAMPLES = {
    "en": "A watercolor fox sits on a mossy rock at sunrise, surrounded by glowing fireflies.",
    "ja": "朝日の中、苔むした岩に座る水彩の狐。周囲には光るホタルが舞っている。",
}

TOKEN_COUNTER_CSS = """
.token-counter-wrapper {
    gap: 16px !important;
}
.token-counter-top-controls {
    gap: 12px !important;
    align-items: stretch;
}
.token-counter-actions {
    gap: 8px !important;
}
.token-counter-actions .gr-button {
    height: 44px;
}
.token-counter-stats .token-stats-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
    gap: 12px;
}
.token-counter-stats .token-stat-card {
    border-radius: 14px;
    padding: 14px;
    background: linear-gradient(180deg, rgba(129, 140, 248, 0.12), rgba(59, 130, 246, 0.06));
    border: 1px solid rgba(148, 163, 184, 0.4);
    backdrop-filter: blur(6px);
}
.token-counter-stats .token-stat-card.characters {
    background: linear-gradient(180deg, rgba(45, 212, 191, 0.12), rgba(16, 185, 129, 0.04));
}
.token-counter-stats .token-stat-title {
    font-size: 0.82rem;
    font-weight: 600;
    opacity: 0.76;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}
.token-counter-stats .token-stat-value {
    font-size: 1.8rem;
    font-weight: 700;
    margin-top: 6px;
}
.token-counter-stats .token-stat-sub {
    font-size: 0.88rem;
    margin-top: 6px;
    opacity: 0.75;
}
.token-counter-stats .token-stat-status {
    font-size: 0.78rem;
    margin-top: 8px;
    font-weight: 600;
}
.token-counter-stats .token-stat-status.ok {
    color: #047857;
}
.token-counter-stats .token-stat-status.warn {
    color: #b91c1c;
}
.token-counter-summary {
    font-size: 0.95rem;
}
.token-counter-visual .token-preview {
    border: 1px solid rgba(148, 163, 184, 0.4);
    border-radius: 16px;
    padding: 16px;
    background: var(--block-background-fill, #f8fafc);
    box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.5);
}
.token-preview-header {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    margin-bottom: 10px;
}
.token-preview-heading {
    font-weight: 700;
    font-size: 1rem;
}
.token-preview-mode {
    font-size: 0.85rem;
    opacity: 0.7;
}
.token-chip-wrap {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
}
.token-chip {
    border-radius: 12px;
    padding: 6px 10px;
    font-family: var(--font-mono, "JetBrains Mono", "SFMono-Regular", monospace);
    font-size: 0.85rem;
    white-space: pre;
    color: #0f172a;
    box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.6);
}
.token-preview-empty {
    display: flex;
    align-items: center;
    justify-content: center;
    min-height: 120px;
    border-radius: 16px;
    border: 1px dashed rgba(148, 163, 184, 0.55);
    background: rgba(148, 163, 184, 0.1);
    padding: 16px;
}
.token-preview-placeholder {
    font-size: 0.95rem;
    opacity: 0.8;
    text-align: center;
}
.token-counter-info {
    font-size: 0.9rem;
    opacity: 0.82;
}
.token-counter-textbox textarea {
    min-height: 220px;
    font-size: 1rem;
}
.token-counter-actions .gr-button {
    flex: 1 1 auto;
}
.token-counter-actions .gr-button:first-child {
    flex: 1.6 1 0;
}
.token-counter-top-controls > .gradio-column {
    min-width: 0;
}
"""


def _resolve_lang(lang_choice: str) -> str:
    return LANG_CHOICES.get(lang_choice, "en")


def _decode_single_token(tokenizer, token_id: int) -> str:
    try:
        return tokenizer.decode([token_id], clean_up_tokenization_spaces=False)
    except Exception:
        try:
            return tokenizer.convert_ids_to_tokens([token_id])[0]
        except Exception:
            return ""


def _present_display_token(token_str: str, decoded: str) -> str:
    token_str = token_str or ""
    decoded = (decoded or "").replace("\r", "")
    if decoded:
        return decoded
    if token_str in SPECIAL_TOKEN_DISPLAY:
        return SPECIAL_TOKEN_DISPLAY[token_str]
    if token_str.endswith("</w>"):
        return token_str[:-4]
    if token_str and token_str[0] in {"Ġ", "▁", "\u0120"}:
        return f" {token_str[1:]}"
    return token_str


def _format_token_display(text: str) -> str:
    if text is None:
        text = ""
    text = text.replace("\r", "")
    text = text.replace("\n", "⏎")
    text = text.replace("\t", "⇥")
    if text == "":
        return "∅"
    return html.escape(text)


def _build_token_report(key: str, prompt: str) -> Dict[str, Any]:
    spec = TOKENIZER_SPECS[key]
    tokenizer = spec["getter"]()
    ids = _extract_ids(tokenizer, prompt)
    tokens = tokenizer.convert_ids_to_tokens(ids)
    decoded_tokens = [_decode_single_token(tokenizer, tid) for tid in ids]
    display_tokens = [
        _present_display_token(tok, decoded)
        for tok, decoded in zip(tokens, decoded_tokens)
    ]
    length = len(ids)
    limit = spec["limit"]
    over = max(0, length - limit)
    effective = min(length, limit)
    usable = max(effective - 2, 0)
    blocks = max(1, (length + limit - 1) // limit) if length else 0
    return {
        "ids": ids,
        "tokens": tokens,
        "decoded_tokens": decoded_tokens,
        "display_tokens": display_tokens,
        "length": length,
        "limit": limit,
        "over": over,
        "usable": usable,
        "blocks": blocks,
        "label_en": spec["label_en"],
        "label_ja": spec["label_ja"],
    }


def _render_stats_html(char_count: int, lang: str, reports: Dict[str, Dict[str, Any]]) -> str:
    if not reports:
        return ""
    cards: List[str] = []
    for key in MODEL_KEY_ORDER:
        report = reports.get(key)
        if report is None:
            continue
        title = (
            f"{report['label_en']} tokens"
            if lang == "en"
            else f"{report['label_ja']}トークン"
        )
        usable = report["usable"]
        limit = report["limit"]
        over = report["over"]
        if lang == "en":
            prompt_tokens = f"{usable} prompt token(s)" if usable else "0 prompt tokens"
            status = "Within limit" if over <= 0 else f"Truncates {over} token(s)"
            sub = f"{prompt_tokens} · limit {limit}"
        else:
            prompt_tokens = f"プロンプト {usable} トークン" if usable else "プロンプト 0 トークン"
            status = "上限内" if over <= 0 else f"超過 {over} トークンは切り捨て"
            sub = f"{prompt_tokens} / 上限 {limit}"
        status_class = "ok" if over <= 0 else "warn"
        cards.append(
            """
            <div class="token-stat-card">
                <div class="token-stat-title">{title}</div>
                <div class="token-stat-value">{value}</div>
                <div class="token-stat-sub">{sub}</div>
                <div class="token-stat-status {status_class}">{status}</div>
            </div>
            """.format(
                title=html.escape(title),
                value=report["length"],
                sub=html.escape(sub),
                status_class=status_class,
                status=html.escape(status),
            )
        )
    char_title = "Characters" if lang == "en" else "文字数"
    char_sub = (
        "Spaces and punctuation included."
        if lang == "en"
        else "スペースや句読点も含まれます。"
    )
    cards.append(
        """
        <div class="token-stat-card characters">
            <div class="token-stat-title">{title}</div>
            <div class="token-stat-value">{value}</div>
            <div class="token-stat-sub">{sub}</div>
            <div class="token-stat-status ok">&nbsp;</div>
        </div>
        """.format(
            title=html.escape(char_title),
            value=char_count,
            sub=html.escape(char_sub),
        )
    )
    return '<div class="token-stats-grid">' + "".join(cards) + "</div>"


def _build_summary_markdown(lang: str, char_count: int, reports: Dict[str, Dict[str, Any]]) -> str:
    if not reports:
        return ""
    lines: List[str] = []
    if lang == "en":
        lines.append(f"**Characters:** {char_count}")
    else:
        lines.append(f"**文字数:** {char_count}")
    for key in MODEL_KEY_ORDER:
        report = reports.get(key)
        if report is None:
            continue
        label = report["label_en"] if lang == "en" else report["label_ja"]
        length = report["length"]
        usable = report["usable"]
        limit = report["limit"]
        over = report["over"]
        if lang == "en":
            usable_part = f"{usable} prompt token(s)" if usable else "0 prompt tokens"
            line = (
                f"- **{label}** · {length} total tokens ({usable_part}) / limit {limit}"
            )
            line += " — within limit" if over <= 0 else f" — truncates {over} token(s)"
        else:
            usable_part = f"プロンプト {usable} トークン" if usable else "プロンプト 0 トークン"
            line = (
                f"- **{label}** · 合計{length}トークン（{usable_part}） / 上限 {limit}"
            )
            line += " — 上限内" if over <= 0 else f" — 超過 {over} トークンは切り捨て"
        lines.append(line)
    return "\n".join(lines)


def _render_empty_preview(message: str) -> str:
    return (
        '<div class="token-preview token-preview-empty">'
        f'<div class="token-preview-placeholder">{html.escape(message)}</div>'
        '</div>'
    )


def _render_token_preview(
    state: Dict[str, Any],
    model_choice: str,
    display_mode: str,
    lang_choice: str,
) -> str:
    lang = _resolve_lang(lang_choice)
    if not state or "reports" not in state:
        message = TOKEN_COUNTER_EMPTY_SUMMARY.get(lang, TOKEN_COUNTER_EMPTY_SUMMARY["en"])
        return _render_empty_preview(message)
    key = MODEL_CHOICE_TO_KEY.get(model_choice, MODEL_KEY_ORDER[0])
    report = state["reports"].get(key)
    if report is None:
        message = TOKEN_COUNTER_EMPTY_SUMMARY.get(lang, TOKEN_COUNTER_EMPTY_SUMMARY["en"])
        return _render_empty_preview(message)
    mode_key = DISPLAY_MODE_TO_KEY.get(display_mode, "text")
    heading = (
        f"{report['label_en']} tokens"
        if lang == "en"
        else f"{report['label_ja']}トークン"
    )
    mode_label = DISPLAY_MODE_LABELS.get(mode_key, DISPLAY_MODE_LABELS["text"])[lang]
    if mode_key == "ids":
        items = [str(x) for x in report["ids"]]
    else:
        items = report["display_tokens"]
    chips: List[str] = []
    for idx, item in enumerate(items):
        color = TOKEN_COLOR_PALETTE[idx % len(TOKEN_COLOR_PALETTE)]
        text_display = _format_token_display(item)
        token_id = report["ids"][idx] if idx < len(report["ids"]) else ""
        if mode_key == "ids":
            tooltip_raw = report["display_tokens"][idx] if idx < len(report["display_tokens"]) else ""
        else:
            tooltip_raw = f"ID {token_id}"
        tooltip = html.escape(tooltip_raw if tooltip_raw else "", quote=True)
        chips.append(
            f'<span class="token-chip" style="background:{color}" title="{tooltip}">{text_display}</span>'
        )
    if not chips:
        message = TOKEN_COUNTER_EMPTY_SUMMARY.get(lang, TOKEN_COUNTER_EMPTY_SUMMARY["en"])
        chips_html = f'<span class="token-chip">{html.escape(message)}</span>'
    else:
        chips_html = "".join(chips)
    return (
        '<div class="token-preview">'
        f'<div class="token-preview-header"><span class="token-preview-heading">{html.escape(heading)}</span>'
        f'<span class="token-preview-mode">{html.escape(mode_label)}</span></div>'
        f'<div class="token-chip-wrap">{chips_html}</div>'
        '</div>'
    )


def update_token_preview(
    state: Dict[str, Any],
    model_choice: str,
    display_mode: str,
    lang_choice: str,
) -> str:
    return _render_token_preview(state, model_choice, display_mode, lang_choice)


def token_count_handle(
    text: str,
    lang_choice: str,
    model_choice: str,
    display_mode: str,
) -> Tuple[str, str, str, Dict[str, Any]]:
    lang = _resolve_lang(lang_choice)
    if CLIPTokenizer is None:
        message = (
            "transformers/CLIPTokenizer unavailable. Install transformers to enable token counting."
            if lang == "en"
            else "transformers/CLIPTokenizerが見つかりません。token countingを使うにはtransformersをインストールしてください。"
        )
        preview = _render_empty_preview(message)
        return "", message, preview, {}

    prompt = text or ""
    if not (prompt.strip()):
        message = TOKEN_COUNTER_EMPTY_SUMMARY.get(lang, TOKEN_COUNTER_EMPTY_SUMMARY["en"])
        preview = _render_empty_preview(message)
        return "", message, preview, {}

    char_count = len(prompt)

    try:
        reports: Dict[str, Dict[str, Any]] = {
            "sd15": _build_token_report("sd15", prompt),
        }
    except Exception as exc:
        message = (
            f"Failed to load SD1.5 tokenizer: {exc}"
            if lang == "en"
            else f"SD1.5トークナイザの読み込みに失敗しました: {exc}"
        )
        preview = _render_empty_preview(message)
        return "", message, preview, {}

    try:
        reports["sdxl_primary"] = _build_token_report("sdxl_primary", prompt)
        reports["sdxl_secondary"] = _build_token_report("sdxl_secondary", prompt)
    except Exception as exc:
        message = (
            f"Failed to load SDXL tokenizers: {exc}"
            if lang == "en"
            else f"SDXLトークナイザの読み込みに失敗しました: {exc}"
        )
        preview = _render_empty_preview(message)
        return "", message, preview, {}

    stats_html = _render_stats_html(char_count, lang, reports)
    summary_md = _build_summary_markdown(lang, char_count, reports)
    state = {"char_count": char_count, "reports": reports}
    preview_html = _render_token_preview(state, model_choice, display_mode, lang_choice)
    return stats_html, summary_md, preview_html, state

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
    )

with gr.Blocks(title='Generate & Describe Images (SD/SDXL, WD-EVA02, BLIP2)', css=TOKEN_COUNTER_CSS) as demo:
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
                scheduler = gr.Dropdown(choices=['euler','ddim','dpmpp2m'], value='euler', label='Sampler (scheduler)')
                seed = gr.Textbox(label='Seed (leave blank for random)', value='')

            with gr.Accordion("Auto description settings", open=False):
                with gr.Row():
                    th_general = gr.Slider(0.0, 1.0, 0.55, 0.01, label="General tag sensitivity (lower = more)")
                    th_character = gr.Slider(0.0, 1.0, 0.75, 0.01, label="Character tag sensitivity (lower = more)")
                with gr.Row():
                    blip_prompt = gr.Textbox(label="Caption prompt (optional)", value="")
                    max_tokens = gr.Slider(8, 120, 40, 1, label="Caption length (max tokens)")

            btn = gr.Button('Generate Image')
            out_img = gr.Image(label='Generated image', type='pil', interactive=False, sources=[])
            # Added: manual description button
            btn_interrogate = gr.Button('Describe Image')
            # Manually-triggered description outputs
            out_wd_sentence = gr.Textbox(label="Tag summary", lines=6, show_copy_button=True)
            out_blip = gr.Textbox(label="Caption", lines=6, show_copy_button=True)
            out_wd_table = gr.Markdown(label="Top tags")

            model_sel_name.change(on_model_change, inputs=[model_sel_name], outputs=[width, height, guidance, steps])

            # 1) Generate image only
            btn.click(
                lambda n, *args: generate(key_by_name.get(n, DEFAULT_KEY), *args),
                inputs=[model_sel_name, prompt, neg, steps, guidance, width, height, scheduler, seed],
                outputs=[out_img],
            )

            # 2) Manually run img2text on the image above when needed
            btn_interrogate.click(
                img2text_handle,
                inputs=[out_img, th_general, th_character, blip_prompt, max_tokens],
                outputs=[out_wd_sentence, out_blip, out_wd_table],
            )

        with gr.Tab("Describe Image"):
            in_img = gr.Image(label="Image", type="pil")
            with gr.Row():
                th_general2 = gr.Slider(0.0, 1.0, 0.55, 0.01, label="General tag sensitivity (lower = more)")
                th_character2 = gr.Slider(0.0, 1.0, 0.75, 0.01, label="Character tag sensitivity (lower = more)")
            with gr.Row():
                blip_prompt2 = gr.Textbox(label="Caption prompt (optional)", value="")
                max_tokens2 = gr.Slider(8, 120, 40, 1, label="Caption length (max tokens)")

            btn2 = gr.Button("Describe Image")
            out_wd_sentence2 = gr.Textbox(label="Tag summary", lines=6, show_copy_button=True)
            out_blip2 = gr.Textbox(label="Caption", lines=6, show_copy_button=True)
            out_wd_table2 = gr.Markdown(label="Top tags")

            btn2.click(
                img2text_handle,
                inputs=[in_img, th_general2, th_character2, blip_prompt2, max_tokens2],
                outputs=[out_wd_sentence2, out_blip2, out_wd_table2]
            )

        # Danbooru tag lookup (JA-friendly)
        with gr.Tab("Danbooru用語（タグ検索）"):
            gr.Markdown("## Danbooru 用語検索\n- 英語/日本語で検索できます。\n- 候補から英語タグを選んで、下のボックスへ追加します。")
            with gr.Row():
                tag_query = gr.Textbox(label="検索（英語/日本語）", placeholder="例: girl / 女の子 / hair", lines=1)
                tag_max = gr.Slider(5, 200, value=50, step=5, label="最大件数")
            with gr.Row():
                tag_table = gr.Dataframe(headers=["英語タグ", "日本語"], interactive=False, row_count=5, col_count=2)
            tag_choices = gr.CheckboxGroup(choices=[], label="候補タグ")
            with gr.Row():
                tag_add = gr.Button("追加")
                tag_clear = gr.Button("クリア")
            tag_prompt_box = gr.Textbox(label="選択済みタグ", show_copy_button=True)

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

        with gr.Tab("Token Counter / トークンカウンター"):
            token_state = gr.State({})
            empty_preview_ja = _render_empty_preview(TOKEN_COUNTER_EMPTY_SUMMARY["ja"])
            with gr.Column(elem_classes=["token-counter-wrapper"]):
                with gr.Row(elem_classes=["token-counter-top-controls"]):
                    with gr.Column(scale=1):
                        token_lang = gr.Radio(
                            choices=list(LANG_CHOICES.keys()),
                            value="日本語",
                            label="Language / 言語",
                            elem_classes=["token-lang-radio"],
                        )
                    with gr.Column(scale=1):
                        token_model = gr.Radio(
                            choices=MODEL_RADIO_CHOICES,
                            value=MODEL_RADIO_CHOICES[0],
                            label="トークン表示",
                            elem_classes=["token-model-radio"],
                        )
                    with gr.Column(scale=1):
                        token_display_mode = gr.Radio(
                            choices=DISPLAY_MODE_RADIO_CHOICES,
                            value=DISPLAY_MODE_RADIO_CHOICES[0],
                            label="表示モード",
                            elem_classes=["token-display-radio"],
                        )
                token_input = gr.Textbox(
                    label="入力テキスト",
                    lines=8,
                    placeholder="プロンプトまたはネガティブプロンプトを入力してください…",
                    show_copy_button=True,
                    elem_classes=["token-counter-textbox"],
                )
                with gr.Row(elem_classes=["token-counter-actions"]):
                    token_button = gr.Button("トークン数を計算", variant="primary")
                    token_clear = gr.Button("クリア")
                    token_example = gr.Button("サンプルを表示")
                token_stats = gr.HTML(value="", elem_classes=["token-counter-stats"])
                token_summary = gr.Markdown(
                    value=TOKEN_COUNTER_EMPTY_SUMMARY["ja"],
                    elem_classes=["token-counter-summary"],
                )
                token_visual = gr.HTML(
                    value=empty_preview_ja,
                    elem_classes=["token-counter-visual"],
                )
                token_info = gr.Markdown(
                    TOKEN_LIMIT_INFO_MD["ja"],
                    elem_classes=["token-counter-info"],
                )

            def _on_token_clear(lang_choice: str):
                lang = _resolve_lang(lang_choice)
                message = TOKEN_COUNTER_EMPTY_SUMMARY.get(lang, TOKEN_COUNTER_EMPTY_SUMMARY["en"])
                preview = _render_empty_preview(message)
                return "", "", message, preview, {}

            def _on_token_example(lang_choice: str, model_choice: str, display_mode_choice: str):
                lang = _resolve_lang(lang_choice)
                sample_text = TOKEN_COUNTER_EXAMPLES.get(lang, TOKEN_COUNTER_EXAMPLES["en"])
                stats, summary, preview, state = token_count_handle(
                    sample_text,
                    lang_choice,
                    model_choice,
                    display_mode_choice,
                )
                return sample_text, stats, summary, preview, state

            def _on_token_lang_change(
                lang_choice: str,
                current_text: str,
                model_choice: str,
                display_mode_choice: str,
            ):
                lang = _resolve_lang(lang_choice)
                info = TOKEN_LIMIT_INFO_MD.get(lang, TOKEN_LIMIT_INFO_MD["en"])
                textbox_update = gr.update(
                    label="Input text" if lang == "en" else "入力テキスト",
                    placeholder="Type your prompt or negative prompt here..." if lang == "en" else "プロンプトまたはネガティブプロンプトを入力してください…",
                )
                button_update = gr.update(value="Count tokens" if lang == "en" else "トークン数を計算")
                clear_update = gr.update(value="Clear" if lang == "en" else "クリア")
                example_update = gr.update(value="Show example" if lang == "en" else "サンプルを表示")
                model_update = gr.update(label="Tokenizer view" if lang == "en" else "トークン表示")
                display_update = gr.update(label="Display mode" if lang == "en" else "表示モード")
                if (current_text or "").strip():
                    stats, summary, preview, state = token_count_handle(
                        current_text,
                        lang_choice,
                        model_choice,
                        display_mode_choice,
                    )
                else:
                    message = TOKEN_COUNTER_EMPTY_SUMMARY.get(lang, TOKEN_COUNTER_EMPTY_SUMMARY["en"])
                    stats = ""
                    summary = message
                    preview = _render_empty_preview(message)
                    state = {}
                return (
                    gr.update(value=info),
                    textbox_update,
                    button_update,
                    clear_update,
                    example_update,
                    model_update,
                    display_update,
                    stats,
                    summary,
                    preview,
                    state,
                )

            token_button.click(
                token_count_handle,
                inputs=[token_input, token_lang, token_model, token_display_mode],
                outputs=[token_stats, token_summary, token_visual, token_state],
            )
            token_input.submit(
                token_count_handle,
                inputs=[token_input, token_lang, token_model, token_display_mode],
                outputs=[token_stats, token_summary, token_visual, token_state],
            )
            token_model.change(
                update_token_preview,
                inputs=[token_state, token_model, token_display_mode, token_lang],
                outputs=[token_visual],
            )
            token_display_mode.change(
                update_token_preview,
                inputs=[token_state, token_model, token_display_mode, token_lang],
                outputs=[token_visual],
            )
            token_clear.click(
                _on_token_clear,
                inputs=[token_lang],
                outputs=[token_input, token_stats, token_summary, token_visual, token_state],
            )
            token_example.click(
                _on_token_example,
                inputs=[token_lang, token_model, token_display_mode],
                outputs=[token_input, token_stats, token_summary, token_visual, token_state],
            )
            token_lang.change(
                _on_token_lang_change,
                inputs=[token_lang, token_input, token_model, token_display_mode],
                outputs=[
                    token_info,
                    token_input,
                    token_button,
                    token_clear,
                    token_example,
                    token_model,
                    token_display_mode,
                    token_stats,
                    token_summary,
                    token_visual,
                    token_state,
                ],
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

    demo.queue(max_size=32).launch(server_name='0.0.0.0', server_port=7860, share=False)
