# app_multi_ext.py
# -*- coding: utf-8 -*-
import os, json, csv, torch, gradio as gr
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

    def _format_line(label_en: str, label_ja: str, length: int, limit: int) -> str:
        over = max(0, length - limit)
        blocks = max(1, (length + limit - 1) // limit)
        effective = min(length, limit)
        usable = max(effective - 2, 0)
        if over > 0:
            status_en = f"truncates {over} token(s)"
            status_ja = f"超過分{over}トークンが切り捨てられます"
        else:
            status_en = "fits within the limit"
            status_ja = "制限内に収まります"
        if lang == "en":
            return f"- {label_en}: **{length}** tokens (usable {usable}/{limit}, blocks {blocks}) – {status_en}."
        return f"- {label_ja}: トークン数{length}（有効{usable}/{limit}・ブロック{blocks}） – {status_ja}。"

    header = "### Token counts" if lang == "en" else "### トークン数"
    lines = [header]
    lines.append(_format_line("SD1.5 CLIP", "SD1.5のCLIP", len(sd15_ids), SD15_TOKEN_LIMIT))
    lines.append(_format_line("SDXL text encoder 1", "SDXLテキストエンコーダ1", len(sdxl_ids_primary), SDXL_TOKEN_LIMIT))
    lines.append(_format_line("SDXL text encoder 2", "SDXLテキストエンコーダ2", len(sdxl_ids_secondary), SDXL_TOKEN_LIMIT))

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
    )

with gr.Blocks(title='Generate & Describe Images (SD/SDXL, WD-EVA02, BLIP2)') as demo:
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

    demo.queue(max_size=32).launch(server_name='0.0.0.0', server_port=7860, share=False)
