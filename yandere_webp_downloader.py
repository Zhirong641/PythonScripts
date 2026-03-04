# -*- coding: utf-8 -*-
"""
yande.re 图片下载器（支持 WebP 转码、断点续抓、JSON sidecar、manifest）

示例：
1) 按标签下载 300 张，保存 sidecar + manifest
   python yandere_webp_downloader.py \
     --tags "shiratama rating:s" \
     --max-posts 300 \
     --out ./yandere_shiratama \
     --save-json --manifest --resume

2) 只下载站内本身是 webp 的图片
   python yandere_webp_downloader.py \
     --tags "original" \
     --only-webp \
     --out ./yandere_webp_only

3) 下载后统一转 webp
   python yandere_webp_downloader.py \
     --tags "1girl" \
     --convert-to-webp --quality 88 \
     --out ./yandere_1girl_webp
"""

import argparse
import csv
import hashlib
import json
import math
import os
import re
import sys
import threading
import time
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Set, Tuple

import requests
from requests.adapters import HTTPAdapter

try:
    from requests.adapters import Retry  # requests>=2.29
except ImportError:
    from urllib3.util.retry import Retry  # type: ignore

try:
    from PIL import Image

    PIL_AVAILABLE = True
except Exception:
    Image = None  # type: ignore
    PIL_AVAILABLE = False

YANDE_API = "https://yande.re/post.json"
YANDE_TAG_API = "https://yande.re/tag.json"
YANDE_POST_URL = "https://yande.re/post/show/{post_id}"
WD_REPO_DEFAULT = "SmilingWolf/wd-eva02-large-tagger-v3"
LOCK = threading.Lock()
MAX_IMAGE_PIXELS = 10_000_000
SUPPORTED_IMAGE_EXTENSIONS = {
    "jpg",
    "jpeg",
    "png",
    "webp",
    "bmp",
    "gif",
    "tif",
    "tiff",
    "jfif",
}
TAG_TYPE_TO_BUCKET = {
    0: "general",
    1: "artist",
    2: "general",
    3: "copyright",
    4: "character",
    # yande.re 的 type 编号与 danbooru 并非完全一致；未知类型先回落到 general，
    # 避免把普通语义标签误写进 meta。
    5: "general",
    6: "general",
}

WD_TAG_GENERAL = 0
WD_TAG_CHARACTER = 4
WD_TAG_RATING = 9
MANIFEST_FIELDNAMES = [
    "id",
    "post_url",
    "md5",
    "rating",
    "score",
    "source",
    "width",
    "height",
    "file_size",
    "created_at",
    "saved_path",
    "saved_ext",
    "tags_general",
    "tags_character",
    "tags_copyright",
    "tags_artist",
    "tags_meta",
    "tags_all",
    "tags",
    "artist_input",
    "artist_canonical",
]


def build_manifest_key(post_id: Optional[int], artist_canonical: Optional[str]) -> Optional[str]:
    if post_id is None:
        return None
    return f"{post_id}::{artist_canonical or ''}"


def is_supported_image_ext(ext: Optional[str]) -> bool:
    return bool(ext and ext.lower() in SUPPORTED_IMAGE_EXTENSIONS)


def _calc_resize_dimensions(width: int, height: int, max_pixels: int) -> Optional[Tuple[int, int]]:
    if max_pixels <= 0 or width <= 0 or height <= 0:
        return None
    total = width * height
    if total <= max_pixels:
        return None
    scale = math.sqrt(max_pixels / float(total))
    new_w = max(1, int(width * scale))
    new_h = max(1, int(height * scale))
    while new_w * new_h > max_pixels and new_w > 1 and new_h > 1:
        if new_w >= new_h:
            new_w -= 1
        else:
            new_h -= 1
    return new_w, new_h


def _resize_image_object(im, max_pixels: Optional[int]):
    if not max_pixels or max_pixels <= 0:
        return im
    dims = _calc_resize_dimensions(*im.size, max_pixels=max_pixels)
    if not dims:
        return im
    return im.resize(dims, Image.LANCZOS)


def limit_image_pixels_inplace(path: str, max_pixels: Optional[int] = MAX_IMAGE_PIXELS) -> bool:
    if not PIL_AVAILABLE or not max_pixels or max_pixels <= 0:
        return False
    try:
        with Image.open(path) as im:
            new_im = _resize_image_object(im, max_pixels)
            if new_im is im:
                return False
            new_im.save(path)
            new_im.close()
        return True
    except Exception:
        return False


def _build_retry() -> Retry:
    common_kwargs = dict(
        total=5,
        backoff_factor=0.5,
        status_forcelist=[429, 500, 502, 503, 504],
        respect_retry_after_header=True,
    )
    methods = frozenset(["GET"])
    if "allowed_methods" in Retry.__init__.__code__.co_varnames:  # type: ignore[attr-defined]
        return Retry(allowed_methods=methods, **common_kwargs)
    return Retry(method_whitelist=methods, **common_kwargs)  # type: ignore[arg-type]


def build_session() -> requests.Session:
    s = requests.Session()
    retries = _build_retry()
    s.mount("https://", HTTPAdapter(max_retries=retries))
    s.headers.update({"User-Agent": "YandereImageDownloader/1.0"})
    return s


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_state(state_path: Optional[str], state: Dict):
    if not state_path:
        return
    with LOCK:
        with open(state_path, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)


def load_state(state_path: Optional[str]) -> Dict:
    if not state_path or not os.path.exists(state_path):
        return {}
    try:
        with open(state_path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    return {}


def split_tags(s: Optional[str]) -> List[str]:
    if not s:
        return []
    return [t for t in str(s).strip().split() if t]


def normalize_created_at(created_at) -> str:
    """标准化 created_at，统一输出 ISO8601（UTC）。"""
    if created_at is None:
        return ""
    s = str(created_at).strip()
    if not s:
        return ""

    # 纯数字：按 Unix timestamp 处理（10 位秒，13 位毫秒）
    if re.fullmatch(r"-?\d+", s):
        try:
            n = int(s)
            if abs(n) > 10**12:
                n = int(n / 1000)
            dt = datetime.fromtimestamp(n, tz=timezone.utc)
            return dt.isoformat()
        except Exception:
            return ""

    # ISO8601
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).isoformat()
    except Exception:
        return ""


def get_post_tags_list(post: Dict) -> List[str]:
    raw = post.get("tags")
    if isinstance(raw, str):
        return split_tags(raw)
    if isinstance(raw, list):
        out: List[str] = []
        for x in raw:
            if isinstance(x, str) and x.strip():
                out.append(x.strip())
            elif isinstance(x, dict):
                name = str(x.get("name", x.get("tag", ""))).strip()
                if name:
                    out.append(name)
        return out
    if isinstance(raw, dict):
        names = [str(k).strip() for k in raw.keys() if str(k).strip()]
        return names
    return []


def _safe_int(v) -> Optional[int]:
    try:
        return int(v)
    except Exception:
        return None


def _parse_tag_type_map(raw) -> Dict[str, int]:
    """
    兼容多种结构：
    - {"tag_name": 0}
    - {"tag_name": {"type": 0}}
    - [{"name":"tag_name","type":0}, ...]
    """
    out: Dict[str, int] = {}
    if isinstance(raw, dict):
        for k, v in raw.items():
            name = str(k).strip()
            if not name:
                continue
            if isinstance(v, dict):
                t = _safe_int(v.get("type", v.get("type_id", v.get("category"))))
            else:
                t = _safe_int(v)
            if t is not None:
                out[name] = t
    elif isinstance(raw, list):
        for item in raw:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name", item.get("tag", ""))).strip()
            if not name:
                continue
            t = _safe_int(item.get("type", item.get("type_id", item.get("category"))))
            if t is not None:
                out[name] = t
    return out


def extract_tags_by_category(post: Dict) -> Dict[str, List[str]]:
    buckets = {
        "general": [],
        "artist": [],
        "copyright": [],
        "character": [],
        "meta": [],
    }
    all_tags = get_post_tags_list(post)

    # 如果帖子内已带 tag_string_xxx，优先使用。
    if any(post.get(k) for k in ("tag_string_general", "tag_string_artist", "tag_string_copyright", "tag_string_character", "tag_string_meta")):
        buckets["general"] = split_tags(post.get("tag_string_general"))
        buckets["artist"] = split_tags(post.get("tag_string_artist"))
        buckets["copyright"] = split_tags(post.get("tag_string_copyright"))
        buckets["character"] = split_tags(post.get("tag_string_character"))
        buckets["meta"] = split_tags(post.get("tag_string_meta"))
        return buckets

    type_map = _parse_tag_type_map(post.get("tags_map", {}))
    # 兼容 post["tags"] 本身是 map/list 的情况
    if not type_map and not isinstance(post.get("tags"), str):
        type_map = _parse_tag_type_map(post.get("tags"))

    if type_map:
        for tag in all_tags:
            t = type_map.get(tag)
            bucket = TAG_TYPE_TO_BUCKET.get(t, "general") if t is not None else "general"
            if bucket in buckets:
                buckets[bucket].append(tag)
            else:
                buckets["general"].append(tag)
    else:
        buckets["general"] = all_tags

    # 兜底修正：从角色标签 `name_(copyright)` 推导版权标签。
    # 例：bremerton_(azur_lane) -> azur_lane
    inferred_copyrights: Set[str] = set()
    for ctag in buckets["character"]:
        m = re.match(r"^.+_\(([^()]+)\)$", ctag)
        if not m:
            continue
        inferred = m.group(1).strip()
        if inferred:
            inferred_copyrights.add(inferred)

    if inferred_copyrights:
        cp_set = set(buckets["copyright"])
        gen_set = set(buckets["general"])
        for cp in inferred_copyrights:
            if cp not in cp_set:
                buckets["copyright"].append(cp)
                cp_set.add(cp)
            if cp in gen_set:
                buckets["general"] = [t for t in buckets["general"] if t != cp]
                gen_set.discard(cp)

    return buckets


def _dedupe_preserve(items: List[str]) -> List[str]:
    seen: Set[str] = set()
    out: List[str] = []
    for x in items:
        s = str(x).strip()
        if not s or s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


def compose_all_tags(tags_by_category: Dict[str, List[str]]) -> List[str]:
    merged: List[str] = []
    for key in ("general", "character", "copyright", "artist", "meta"):
        merged.extend(tags_by_category.get(key, []))
    return _dedupe_preserve(merged)


def _normalize_tag_type_value(v) -> Optional[int]:
    i = _safe_int(v)
    if i is not None:
        return i
    if isinstance(v, str):
        m = {
            "general": 0,
            "artist": 1,
            "copyright": 3,
            "character": 4,
            "meta": 5,
        }
        return m.get(v.strip().lower())
    return None


def fetch_tag_type(
    s: requests.Session,
    tag_name: str,
    cache: Dict[str, Optional[int]],
    debug: bool = False,
) -> Optional[int]:
    if tag_name in cache:
        return cache[tag_name]
    t: Optional[int] = None
    # moebooru 常见写法：tag.json?name=<tag>
    try:
        r = s.get(YANDE_TAG_API, params={"name": tag_name, "limit": 1}, timeout=20)
        if debug:
            req = r.request
            print(f"[DEBUG] {req.method} {req.url} -> {r.status_code}")
        r.raise_for_status()
        data = r.json()
        if isinstance(data, list) and data:
            # 只接受精确同名匹配，避免 fuzzy 命中导致分类串列。
            exact_row = None
            for item in data:
                if not isinstance(item, dict):
                    continue
                name = str(item.get("name", "")).strip()
                if name == tag_name:
                    exact_row = item
                    break
            if exact_row is None and isinstance(data[0], dict):
                first_name = str(data[0].get("name", "")).strip()
                if first_name == tag_name:
                    exact_row = data[0]
            if exact_row is not None:
                t = _normalize_tag_type_value(
                    exact_row.get("type", exact_row.get("type_id", exact_row.get("category")))
                )
        elif isinstance(data, dict):
            name = str(data.get("name", "")).strip()
            if (not name) or name == tag_name:
                t = _normalize_tag_type_value(data.get("type", data.get("type_id", data.get("category"))))
    except Exception:
        t = None
    cache[tag_name] = t
    return t


def enrich_post_tags_map_with_lookup(
    s: requests.Session,
    post: Dict,
    tag_type_cache: Dict[str, Optional[int]],
    debug: bool = False,
):
    tags = get_post_tags_list(post)
    if not tags:
        return

    existing_map = _parse_tag_type_map(post.get("tags_map", {}))
    if not existing_map and not isinstance(post.get("tags"), str):
        existing_map = _parse_tag_type_map(post.get("tags"))

    changed = False
    for tname in tags:
        if tname in existing_map and existing_map[tname] is not None:
            continue
        t = fetch_tag_type(s, tname, cache=tag_type_cache, debug=debug)
        if t is not None:
            existing_map[tname] = t
            changed = True

    if changed or existing_map:
        post["tags_map"] = existing_map


def pick_best_url(post: Dict) -> Optional[str]:
    for key in ("file_url", "jpeg_url", "sample_url"):
        v = post.get(key)
        if isinstance(v, str) and v.strip():
            if v.startswith("//"):
                return "https:" + v
            return v
    return None


def ext_from_url(url: str) -> str:
    base = url.split("?")[0].split("#")[0]
    if "." in base:
        return base.rsplit(".", 1)[-1].lower()
    return "bin"


def build_filename(post: Dict, url: str, force_ext: Optional[str] = None) -> str:
    pid = post.get("id")
    md5v = post.get("md5") or hashlib.md5(url.encode("utf-8")).hexdigest()
    ext = (force_ext or post.get("file_ext") or ext_from_url(url)).lower()
    return f"{pid}_{md5v}.{ext}"


def _safe_remove(path: str):
    try:
        if os.path.exists(path):
            os.remove(path)
    except Exception:
        pass


def _fmt_err(e: Exception) -> str:
    msg = str(e).strip()
    if msg:
        return f"{e.__class__.__name__}: {msg}"
    return e.__class__.__name__


def download_one(
    s: requests.Session,
    url: str,
    out_path: str,
    timeout: int = 45,
    retries: int = 2,
    retry_wait: float = 0.8,
) -> Tuple[bool, Optional[str]]:
    tmp_path = out_path + ".part"
    _safe_remove(tmp_path)
    last_err: Optional[str] = None
    attempts = max(0, retries) + 1
    for i in range(attempts):
        try:
            with s.get(url, stream=True, timeout=timeout) as r:
                r.raise_for_status()
                with open(tmp_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1 << 16):
                        if chunk:
                            f.write(chunk)
            os.replace(tmp_path, out_path)
            return True, None
        except Exception as e:
            last_err = _fmt_err(e)
            _safe_remove(tmp_path)
            if i < attempts - 1:
                time.sleep(max(0.0, retry_wait) * (i + 1))
    return False, last_err


def _image_has_alpha(im) -> bool:
    bands = getattr(im, "getbands", lambda: tuple())()
    if bands and "A" in bands:
        return True
    if getattr(im, "mode", "") == "P" and "transparency" in getattr(im, "info", {}):
        return True
    return False


def _prepare_webp_image(im):
    if _image_has_alpha(im):
        return im.convert("RGBA")
    if im.mode != "RGB":
        return im.convert("RGB")
    return im


def convert_to_webp(src_path: str, dst_path: str, quality: int = 85, max_pixels: Optional[int] = MAX_IMAGE_PIXELS) -> bool:
    if not PIL_AVAILABLE:
        return False
    try:
        with Image.open(src_path) as im:
            im = _resize_image_object(im, max_pixels)
            im = _prepare_webp_image(im)
            im.save(dst_path, "WEBP", quality=quality, method=6)
        return True
    except Exception:
        return False


class WdEva02GeneralTagger:
    """使用 SmilingWolf/wd-eva02-large-tagger-v3 推理 general tags。"""

    def __init__(self, repo_id: str, use_gpu: bool, general_threshold: float):
        self.repo_id = repo_id
        self.use_gpu = use_gpu
        self.general_threshold = general_threshold
        self._lock = threading.Lock()
        self._ready = False
        self._session = None
        self._tag_names: List[str] = []
        self._tag_cats: List[int] = []
        self._nhwc = True
        self._input_name = ""

    def _ensure_ready(self):
        if self._ready:
            return
        try:
            import numpy as np  # type: ignore
            import onnxruntime as ort  # type: ignore
            from huggingface_hub import hf_hub_download  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "启用 WD general 替换需要安装依赖：onnxruntime, huggingface_hub, numpy"
            ) from e

        model_path = hf_hub_download(repo_id=self.repo_id, filename="model.onnx")
        tags_path = hf_hub_download(repo_id=self.repo_id, filename="selected_tags.csv")
        names, cats = self._load_wd_tags(tags_path)
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if self.use_gpu else ["CPUExecutionProvider"]
        session = ort.InferenceSession(model_path, providers=providers)
        self._session = session
        self._tag_names = names
        self._tag_cats = cats
        self._input_name = session.get_inputs()[0].name
        self._nhwc = self._is_nhwc_input(session)
        self._np = np
        self._ready = True

    @staticmethod
    def _load_wd_tags(tags_csv_path: str) -> Tuple[List[str], List[int]]:
        names: List[str] = []
        cats: List[int] = []
        with open(tags_csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                name = str(row.get("name", "")).strip()
                if not name:
                    continue
                names.append(name)
                cat_raw = row.get("category") or row.get("Category") or "0"
                try:
                    cats.append(int(cat_raw))
                except Exception:
                    low = str(cat_raw).strip().lower()
                    if low.startswith("char"):
                        cats.append(WD_TAG_CHARACTER)
                    elif low.startswith("rating"):
                        cats.append(WD_TAG_RATING)
                    else:
                        cats.append(WD_TAG_GENERAL)
        return names, cats

    @staticmethod
    def _is_nhwc_input(session) -> bool:
        shape = session.get_inputs()[0].shape
        if len(shape) != 4:
            return True
        try:
            if shape[3] == 3:
                return True
            if shape[1] == 3:
                return False
        except Exception:
            pass
        return True

    @staticmethod
    def _pad_to_square_white(img_rgb):
        w, h = img_rgb.size
        if w == h:
            return img_rgb
        m = max(w, h)
        canvas = Image.new("RGB", (m, m), (255, 255, 255))
        canvas.paste(img_rgb, ((m - w) // 2, (m - h) // 2))
        return canvas

    def _preprocess(self, image_path: str):
        with Image.open(image_path) as im:
            if im.mode != "RGBA":
                im = im.convert("RGBA")
            canvas = Image.new("RGBA", im.size, (255, 255, 255, 255))
            canvas.alpha_composite(im)
            im = canvas.convert("RGB")
            im = self._pad_to_square_white(im)
            if im.size != (448, 448):
                im = im.resize((448, 448), Image.BICUBIC)
            arr = self._np.asarray(im, dtype=self._np.float32)
            if self._nhwc:
                arr = arr[:, :, ::-1]  # BGR
                arr = self._np.expand_dims(arr, axis=0).astype(self._np.float32)
            else:
                arr = arr / 255.0
                arr = arr * 2.0 - 1.0
                arr = self._np.transpose(arr, (2, 0, 1))
                arr = self._np.expand_dims(arr, axis=0).astype(self._np.float32)
            return arr

    def infer_general_tags(self, image_path: str, limit: int = 128) -> List[str]:
        with self._lock:
            self._ensure_ready()
            x = self._preprocess(image_path)
            probs = self._session.run(None, {self._input_name: x})[0][0]
            pairs: List[Tuple[str, float]] = []
            for idx, name in enumerate(self._tag_names):
                cat = self._tag_cats[idx]
                if cat != WD_TAG_GENERAL:
                    continue
                p = float(probs[idx])
                if p >= self.general_threshold:
                    pairs.append((name, p))
            pairs.sort(key=lambda t: t[1], reverse=True)
            return [n for n, _ in pairs[:limit]]


def should_keep_post_as_webp(post: Dict, only_webp: bool) -> bool:
    if not only_webp:
        return True
    ext = str(post.get("file_ext", "")).lower()
    if ext == "webp":
        return True
    url = pick_best_url(post)
    return bool(url and ext_from_url(url) == "webp")


def build_post_url(post_id: int) -> str:
    return YANDE_POST_URL.format(post_id=post_id)


def write_json_sidecar(
    post: Dict,
    image_path: str,
    final_ext: str,
    artist_input: Optional[str] = None,
    artist_canonical: Optional[str] = None,
    general_tags_override: Optional[List[str]] = None,
):
    js_path = os.path.splitext(image_path)[0] + ".json"
    tags_by_category = extract_tags_by_category(post)
    if general_tags_override is not None:
        tags_by_category["general"] = _dedupe_preserve(general_tags_override)
    tags = compose_all_tags(tags_by_category)
    data = {
        "id": post.get("id"),
        "post_url": build_post_url(post.get("id")),
        "md5": post.get("md5"),
        "rating": post.get("rating"),
        "score": post.get("score"),
        "source": post.get("source"),
        "file": {
            "original_ext": post.get("file_ext") or ext_from_url(pick_best_url(post) or ""),
            "saved_ext": final_ext,
            "width": post.get("width"),
            "height": post.get("height"),
            "file_size": post.get("file_size"),
        },
        "created_at": normalize_created_at(post.get("created_at")),
        "tags": tags,
        "tags_string": " ".join(tags),
        "tags_by_category": tags_by_category,
    }
    if artist_input or artist_canonical:
        data["artist_context"] = {
            "artist_input": artist_input or "",
            "artist_canonical": artist_canonical or "",
        }
    with open(js_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def append_manifest_row(
    manifest_path: str,
    post: Dict,
    saved_path: str,
    final_ext: str,
    artist_input: Optional[str] = None,
    artist_canonical: Optional[str] = None,
    existing_ids: Optional[Set[str]] = None,
    general_tags_override: Optional[List[str]] = None,
):
    pid = int(post.get("id"))
    key = build_manifest_key(pid, artist_canonical)
    with LOCK:
        if existing_ids is not None and key and key in existing_ids:
            return

        exists = os.path.exists(manifest_path)
        tags_by_category = extract_tags_by_category(post)
        if general_tags_override is not None:
            tags_by_category["general"] = _dedupe_preserve(general_tags_override)
        tags_all = " ".join(compose_all_tags(tags_by_category))
        row = {
            "id": pid,
            "post_url": build_post_url(pid),
            "md5": post.get("md5"),
            "rating": post.get("rating"),
            "score": post.get("score"),
            "source": post.get("source"),
            "width": post.get("width"),
            "height": post.get("height"),
            "file_size": post.get("file_size"),
            "created_at": normalize_created_at(post.get("created_at")),
            "saved_path": os.path.abspath(saved_path),
            "saved_ext": final_ext,
            "tags_general": " ".join(tags_by_category["general"]),
            "tags_character": " ".join(tags_by_category["character"]),
            "tags_copyright": " ".join(tags_by_category["copyright"]),
            "tags_artist": " ".join(tags_by_category["artist"]),
            "tags_meta": " ".join(tags_by_category["meta"]),
            "tags_all": tags_all,
            "tags": tags_all,
            "artist_input": artist_input or "",
            "artist_canonical": artist_canonical or "",
        }
        with open(manifest_path, "a", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=MANIFEST_FIELDNAMES)
            if not exists:
                w.writeheader()
            w.writerow(row)
        if existing_ids is not None and key:
            existing_ids.add(key)


def load_manifest_ids(manifest_path: Optional[str]) -> Set[str]:
    ids: Set[str] = set()
    if not manifest_path or not os.path.exists(manifest_path):
        return ids
    try:
        with open(manifest_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if not row:
                    continue
                try:
                    pid = int(row.get("id", ""))
                    key = build_manifest_key(pid, row.get("artist_canonical") or None)
                    if key:
                        ids.add(key)
                except Exception:
                    continue
    except Exception as e:
        print(f"[WARN] 读取 manifest 失败：{e}", file=sys.stderr)
    return ids


def ensure_manifest_schema(manifest_path: str):
    """若 manifest 列与当前定义不一致，自动重写为标准表头。"""
    if not os.path.exists(manifest_path):
        return
    try:
        with open(manifest_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            old_fields = list(reader.fieldnames or [])
            if not old_fields:
                return
            missing = [c for c in MANIFEST_FIELDNAMES if c not in old_fields]
            extra = [c for c in old_fields if c not in MANIFEST_FIELDNAMES]
            rows = list(reader)
            needs_created_at_normalize = False
            for row in rows[:200]:
                raw = (row.get("created_at") or "").strip()
                if raw and re.fullmatch(r"-?\d+", raw):
                    needs_created_at_normalize = True
                    break
            if not missing and not extra and not needs_created_at_normalize:
                return
    except Exception as e:
        print(f"[WARN] 读取旧 manifest 结构失败：{e}", file=sys.stderr)
        return

    try:
        tmp_path = manifest_path + ".schema_tmp"
        with open(tmp_path, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=MANIFEST_FIELDNAMES)
            w.writeheader()
            for row in rows:
                old_tags = row.get("tags", "") or ""
                out = {k: row.get(k, "") for k in MANIFEST_FIELDNAMES}
                out["created_at"] = normalize_created_at(out.get("created_at"))
                if not out.get("tags_all"):
                    out["tags_all"] = old_tags
                if not out.get("tags_general"):
                    out["tags_general"] = old_tags
                if not out.get("tags"):
                    out["tags"] = old_tags
                w.writerow(out)
        os.replace(tmp_path, manifest_path)
        print(
            f"[INFO] manifest 已升级表头，新增列：{missing}，移除列：{extra}，"
            f"created_at 标准化：{needs_created_at_normalize}"
        )
    except Exception as e:
        print(f"[WARN] 升级 manifest 表头失败：{e}", file=sys.stderr)


def backfill_manifest_tags(
    manifest_path: str,
    s_meta: requests.Session,
    tag_type_cache: Dict[str, Optional[int]],
    debug: bool = False,
):
    if not os.path.exists(manifest_path):
        print(f"[WARN] manifest 不存在，跳过回填：{manifest_path}", file=sys.stderr)
        return
    try:
        with open(manifest_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            fields = list(reader.fieldnames or [])
    except Exception as e:
        print(f"[WARN] 读取 manifest 失败，无法回填：{e}", file=sys.stderr)
        return

    if not fields:
        return

    changed = 0
    for row in rows:
        row["created_at"] = normalize_created_at(row.get("created_at"))
        post = {
            "tags": row.get("tags_all") or row.get("tags") or "",
            "tags_map": {},
        }
        enrich_post_tags_map_with_lookup(s_meta, post, tag_type_cache=tag_type_cache, debug=debug)
        tags_by_category = extract_tags_by_category(post)
        old = (
            row.get("tags_general", ""),
            row.get("tags_character", ""),
            row.get("tags_copyright", ""),
            row.get("tags_artist", ""),
            row.get("tags_meta", ""),
        )
        row["tags_general"] = " ".join(tags_by_category["general"])
        row["tags_character"] = " ".join(tags_by_category["character"])
        row["tags_copyright"] = " ".join(tags_by_category["copyright"])
        row["tags_artist"] = " ".join(tags_by_category["artist"])
        row["tags_meta"] = " ".join(tags_by_category["meta"])
        new = (
            row["tags_general"],
            row["tags_character"],
            row["tags_copyright"],
            row["tags_artist"],
            row["tags_meta"],
        )
        if new != old:
            changed += 1

    tmp_path = manifest_path + ".backfill_tmp"
    try:
        with open(tmp_path, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for row in rows:
                w.writerow(row)
        os.replace(tmp_path, manifest_path)
        print(f"[INFO] manifest 标签分类回填完成：更新 {changed}/{len(rows)} 行")
    except Exception as e:
        print(f"[WARN] 写回 manifest 失败：{e}", file=sys.stderr)


def api_fetch_posts(
    s: requests.Session,
    tags: str,
    page: int,
    limit: int,
    debug: bool = False,
) -> List[Dict]:
    params_v2 = {
        "tags": tags,
        "page": page,
        "limit": max(1, min(limit, 100)),
        "api_version": 2,
        "include_tags": 1,
    }
    params_v1 = {
        "tags": tags,
        "page": page,
        "limit": max(1, min(limit, 100)),
    }

    def _do(params: Dict) -> object:
        r = s.get(YANDE_API, params=params, timeout=30)
        if debug:
            req = r.request
            print(f"[DEBUG] {req.method} {req.url} -> {r.status_code}")
        r.raise_for_status()
        return r.json()

    def _normalize_posts(data: object) -> List[Dict]:
        if isinstance(data, list):
            return [p for p in data if isinstance(p, dict)]

        if not isinstance(data, dict):
            return []

        posts = data.get("posts")
        if not isinstance(posts, list):
            return []
        posts = [p for p in posts if isinstance(p, dict)]
        global_tag_map = _parse_tag_type_map(data.get("tags"))
        if global_tag_map:
            for p in posts:
                # 注入到 post，供 manifest/sidecar 分类使用
                p["tags_map"] = global_tag_map
        return posts

    try:
        data = _do(params_v2)
        posts = _normalize_posts(data)
        if posts:
            return posts
    except requests.HTTPError as e:
        if debug:
            print(f"[DEBUG] v2 接口失败，将回退 v1：{e}", file=sys.stderr)

    try:
        data = _do(params_v1)
        return _normalize_posts(data)
    except requests.HTTPError:
        raise


def load_alias_map(path: Optional[str]) -> Dict[str, str]:
    """支持 CSV/TSV（两列）、JSON（dict 或 [a,b] 列表）、纯文本 a->b。"""
    if not path:
        return {}

    def add_pair(dst: Dict[str, str], a: str, b: str):
        a = a.strip()
        b = b.strip()
        if a:
            dst[a] = b

    alias: Dict[str, str] = {}
    ext = os.path.splitext(path)[-1].lower()
    try:
        if ext in (".csv", ".tsv"):
            delim = "\t" if ext == ".tsv" else ","
            with open(path, "r", encoding="utf-8") as f:
                reader = csv.reader(f, delimiter=delim)
                for row in reader:
                    if not row or len(row) < 2:
                        continue
                    add_pair(alias, str(row[0]), str(row[1]))
        elif ext == ".json":
            with open(path, "r", encoding="utf-8") as f:
                obj = json.load(f)
            if isinstance(obj, dict):
                for k, v in obj.items():
                    add_pair(alias, str(k), str(v))
            elif isinstance(obj, list):
                for item in obj:
                    if isinstance(item, (list, tuple)) and len(item) >= 2:
                        add_pair(alias, str(item[0]), str(item[1]))
        else:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    s = line.strip()
                    if not s or s.startswith("#") or "->" not in s:
                        continue
                    a, b = s.split("->", 1)
                    add_pair(alias, a, b)
    except Exception as e:
        print(f"[WARN] 读取 alias-map 失败：{e}", file=sys.stderr)
    return alias


def load_artists_list(path: str) -> List[str]:
    """
    支持 txt/csv/tsv/json。
    - txt: 每行一个作者，支持 # 注释
    - csv/tsv: 每行读取作者；优先列名 artist/canonical/name/tag，否则取第一列
    - json: list[str] 或 {name: canonical}（取 key）
    """
    artists: List[str] = []
    seen: Set[str] = set()
    ext = os.path.splitext(path)[-1].lower()

    def add_artist(v: str):
        s = v.strip()
        if s and s not in seen:
            seen.add(s)
            artists.append(s)

    try:
        if ext in (".csv", ".tsv"):
            delim = "\t" if ext == ".tsv" else ","
            with open(path, "r", encoding="utf-8", newline="") as f:
                rows = list(csv.reader(f, delimiter=delim))

            if not rows:
                return artists

            header_keys = {"artist", "canonical", "name", "tag"}
            first_row = [str(x).strip() for x in rows[0] if str(x).strip()]
            normalized = {x.lower() for x in first_row}
            header_hit = bool(normalized & header_keys)

            # 明确命中常见表头时，用 DictReader；否则按“每行第一列是作者”处理。
            if header_hit and len(rows) > 1:
                with open(path, "r", encoding="utf-8", newline="") as f:
                    reader = csv.DictReader(f, delimiter=delim)
                    for row in reader:
                        if not row:
                            continue
                        candidate = None
                        for k in ("artist", "canonical", "name", "tag"):
                            if row.get(k):
                                candidate = row[k]
                                break
                        if candidate is None:
                            vals = [v for v in row.values() if v]
                            candidate = vals[0] if vals else None
                        if candidate:
                            add_artist(str(candidate))
            else:
                for row in rows:
                    if not row:
                        continue
                    add_artist(str(row[0]))
        elif ext == ".json":
            with open(path, "r", encoding="utf-8") as f:
                obj = json.load(f)
            if isinstance(obj, list):
                for item in obj:
                    if isinstance(item, str):
                        add_artist(item)
                    elif isinstance(item, dict):
                        for k in ("artist", "canonical", "name", "tag"):
                            if k in item and item[k]:
                                add_artist(str(item[k]))
                                break
            elif isinstance(obj, dict):
                for k in obj.keys():
                    add_artist(str(k))
        else:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    s = line.strip()
                    if not s or s.startswith("#"):
                        continue
                    add_artist(s)
    except Exception as e:
        print(f"[WARN] 读取 artists-file 失败：{e}", file=sys.stderr)
    return artists


def normalize_artist(input_name: str, alias_map: Dict[str, str]) -> str:
    return alias_map.get(input_name, input_name)


def sanitize_folder_name(name: str) -> str:
    bad = '<>:"/\\|?*'
    for ch in bad:
        name = name.replace(ch, "_")
    name = name.strip().rstrip(".")
    return name or "_"


def collect_download_tasks(
    s_meta: requests.Session,
    tags: str,
    out_dir: str,
    max_posts: int,
    only_webp: bool,
    convert_to_webp_flag: bool,
    save_json: bool,
    manifest_path: Optional[str],
    manifest_ids: Optional[Set[str]],
    interval: float,
    state_path: Optional[str],
    start_page: int,
    debug: bool = False,
    artist_input: Optional[str] = None,
    artist_canonical: Optional[str] = None,
    tag_type_cache: Optional[Dict[str, Optional[int]]] = None,
    enable_tag_type_lookup: bool = True,
    general_tagger: Optional[WdEva02GeneralTagger] = None,
) -> Tuple[List[Tuple[Dict, str, str]], int]:
    ensure_dir(out_dir)

    state = load_state(state_path)
    current_page = int(state.get("next_page", start_page)) if state_path else start_page
    current_page = max(1, current_page)

    to_download: List[Tuple[Dict, str, str]] = []
    fetched_posts = 0

    while fetched_posts < max_posts:
        need = min(100, max_posts - fetched_posts)
        try:
            posts = api_fetch_posts(s_meta, tags=tags, page=current_page, limit=need, debug=debug)
        except requests.HTTPError as e:
            print(f"请求失败: {e}", file=sys.stderr)
            break

        if not posts:
            print("没有更多结果。")
            break

        for post in posts:
            if enable_tag_type_lookup:
                enrich_post_tags_map_with_lookup(
                    s_meta,
                    post,
                    tag_type_cache if tag_type_cache is not None else {},
                    debug=debug,
                )
            pid = post.get("id")
            if not isinstance(pid, int):
                continue
            if not should_keep_post_as_webp(post, only_webp):
                continue
            manifest_key = build_manifest_key(pid, artist_canonical)
            if manifest_ids and manifest_key and manifest_key in manifest_ids:
                continue

            url = pick_best_url(post)
            if not url:
                continue

            file_ext = str(post.get("file_ext") or ext_from_url(url)).lower()
            if not is_supported_image_ext(file_ext):
                if debug:
                    print(f"[DEBUG] 跳过非图片 post {pid} (ext={file_ext})")
                continue

            final_ext = "webp" if convert_to_webp_flag else None
            filename = build_filename(post, url, force_ext=final_ext)
            dst = os.path.join(out_dir, filename)

            if os.path.exists(dst):
                general_override: Optional[List[str]] = None
                if general_tagger is not None:
                    try:
                        general_override = general_tagger.infer_general_tags(dst)
                    except Exception as e:
                        if debug:
                            print(f"[DEBUG] WD general 推理失败（跳过覆盖）post {pid}: {_fmt_err(e)}")
                if save_json:
                    write_json_sidecar(
                        post,
                        dst,
                        final_ext or file_ext,
                        artist_input=artist_input,
                        artist_canonical=artist_canonical,
                        general_tags_override=general_override,
                    )
                if manifest_path:
                    append_manifest_row(
                        manifest_path,
                        post,
                        dst,
                        final_ext or file_ext,
                        artist_input=artist_input,
                        artist_canonical=artist_canonical,
                        existing_ids=manifest_ids,
                        general_tags_override=general_override,
                    )
                continue

            to_download.append((post, url, dst))

        fetched_posts += len(posts)
        current_page += 1

        if state_path:
            save_state(state_path, {"next_page": current_page})

        time.sleep(interval)

        if len(to_download) >= max_posts:
            to_download = to_download[:max_posts]
            break

    return to_download, current_page


def execute_downloads(
    items: List[Tuple[Dict, str, str]],
    workers: int,
    convert_to_webp_flag: bool,
    quality: int,
    save_json: bool,
    manifest_path: Optional[str],
    manifest_ids: Optional[Set[str]] = None,
    artist_input: Optional[str] = None,
    artist_canonical: Optional[str] = None,
    download_timeout: int = 45,
    download_retries: int = 2,
    general_tagger: Optional[WdEva02GeneralTagger] = None,
) -> int:
    s = build_session()

    def after_success(
        post: Dict,
        saved_path: str,
        final_ext: str,
        general_tags_override: Optional[List[str]] = None,
    ):
        if save_json:
            write_json_sidecar(
                post,
                saved_path,
                final_ext,
                artist_input=artist_input,
                artist_canonical=artist_canonical,
                general_tags_override=general_tags_override,
            )
        if manifest_path:
            append_manifest_row(
                manifest_path,
                post,
                saved_path,
                final_ext,
                artist_input=artist_input,
                artist_canonical=artist_canonical,
                existing_ids=manifest_ids,
                general_tags_override=general_tags_override,
            )

    def worker(item: Tuple[Dict, str, str]) -> Tuple[bool, Optional[str], int]:
        post, url, dst = item
        post_id = int(post.get("id", -1))

        def infer_general(saved_path: str) -> Tuple[Optional[List[str]], Optional[str]]:
            if general_tagger is None:
                return None, None
            try:
                return general_tagger.infer_general_tags(saved_path), None
            except Exception as e:
                return None, f"WD general 推理失败: {_fmt_err(e)}"

        if convert_to_webp_flag:
            tmp_src = dst + ".orig"
            tmp_src_part = tmp_src + ".part"
            if not os.path.exists(tmp_src):
                ok, err = download_one(
                    s,
                    url,
                    tmp_src,
                    timeout=download_timeout,
                    retries=download_retries,
                )
                if not ok:
                    _safe_remove(tmp_src_part)
                    return False, f"下载失败: {err}", post_id

            if convert_to_webp(tmp_src, dst, quality=quality, max_pixels=MAX_IMAGE_PIXELS):
                _safe_remove(tmp_src)
                _safe_remove(tmp_src_part)
                gen_tags, wd_warn = infer_general(dst)
                after_success(post, dst, "webp", general_tags_override=gen_tags)
                return True, wd_warn, post_id

            limit_image_pixels_inplace(tmp_src, MAX_IMAGE_PIXELS)
            ext = ext_from_url(url)
            fallback = dst.rsplit(".", 1)[0] + "." + ext
            try:
                os.replace(tmp_src, fallback)
            except Exception:
                _safe_remove(tmp_src_part)
                return False, "转 webp 失败，且原图回退保存失败", post_id
            _safe_remove(tmp_src_part)
            gen_tags, wd_warn = infer_general(fallback)
            msg = "转 webp 失败，已回退保存原图"
            if wd_warn:
                msg = f"{msg}; {wd_warn}"
            after_success(post, fallback, ext, general_tags_override=gen_tags)
            return True, msg, post_id

        ok, err = download_one(
            s,
            url,
            dst,
            timeout=download_timeout,
            retries=download_retries,
        )
        if ok:
            limit_image_pixels_inplace(dst, MAX_IMAGE_PIXELS)
            gen_tags, wd_warn = infer_general(dst)
            after_success(post, dst, ext_from_url(url), general_tags_override=gen_tags)
            return True, wd_warn, post_id
        return False, f"下载失败: {err}", post_id

    ok = 0
    done = 0
    failed: List[str] = []
    warnings: List[str] = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        futures = {ex.submit(worker, it): it for it in items}
        for fut in as_completed(futures):
            success, msg, post_id = fut.result()
            done += 1
            if success:
                ok += 1
                if msg:
                    warnings.append(f"post {post_id}: {msg}")
            else:
                failed.append(f"post {post_id}: {msg or 'unknown error'}")
            if done % 25 == 0 or done == len(items):
                print(f"... 已完成 {done}/{len(items)}（成功 {ok}）")

    if warnings:
        print(f"[WARN] 有 {len(warnings)} 项发生转码回退，已保存原图。")
        for line in warnings[:10]:
            print(f"[WARN] {line}")
        if len(warnings) > 10:
            print(f"[WARN] ... 其余 {len(warnings) - 10} 项省略")
    if failed:
        print(f"[WARN] 有 {len(failed)} 项下载失败。")
        for line in failed[:20]:
            print(f"[WARN] {line}")
        if len(failed) > 20:
            print(f"[WARN] ... 其余 {len(failed) - 20} 项省略")
    return ok


def main():
    parser = argparse.ArgumentParser("yande.re Image Downloader")
    parser.add_argument("--tags", type=str, default=None, help="yande.re 搜索标签（空格分隔）")
    parser.add_argument("--artists-file", type=str, default=None, help="多作者模式：作者列表文件（支持 csv/txt/json）")
    parser.add_argument("--alias-map", type=str, default=None, help="作者别名映射文件（CSV/TSV/JSON/文本 a->b）")
    parser.add_argument("--base-tags", type=str, default="", help="多作者统一追加标签（如 rating:s）")
    parser.add_argument("--max-per-artist", type=int, default=200, help="多作者模式每位作者最多抓取数")
    parser.add_argument("--out", type=str, required=True, help="输出目录")
    parser.add_argument("--max-posts", type=int, default=200, help="最多抓取的帖子数")
    parser.add_argument("--start-page", type=int, default=1, help="起始页（默认 1）")

    parser.add_argument("--only-webp", action="store_true", help="仅下载源文件为 webp 的图片")
    parser.add_argument("--convert-to-webp", action="store_true", help="将下载结果统一转成 webp")
    parser.add_argument("--quality", type=int, default=85, help="webp 质量（转码时生效）")

    parser.add_argument("--workers", type=int, default=8, help="下载并发数")
    parser.add_argument("--download-timeout", type=int, default=45, help="单文件下载超时秒数（默认 45）")
    parser.add_argument("--download-retries", type=int, default=2, help="单文件下载重试次数（默认 2）")
    parser.add_argument("--replace-general-with-wd", action="store_true", help="用 WD-EVA02 结果替换 general tags")
    parser.add_argument("--wd-repo", type=str, default=WD_REPO_DEFAULT, help="WD 模型仓库（默认 SmilingWolf/wd-eva02-large-tagger-v3）")
    parser.add_argument("--wd-general-threshold", type=float, default=0.35, help="WD general 阈值（默认 0.35）")
    parser.add_argument("--wd-use-gpu", action="store_true", help="WD 推理使用 GPU（onnxruntime CUDA）")
    parser.add_argument("--no-tag-type-lookup", action="store_true", help="禁用在线标签类型查询（分类列将可能为空）")
    parser.add_argument("--backfill-manifest-tags", action="store_true", help="启动时先回填 manifest 的分类列")
    parser.add_argument("--resume", action="store_true", help="断点续抓，写入 state.json")
    parser.add_argument(
        "--request-interval",
        type=float,
        default=0.6,
        help="API 请求间隔秒（默认 0.6）",
    )
    parser.add_argument("--save-json", action="store_true", help="为每张图保存同名 .json")
    parser.add_argument("--manifest", action="store_true", help="生成/追加 manifest.csv")
    parser.add_argument(
        "--manifest-path",
        type=str,
        default=None,
        help="manifest 输出路径（默认：<out>/manifest.csv）",
    )
    parser.add_argument("--debug", action="store_true", help="打印调试信息")

    args = parser.parse_args()

    if args.convert_to_webp and not PIL_AVAILABLE:
        print("未检测到 Pillow；请先 `pip install pillow` 或关闭 --convert-to-webp", file=sys.stderr)
        sys.exit(1)

    ensure_dir(args.out)

    if args.manifest:
        manifest_path = args.manifest_path or os.path.join(args.out, "manifest.csv")
        parent = os.path.dirname(manifest_path)
        if parent:
            ensure_dir(parent)
        ensure_manifest_schema(manifest_path)
        manifest_ids = load_manifest_ids(manifest_path)
    else:
        manifest_path = None
        manifest_ids = None

    s_meta = build_session()
    tag_type_cache: Dict[str, Optional[int]] = {}
    general_tagger: Optional[WdEva02GeneralTagger] = None
    if args.replace_general_with_wd:
        general_tagger = WdEva02GeneralTagger(
            repo_id=args.wd_repo,
            use_gpu=args.wd_use_gpu,
            general_threshold=float(args.wd_general_threshold),
        )
        print(
            f">> WD general 替换已启用: repo={args.wd_repo}, "
            f"th={args.wd_general_threshold}, use_gpu={args.wd_use_gpu}"
        )

    if args.backfill_manifest_tags and manifest_path:
        backfill_manifest_tags(
            manifest_path=manifest_path,
            s_meta=s_meta,
            tag_type_cache=tag_type_cache,
            debug=args.debug,
        )

    if args.artists_file:
        alias_map = load_alias_map(args.alias_map)
        artists = load_artists_list(args.artists_file)
        if not artists:
            print("artists-file 未读取到任何作者。", file=sys.stderr)
            sys.exit(2)

        print(f">> 多作者模式：{len(artists)} 位作者；base-tags='{args.base_tags}'")
        if alias_map:
            print(f">> alias-map：{len(alias_map)} 条")
        print(
            f">> mode: only_webp={args.only_webp}, convert_to_webp={args.convert_to_webp}, "
            f"workers={args.workers}, save_json={args.save_json}, manifest={args.manifest}"
        )

        total_ok = 0
        for i, input_name in enumerate(artists, 1):
            canonical = normalize_artist(input_name, alias_map)
            tags = canonical
            if args.base_tags.strip():
                tags = f"{tags} {args.base_tags.strip()}"

            folder = sanitize_folder_name(canonical)
            out_dir = os.path.join(args.out, folder)
            ensure_dir(out_dir)
            state_path = os.path.join(out_dir, f"state_{folder}.json") if args.resume else None

            print(f"\n== [{i}/{len(artists)}] {input_name} -> {canonical}")
            print(f">> tags={tags} | out={out_dir} | max={args.max_per_artist}")

            items, next_page = collect_download_tasks(
                s_meta=s_meta,
                tags=tags,
                out_dir=out_dir,
                max_posts=max(1, args.max_per_artist),
                only_webp=args.only_webp,
                convert_to_webp_flag=args.convert_to_webp,
                save_json=args.save_json,
                manifest_path=manifest_path,
                manifest_ids=manifest_ids,
                interval=max(0.0, args.request_interval),
                state_path=state_path,
                start_page=max(1, args.start_page),
                debug=args.debug,
                artist_input=input_name,
                artist_canonical=canonical,
                tag_type_cache=tag_type_cache,
                enable_tag_type_lookup=not args.no_tag_type_lookup,
                general_tagger=general_tagger,
            )

            if not items:
                print(">> 没有可下载的条目。")
                continue

            print(f">> 待下载: {len(items)}")
            ok = execute_downloads(
                items=items,
                workers=args.workers,
                convert_to_webp_flag=args.convert_to_webp,
                quality=args.quality,
                save_json=args.save_json,
                manifest_path=manifest_path,
                manifest_ids=manifest_ids,
                artist_input=input_name,
                artist_canonical=canonical,
                download_timeout=max(5, args.download_timeout),
                download_retries=max(0, args.download_retries),
                general_tagger=general_tagger,
            )
            total_ok += ok
            print(f">> {canonical} 完成：{ok}/{len(items)}")
            if state_path:
                print(f">> 已保存断点：{state_path} (next_page={next_page})")

            time.sleep(max(0.0, args.request_interval))

        print(f"\n== 全部作者完成。合计成功：{total_ok}")
        return

    if not args.tags:
        print("请提供 --tags 或 --artists-file。", file=sys.stderr)
        sys.exit(2)

    state_path = os.path.join(args.out, "state.json") if args.resume else None

    print(
        f">> tags={args.tags} | out={args.out} | max_posts={args.max_posts} | "
        f"start_page={args.start_page}"
    )
    print(
        f">> mode: only_webp={args.only_webp}, convert_to_webp={args.convert_to_webp}, "
        f"workers={args.workers}, save_json={args.save_json}, manifest={args.manifest}"
    )

    items, next_page = collect_download_tasks(
        s_meta=s_meta,
        tags=args.tags,
        out_dir=args.out,
        max_posts=max(1, args.max_posts),
        only_webp=args.only_webp,
        convert_to_webp_flag=args.convert_to_webp,
        save_json=args.save_json,
        manifest_path=manifest_path,
        manifest_ids=manifest_ids,
        interval=max(0.0, args.request_interval),
        state_path=state_path,
        start_page=max(1, args.start_page),
        debug=args.debug,
        artist_input=None,
        artist_canonical=None,
        tag_type_cache=tag_type_cache,
        enable_tag_type_lookup=not args.no_tag_type_lookup,
        general_tagger=general_tagger,
    )

    if not items:
        print("没有可下载的条目。")
        return

    print(f">> 待下载: {len(items)}")
    ok = execute_downloads(
        items=items,
        workers=args.workers,
        convert_to_webp_flag=args.convert_to_webp,
        quality=args.quality,
        save_json=args.save_json,
        manifest_path=manifest_path,
        manifest_ids=manifest_ids,
        artist_input=None,
        artist_canonical=None,
        download_timeout=max(5, args.download_timeout),
        download_retries=max(0, args.download_retries),
        general_tagger=general_tagger,
    )
    print(f">> 完成：{ok}/{len(items)}")
    if state_path:
        print(f">> 已保存断点：{state_path} (next_page={next_page})")


if __name__ == "__main__":
    main()
