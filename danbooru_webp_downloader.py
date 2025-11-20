# -*- coding: utf-8 -*-
"""
Danbooru 图片下载器（含完整 tags 元数据）——多作者批量模式增强版
----------------------------------------------------------------
新增能力：
- --artists-file：从文件中读取大量作者名，逐个下载到 out/<canonical>/ 目录
- --alias-map：可选，提供别名→正式名映射（CSV/TSV/JSON/纯文本 "a->b"）
- --base-tags：可选，追加统一搜索限定（如 "rating:safe"）
- 共用一个 manifest（默认写到 out/manifest.csv），额外记录 artist_input 与 artist_canonical
- 每个作者独立断点（state_<canonical>.json），互不影响，可随时续抓
- --max-per-artist 控制每个作者抓取上限；未使用 --artists-file 时保留原先单标签模式（--max-posts）

用法示例：
1) 批量作者、共享 manifest、每人 300 张、rating 限定、转 webp 并保存 JSON：
   python danbooru_webp_downloader.py \
     --artists-file artists.txt \
     --alias-map alias.csv \
     --base-tags "rating:safe" \
     --max-per-artist 300 \
     --out ./dan_artists \
     --convert-to-webp --save-json --manifest \
     --login YOUR_NAME --api-key YOUR_KEY

2) 仍可用原有单标签模式（保持兼容）：
   python danbooru_webp_downloader.py --tags "1girl solo" --max-posts 200 \
     --out ./danbooru_1girl --convert-to-webp --save-json --manifest
"""
import os
import sys
import time
import json
import csv
import argparse
import hashlib
import math
import threading
from typing import List, Optional, Dict, Tuple, Iterable, Set
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from requests.adapters import HTTPAdapter
try:
    from requests.adapters import Retry  # requests>=2.29
except ImportError:
    from urllib3.util.retry import Retry  # type: ignore

# 可选：用于把 jpg/png 转成 webp
try:
    from PIL import Image
    PIL_AVAILABLE = True
except Exception:
    Image = None  # type: ignore
    PIL_AVAILABLE = False

DANBOORU_API = "https://danbooru.donmai.us/posts.json"
LOCK = threading.Lock()
MAX_IMAGE_PIXELS = 10_000_000  # 10MP 上限
SUPPORTED_IMAGE_EXTENSIONS = {
    "jpg", "jpeg", "png", "webp", "bmp", "gif", "tif", "tiff", "jfif"
}

def is_supported_image_ext(ext: Optional[str]) -> bool:
    if not ext:
        return False
    return ext.lower() in SUPPORTED_IMAGE_EXTENSIONS

def _calc_resize_dimensions(width: int, height: int, max_pixels: int) -> Optional[Tuple[int, int]]:
    if max_pixels <= 0 or width <= 0 or height <= 0:
        return None
    total = width * height
    if total <= max_pixels:
        return None
    scale = math.sqrt(max_pixels / float(total))
    new_w = max(1, int(width * scale))
    new_h = max(1, int(height * scale))
    # 防止取整后仍超过限制
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

# ============== 工具与会话 ==============
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
    s.headers.update({"User-Agent": "DanbooruWebpDownloader/3.0"})
    return s

def ensure_dir(d: str):
    os.makedirs(d, exist_ok=True)

def save_state(state_path: str, last_id: Optional[int]):
    if not state_path:
        return
    with LOCK:
        with open(state_path, "w", encoding="utf-8") as f:
            json.dump({"last_id": last_id}, f)

def load_state(state_path: str) -> Optional[int]:
    if not state_path or not os.path.exists(state_path):
        return None
    try:
        with open(state_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data.get("last_id")
    except Exception:
        return None

def pick_best_url(post: Dict) -> Optional[str]:
    for key in ("file_url", "large_file_url"):
        url = post.get(key)
        if url and isinstance(url, str):
            return url
    return None

def ext_from_url(url: str) -> str:
    base = url.split("?")[0].split("#")[0]
    if "." in base:
        return base.rsplit(".", 1)[-1].lower()
    return "bin"

def build_filename(post: Dict, url: str, force_ext: Optional[str] = None) -> str:
    pid = post.get("id")
    md5v = post.get("md5") or hashlib.md5(url.encode("utf-8")).hexdigest()
    ext = (force_ext or ext_from_url(url)).lower()
    return f"{pid}_{md5v}.{ext}"

def download_one(s: requests.Session, url: str, out_path: str, timeout: int = 30) -> bool:
    try:
        with s.get(url, stream=True, timeout=timeout) as r:
            r.raise_for_status()
            tmp = out_path + ".part"
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 16):
                    if chunk:
                        f.write(chunk)
            os.replace(tmp, out_path)
        return True
    except Exception:
        return False

def convert_to_webp(src_path: str, dst_path: str, quality: int = 85,
                    max_pixels: Optional[int] = MAX_IMAGE_PIXELS) -> bool:
    if not PIL_AVAILABLE:
        return False
    try:
        with Image.open(src_path) as im:
            im = _resize_image_object(im, max_pixels)
            if im.mode not in ("RGB",):
                im = im.convert("RGB")
            im.save(dst_path, "WEBP", quality=quality, method=6)
        return True
    except Exception:
        return False

def should_keep_post_as_webp(post: Dict, only_webp: bool) -> bool:
    if not only_webp:
        return True
    ext = str(post.get("file_ext", "")).lower()
    if ext == "webp":
        return True
    url = pick_best_url(post)
    return (url is not None) and (ext_from_url(url) == "webp")

# ============== 标签与清单 ==============
def split_tags(s: Optional[str]) -> List[str]:
    if not s:
        return []
    return [t for t in s.strip().split() if t]

def build_post_url(post_id: int) -> str:
    return f"https://danbooru.donmai.us/posts/{post_id}"

def write_json_sidecar(post: Dict, image_path: str, final_ext: str,
                       artist_input: Optional[str] = None,
                       artist_canonical: Optional[str] = None):
    base = os.path.splitext(image_path)[0]
    js_path = base + ".json"
    data = {
        "id": post.get("id"),
        "post_url": build_post_url(post.get("id")),
        "md5": post.get("md5"),
        "rating": post.get("rating"),
        "score": post.get("score"),
        "fav_count": post.get("fav_count"),
        "source": post.get("source"),
        "file": {
            "original_ext": post.get("file_ext"),
            "saved_ext": final_ext,
            "image_width": post.get("image_width"),
            "image_height": post.get("image_height"),
            "file_size": post.get("file_size"),
        },
        "created_at": post.get("created_at"),
        "uploader_id": post.get("uploader_id"),
        "tags": {
            "all":           split_tags(post.get("tag_string")),
            "general":       split_tags(post.get("tag_string_general")),
            "character":     split_tags(post.get("tag_string_character")),
            "copyright":     split_tags(post.get("tag_string_copyright")),
            "artist":        split_tags(post.get("tag_string_artist")),
            "meta":          split_tags(post.get("tag_string_meta")),
        },
    }
    if artist_input or artist_canonical:
        data["artist_context"] = {
            "artist_input": artist_input,
            "artist_canonical": artist_canonical,
        }
    with open(js_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def append_manifest_row(manifest_path: str, post: Dict, saved_path: str, final_ext: str,
                        artist_input: Optional[str] = None,
                        artist_canonical: Optional[str] = None,
                        existing_ids: Optional[Set[str]] = None):
    post_id = post.get("id")
    post_id_str = str(post_id) if post_id is not None else ""
    with LOCK:
        if existing_ids is not None and post_id_str and post_id_str in existing_ids:
            return
        exists = os.path.exists(manifest_path)
        fieldnames = [
            "id","post_url","md5","rating","score","fav_count","source",
            "image_width","image_height","file_size","created_at","uploader_id",
            "saved_path","saved_ext",
            "tags_general","tags_character","tags_copyright","tags_artist","tags_meta","tags_all",
            "artist_input","artist_canonical"
        ]
        row = {
            "id": post.get("id"),
            "post_url": build_post_url(post.get("id")),
            "md5": post.get("md5"),
            "rating": post.get("rating"),
            "score": post.get("score"),
            "fav_count": post.get("fav_count"),
            "source": post.get("source"),
            "image_width": post.get("image_width"),
            "image_height": post.get("image_height"),
            "file_size": post.get("file_size"),
            "created_at": post.get("created_at"),
            "uploader_id": post.get("uploader_id"),
            "saved_path": os.path.abspath(saved_path),
            "saved_ext": final_ext,
            "tags_general": " ".join(split_tags(post.get("tag_string_general"))),
            "tags_character": " ".join(split_tags(post.get("tag_string_character"))),
            "tags_copyright": " ".join(split_tags(post.get("tag_string_copyright"))),
            "tags_artist": " ".join(split_tags(post.get("tag_string_artist"))),
            "tags_meta": " ".join(split_tags(post.get("tag_string_meta"))),
            "tags_all": " ".join(split_tags(post.get("tag_string"))),
            "artist_input": artist_input or "",
            "artist_canonical": artist_canonical or "",
        }
        with open(manifest_path, "a", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            if not exists:
                w.writeheader()
            w.writerow(row)
        if existing_ids is not None and post_id_str:
            existing_ids.add(post_id_str)

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
                pid = row.get("id")
                if pid:
                    ids.add(str(pid).strip())
    except Exception as e:
        print(f"[WARN] 读取 manifest 失败：{e}", file=sys.stderr)
    return ids

# ============== API 抓取 ==============
def api_fetch_posts(
    s: requests.Session,
    tags: str,
    page_before_id: Optional[int],
    limit: int,
    login: Optional[str],
    api_key: Optional[str],
    debug: bool = False,
) -> List[Dict]:
    params = {
        "tags": tags,
        "limit": max(1, min(limit, 100)),
        "only": ",".join([
            "id","md5","file_ext","file_url","large_file_url",
            "image_width","image_height","file_size","rating",
            "tag_string","tag_string_general","tag_string_character",
            "tag_string_copyright","tag_string_artist","tag_string_meta",
            "score","fav_count","source","created_at","uploader_id",
        ])
    }
    if page_before_id:
        params["page"] = f"b{page_before_id}"

    def _do(auth_tuple):
        r = s.get(DANBOORU_API, params=params, auth=auth_tuple, timeout=30)
        if debug:
            req = r.request
            print(f"[DEBUG] {req.method} {req.url} -> {r.status_code}")
        r.raise_for_status()
        return r.json()

    try:
        return _do((login, api_key) if (login and api_key) else None)
    except requests.HTTPError as e:
        status = getattr(e.response, "status_code", None)
        if status in (401, 403):
            print("认证失败或权限不足（401/403）。以匿名方式重试一次……", file=sys.stderr)
            return _do(None)
        raise

# ============== 别名映射与作者列表 ==============
def load_alias_map(path: Optional[str]) -> Dict[str, str]:
    """支持 CSV/TSV（两列）、JSON（obj 或 list of [a,b]）、纯文本（'a->b' 每行）。"""
    base_map = {
        # 你的示例里的常见别名
        "shiratama": "shiratama_(shiratamaco)",
        "momizi": "momiji",
    }
    if not path:
        return base_map

    def add_pair(d: Dict[str, str], a: str, b: str):
        a = a.strip()
        b = b.strip()
        if a:
            d[a] = b

    d = dict(base_map)
    ext = os.path.splitext(path)[-1].lower()

    try:
        if ext in (".csv", ".tsv"):
            delim = "\t" if ext == ".tsv" else ","
            with open(path, "r", encoding="utf-8") as f:
                r = csv.reader(f, delimiter=delim)
                for row in r:
                    if not row or len(row) < 2:
                        continue
                    add_pair(d, row[0], row[1])
        elif ext == ".json":
            with open(path, "r", encoding="utf-8") as f:
                obj = json.load(f)
            if isinstance(obj, dict):
                for k, v in obj.items():
                    add_pair(d, str(k), str(v))
            elif isinstance(obj, list):
                for it in obj:
                    if isinstance(it, (list, tuple)) and len(it) >= 2:
                        add_pair(d, str(it[0]), str(it[1]))
        else:
            # 纯文本：支持注释与空行，每行 "a->b"
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    if "->" in line:
                        a, b = line.split("->", 1)
                        add_pair(d, a, b)
    except Exception as e:
        print(f"[WARN] 读取别名映射失败：{e}", file=sys.stderr)
    return d

def load_artists_list(path: str) -> List[str]:
    """每行一个作者名，支持注释 '#'，自动去重、去空。"""
    artists: List[str] = []
    seen = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            if s not in seen:
                seen.add(s)
                artists.append(s)
    return artists

def normalize_artist(input_name: str, alias_map: Dict[str, str]) -> str:
    # 优先别名映射；否则保持原样
    return alias_map.get(input_name, input_name)

def sanitize_folder_name(name: str) -> str:
    # 简单处理：替换文件名不安全字符
    bad = '<>:"/\\|?*'
    for ch in bad:
        name = name.replace(ch, "_")
    return name.strip().rstrip(".")

# ============== 单作者下载流程（复用原逻辑） ==============
def collect_download_tasks_for_tags(
    s_meta: requests.Session,
    tags: str,
    out_dir: str,
    max_posts: int,
    only_webp: bool,
    convert_to_webp: bool,
    save_json: bool,
    manifest_path: Optional[str],
    manifest_ids: Optional[Set[str]],
    login: Optional[str],
    api_key: Optional[str],
    interval: float,
    state_path: Optional[str],
    debug: bool = False,
    artist_input: Optional[str] = None,
    artist_canonical: Optional[str] = None,
) -> Tuple[List[Tuple[Dict, str, str]], Optional[int]]:
    ensure_dir(out_dir)
    last_id = load_state(state_path)

    to_download: List[Tuple[Dict, str, str]] = []
    fetched = 0
    oldest_id = last_id

    while fetched < max_posts:
        need = min(100, max_posts - fetched)
        try:
            posts = api_fetch_posts(
                s_meta, tags, page_before_id=oldest_id, limit=need,
                login=login, api_key=api_key, debug=debug
            )
        except requests.HTTPError as e:
            print(f"请求失败: {e}", file=sys.stderr)
            break

        if not posts:
            print("没有更多结果。")
            break

        for p in posts:
            if not should_keep_post_as_webp(p, only_webp):
                continue
            post_id = p.get("id")
            post_id_str = str(post_id) if post_id is not None else ""
            if manifest_ids and post_id_str and post_id_str in manifest_ids:
                continue
            url = pick_best_url(p)
            if not url:
                continue
            file_ext = (p.get("file_ext") or "").lower()
            if not file_ext:
                file_ext = ext_from_url(url)
            if not is_supported_image_ext(file_ext):
                if debug:
                    print(f"[DEBUG] 跳过非图片 post {post_id} (ext={file_ext})")
                continue

            final_ext = "webp" if convert_to_webp else None
            fname = build_filename(p, url, force_ext=final_ext)
            dst = os.path.join(out_dir, fname)
            if os.path.exists(dst):
                # 已存在：可选写 sidecar/manifest（幂等）
                if save_json:
                    write_json_sidecar(
                        p,
                        dst,
                        (final_ext or ext_from_url(url)),
                        artist_input=artist_input,
                        artist_canonical=artist_canonical,
                    )
                if manifest_path:
                    append_manifest_row(
                        manifest_path,
                        p,
                        dst,
                        (final_ext or ext_from_url(url)),
                        artist_input=artist_input,
                        artist_canonical=artist_canonical,
                        existing_ids=manifest_ids,
                    )
                continue

            to_download.append((p, url, dst))

        fetched += len(posts)
        oldest_id = min(int(p["id"]) for p in posts)
        if state_path:
            save_state(state_path, oldest_id)

        time.sleep(interval)

        if len(to_download) >= max_posts:
            to_download = to_download[:max_posts]
            break

    return to_download, oldest_id

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
) -> int:
    s = build_session()

    def after_success(p: Dict, saved_path: str, final_ext: str):
        if save_json:
            write_json_sidecar(p, saved_path, final_ext,
                               artist_input=artist_input, artist_canonical=artist_canonical)
        if manifest_path:
            append_manifest_row(manifest_path, p, saved_path, final_ext,
                                artist_input=artist_input, artist_canonical=artist_canonical,
                                existing_ids=manifest_ids)

    def worker(item: Tuple[Dict, str, str]) -> bool:
        p, url, dst = item
        if convert_to_webp_flag:
            tmp_src = dst + ".orig"
            if (not os.path.exists(tmp_src)) and (not download_one(s, url, tmp_src)):
                return False
            if convert_to_webp(tmp_src, dst, quality=quality, max_pixels=MAX_IMAGE_PIXELS):
                try:
                    os.remove(tmp_src)
                except Exception:
                    pass
                after_success(p, dst, "webp")
                return True
            else:
                limit_image_pixels_inplace(tmp_src, MAX_IMAGE_PIXELS)
                ext = ext_from_url(url)
                fallback = dst.rsplit(".", 1)[0] + "." + ext
                try:
                    os.replace(tmp_src, fallback)
                except Exception:
                    pass
                after_success(p, fallback, ext)
                return False
        else:
            if download_one(s, url, dst):
                limit_image_pixels_inplace(dst, MAX_IMAGE_PIXELS)
                after_success(p, dst, ext_from_url(url))
                return True
            return False

    ok = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(worker, it): it for it in items}
        for fut in as_completed(futures):
            success = fut.result()
            ok += 1 if success else 0
            if ok and ok % 25 == 0:
                print(f"... 已完成 {ok}/{len(items)}")
    return ok

# ============== 主流程 ==============
def main():
    parser = argparse.ArgumentParser("Danbooru WebP Downloader (with tags) — multi-artist enhanced")
    # 原单标签模式（保持兼容）
    parser.add_argument("--tags", type=str, default=None,
                        help="Danbooru 搜索标签（空格分隔，如 '1girl solo rating:safe'）")

    # 新的多作者模式
    parser.add_argument("--artists-file", type=str, default=None,
                        help="包含作者名的文件，每行一个；支持#注释、去重")
    parser.add_argument("--alias-map", type=str, default=None,
                        help="别名映射文件（CSV/TSV两列、JSON、或 'a->b' 文本）")
    parser.add_argument("--base-tags", type=str, default="",
                        help="所有作者统一追加的搜索限定（如 'rating:safe'）")
    parser.add_argument("--max-per-artist", type=int, default=200,
                        help="每个作者最多抓取的帖子数（用于多作者模式）")

    parser.add_argument("--out", type=str, required=True, help="输出根目录")
    parser.add_argument("--max-posts", type=int, default=200,
                        help="单标签模式下最多抓取的帖子数（兼容旧用法）")
    parser.add_argument("--login", type=str, default=None, help="Danbooru 用户名（可选）")
    parser.add_argument("--api-key", type=str, default=None, help="Danbooru API Key（可选）")

    parser.add_argument("--only-webp", action="store_true",
                        help="仅下载站上本身就是 webp 的文件")
    parser.add_argument("--convert-to-webp", action="store_true",
                        help="将 jpg/png 等转码为 webp 保存")
    parser.add_argument("--quality", type=int, default=85, help="webp 质量（转码时生效）")

    parser.add_argument("--workers", type=int, default=8, help="下载并发数")
    parser.add_argument("--resume", action="store_true",
                        help="断点续抓：多作者模式下将写 state_<canonical>.json")
    parser.add_argument("--request-interval", type=float, default=None,
                        help="API 请求间隔秒（默认：匿名 1.0；登录 0.35）")
    parser.add_argument("--save-json", action="store_true",
                        help="为每张图片写同名 .json（完整 tags 与元数据）")
    parser.add_argument("--manifest", action="store_true",
                        help="生成/追加 manifest.csv 汇总（多作者模式：共用一个清单）")
    parser.add_argument("--manifest-path", type=str, default=None,
                        help="manifest 输出路径（默认：<out>/manifest.csv）")
    parser.add_argument("--debug", action="store_true", help="打印调试信息（请求 URL 等）")

    args = parser.parse_args()

    if args.convert_to_webp and not PIL_AVAILABLE:
        print("未检测到 Pillow；请先 `pip install pillow` 或关闭 --convert-to-webp", file=sys.stderr)
        sys.exit(1)

    ensure_dir(args.out)
    s_meta = build_session()
    interval = args.request_interval if args.request_interval is not None else (
        0.35 if (args.login and args.api_key) else 1.0
    )

    # manifest 统一路径
    manifest_path: Optional[str] = None
    manifest_ids: Optional[Set[str]] = None
    if args.manifest:
        manifest_path = args.manifest_path or os.path.join(args.out, "manifest.csv")
        # 提前创建文件夹
        ensure_dir(os.path.dirname(manifest_path))
        manifest_ids = load_manifest_ids(manifest_path)

    # ========= 分支1：多作者模式 =========
    if args.artists_file:
        alias_map = load_alias_map(args.alias_map)
        artists = load_artists_list(args.artists_file)
        print(f">> 多作者模式：{len(artists)} 位作者；base-tags='{args.base_tags}'")
        if alias_map:
            print(f">> 别名映射：{len(alias_map)} 条")

        if args.convert_to_webp:
            print(f">> 转码为 webp，质量={args.quality}")
        mode_line = f"only_webp={args.only_webp}, workers={args.workers}, save_json={args.save_json}, manifest={args.manifest}"
        print(f">> 模式：{mode_line}")

        total_ok = 0
        for i, input_name in enumerate(artists, 1):
            canonical = normalize_artist(input_name, alias_map)
            # Danbooru 查询标签：artist:canonical
            # 注意 canonical 可能包含空格/括号，Danbooru 接受
            tags = f"{canonical}"
            if args.base_tags.strip():
                tags = f"{tags} {args.base_tags.strip()}"

            folder = sanitize_folder_name(canonical)
            out_dir = os.path.join(args.out, folder)
            ensure_dir(out_dir)
            state_path = os.path.join(out_dir, f"state_{folder}.json") if args.resume else None

            print(f"\n== [{i}/{len(artists)}] {input_name} -> {canonical}")
            print(f">> tags: {tags}")
            print(f">> out : {out_dir}")
            print(f">> max : {args.max_per_artist}")

            items, oldest_id = collect_download_tasks_for_tags(
                s_meta=s_meta,
                tags=tags,
                out_dir=out_dir,
                max_posts=args.max_per_artist,
                only_webp=args.only_webp,
                convert_to_webp=args.convert_to_webp,
                save_json=args.save_json,
                manifest_path=manifest_path,
                manifest_ids=manifest_ids,
                login=args.login,
                api_key=args.api_key,
                interval=interval,
                state_path=state_path,
                debug=args.debug,
                artist_input=input_name,
                artist_canonical=canonical,
            )

            if not items:
                print(">> 没有可下载的条目。")
                continue

            print(f">> 待下载: {len(items)} 个文件")
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
            )
            total_ok += ok
            if state_path:
                print(f">> 断点位（{canonical} 最旧 post_id）保存在: {state_path}")
            print(f">> {canonical} 完成：{ok}/{len(items)}")

            # 友好控制节奏，避免频繁打爆限流
            time.sleep(interval)

        print(f"\n== 全部作者完成。合计下载成功：{total_ok} 张。")
        return

    # ========= 分支2：原单标签模式（兼容） =========
    # 若未提供 --artists-file，则要求 --tags
    if not args.tags:
        print("请提供 --tags 或 --artists-file。", file=sys.stderr)
        sys.exit(2)

    print(f">> 单标签模式")
    print(f">> tags: {args.tags}")
    print(f">> out : {args.out}")
    print(f">> max : {args.max_posts}")
    print(f">> mode: only_webp={args.only_webp}, convert_to_webp={args.convert_to_webp}, "
          f"workers={args.workers}, save_json={args.save_json}, manifest={args.manifest}")
    if args.login and args.api_key:
        print(">> auth: using login + api-key")

    state_path = os.path.join(args.out, "state.json") if args.resume else None
    items, oldest_id = collect_download_tasks_for_tags(
        s_meta=s_meta,
        tags=args.tags,
        out_dir=args.out,
        max_posts=args.max_posts,
        only_webp=args.only_webp,
        convert_to_webp=args.convert_to_webp,
        save_json=args.save_json,
        manifest_path=manifest_path,
        manifest_ids=manifest_ids,
        login=args.login,
        api_key=args.api_key,
        interval=interval,
        state_path=state_path,
        debug=args.debug,
        artist_input=None,
        artist_canonical=None,
    )

    if not items:
        print("没有可下载的条目。")
        return

    print(f">> 待下载: {len(items)} 个文件")
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
    )
    print(f">> 完成：{ok}/{len(items)}")
    if state_path:
        print(f">> 断点位（最旧 post_id）保存在: {state_path}")

if __name__ == "__main__":
    main()
