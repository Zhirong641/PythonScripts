# -*- coding: utf-8 -*-
"""
Danbooru 图片下载器（含完整 tags 元数据）
---------------------------------------
特性：
- 通过官方 API /posts.json 抓取，避免解析 HTML
- 分页：page=b{last_id} 向更旧内容翻页，稳定不丢不重
- 并发下载、断点续抓（记录 last_id）、速率限制、自动重试
- 可选：仅下载站点原生 WebP；或把 jpg/png 转成 WebP 保存
- 为每张图片写 <basename>.json sidecar（包含完整 tags 与常用元数据）
- 可选生成 manifest.csv 汇总

用法示例：
1) 抓 200 张并转为 WebP、保存 JSON 与 manifest：
   python danbooru_webp_downloader.py --tags "1girl solo" --max-posts 200 \
     --out ./danbooru_1girl --convert-to-webp --save-json --manifest

2) 只下载站上原生 webp（不转码），匿名访问：
   python danbooru_webp_downloader.py --tags "rating:safe scenic" \
     --max-posts 100 --out ./safe_webp --only-webp

3) 带账号 + API Key（推荐，配额更高）：
   python danbooru_webp_downloader.py --tags "rating:questionable" --max-posts 300 \
     --out ./q --convert-to-webp --login YOUR_NAME --api-key YOUR_KEY --save-json --manifest
"""

import os
import sys
import time
import json
import csv
import argparse
import hashlib
import threading
from typing import List, Optional, Dict, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from requests.adapters import HTTPAdapter

try:
    from requests.adapters import Retry  # requests>=2.29 re-exports urllib3 Retry
except ImportError:  # pragma: no cover - very old requests
    from urllib3.util.retry import Retry  # type: ignore

# 可选：用于把 jpg/png 转成 webp
try:
    from PIL import Image
    PIL_AVAILABLE = True
except Exception:
    PIL_AVAILABLE = False

DANBOORU_API = "https://danbooru.donmai.us/posts.json"
LOCK = threading.Lock()

# --------------- 基础工具 ---------------

def _build_retry() -> Retry:
    common_kwargs = dict(
        total=5,
        backoff_factor=0.5,
        status_forcelist=[429, 500, 502, 503, 504],
        respect_retry_after_header=True,
    )

    methods = frozenset(["GET"])

    # urllib3 < 1.26 使用 method_whitelist；>=1.26 使用 allowed_methods。
    if "allowed_methods" in Retry.__init__.__code__.co_varnames:  # type: ignore[attr-defined]
        return Retry(allowed_methods=methods, **common_kwargs)
    return Retry(method_whitelist=methods, **common_kwargs)  # type: ignore[arg-type]


def build_session() -> requests.Session:
    s = requests.Session()
    retries = _build_retry()
    s.mount("https://", HTTPAdapter(max_retries=retries))
    s.headers.update({"User-Agent": "DanbooruWebpDownloader/2.0"})
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
    """优先原图 file_url，退到 large_file_url。"""
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

def convert_to_webp(src_path: str, dst_path: str, quality: int = 85) -> bool:
    if not PIL_AVAILABLE:
        return False
    try:
        with Image.open(src_path) as im:
            # 统一转 RGB 保存（若是 RGBA/P 等）
            if im.mode not in ("RGB",):
                im = im.convert("RGB")
            im.save(dst_path, "WEBP", quality=quality, method=6)
        return True
    except Exception:
        return False

def should_keep_post_as_webp(post: Dict, only_webp: bool) -> bool:
    """only_webp=True 时，只接受 file_ext 为 webp（或直链扩展为 webp）。"""
    if not only_webp:
        return True
    ext = str(post.get("file_ext", "")).lower()
    if ext == "webp":
        return True
    url = pick_best_url(post)
    return (url is not None) and (ext_from_url(url) == "webp")

# --------------- 标签与清单 ---------------

def split_tags(s: Optional[str]) -> List[str]:
    if not s:
        return []
    return [t for t in s.strip().split() if t]

def build_post_url(post_id: int) -> str:
    return f"https://danbooru.donmai.us/posts/{post_id}"

def write_json_sidecar(post: Dict, image_path: str, final_ext: str):
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
        }
    }
    with open(js_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def append_manifest_row(manifest_path: str, post: Dict, saved_path: str, final_ext: str):
    exists = os.path.exists(manifest_path)
    fieldnames = [
        "id","post_url","md5","rating","score","fav_count","source",
        "image_width","image_height","file_size","created_at","uploader_id",
        "saved_path","saved_ext",
        "tags_general","tags_character","tags_copyright","tags_artist","tags_meta","tags_all"
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
    }
    with open(manifest_path, "a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            w.writeheader()
        w.writerow(row)

# --------------- API 抓取 ---------------

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

    # 先按用户提供的认证请求
    try:
        return _do((login, api_key) if (login and api_key) else None)
    except requests.HTTPError as e:
        status = getattr(e.response, "status_code", None)
        if status in (401, 403):
            print("认证失败或权限不足（401/403）。将以匿名方式重试一次……", file=sys.stderr)
            return _do(None)
        raise

# --------------- 主流程 ---------------

def main():
    parser = argparse.ArgumentParser("Danbooru WebP Downloader (with tags)")
    parser.add_argument("--tags", type=str, default="rating:safe",
                        help="Danbooru 搜索标签（空格分隔，如 '1girl solo rating:safe'）")
    parser.add_argument("--out", type=str, required=True, help="输出目录")
    parser.add_argument("--max-posts", type=int, default=200, help="最多抓取的帖子数")
    parser.add_argument("--login", type=str, default=None, help="Danbooru 用户名（可选）")
    parser.add_argument("--api-key", type=str, default=None, help="Danbooru API Key（可选）")

    parser.add_argument("--only-webp", action="store_true",
                        help="仅下载站上本身就是 webp 的文件")
    parser.add_argument("--convert-to-webp", action="store_true",
                        help="将 jpg/png 等转码为 webp 保存")
    parser.add_argument("--quality", type=int, default=85, help="webp 质量（转码时生效）")

    parser.add_argument("--workers", type=int, default=8, help="下载并发数")
    parser.add_argument("--resume", action="store_true",
                        help="开启断点续抓（输出目录生成 state.json）")
    parser.add_argument("--request-interval", type=float, default=None,
                        help="API 请求间隔秒（默认：匿名 1.0；登录 0.35）")
    parser.add_argument("--save-json", action="store_true",
                        help="为每张图片写同名 .json（完整 tags 与元数据）")
    parser.add_argument("--manifest", action="store_true",
                        help="生成/追加 manifest.csv 汇总")
    parser.add_argument("--debug", action="store_true", help="打印调试信息（请求 URL 等）")

    args = parser.parse_args()

    if args.convert_to_webp and not PIL_AVAILABLE:
        print("未检测到 Pillow；请先 `pip install pillow` 或关闭 --convert-to-webp", file=sys.stderr)
        sys.exit(1)

    ensure_dir(args.out)
    state_path = os.path.join(args.out, "state.json") if args.resume else None
    last_id = load_state(state_path)

    s_meta = build_session()
    interval = args.request_interval
    if interval is None:
        interval = 0.35 if (args.login and args.api_key) else 1.0

    print(f">> tags: {args.tags}")
    print(f">> out : {args.out}")
    print(f">> max : {args.max_posts}")
    print(f">> mode: only_webp={args.only_webp}, convert_to_webp={args.convert_to_webp}, "
          f"workers={args.workers}, save_json={args.save_json}, manifest={args.manifest}")
    if args.login and args.api_key:
        print(">> auth: using login + api-key")

    to_download: List[Tuple[Dict, str, str]] = []
    fetched = 0
    oldest_id = last_id

    while fetched < args.max_posts:
        need = min(100, args.max_posts - fetched)
        try:
            posts = api_fetch_posts(
                s_meta, args.tags, page_before_id=oldest_id, limit=need,
                login=args.login, api_key=args.api_key, debug=args.debug
            )
        except requests.HTTPError as e:
            print(f"请求失败: {e}", file=sys.stderr)
            break

        if not posts:
            print("没有更多结果。")
            break

        for p in posts:
            if not should_keep_post_as_webp(p, args.only_webp):
                continue
            url = pick_best_url(p)
            if not url:
                continue

            final_ext = "webp" if args.convert_to_webp else None
            fname = build_filename(p, url, force_ext=final_ext)
            dst = os.path.join(args.out, fname)
            if os.path.exists(dst):
                # 已存在目标文件则可选写 sidecar/manifest（保证幂等）
                if args.save_json:
                    write_json_sidecar(p, dst, (final_ext or ext_from_url(url)))
                if args.manifest:
                    append_manifest_row(os.path.join(args.out, "manifest.csv"), p, dst,
                                        (final_ext or ext_from_url(url)))
                continue

            to_download.append((p, url, dst))

        fetched += len(posts)
        oldest_id = min(int(p["id"]) for p in posts)
        if state_path:
            save_state(state_path, oldest_id)

        # 控制抓取节奏，避免触发限流
        time.sleep(interval)

        if len(to_download) >= args.max_posts:
            to_download = to_download[:args.max_posts]
            break

    if not to_download:
        print("没有可下载的条目。")
        return

    print(f">> 待下载: {len(to_download)} 个文件")

    # 下载会话分离（与 meta 抓取分开）
    s = build_session()
    manifest_path = os.path.join(args.out, "manifest.csv") if args.manifest else None

    def after_success(p: Dict, saved_path: str, final_ext: str):
        if args.save_json:
            write_json_sidecar(p, saved_path, final_ext)
        if manifest_path:
            append_manifest_row(manifest_path, p, saved_path, final_ext)

    def worker(item: Tuple[Dict, str, str]) -> bool:
        p, url, dst = item
        if args.convert_to_webp:
            tmp_src = dst + ".orig"
            if (not os.path.exists(tmp_src)) and (not download_one(s, url, tmp_src)):
                return False
            if convert_to_webp(tmp_src, dst, quality=args.quality):
                try:
                    os.remove(tmp_src)
                except Exception:
                    pass
                after_success(p, dst, "webp")
                return True
            else:
                # 转码失败：保留原始扩展名
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
                after_success(p, dst, ext_from_url(url))
                return True
            return False

    ok = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(worker, item): item for item in to_download}
        for fut in as_completed(futures):
            success = fut.result()
            ok += 1 if success else 0
            if ok % 25 == 0:
                print(f"... 已完成 {ok}/{len(to_download)}")

    print(f">> 完成：{ok}/{len(to_download)}")
    if state_path:
        print(f">> 断点位（最旧 post_id）保存在: {state_path}")

if __name__ == "__main__":
    main()
