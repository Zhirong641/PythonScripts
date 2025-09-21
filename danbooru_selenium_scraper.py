# -*- coding: utf-8 -*-
"""
Danbooru 网页抓取器（支持 Q/E，原图失败回退 sample，默认转为 WebP）
-------------------------------------------------------------------
依赖:
  pip install selenium requests beautifulsoup4 pillow

用法示例:
1) 匿名抓 100 张 '1girl solo'，转 webp + 保存 JSON + manifest
   python danbooru_selenium_scraper.py --tags "1girl solo" --max-posts 100 \
     --out ./dl_1girl --save-json --manifest --headless

2) 登录后抓显式内容（explicit），转 webp
   python danbooru_selenium_scraper.py --username YOUR_LOGIN --password YOUR_PASS \
     --tags "rating:explicit" --max-posts 50 --out ./dl_exp --save-json --headless

参数要点:
- 不传 --username/--password 则匿名；传了则用 Selenium 登录拿到 cookie 再抓。
- 默认启用“转 WebP”；若想保留原格式可加 --no-convert。
"""

import os, sys, time, json, csv, argparse, hashlib, re
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlencode

import requests
from bs4 import BeautifulSoup

# ---- 可选：图片转 WebP ----
try:
    from PIL import Image
    PIL_OK = True
except Exception:
    PIL_OK = False

# ---- Selenium（可选）----
from selenium import webdriver
from selenium.webdriver.chrome.options import Options as ChromeOptions
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

BASE = "https://danbooru.donmai.us"

# ================= 工具函数 =================

def ensure_dir(d: str):
    os.makedirs(d, exist_ok=True)

def md5(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()

def ext_from_url(url: str) -> str:
    u = url.split("?", 1)[0].split("#", 1)[0]
    return u.rsplit(".", 1)[-1].lower() if "." in u else "bin"

def build_requests_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": "DanbooruSeleniumScraper/2.0"})
    return s

def build_driver(headless: bool = False) -> webdriver.Chrome:
    opts = ChromeOptions()
    if headless:
        opts.add_argument("--headless=new")
    opts.add_argument("--disable-gpu")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--window-size=1280,800")
    opts.add_argument("--lang=en-US")
    driver = webdriver.Chrome(options=opts)
    return driver

def selenium_login_and_get_cookies(username: str, password: str, headless=False) -> List[dict]:
    driver = build_driver(headless=headless)
    driver.get(f"{BASE}/session/new")
    wait = WebDriverWait(driver, 20)
    name_in = wait.until(EC.presence_of_element_located((By.NAME, "name")))
    pwd_in  = driver.find_element(By.NAME, "password")
    name_in.clear(); name_in.send_keys(username)
    pwd_in.clear();  pwd_in.send_keys(password)
    submit = driver.find_element(By.CSS_SELECTOR, "form input[type=submit], form button[type=submit]")
    submit.click()
    # 认为登录成功的条件：页面不再是 /session/new，或出现 My Account
    wait.until(lambda d: "/session" not in d.current_url or "My Account" in d.page_source)
    cookies = driver.get_cookies()
    driver.quit()
    return cookies

def apply_cookies_to_session(s: requests.Session, cookies: List[dict]):
    for c in cookies:
        domain = c.get("domain") or "danbooru.donmai.us"
        path = c.get("path") or "/"
        s.cookies.set(c["name"], c["value"], domain=domain, path=path)

def download_binary(s: requests.Session, url: str, path: str, referer: str = None,
                    timeout: int = 30, retries: int = 3, backoff: float = 1.0) -> bool:
    """带 Referer 的下载器，403/5xx 指数退避重试"""
    headers = {"User-Agent": s.headers.get("User-Agent", "DanbooruScraper/2.0")}
    if referer:
        headers["Referer"] = referer

    for i in range(retries):
        try:
            r = s.get(url, stream=True, timeout=timeout, headers=headers)
            if r.status_code == 403 and "Referer" not in headers:
                # 部分 CDN 需要 Referer，兜底用站点主页
                headers["Referer"] = BASE + "/"
                r = s.get(url, stream=True, timeout=timeout, headers=headers)
            r.raise_for_status()
            tmp = path + ".part"
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(1 << 16):
                    if chunk:
                        f.write(chunk)
            os.replace(tmp, path)
            return True
        except requests.HTTPError:
            if i < retries - 1:
                time.sleep(backoff * (2 ** i))
                continue
            return False
        except Exception:
            if i < retries - 1:
                time.sleep(backoff * (2 ** i))
                continue
            return False

def convert_to_webp(src: str, dst: str, quality: int = 85) -> bool:
    if not PIL_OK:
        return False
    try:
        with Image.open(src) as im:
            if im.mode not in ("RGB",):
                im = im.convert("RGB")
            im.save(dst, "WEBP", quality=quality, method=6)
        return True
    except Exception:
        return False

def split_tags(s: Optional[str]) -> List[str]:
    if not s:
        return []
    return [t for t in s.strip().split() if t]

# ================= 解析函数 =================

def parse_listing_for_post_ids(html: str) -> List[int]:
    """列表页提取 post id"""
    soup = BeautifulSoup(html, "html.parser")
    ids: List[int] = []
    for art in soup.select("article.post-preview"):
        pid = art.get("data-id")
        if pid and pid.isdigit():
            ids.append(int(pid))
    if not ids:
        # 兜底: /posts/<id>
        for a in soup.select("a[href^='/posts/']"):
            m = re.match(r"^/posts/(\d+)$", a.get("href", ""))
            if m:
                v = int(m.group(1))
                if v not in ids:
                    ids.append(v)
    return ids

def parse_post_page(html: str) -> Dict:
    """
    详情页解析：
      - 原图 URL（#image-download-link 或 container 的 data-file-url / data-large-file-url）
      - sample URL（data-normal-file-url / <img id="image">）
      - 宽高（data-large-width/height）
      - 标签分类（#tag-list li.category-*）
    """
    soup = BeautifulSoup(html, "html.parser")
    data: Dict = {
        "file_url": None,
        "sample_url": None,
        "image_width": None,
        "image_height": None,
        "tags": {"general": [], "character": [], "copyright": [], "artist": [], "meta": []},
    }

    # 原图链接
    a = soup.select_one("a#image-download-link, a[href*='original']")
    if a and a.get("href"):
        href = a["href"]
        data["file_url"] = href if href.startswith("http") else BASE + href

    # 容器 data-xxx-url
    div = soup.select_one("#image-container, div.image-container")
    if div:
        # 原图候选
        for k in ("data-file-url", "data-large-file-url"):
            if not data["file_url"] and div.get(k):
                u = div[k]; data["file_url"] = u if u.startswith("http") else BASE + u
        # sample 候选
        for k in ("data-normal-file-url", "data-preview-file-url", "data-large-file-url"):
            if not data["sample_url"] and div.get(k):
                u = div[k]; data["sample_url"] = u if u.startswith("http") else BASE + u
        # 尺寸
        w = div.get("data-large-width") or div.get("data-width")
        h = div.get("data-large-height") or div.get("data-height")
        if w and w.isdigit(): data["image_width"] = int(w)
        if h and h.isdigit(): data["image_height"] = int(h)

    # <img id="image"> 作为 sample 兜底
    img = soup.select_one("img#image")
    if img and img.get("src") and not data["sample_url"]:
        u = img["src"]; data["sample_url"] = u if u.startswith("http") else BASE + u

    # 标签
    cat_map = {"category-0":"general","category-1":"artist","category-3":"copyright",
               "category-4":"character","category-5":"meta"}
    for li in soup.select("#tag-list li"):
        cls = [c for c in li.get("class", []) if c.startswith("category-")]
        if not cls: continue
        key = cat_map.get(cls[0])
        if not key: continue
        a = li.find("a", href=True)
        if not a: continue
        tag = a.get_text(strip=True).replace(" ", "_")
        if tag: data["tags"][key].append(tag)

    # 兜底：页面上的 data-tags 大串
    if not any(data["tags"].values()):
        m = re.search(r'data-tags="([^"]+)"', html)
        if m:
            all_tags = split_tags(m.group(1))
            data["tags"]["general"] = all_tags

    return data

# ================= 主流程 =================

def main():
    ap = argparse.ArgumentParser("Danbooru Selenium scraper (to WebP)")
    ap.add_argument("--username", help="Danbooru 登录名（可选）")
    ap.add_argument("--password", help="密码（可选，和 --username 一起用）")
    ap.add_argument("--headless", action="store_true", help="无头浏览器模式（登录时有用）")

    ap.add_argument("--tags", default="rating:safe", help="搜索标签（空格分隔）")
    ap.add_argument("--max-posts", type=int, default=100, help="最多下载数量")
    ap.add_argument("--out", required=True, help="输出目录")
    ap.add_argument("--delay", type=float, default=1.0, help="每个请求间隔秒（建议≥1.0，避免风控）")

    ap.add_argument("--only-webp", action="store_true", help="只保存原生 webp（会少很多）")
    # 默认启用转换为 webp；用 --no-convert 可关闭
    ap.add_argument("--convert-to-webp", dest="convert_to_webp", action="store_true", default=True,
                    help="将图片转为 webp（默认启用）")
    ap.add_argument("--no-convert", dest="convert_to_webp", action="store_false",
                    help="禁用 webp 转换，保留原格式")
    ap.add_argument("--quality", type=int, default=85, help="webp 质量（转码时生效）")
    ap.add_argument("--min-width", type=int, default=0, help="最小宽度过滤")
    ap.add_argument("--min-height", type=int, default=0, help="最小高度过滤")
    ap.add_argument("--save-json", action="store_true", help="为每张图片写 <basename>.json（标签与元数据）")
    ap.add_argument("--manifest", action="store_true", help="生成/追加 manifest.csv")
    args = ap.parse_args()

    if args.convert_to_webp and not PIL_OK:
        print("缺少 Pillow；请先 `pip install pillow`，或使用 --no-convert 关闭转码。", file=sys.stderr)
        sys.exit(1)

    ensure_dir(args.out)

    # 可选登录：提高可见范围（Q/E）
    s = build_requests_session()
    if args.username and args.password:
        print(">> Selenium 登录中 …")
        try:
            cookies = selenium_login_and_get_cookies(args.username, args.password, headless=args.headless)
            apply_cookies_to_session(s, cookies)
            print(">> 登录完成，已注入 Cookie")
        except Exception as e:
            print(">> 登录失败，改为匿名抓取：", e, file=sys.stderr)

    # 收集 post id（稳定翻页 b{last_id}）
    tags_q = "+".join(t for t in args.tags.split() if t)
    post_ids: List[int] = []
    last_id: Optional[int] = None
    while len(post_ids) < args.max_posts:
        params = {"tags": tags_q}
        if last_id:
            params["page"] = f"b{last_id}"
        url = f"{BASE}/posts?{urlencode(params)}"
        r = s.get(url, timeout=30)
        if not r.ok:
            print("列表页失败：", r.status_code)
            break
        ids = parse_listing_for_post_ids(r.text)
        if not ids:
            print("没有更多结果。")
            break
        for pid in ids:
            if pid not in post_ids:
                post_ids.append(pid)
        last_id = min(ids)
        print(f"... 收集到 {len(post_ids)} 个 id（last_id={last_id}）")
        if len(post_ids) >= args.max_posts:
            post_ids = post_ids[:args.max_posts]
            break
        time.sleep(args.delay)

    if not post_ids:
        print("没有可下载的条目。")
        return

    # manifest
    manifest_path = os.path.join(args.out, "manifest.csv") if args.manifest else None
    if manifest_path and not os.path.exists(manifest_path):
        with open(manifest_path, "w", encoding="utf-8", newline="") as f:
            csv.DictWriter(f, fieldnames=[
                "id","post_url","saved_path","saved_ext","image_width","image_height",
                "tags_general","tags_character","tags_copyright","tags_artist","tags_meta","tags_all"
            ]).writeheader()

    # 逐帖下载
    ok = 0
    for idx, pid in enumerate(post_ids, 1):
        post_url = f"{BASE}/posts/{pid}"
        r = s.get(post_url, timeout=30)
        if not r.ok:
            print(f"[{idx}/{len(post_ids)}] 详情页失败 {pid}: {r.status_code}")
            time.sleep(args.delay); continue

        parsed = parse_post_page(r.text)
        file_url   = parsed.get("file_url")
        sample_url = parsed.get("sample_url")

        # 分辨率过滤
        iw, ih = parsed.get("image_width"), parsed.get("image_height")
        if (args.min_width and (not iw or iw < args.min_width)) or \
           (args.min_height and (not ih or ih < args.min_height)):
            print(f"[{idx}] 分辨率不达标，跳过 {pid} ({iw}x{ih})")
            time.sleep(args.delay); continue

        # 选择 URL：优先原图，失败回退 sample
        chosen_url = file_url or sample_url
        if not chosen_url:
            print(f"[{idx}] 找不到可下载 URL（可能权限或防盗链）：{pid}")
            time.sleep(args.delay); continue

        ext = ext_from_url(chosen_url)
        if args.only_webp and ext != "webp":
            print(f"[{idx}] 非 webp，跳过 {pid}")
            time.sleep(args.delay); continue

        fname_base = f"{pid}_{md5(chosen_url)}"
        dst = os.path.join(args.out, (fname_base + ".webp") if args.convert_to_webp
                                          else (fname_base + f".{ext}"))

        success = True
        if not os.path.exists(dst):
            if args.convert_to_webp:
                # 先下原始，再转 webp
                tmp = dst + ".orig"
                # 先试 chosen（原图优先），失败换 sample
                if (not os.path.exists(tmp)) and (not download_binary(s, chosen_url, tmp, referer=post_url)):
                    if chosen_url != sample_url and sample_url:
                        if download_binary(s, sample_url, tmp, referer=post_url):
                            chosen_url = sample_url
                            ext = ext_from_url(chosen_url)
                        else:
                            success = False
                    else:
                        success = False
                if success:
                    if convert_to_webp(tmp, dst, quality=args.quality):
                        try: os.remove(tmp)
                        except Exception: pass
                    else:
                        # 转码失败：保留原格式
                        os.replace(tmp, os.path.join(args.out, f"{fname_base}.{ext}"))
                        success = False
            else:
                # 不转码：直接下，带 referer；失败回退 sample
                if not download_binary(s, chosen_url, dst, referer=post_url):
                    if chosen_url != sample_url and sample_url and \
                       download_binary(s, sample_url, dst, referer=post_url):
                        chosen_url = sample_url
                        ext = ext_from_url(chosen_url)
                    else:
                        success = False

        if success:
            # 组织标签与 sidecar
            tags = parsed["tags"]
            tags_all = list({t for cat in tags.values() for t in cat})
            if args.save_json:
                side = {
                    "id": pid,
                    "post_url": post_url,
                    "file_url": chosen_url,
                    "saved_path": os.path.abspath(dst),
                    "saved_ext": "webp" if args.convert_to_webp else ext,
                    "image_width": iw, "image_height": ih,
                    "tags": tags, "tags_all": tags_all
                }
                with open(os.path.splitext(dst)[0] + ".json", "w", encoding="utf-8") as f:
                    json.dump(side, f, ensure_ascii=False, indent=2)

            if manifest_path:
                with open(manifest_path, "a", encoding="utf-8", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=[
                        "id","post_url","saved_path","saved_ext","image_width","image_height",
                        "tags_general","tags_character","tags_copyright","tags_artist","tags_meta","tags_all"
                    ])
                    w.writerow({
                        "id": pid, "post_url": post_url,
                        "saved_path": os.path.abspath(dst),
                        "saved_ext": "webp" if args.convert_to_webp else ext,
                        "image_width": iw, "image_height": ih,
                        "tags_general": " ".join(tags.get("general", [])),
                        "tags_character": " ".join(tags.get("character", [])),
                        "tags_copyright": " ".join(tags.get("copyright", [])),
                        "tags_artist": " ".join(tags.get("artist", [])),
                        "tags_meta": " ".join(tags.get("meta", [])),
                        "tags_all": " ".join(tags_all),
                    })

            print(f"[{idx}] OK {pid} -> {os.path.basename(dst)}")
            ok += 1
        else:
            print(f"[{idx}] 下载失败 {pid}")

        time.sleep(args.delay)

    print(f">> 完成：{ok}/{len(post_ids)}")

if __name__ == "__main__":
    # 依赖：bs4 在上方使用，这里延迟导入，避免没装时报错不明显
    try:
        from bs4 import BeautifulSoup  # noqa
    except Exception:
        print("缺少依赖：beautifulsoup4。请先 `pip install beautifulsoup4`。", file=sys.stderr)
        sys.exit(1)
    main()
