import time
import requests
import os
import re
import csv
import atexit
import argparse
import json
from pathlib import Path
from urllib.parse import urljoin, urlparse
from datetime import datetime
from hitomi_download_state import (
    read_duplicate_ids, read_legacy_ids, read_recorded_image_keys,
    read_verified_cumulative_ids,
)
from hitomi_targets import (
    canonical_gallery_type, parse_gg_routing, parse_hitomi_url,
    search_url_for_title, title_matches,
)
# base_url = "https://hitomi.la/group/unisonshift-all.html"
base_urls = [
      "https://hitomi.la/group/carol%20works-all.html",
      "https://hitomi.la/group/mado%20soft-all.html",
      "https://hitomi.la/group/canvas%20garden-all.html",
      "https://hitomi.la/group/animal%20herb-all.html",
      "https://hitomi.la/group/sprite-all.html",
      "https://hitomi.la/group/makura-all.html",
      "https://hitomi.la/search.html?meikura",
      "https://hitomi.la/search.html?koisuru%20kimochi%20no%20kasanekata",
      "https://hitomi.la/group/circus-all.html",
      "https://hitomi.la/group/bug%20system-all.html",
      "https://hitomi.la/group/asa%20project-all.html"



]
allowded_type_list = ["Game CG", "Image Set", "Artist CG"]
allowded_type_set = {t.lower() for t in allowded_type_list}

log = open("log.txt", 'a')
log.write(str(datetime.now()) + "\n")
log.flush()

output_csv_path = Path("hitomi_260303_diff.csv")
csv_file = None
csv_writer = None
recorded_image_keys = set()
chrome_driver_path = None

def make_driver():
    global chrome_driver_path, By, WebDriverWait
    try:
        from selenium import webdriver
        from selenium.webdriver.chrome.service import Service
        from selenium.webdriver.common.by import By
        from selenium.webdriver.chrome.options import Options
        from selenium.webdriver.support.ui import WebDriverWait
        from webdriver_manager.chrome import ChromeDriverManager
    except ImportError as exc:
        raise RuntimeError(
            "Listing and title searches require selenium and webdriver-manager"
        ) from exc

    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--disable-extensions")
    chrome_options.add_argument("--blink-settings=imagesEnabled=false")
    chrome_options.add_argument("--window-size=1280,900")
    chrome_options.page_load_strategy = "eager"
    if chrome_driver_path is None:
        chrome_driver_path = ChromeDriverManager().install()
    driver = webdriver.Chrome(service=Service(chrome_driver_path), options=chrome_options)
    driver.set_page_load_timeout(45)
    return driver

def write_image_row(row):
    global csv_file, csv_writer
    key = (str(row[4]), int(row[5]))
    if key in recorded_image_keys:
        return False
    if csv_file is None:
        output_csv_path.parent.mkdir(parents=True, exist_ok=True)
        csv_file = output_csv_path.open("a", newline="", encoding="utf-8")
        csv_writer = csv.writer(csv_file)
    csv_writer.writerow(row)
    csv_file.flush()
    recorded_image_keys.add(key)
    return True

def close_driver(driver):
    if driver is not None:
        try:
            driver.quit()
        except Exception as exc:
            print(f"Could not close Chrome cleanly: {exc}")

drive1 = None
drive2 = None

def cleanup():
    close_driver(drive1)
    close_driver(drive2)
    session.close()
    log.close()
    if csv_file is not None:
        csv_file.close()

atexit.register(cleanup)

def get_text(el):
    if el is None:
        return ""
    # Prefer rendered text (keeps original case), fallback to raw textContent
    text = (el.text or "").strip()
    # if not text:
    #     text = (el.get_attribute("innerText") or "").strip()
    if not text:
        text = (el.get_attribute("textContent") or "").strip()
    # Normalize whitespace
    return " ".join(text.split())

def wait_for_page_ready(driver, timeout=20):
    WebDriverWait(driver, timeout).until(
        lambda d: d.execute_script("return document.readyState") == "complete"
    )

def load_list_page(driver, url, timeout=20):
    driver.get(url)
    wait_for_page_ready(driver, timeout)
    # Trigger lazy rendering
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    time.sleep(0.5)
    driver.execute_script("window.scrollTo(0, 0);")
    # Search pages render asynchronously, including their empty-result state.
    def results_loaded(d):
        items = d.find_elements(By.CSS_SELECTOR, "div.gallery-content h1.lillie a")
        if items:
            non_empty = sum(1 for it in items if get_text(it))
            if non_empty >= min(3, len(items)):
                return "results"
        counts = d.find_elements(By.ID, "number-of-results")
        if counts and get_text(counts[0]).startswith("0 Result"):
            return "empty"
        return False
    return WebDriverWait(driver, timeout).until(results_loaded) == "results"

# 伪装请求头
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36",
    "Referer": "https://hitomi.la/",
}

session = requests.Session()

def completion_marker(game_id):
    return Path("webp_complete") / str(game_id)

def migrate_completion_markers():
    image_root = Path("webp")
    if not image_root.is_dir():
        return 0
    moved = 0
    for directory in image_root.iterdir():
        if not directory.is_dir():
            continue
        old_marker = directory / ".complete"
        if not old_marker.is_file():
            continue
        new_marker = completion_marker(directory.name)
        new_marker.parent.mkdir(exist_ok=True)
        if new_marker.exists():
            if new_marker.read_bytes() != old_marker.read_bytes():
                raise RuntimeError(f"Conflicting completion markers for {directory.name}")
            old_marker.unlink()
        else:
            old_marker.replace(new_marker)
        moved += 1
    return moved

def gallery_info(game_id):
    gallery_response = session.get(
        f"https://ltn.gold-usergeneratedcontent.net/galleries/{game_id}.js",
        timeout=(10, 30),
    )
    gallery_response.raise_for_status()
    gallery_match = re.fullmatch(
        r"\s*var galleryinfo\s*=\s*(\{.*\})\s*;?\s*",
        gallery_response.text,
        re.DOTALL,
    )
    if gallery_match is None:
        raise ValueError(f"Unexpected gallery metadata for {game_id}")
    info = json.loads(gallery_match.group(1))
    if str(info.get("id")) != game_id or not info.get("files"):
        raise ValueError(f"Missing gallery files for {game_id}")
    return info

def gallery_info_and_urls(game_id):
    info = gallery_info(game_id)
    gg_response = session.get(
        "https://ltn.gold-usergeneratedcontent.net/gg.js", timeout=(10, 30)
    )
    gg_response.raise_for_status()
    prefix, default_route, route_overrides = parse_gg_routing(gg_response.text)
    urls = []
    for image in info["files"]:
        image_hash = image["hash"]
        if re.fullmatch(r"[0-9a-f]{64}", image_hash) is None:
            raise ValueError(f"Unexpected image hash for {game_id}")
        slot = int(image_hash[-1] + image_hash[-3:-1], 16)
        subdomain = 1 + route_overrides.get(slot, default_route)
        urls.append(
            f"https://w{subdomain}.gold-usergeneratedcontent.net/"
            f"{prefix}{slot}/{image_hash}.webp"
        )
    return info, urls

def gallery_record_for_id(game_id):
    info = gallery_info(game_id)
    gallery_path = info.get("galleryurl") or f"/galleries/{game_id}.html"
    link = urljoin("https://hitomi.la/", gallery_path)
    kind = {"gamecg": "Game CG", "artistcg": "Artist CG", "imageset": "Image Set"}.get(
        info.get("type"), info.get("type", "")
    )
    return info, info.get("title") or "", link, kind

def download_gallery(base_url, title, link, kind, game_id):
    global drive2
    directory = Path("webp") / game_id
    for gallery_attempt in range(1, 4):
        try:
            _, image_urls = gallery_info_and_urls(game_id)
            directory.mkdir(parents=True, exist_ok=True)
            print(f"Downloading game {game_id}: {len(image_urls)} images", flush=True)
            for img_index, img_url in enumerate(image_urls, start=1):
                if os.path.exists("stop"):
                    raise SystemExit("Stop download")
                existing = [p for p in directory.glob(f"image_{img_index}.*")
                            if p.suffix != ".part" and p.stat().st_size > 0]
                if existing:
                    write_image_row([base_url, title, link, kind, game_id, img_index])
                    continue
                if not img_url.startswith("http"):
                    raise ValueError(f"Unexpected image URL: {img_url}")
                extension = Path(urlparse(img_url).path).suffix.lower()
                if extension not in {".webp", ".jpg", ".jpeg", ".png", ".gif", ".avif"}:
                    raise ValueError(f"Unexpected image extension: {img_url}")
                destination = directory / f"image_{img_index}{extension}"
                temporary = destination.with_name(destination.name + ".part")
                for request_attempt in range(1, 5):
                    try:
                        with session.get(img_url, headers=headers, timeout=(10, 40), stream=True) as response:
                            response.raise_for_status()
                            with temporary.open("wb") as output:
                                for chunk in response.iter_content(chunk_size=65536):
                                    if chunk:
                                        output.write(chunk)
                            if temporary.stat().st_size == 0:
                                raise ValueError("Empty image")
                            temporary.replace(destination)
                        write_image_row([base_url, title, link, kind, game_id, img_index])
                        break
                    except (requests.RequestException, OSError, ValueError) as exc:
                        temporary.unlink(missing_ok=True)
                        log.write(f"[WARN] Image {game_id}/{img_index} request {request_attempt}/4: {exc}\n")
                        log.flush()
                        if request_attempt == 4:
                            raise
                        time.sleep(request_attempt * 2)
            if not all(any(p.stat().st_size > 0 for p in directory.glob(f"image_{index}.*")
                           if p.suffix != ".part") for index in range(1, len(image_urls) + 1)):
                raise RuntimeError(f"Incomplete gallery {game_id}")
            if not all((game_id, index) in recorded_image_keys for index in range(1, len(image_urls) + 1)):
                raise RuntimeError(f"Missing image CSV rows for gallery {game_id}")
            marker = completion_marker(game_id)
            marker.parent.mkdir(exist_ok=True)
            marker.write_text(f"{len(image_urls)}\n", encoding="ascii")
            print(f"Completed game {game_id}: {len(image_urls)} images", flush=True)
            return True
        except Exception as exc:
            print(f"Game {game_id} attempt {gallery_attempt}/3 failed: {exc}", flush=True)
            log.write(f"[ERR] Game {game_id} attempt {gallery_attempt}/3 failed: {exc}\n")
            log.flush()
            if gallery_attempt < 3:
                time.sleep(gallery_attempt * 2)
    return False

def retry_failed_galleries(known_complete_ids):
    failed_path = Path("hitomi_failed.txt")
    if not failed_path.exists():
        return 0
    failed_ids = list(dict.fromkeys(failed_path.read_text(encoding="utf-8").split()))
    remaining = []
    for game_id in failed_ids:
        if completion_marker(game_id).exists() or game_id in known_complete_ids:
            continue
        try:
            info, title, link, kind = gallery_record_for_id(game_id)
            group = next((group.get("url") for group in info.get("groups") or []
                          if group.get("url")), None)
            base_url = "https://hitomi.la" + group if group else f"https://hitomi.la/reader/{game_id}.html"
            if download_gallery(base_url, title, link, kind, game_id):
                known_complete_ids.add(game_id)
            else:
                remaining.append(game_id)
        except Exception as exc:
            print(f"Could not retry game {game_id}: {exc}", flush=True)
            remaining.append(game_id)
    if remaining:
        failed_path.write_text("\n".join(remaining) + "\n", encoding="utf-8")
    else:
        failed_path.unlink()
    return len(remaining)

parser = argparse.ArgumentParser(description="Download Hitomi galleries")
parser.add_argument("--url", action="append",
                    help="Hitomi listing or gallery URL; may be repeated")
parser.add_argument("--id", dest="gallery_ids", action="append",
                    help="Download one gallery ID directly; may be repeated")
parser.add_argument("--title", dest="titles", action="append",
                    help="Search for this title and download matching results; may be repeated")
parser.add_argument("--title-match", choices=("exact", "contains"), default="exact",
                    help="How --title matches search results (default: exact)")
parser.add_argument("--type", dest="selected_types", action="append",
                    choices=allowded_type_list,
                    help="Restrict listing/title downloads to this type; may be repeated")
parser.add_argument("--list-only", action="store_true",
                    help="Show selected galleries without downloading images")
parser.add_argument("--output-csv", type=Path, default=output_csv_path,
                    help="CSV for newly downloaded images")
parser.add_argument("--retry-failed", action="store_true",
                    help="Retry galleries listed in hitomi_failed.txt")
parser.add_argument("--ids-csv", type=Path, default=Path("ids.csv"),
                    help="Optional legacy CSV with gallery IDs in column five")
parser.add_argument("--duplicates-csv", type=Path,
                    default=Path(__file__).resolve().parent / "config/hitomi_duplicate_gallery_ids.csv",
                    help="Known duplicate gallery IDs to skip")
default_data_root = Path(__file__).resolve().parent.parent / "data"
parser.add_argument("--existing-csv", type=Path,
                    default=default_data_root / "hitomi_260801.csv",
                    help="Existing cumulative six-column image CSV")
parser.add_argument("--existing-images-root", type=Path,
                    default=default_data_root / "webp",
                    help="Image root corresponding to --existing-csv")
args = parser.parse_args()
output_csv_path = args.output_csv
recorded_image_keys = set() if args.list_only else read_recorded_image_keys(output_csv_path)
active_type_set = (
    {value.lower() for value in args.selected_types}
    if args.selected_types else allowded_type_set
)
if args.retry_failed and (args.gallery_ids or args.titles or args.url or args.list_only):
    parser.error("--retry-failed cannot be combined with target selection or --list-only")
for game_id in args.gallery_ids or []:
    if not re.fullmatch(r"[1-9]\d*", game_id):
        parser.error(f"Invalid gallery ID: {game_id}")
for title in args.titles or []:
    if not title.strip():
        parser.error("--title must not be empty")

direct_url_ids = []
listing_urls = []
for url in args.url or []:
    try:
        target_kind, value = parse_hitomi_url(url)
    except ValueError as exc:
        parser.error(str(exc))
    if target_kind == "gallery":
        direct_url_ids.append(value)
    else:
        listing_urls.append(value)
direct_ids = list(dict.fromkeys((args.gallery_ids or []) + direct_url_ids))
list_targets = [(url, None) for url in listing_urls]
list_targets += [
    (search_url_for_title(title), title)
    for title in dict.fromkeys(args.titles or [])
]
explicit_selection = bool(args.url or args.gallery_ids or args.titles)
if not explicit_selection:
    list_targets = [(url, None) for url in base_urls]

migrated_markers = 0 if args.list_only else migrate_completion_markers()
if migrated_markers:
    print(f"Moved {migrated_markers} completion markers out of webp.", flush=True)
legacy_ids = read_legacy_ids(args.ids_csv)
cumulative_ids = read_verified_cumulative_ids(args.existing_csv, args.existing_images_root)
duplicate_ids = read_duplicate_ids(args.duplicates_csv)
known_complete_ids = legacy_ids | cumulative_ids | set(duplicate_ids)
print(
    f"Loaded {len(known_complete_ids)} gallery IDs to skip "
    f"({len(legacy_ids)} from optional ids.csv, "
    f"{len(cumulative_ids)} verified from cumulative images/CSV, "
    f"{len(duplicate_ids)} known duplicates); "
    "completion markers are also checked.",
    flush=True,
)
if not known_complete_ids and not any(Path("webp_complete").glob("*")):
    print("[WARN] No completed-ID source found; supply --existing-csv and "
          "--existing-images-root if prior downloads are elsewhere.", flush=True)

if args.retry_failed:
    remaining_failures = retry_failed_galleries(known_complete_ids)
    if remaining_failures:
        print(f"{remaining_failures} galleries remain in hitomi_failed.txt", flush=True)
    else:
        print("All failed galleries have been recovered.", flush=True)
    raise SystemExit(1 if remaining_failures else 0)

selection_errors = 0
for game_id in direct_ids:
    if game_id in duplicate_ids:
        print(f"Game {game_id} duplicates {duplicate_ids[game_id]}; skipping", flush=True)
        continue
    if not args.list_only and (completion_marker(game_id).exists() or game_id in known_complete_ids):
        print(f"Game {game_id} exists", flush=True)
        continue
    try:
        _, title, link, kind = gallery_record_for_id(game_id)
    except Exception as exc:
        print(f"Could not find gallery {game_id}: {exc}", flush=True)
        selection_errors += 1
        continue
    if args.list_only:
        print(f"ID {game_id}: {title} | {kind} | {link}", flush=True)
        continue
    if download_gallery(f"https://hitomi.la/reader/{game_id}.html", title, link, kind, game_id):
        known_complete_ids.add(game_id)
    else:
        with open("hitomi_failed.txt", "a", encoding="utf-8") as failures:
            failures.write(f"{game_id}\n")
        selection_errors += 1

for base_url, requested_title in list_targets:
    print(f"Processing base URL: {base_url}")
    log.write(f"[DBG] Processing base URL: {base_url}\n")
    log.flush()
    matched_title_count = 0
    eligible_title_count = 0
    # Open the webpage
    for i in range(10):
        try:
            print(f"Loading base URL: {base_url}, attempt {i+1}")
            if drive1 is None:
                drive1 = make_driver()
            has_results = load_list_page(drive1, base_url)
            if not has_results:
                items = []
                total_page_count = 1
                break
            next_page = drive1.find_elements(By.CSS_SELECTOR, "div.page-container.page-top a")
            items = drive1.find_elements(By.CSS_SELECTOR, "div.gallery-content h1.lillie a")
            if not items or get_text(items[0]) == "":
                print(f"No items found on base URL: {base_url}, retrying...")
                log.write(f"[ERR] No items found on base URL: {base_url}, retrying...\n")
                log.flush()
                continue
            total_page_count = 1
            for page in next_page:
                page_text = get_text(page)
                if page_text.isdigit():
                    total_page_count = max(total_page_count, int(page_text))
            result_counts = drive1.find_elements(By.ID, "number-of-results")
            if result_counts:
                count_match = re.match(r"([\d,]+)\s+Results?", get_text(result_counts[0]))
                if count_match:
                    result_count = int(count_match.group(1).replace(",", ""))
                    total_page_count = max(total_page_count, (result_count + 24) // 25)
            break
        except Exception as e:
            print(f"Failed to load base URL: {base_url}, attempt {i+1}, error: {e}")
            log.write(f"[ERR] Failed to load base URL: {base_url}, attempt {i+1}\n")
            log.flush()
            close_driver(drive1)
            drive1 = make_driver()
    else:
        print(f"Failed to load base URL after 10 attempts: {base_url}")
        log.write(f"[ERR] Skipping base URL: {base_url}\n")
        log.flush()
        selection_errors += 1
        continue

    print('Total page count: ' + str(total_page_count))
    page_number = 1
    url = ""
    while True:
        # 查找当前页面中的所有项目
        print(f"Page: {page_number}")
        
        if len(url) > 0:
            for i in range(5):
                try:
                    print(f"Loading Page {page_number} for {i}st times. url: {url}")
                    has_results = load_list_page(drive1, url)
                    if not has_results:
                        items = []
                        print(f"No results on page {page_number}")
                        break
                    items = drive1.find_elements(By.CSS_SELECTOR, "div.gallery-content h1.lillie a")
                    if not items or get_text(items[0]) == "":
                        print(f"No items found on page {page_number}, retrying...")
                        continue
                    print(f"Found {len(items)} items on page {page_number}")
                    break
                except Exception as e:
                    log.write(f"[ERR] Failed to find items, page: {page_number}, times: {i}\n")
                    log.flush()
                    close_driver(drive1)
                    drive1 = make_driver()
                    continue
            else:
                log.write(f"[ERR] Skipping failed page: {url}\n")
                log.flush()
                selection_errors += 1
                break

        try:
            descs = drive1.find_elements(By.CSS_SELECTOR, "div.gallery-content table.dj-desc")
        except Exception as e:
            log.write(f"[WARN] Failed to find descs, page: {page_number}\n")
            log.flush()
            descs = []
        for i in range(len(items)):
            item = items[i]
            title = get_text(item)  # 获取项目标题
            if not title:
                title = (item.get_attribute("title") or item.get_attribute("data-title") or "").strip()
            link = item.get_attribute("href")    # 获取项目链接
            if requested_title:
                candidate_titles = (
                    title,
                    item.get_attribute("title") or "",
                    item.get_attribute("data-title") or "",
                )
                if not any(value and title_matches(value, requested_title, args.title_match)
                           for value in candidate_titles):
                    continue
            if requested_title:
                matched_title_count += 1
            type = ''
            # Try to find desc table within the same gallery card as the item
            try:
                card = drive1.execute_script("return arguments[0].closest('div.gallery')", item)
                if card:
                    desc = card.find_element(By.CSS_SELECTOR, "table.dj-desc")
                    tds = desc.find_elements(By.TAG_NAME, "td")
                    if len(tds) > 3:
                        type = get_text(tds[3])
            except Exception:
                pass
            # Fallback to index-based mapping if needed
            if not type and i < len(descs):
                tds = descs[i].find_elements(By.TAG_NAME, "td")
                if len(tds) > 3:
                    type = get_text(tds[3])
            if args.list_only:
                print(f"Match: {title} | {type} | {link}", flush=True)
                continue
            if os.path.exists("stop"):
                print("Stop download, exiting...")
                log.write("Stop download, exiting...\n")
                log.flush()
                exit()
            print(f"Title: {title}, Link: {link}, Type: {type}")
            canonical_type = canonical_gallery_type(type, allowded_type_list)
            if not canonical_type or canonical_type.lower() not in active_type_set:
                print(f"No need to download type: {type}")
                continue
            if requested_title:
                eligible_title_count += 1
            match = re.search(r'-(\d+)\.html', link)
            if match:
                game_id = match.group(1)
                if game_id in duplicate_ids:
                    print(f"Game {game_id} duplicates {duplicate_ids[game_id]}; skipping")
                    continue
                if completion_marker(game_id).exists() or game_id in known_complete_ids:
                    print(f"Game {game_id} exists")
                    log.write(f"[INFO] Game {game_id} exists\n")
                    continue
                if download_gallery(base_url, title, link, canonical_type, game_id):
                    known_complete_ids.add(game_id)
                else:
                    with open("hitomi_failed.txt", "a", encoding="utf-8") as failures:
                        failures.write(f"{game_id}\n")
            else:
                print(f"[WARN] No game ID found in link: {link}")
                log.write(f"[WARN] No game ID found in link: {link}\n")
                log.flush()

        print("---------------")
        
        if page_number < total_page_count:
            page_number += 1
        else:
            break
        if "/search.html" in base_url:
            url = f"{base_url}#{page_number}"
        else:
            url = f"{base_url}?page={page_number}"
        # drive1.get(url)
        # # 等待页面加载完成
        # time.sleep(5)

    if requested_title:
        print(f"Title '{requested_title}': {matched_title_count} matching galleries", flush=True)
        if matched_title_count == 0 or (eligible_title_count == 0 and not args.list_only):
            selection_errors += 1

remaining_failures = selection_errors
if not explicit_selection and not args.list_only:
    remaining_failures += retry_failed_galleries(known_complete_ids)
if args.list_only:
    print(f"Listing finished with {remaining_failures} unresolved targets.")
elif remaining_failures:
    print(f"Finished with {remaining_failures} unresolved targets; check hitomi_failed.txt if downloads failed.")
    log.write(f"[ERR] Finished with {remaining_failures} unresolved targets.\n")
else:
    print("All images have been downloaded successfully.")
    log.write("All images have been downloaded successfully.\n")
log.flush()
raise SystemExit(1 if remaining_failures else 0)
