import time
import requests
# import imageio.v3 as iio
# from PIL import Image
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import os
import re
import csv
from datetime import datetime
from CSVProcessor import CSVProcessor
# base_url = "https://hitomi.la/group/unisonshift-all.html"
base_urls = [
    "https://hitomi.la/group/crystalia-all.html",
    "https://hitomi.la/group/escude-all.html",
    "https://hitomi.la/group/asa%20project-all.html",
    "https://hitomi.la/search.html?artist%3Ago-1%20amane",
    "https://hitomi.la/search.html?artist%3Ago-1%20riruka",
    "https://hitomi.la/artist/k-ko-all.html",
    "https://hitomi.la/artist/bekotarou-all.html",
    "https://hitomi.la/group/monako-all.html",
]

allowded_type_list = ["Game CG", "Image Set", "Artist CG"]

log = open("check_results.txt", 'w')
log.write(str(datetime.now()) + "\n")
log.flush()

csv_file_path = "artists.csv"
id_artists = {}
if os.path.exists(csv_file_path):
    with open(csv_file_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        # next(reader)  # 跳过表头
        for row in reader:
            key, value = row
            id_artists[key] = value  # key:value 形式存入字典
print("Loaded artists len:", len(id_artists))

csv_file = open(csv_file_path, 'a', newline='')
csv_writer = csv.writer(csv_file)


# Setup Chrome options
chrome_options = Options()
chrome_options.add_argument("--headless") # Run in headless mode (no GUI)
chrome_options.add_argument('--no-sandbox')
chrome_options.add_argument('--disable-dev-shm-usage')
chrome_options.add_argument('--window-size=1920,3000')

# Path to the ChromeDriver
chrome_driver_path = "./chromedriver-linux64/chromedriver"  # Update this path

# Initialize the WebDriver
# service = Service(executable_path=chrome_driver_path)
service = Service(ChromeDriverManager().install())  # Automatically manage ChromeDriver
drive = webdriver.Chrome(service=service, options=chrome_options)

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
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    time.sleep(0.5)
    driver.execute_script("window.scrollTo(0, 0);")
    WebDriverWait(driver, timeout).until(
        EC.presence_of_all_elements_located((By.CSS_SELECTOR, "div.gallery-content h1.lillie a"))
    )
    def titles_loaded(d):
        items = d.find_elements(By.CSS_SELECTOR, "div.gallery-content h1.lillie a")
        if not items:
            return False
        non_empty = sum(1 for it in items if get_text(it))
        return non_empty >= min(3, len(items))
    WebDriverWait(driver, timeout).until(titles_loaded)

# 伪装请求头
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36",
    "Referer": "https://hitomi.la/",
}

for base_url in base_urls:
    print(f"Processing base URL: {base_url}")
    log.write(f"Processing base URL: {base_url}\n")
    # Open the webpage
    for i in range(10):
        try:
            print(f"Loading base URL: {base_url}, attempt {i+1}")
            load_list_page(drive, base_url)
            next_page = drive.find_elements(By.CSS_SELECTOR, "div.page-container.page-top a")
            items = drive.find_elements(By.CSS_SELECTOR, "div.gallery-content h1.lillie a")
            artist_lists = drive.find_elements(By.CSS_SELECTOR, "div.gallery-content div.artist-list")
            if not items or get_text(items[0]) == "" or not artist_lists or len(items) != len(artist_lists):
                print(f"No items found on base URL: {base_url}, retrying...")
                log.write(f"[ERR] No items found on base URL: {base_url}, retrying...\n")
                log.flush()
                continue
            total_page_count = 0
            for page in next_page:
                # print(page.text)
                if int(page.text) > total_page_count:
                    total_page_count = int(page.text)
            break
        except Exception as e:
            print(f"Failed to load base URL: {base_url}, attempt {i+1}, error: {e}")
            log.write(f"[ERR] Failed to load base URL: {base_url}, attempt {i+1}\n")
            log.flush()
            if i == 9:
                print("Failed to load base URL after 10 attempts, exiting...")
                exit()

    print('Total page count: ' + str(total_page_count + 1))
    page_number = 1
    url = ""
    while True:
        # 查找当前页面中的所有项目
        print(f"Page: {page_number}")
        
        if len(url) > 0:
            for i in range(100):
                try:
                    print(f"Loading Page {page_number} for {i}st times. url: {url}")
                    load_list_page(drive, url)
                    items = drive.find_elements(By.CSS_SELECTOR, "div.gallery-content h1.lillie a")
                    artist_lists = drive.find_elements(By.CSS_SELECTOR, "div.gallery-content div.artist-list")
                    if not items or get_text(items[0]) == "" or not artist_lists or len(items) != len(artist_lists):
                        print(f"No items found on page {page_number}, retrying...")
                        continue
                    print(f"Found {len(items)} items on page {page_number}")
                    break
                except Exception as e:
                    print(f"Error loading page {page_number}, attempt {i}: {e}")
                    continue

        try:
            descs = drive.find_elements(By.CSS_SELECTOR, "div.gallery-content table.dj-desc")
        except Exception as e:
            print(f"Error finding descriptions on page {page_number}: {e}")
            descs = []
        for i in range(len(items)):
            item = items[i]
            artist_list = artist_lists[i]
            try:
                artists = artist_list.find_elements(By.CSS_SELECTOR, "a")
            except Exception as e:
                artists = []
            artist_names = ""
            if artists:
                for artist in artists:
                    name = get_text(artist)
                    if name != "...":
                        artist_names += name + ", "
            artist_names = artist_names.strip(", ").lower()
            title = get_text(item)  # 获取项目标题
            if not title:
                title = (item.get_attribute("title") or item.get_attribute("data-title") or "").strip()
            link = item.get_attribute("href")    # 获取项目链接
            type = ''
            # Try to find desc table within the same gallery card as the item
            try:
                card = drive.execute_script("return arguments[0].closest('div.gallery')", item)
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
            # print(f"Title: {title}, Link: {link}, Type: {type}")
            if (not type) or (type not in allowded_type_list):
                continue
            match = re.search(r'-(\d+)\.html', link)
            if match:
                game_id = match.group(1)
                if game_id not in id_artists:
                    id_artists[game_id] = artist_names
                    csv_writer.writerow([game_id, artist_names])
                    csv_file.flush()
                    print(f"New artist found: {game_id}, {artist_names}")
            else:
                print(f"[WARN] No game ID found in link: {link}")
        
        if page_number < total_page_count:
            page_number += 1
        else:
            break
        if "/search.html" in base_url:
            url = f"{base_url}#{page_number}"
        else:
            url = f"{base_url}?page={page_number}"
        # drive.get(url)
        # # 等待页面加载完成
        # time.sleep(5)

drive.quit()
print("All done.")
log.close()
exit()
