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
    # "https://hitomi.la/artist/miyase%20mahiro-all.html",
    # "https://hitomi.la/artist/karory-all.html",
    # "https://hitomi.la/artist/oryou-all.html",
    # "https://hitomi.la/artist/minatsuki%20alumi-all.html",
    "https://hitomi.la/search.html?jewel%20princess%20reincarnation",
    "https://hitomi.la/search.html?angelic%20link",
    "https://hitomi.la/search.html?girls%20creation",
    "https://hitomi.la/search.html?muv-luv%20girls",
    "https://hitomi.la/search.html?fruits%20fulcute",
    "https://hitomi.la/search.html?twinkle%20star%20knights"

]
allowded_type_list = ["Game CG", "Image Set", "Artist CG"]
allowded_type_set = {t.lower() for t in allowded_type_list}

log = open("log.txt", 'w')
log.write(str(datetime.now()) + "\n")
log.flush()

csv_file_path = "ids.csv"
csv_reader = CSVProcessor(csv_file_path, has_header=False)
csv_file = open("hitomi_260303_diff.csv", 'a', newline='')
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
drive1 = webdriver.Chrome(service=service, options=chrome_options)
drive2 = webdriver.Chrome(service=service, options=chrome_options)

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
    WebDriverWait(driver, timeout).until(
        EC.presence_of_all_elements_located((By.CSS_SELECTOR, "div.gallery-content h1.lillie a"))
    )
    # Ensure titles are populated (textContent)
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
    log.write(f"[DBG] Processing base URL: {base_url}\n")
    log.flush()
    # Open the webpage
    for i in range(10):
        try:
            print(f"Loading base URL: {base_url}, attempt {i+1}")
            load_list_page(drive1, base_url)
            next_page = drive1.find_elements(By.CSS_SELECTOR, "div.page-container.page-top a")
            items = drive1.find_elements(By.CSS_SELECTOR, "div.gallery-content h1.lillie a")
            if not items or get_text(items[0]) == "":
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
                    load_list_page(drive1, url)
                    items = drive1.find_elements(By.CSS_SELECTOR, "div.gallery-content h1.lillie a")
                    if not items or get_text(items[0]) == "":
                        print(f"No items found on page {page_number}, retrying...")
                        continue
                    print(f"Found {len(items)} items on page {page_number}")
                    break
                except Exception as e:
                    log.write(f"[ERR] Failed to find items, page: {page_number}, times: {i}\n")
                    log.flush()
                    continue

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
            if os.path.exists("stop"):
                print("Stop download, exiting...")
                log.write("Stop download, exiting...\n")
                log.flush()
                drive1.quit()
                drive2.quit()
                exit()
            print(f"Title: {title}, Link: {link}, Type: {type}")
            type_key = type.lower()
            if (not type_key) or (type_key not in allowded_type_set):
                print(f"No need to download type: {type}")
                continue
            match = re.search(r'-(\d+)\.html', link)
            if match:
                game_id = match.group(1)
                if os.path.exists(f"webp/{game_id}") or csv_reader.has_value_in_column_index(4, game_id):
                    print(f"Game {game_id} exists")
                    log.write(f"[INFO] Game {game_id} exists\n")
                    continue
                os.makedirs(f"webp/{game_id}", exist_ok=True)
                title_base_url = f"https://hitomi.la/reader/{game_id}.html"
                drive2.get(title_base_url)
                time.sleep(5)
                try:
                    options = drive2.find_elements(By.CSS_SELECTOR, "#single-page-select option")
                except Exception as e:
                    print("Get options failed")
                    log.write(f"[WARN] Failed for downloading game: {game_id}\n")
                    log.flush()
                    continue
                img_index = 0
                print(f"Page {page_number}-{i}: Downloading images for game: {game_id}, images count: {len(options)}")
                log.write(f"[DBG] Page {page_number}-{i}: Downloading images for game: {game_id}, images count: {len(options)}\n")
                log.flush()
                for option in options:
                    img_index += 1
                    img_page = option.get_attribute("value")
                    drive2.get(title_base_url + "#" + option.get_attribute("value"))
                    # time.sleep(1)
                    try:
                        img_element = drive2.find_element(By.CSS_SELECTOR, "div#comicImages picture img")
                        img_url = img_element.get_attribute("src")
                    except Exception as e:
                        log.write(f"[WARN] Find img url {game_id}/{img_page} failed.\n")
                        log.flush()
                    # if img_index > 2:
                    #     break
                    for retry_count in range(4):
                        response = requests.get(img_url, headers=headers)
                        if response.status_code == 200:
                            img_extension = img_url.split('.')[-1]  # 获取图片的扩展名，例如 'webp'
                            img_name = f"image_{img_index}.{img_extension}"
                            # if os.path.exists(f"webp/{game_id}/{img_name}"):
                            #     print(f"Image {img_name} exists")
                            #     continue
                            with open(f"webp/{game_id}/{img_name}", 'wb') as handler:
                                handler.write(response.content)
                            csv_writer.writerow([base_url, title, link, type, game_id, img_index])
                            # print(f"Image {img_index} downloaded: {img_url}")
                            break
                        else:
                            print(f"Failed to download Image {img_index}: {title_base_url} - HTTP Status: {response.status_code}, retrying {retry_count+1}/4")
                            log.write(f"[WARN] Failed to download Image {img_index}: {title_base_url} - HTTP Status: {response.status_code}, retrying {retry_count+1}/4\n")
                            log.flush()
                            time.sleep(1)
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

drive1.quit()
drive2.quit()
print("All images have been downloaded successfully.")
log.write("All images have been downloaded successfully.\n")
log.close()
csv_file.close()
exit()
