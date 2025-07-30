import time
import requests
# import imageio.v3 as iio
# from PIL import Image
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
import os
import re
import csv
from datetime import datetime
from CSVProcessor import CSVProcessor
# base_url = "https://hitomi.la/group/unisonshift-all.html"
base_urls = [
    "https://hitomi.la/group/cube-all.html"
]
allowded_type_list = ["Game CG", "Image Set"]

log = open("log.txt", 'w')
log.write(str(datetime.now()) + "\n")
log.flush()

csv_file_path = "cglist.csv"
csv_reader = CSVProcessor(csv_file_path, has_header=False)
csv_file = open(csv_file_path, 'a', newline='')
csv_writer = csv.writer(csv_file)

# Setup Chrome options
chrome_options = Options()
chrome_options.add_argument("--headless") # Run in headless mode (no GUI)
chrome_options.add_argument('--no-sandbox')
chrome_options.add_argument('--disable-dev-shm-usage')

# Path to the ChromeDriver
chrome_driver_path = "./chromedriver-linux64/chromedriver"  # Update this path

# Initialize the WebDriver
# service = Service(executable_path=chrome_driver_path)
service = Service(ChromeDriverManager().install())  # Automatically manage ChromeDriver
drive1 = webdriver.Chrome(service=service, options=chrome_options)
drive2 = webdriver.Chrome(service=service, options=chrome_options)

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
            drive1.get(base_url)
            # 等待页面加载完成
            time.sleep(6)
            next_page = drive1.find_elements(By.CSS_SELECTOR, "div.page-container.page-top a")
            items = drive1.find_elements(By.CSS_SELECTOR, "div.gallery-content h1.lillie a")
            if not items or items[0].text == "":
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
                    drive1.get(url)
                    # 等待页面加载完成
                    time.sleep(5)
                    items = drive1.find_elements(By.CSS_SELECTOR, "div.gallery-content h1.lillie a")
                    if not items or items[0].text == "":
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
        for i in range(len(items)):
            item = items[i]
            title = item.text  # 获取项目标题
            link = item.get_attribute("href")    # 获取项目链接
            type = ''
            if i < len(descs):
                desc = descs[i]
                tds = desc.find_elements(By.TAG_NAME, "td")
                if len(tds) > 3:
                    type = tds[3].text
            if os.path.exists("stop"):
                print("Stop download, exiting...")
                log.write("Stop download, exiting...\n")
                log.flush()
                drive1.quit()
                drive2.quit()
                exit()
            print(f"Title: {title}, Link: {link}, Type: {type}")
            if not type in allowded_type_list:
                print(f"No need to download type: {type}")
                continue
            match = re.search(r'-(\d+)\.html', link)
            if match:
                game_id = match.group(1)
                if os.path.exists(f"webp/{game_id}") or csv_reader.has_value_in_column_index(4, game_id):
                    print(f"Game {game_id} exists")
                    log.write(f"[INFO] Game {game_id} exists\n")
                    continue
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
                    csv_writer.writerow([base_url, title, link, type, game_id, img_index])
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

