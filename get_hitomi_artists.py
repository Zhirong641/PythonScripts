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
    # "https://hitomi.la/artist/hiiragi%20ringo-all.html",
    # "https://hitomi.la/artist/kimishima%20ao-all.html",
    # "https://hitomi.la/artist/koku-all.html",
    # "https://hitomi.la/artist/mikagami%20mamizu-all.html",
    # "https://hitomi.la/artist/mutou%20kurihito-all.html",
    # "https://hitomi.la/artist/nanaca%20mai-all.html",
    "https://hitomi.la/artist/shiratama-all.html",
    # "https://hitomi.la/group/akabei%20soft3-all.html",
    # "https://hitomi.la/group/alcot-all.html",
    # "https://hitomi.la/group/applique-all.html",
    # "https://hitomi.la/group/asa%20project-all.html",
    # "https://hitomi.la/group/astronauts-all.html",
    # "https://hitomi.la/group/campus-all.html",
    # "https://hitomi.la/group/crystalia-all.html",
    # "https://hitomi.la/group/cube-all.html",
    # "https://hitomi.la/group/escude-all.html",
    # "https://hitomi.la/group/favorite-all.html",
    # "https://hitomi.la/group/feng-all.html",
    # "https://hitomi.la/group/galette-all.html",
    # "https://hitomi.la/group/giga-all.html",
    # "https://hitomi.la/group/hook-all.html",
    # "https://hitomi.la/group/hulotte-all.html",
    # "https://hitomi.la/group/lass-all.html",
    # "https://hitomi.la/group/lose-all.html",
    # "https://hitomi.la/group/marmalade-all.html",
    # "https://hitomi.la/group/minori-all.html",
    # "https://hitomi.la/group/mirai-all.html",
    # "https://hitomi.la/group/palette-all.html",
    # "https://hitomi.la/group/parasol-all.html",
    # "https://hitomi.la/group/saga%20planets-all.html",
    # "https://hitomi.la/group/smee-all.html",
    # "https://hitomi.la/group/tinkle%20position-all.html",
    # "https://hitomi.la/group/unisonshift-all.html",
    # "https://hitomi.la/group/windmill-all.html",
    # "https://hitomi.la/group/yuzu%20soft-all.html",
    # "https://hitomi.la/search.html?artist%3Aoryou%20type%3Agamecg",
    # "https://hitomi.la/search.html?laplacian",
    # "https://hitomi.la/search.html?moonstone",
    # "https://hitomi.la/search.html?pulltop",
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

# Path to the ChromeDriver
chrome_driver_path = "./chromedriver-linux64/chromedriver"  # Update this path

# Initialize the WebDriver
# service = Service(executable_path=chrome_driver_path)
service = Service(ChromeDriverManager().install())  # Automatically manage ChromeDriver
drive = webdriver.Chrome(service=service, options=chrome_options)

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
            drive.get(base_url)
            # 等待页面加载完成
            time.sleep(5)
            next_page = drive.find_elements(By.CSS_SELECTOR, "div.page-container.page-top a")
            items = drive.find_elements(By.CSS_SELECTOR, "div.gallery-content h1.lillie a")
            artist_lists = drive.find_elements(By.CSS_SELECTOR, "div.gallery-content div.artist-list")
            if not items or items[0].text == "" or not artist_lists or len(items) != len(artist_lists):
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
                    drive.get(url)
                    # 等待页面加载完成
                    time.sleep(5)
                    items = drive.find_elements(By.CSS_SELECTOR, "div.gallery-content h1.lillie a")
                    artist_lists = drive.find_elements(By.CSS_SELECTOR, "div.gallery-content div.artist-list")
                    if not items or items[0].text == "" or not artist_lists or len(items) != len(artist_lists):
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
                    name = artist.get_attribute("textContent").strip()
                    if name != "...":
                        artist_names += name + ", "
            artist_names = artist_names.strip(", ")
            title = item.text  # 获取项目标题
            link = item.get_attribute("href")    # 获取项目链接
            type = ''
            if i < len(descs):
                desc = descs[i]
                tds = desc.find_elements(By.TAG_NAME, "td")
                if len(tds) > 3:
                    type = tds[3].text
            # print(f"Title: {title}, Link: {link}, Type: {type}")
            if not type in allowded_type_list:
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

