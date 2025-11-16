import csv
import os
import time
import argparse
from typing import List, Tuple, Dict

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import WebDriverException
from webdriver_manager.chrome import ChromeDriverManager


HITOMI_BASE = "https://hitomi.la"


class DriverConnectionError(RuntimeError):
    pass


def is_driver_connection_error(exc: Exception) -> bool:
    """
    ChromeDriver 偶尔会失联，此时异常信息里会包含 localhost 连接超时。
    """
    msg = str(exc) if exc else ""
    msg_lower = msg.lower()
    if "httpconnectionpool" in msg_lower and "localhost" in msg_lower:
        return True
    if "chrome not reachable" in msg_lower:
        return True
    if "disconnected: not connected to devtools" in msg_lower:
        return True

    if isinstance(exc, WebDriverException):
        if exc.msg and "chrome not reachable" in exc.msg.lower():
            return True
    return False


def read_input_csv(path: str) -> List[Tuple[List[str], str]]:
    """
    读取输入 CSV 的每一行，返回 (整行, id) 列表。
    默认认为第一列是 id。
    """
    rows: List[Tuple[List[str], str]] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            id_str = row[0].strip()
            # 通常是纯数字，如果有奇怪的行可以在这里过滤
            if not id_str:
                continue
            rows.append((row, id_str))
    print(f"Loaded {len(rows)} rows from {path}")
    return rows


def load_existing_output(path: str) -> Dict[str, str]:
    """
    返回输出 CSV 中已经处理过的 id -> group 映射，用于增量跳过。
    """
    existing: Dict[str, str] = {}
    if not os.path.exists(path):
        return existing

    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            gid = row[0].strip()
            if not gid:
                continue
            group_value = row[2].strip() if len(row) > 2 else ""
            existing[gid] = group_value

    print(f"Detected {len(existing)} existing rows in {path}")
    return existing


def create_driver(headless: bool = True, driver_timeout: float = 120.0) -> webdriver.Chrome:
    chrome_options = Options()
    if headless:
        chrome_options.add_argument("--headless=new")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )

    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=chrome_options)
    driver.set_page_load_timeout(30)
    try:
        driver.command_executor.set_timeout(driver_timeout)
    except Exception:
        pass
    return driver


def wait_for_non_empty_href(
    driver: webdriver.Chrome, element, timeout: float = 8.0
) -> str:
    """
    等待元素的 href 属性出现且非空，减少因为空 href 而重新加载页面。
    """
    wait = WebDriverWait(driver, timeout)
    return wait.until(lambda _: element.get_attribute("href") or False)


def open_url_with_retry(
    driver: webdriver.Chrome,
    url: str,
    max_retries: int = 5,
    sleep_between: float = 3.0,
) -> bool:
    """
    通用的打开 URL 带重试，返回是否成功。
    """
    for attempt in range(1, max_retries + 1):
        try:
            print(f"[open] {url} (attempt {attempt}/{max_retries})")
            driver.get(url)
            return True
        except Exception as e:
            if is_driver_connection_error(e):
                raise DriverConnectionError(str(e)) from e
            print(f"  -> failed to open {url}: {e}")
            if attempt == max_retries:
                return False
            time.sleep(sleep_between)
    return False


def get_group_for_id(
    driver: webdriver.Chrome,
    gallery_id: str,
    max_retries: int = 5,
) -> str:
    reader_url = f"{HITOMI_BASE}/reader/{gallery_id}.html"

    for attempt in range(1, max_retries + 1):
        try:
            # 1) 打开 reader 页面
            if not open_url_with_retry(driver, reader_url, max_retries=3):
                raise RuntimeError("Failed to open reader page")

            # 2) 等待 Gallery Info 链接出现
            wait = WebDriverWait(driver, 15)
            gallery_link = wait.until(
                EC.presence_of_element_located(
                    (
                        By.XPATH,
                        "//a[contains(@class,'brand') and normalize-space()='Gallery Info']",
                    )
                )
            )

            href = gallery_link.get_attribute("href")
            if not href:
                href = wait_for_non_empty_href(driver, gallery_link, timeout=10.0)

            if href.startswith("/"):
                href = HITOMI_BASE + href

            # 3) 打开 Gallery Info 页面
            if not open_url_with_retry(driver, href, max_retries=3):
                raise RuntimeError("Failed to open gallery page")

            # 4) 等待 id="groups" 出现
            wait = WebDriverWait(driver, 15)
            group_td = wait.until(
                EC.presence_of_element_located((By.ID, "groups"))
            )

            raw_text = group_td.text.strip()

            # 尝试从链接列表提取多个 group，用逗号连接
            links = group_td.find_elements(By.TAG_NAME, "a")
            link_groups = [
                link.text.strip().lower() for link in links if link.text and link.text.strip()
            ]

            # === 关键逻辑修改在这里 ===
            if not raw_text and not link_groups:
                # 空字符串 -> 说明还没加载好，当作失败重试
                raise RuntimeError("Group text empty (page not fully loaded?)")

            if raw_text.upper() == "N/A":
                group_text = ""      # 只有 N/A 映射成空串
            elif link_groups:
                group_text = ",".join(link_groups)
            else:
                group_text = raw_text.lower()

            print(f"[id={gallery_id}] group = '{group_text}'")
            return group_text

        except Exception as e:
            if is_driver_connection_error(e):
                raise DriverConnectionError(str(e)) from e
            print(f"[id={gallery_id}] attempt {attempt}/{max_retries} failed: {e}")
            if attempt == max_retries:
                # 实在不行才放弃，这里仍然返回空串
                print(f"[id={gallery_id}] give up, set group = ''")
                return ""
            time.sleep(3.0)

    return ""


def main():
    parser = argparse.ArgumentParser(
        description="使用 Selenium 从 hitomi 获取每个 id 对应的 group，并在 CSV 中追加一列。"
    )
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="输入 CSV 路径（第一列为 id，第二列为 artist）",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="输出 CSV 路径（默认：在输入文件名后加 _with_group）",
    )
    parser.add_argument(
        "--no-headless",
        action="store_true",
        help="调试用：打开非 headless 模式浏览器窗口",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="每处理一个 id 之间的延迟（秒），默认 1.0，可调小提高速度",
    )
    parser.add_argument(
        "--driver-timeout",
        type=float,
        default=10.0,
        help="与 ChromeDriver 通讯的超时时间（秒），默认 10，可调小以更快检测失联",
    )
    args = parser.parse_args()

    input_path = args.input
    if args.output is None:
        root, ext = os.path.splitext(input_path)
        output_path = f"{root}_with_group{ext}"
    else:
        output_path = args.output

    rows = read_input_csv(input_path)

    existing_groups = load_existing_output(output_path)
    existing_ids = set(existing_groups.keys())
    remaining_rows = sum(1 for _, gid in rows if gid not in existing_ids)

    if remaining_rows == 0:
        print(
            f"All {len(rows)} rows already processed in {output_path}. Nothing to do."
        )
        return

    print(
        f"{remaining_rows} rows need processing "
        f"(skipping {len(rows) - remaining_rows} existing rows)."
    )

    driver = None
    delay_between_rows = max(args.delay, 0.0)
    driver_timeout = max(args.driver_timeout, 5.0)

    try:
        driver = create_driver(headless=not args.no_headless, driver_timeout=driver_timeout)
        groups_cache: Dict[str, str] = dict(existing_groups)
        file_mode = "a" if os.path.exists(output_path) else "w"

        with open(output_path, file_mode, newline="", encoding="utf-8") as f_out:
            writer = csv.writer(f_out)
            total = len(rows)

            for idx, (row, gid) in enumerate(rows, start=1):
                print(f"\n===== Row {idx}/{total}  id={gid} =====")
                if gid in existing_ids:
                    print(
                        f"[skip] id={gid} already exists in {output_path}, skip fetching"
                    )
                    continue

                if gid in groups_cache:
                    group = groups_cache[gid]
                else:
                    restart_attempts = 0
                    while True:
                        try:
                            group = get_group_for_id(driver, gid)
                            groups_cache[gid] = group
                            if delay_between_rows > 0:
                                time.sleep(delay_between_rows)
                            break
                        except DriverConnectionError as drv_err:
                            restart_attempts += 1
                            print(
                                f"[driver] connection lost while fetching id={gid}: {drv_err}"
                            )
                            if driver is not None:
                                try:
                                    driver.quit()
                                except Exception:
                                    pass
                            if restart_attempts >= 3:
                                print(
                                    f"[driver] failed to recover after {restart_attempts} attempts, skip id={gid}"
                                )
                                group = ""
                                groups_cache[gid] = group
                                break
                            print("[driver] recreating Chrome driver and retrying...")
                            driver = create_driver(
                                headless=not args.no_headless, driver_timeout=driver_timeout
                            )
                            continue

                base_cols = row[:2]
                extended_row = base_cols + [group]
                writer.writerow(extended_row)

                if idx % 10 == 0:
                    print(f"Processed {idx}/{len(rows)} rows")

        print(f"Done. Output written to: {output_path}")

    finally:
        if driver is not None:
            try:
                driver.quit()
            except Exception:
                pass


if __name__ == "__main__":
    main()
