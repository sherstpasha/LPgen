import os
import requests
from tqdm import tqdm

# Папка для сохранения
SAVE_DIR = "benchmarks"
os.makedirs(SAVE_DIR, exist_ok=True)

# GitHub API
API_URL = "https://api.github.com/repos/ozy4dm/lp-data-netlib/contents/mps_files"
RAW_BASE = "https://raw.githubusercontent.com/ozy4dm/lp-data-netlib/main/mps_files/"


def get_file_list():
    """Получаем список файлов через GitHub API"""
    r = requests.get(API_URL)
    r.raise_for_status()
    data = r.json()
    files = [item["name"] for item in data if item["name"].endswith(".mps")]
    return sorted(files)


def download_file(name):
    """Скачиваем один .mps файл"""
    url = RAW_BASE + name
    path = os.path.join(SAVE_DIR, name)
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        with open(path, "wb") as f:
            f.write(r.content)
        return True
    except Exception as e:
        print(f"Ошибка при загрузке {name}: {e}")
        return False


if __name__ == "__main__":
    files = get_file_list()
    print(f"Найдено {len(files)} задач в Netlib LP (mps_files)\n")

    for name in tqdm(files, desc="📥 Загрузка"):
        download_file(name)

    print(f"\nВсё готово! Файлы сохранены в ./{SAVE_DIR}")
