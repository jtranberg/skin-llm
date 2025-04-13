import os
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
from PIL import Image
from io import BytesIO

# === CONFIG ===
TOPIC_URLS = [
    "https://dermnetnz.org/topics/normal-skin",
    "https://dermnetnz.org/topics/skin-anatomy",
    # Add more DermNet pages that represent healthy skin here
]

SAVE_DIR = "dataset/normal"
IMG_SIZE = (224, 224)
HEADERS = {"User-Agent": "Mozilla/5.0"}

# === SETUP ===
os.makedirs(SAVE_DIR, exist_ok=True)
log_success = []
log_failed = []

def resize_and_crop(image, size=(224, 224)):
    image = image.convert("RGB")
    image.thumbnail((max(size), max(size)))
    w, h = image.size
    left = (w - size[0]) / 2
    top = (h - size[1]) / 2
    right = (w + size[0]) / 2
    bottom = (h + size[1]) / 2
    return image.crop((left, top, right, bottom))

def scrape_from_global_image_index(max_pages=10, keyword_filter=None):
    print(f"\n🌐 Scraping DermNet global image index... (filter: {keyword_filter or 'none'})")

    for page in range(1, max_pages + 1):
        page_url = f"https://dermnetnz.org/images?page={page}"
        print(f"📥 Page {page}: {page_url}")

        try:
            res = requests.get(page_url, headers=HEADERS, timeout=10)
            soup = BeautifulSoup(res.content, "html.parser")
            img_tags = soup.find_all("img")

            found = False
            for img in tqdm(img_tags, desc=f"Page {page}"):
                alt = img.get("alt", "").lower()
                if keyword_filter and keyword_filter not in alt:
                    continue  # Skip if keyword doesn't match

                img_url = img.get("data-src") or img.get("src")
                if not img_url or any(ext in img_url for ext in [".svg", "logo", "icon"]):
                    continue

                if img_url.startswith("//"):
                    img_url = "https:" + img_url
                elif img_url.startswith("/"):
                    img_url = "https://dermnetnz.org" + img_url

                filename = os.path.join(SAVE_DIR, img_url.split("/")[-1].split("?")[0])
                if os.path.exists(filename):
                    continue

                try:
                    img_data = requests.get(img_url, headers=HEADERS).content
                    image = Image.open(BytesIO(img_data))

                    if image.size[0] < 100 or image.size[1] < 100:
                        raise ValueError("Image too small")

                    image = resize_and_crop(image, IMG_SIZE)
                    image.save(filename)
                    log_success.append(filename)
                    found = True
                except Exception as e:
                    log_failed.append((img_url, str(e)))

            if not found:
                print("⚠️ No usable images found on this page.")

        except Exception as e:
            print(f"❌ Error scraping page {page}: {e}")






# === RUN GLOBAL SCRAPER ===
scrape_from_global_image_index(max_pages=10, keyword_filter=None)

# === LOG RESULTS ===
print(f"\n✅ {len(log_success)} images saved to {SAVE_DIR}")
if log_failed:
    print(f"⚠️ {len(log_failed)} failures:")
    for url, err in log_failed[:5]:  # show a few errors
        print(f" - {url} => {err}")

with open("normal_scrape_success.txt", "w") as f:
    f.write("\n".join(log_success))

with open("normal_scrape_failed.txt", "w") as f:
    f.write("\n".join([f"{u} | {e}" for u, e in log_failed]))
