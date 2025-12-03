import requests
from bs4 import BeautifulSoup
import json
import time
from urllib.parse import urlparse, urlunparse

# ===============================
# Helpers
# ===============================

def is_valid_article_url(url: str) -> bool:
    """Filter only real article URLs (no video/photo/podcast/tag...)."""
    invalid_patterns = [
        "video.vnexpress.net",
        "/video/",
        "/photo/",
        "/infographic/",
        "/podcast/",
        "/tag/",
        "/interactive/",
        "thong-ke",   # thống kê dạng bảng
        "dqdoc",      # quảng cáo
    ]
    return not any(p in url for p in invalid_patterns)

def normalize_url(url: str) -> str:
    """Remove fragment (#...) and query parameters from URL."""
    parsed = urlparse(url)
    return urlunparse((parsed.scheme, parsed.netloc, parsed.path, '', '', ''))

# ===============================
# Crawl list of article URLs
# ===============================

def get_article_links():
    headers = {"User-Agent": "Mozilla/5.0 (compatible; MyCrawler/1.0)"}
    links = []

    selectors = [
        "h3.title-news a[href]",
        "h2.title-news a[href]",
        "article.item-news a[href]",
        "div.thumb-art a[href]",
        "a.thumb[href]",
        "a.title-news[href]",
    ]

    for page in range(1, 6):
        url = "https://vnexpress.net/kinh-doanh" if page == 1 else f"https://vnexpress.net/kinh-doanh-p{page}"
        print(f"Fetching page {page} → {url}")
        try:
            resp = requests.get(url, headers=headers, timeout=10)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")

            page_links = []

            for sel in selectors:
                for a in soup.select(sel):
                    href = a.get("href", "").strip()
                    if href.startswith("/"):
                        href = "https://vnexpress.net" + href

                    href = normalize_url(href)

                    if is_valid_article_url(href):
                        page_links.append(href)

            print(f"  → Found {len(page_links)} links")
            links.extend(page_links)

            time.sleep(0.7)

        except Exception as e:
            print(f"Error fetching page {page}: {e}")

    # unique preserving order
    seen = set()
    uniq = []
    for l in links:
        if l not in seen:
            uniq.append(l)
            seen.add(l)

    return uniq

# ===============================
# Crawl single article
# ===============================

def crawl_article(url):
    headers = {"User-Agent": "Mozilla/5.0 (compatible; MyCrawler/1.0)"}

    try:
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        title = soup.select_one("h1.title-detail")
        date = soup.select_one("span.date, span.time")
        content = soup.select_one(".fck_detail")

        raw = content.get_text(" ", strip=True) if content else None

        if not raw:
            # Skip articles with empty content
            return None

        return {
            "title": title.get_text(strip=True) if title else None,
            "date": date.get_text(strip=True) if date else None,
            "content": raw,
            "url": url,
        }

    except Exception as e:
        print(f"Error crawling {url}: {e}")
        return None

# ===============================
# Main
# ===============================

def main():
    print("Collecting article links...")
    links = get_article_links()
    print(f"Total valid links: {len(links)}")

    articles = []

    for i, link in enumerate(links, 1):
        print(f"[{i}/{len(links)}] Crawling: {link}")
        data = crawl_article(link)
        if data:
            articles.append(data)
        else:
            print(f"  ✗ Skipped (no content)")

        time.sleep(0.7)

    # Save file
    out_file = "vnexpress_kinhdoanh.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(articles, f, ensure_ascii=False, indent=2)

    print("\n--- DONE ---")
    print(f"Saved {len(articles)} articles → {out_file}")

if __name__ == "__main__":
    main()
