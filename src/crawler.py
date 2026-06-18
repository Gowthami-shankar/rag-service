# src/crawler.py
import requests
import time
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin
from urllib.robotparser import RobotFileParser

def crawl_site(start_url: str, max_pages: int = 30, crawl_delay_ms: int = 1000):
    """
    Crawls a website starting from a URL, respecting robots.txt and staying within the domain.
    """
    domain = urlparse(start_url).netloc
    queue = [start_url]
    visited = set()
    page_content = {}
    skipped_urls = set()

    robots_url = urljoin(start_url, '/robots.txt')
    rp = RobotFileParser()
    rp.set_url(robots_url)
    rp.read()

    print(f"Starting crawl of {domain}. Respecting robots.txt...")

    while queue and len(visited) < max_pages:
        url = queue.pop(0)
        if url in visited or url in skipped_urls:
            continue

        if not rp.can_fetch('*', url):
            print(f"Skipping (disallowed by robots.txt): {url}")
            skipped_urls.add(url)
            continue
        
        time.sleep(crawl_delay_ms / 1000)

        try:
            print(f"Crawling: {url}")
            response = requests.get(url, timeout=10, headers={'User-Agent': 'RAG-TakeHome-Bot/1.0'})
            response.raise_for_status()

            visited.add(url)
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            
            
            # 1. Define a list of semantic tags to search for, in order of priority.
            content_tags = ['article', 'main', '.main-content', '#content']
            
            main_content = None
            for tag in content_tags:
                main_content = soup.select_one(tag)
                if main_content:
                    break
            
            # 2. If no specific tag is found, fall back to the body,
            #    but first remove common boilerplate tags.
            if not main_content:
                main_content = soup.find('body')
                if main_content:
                    for boilerplate_tag in main_content.select('nav, footer, header, script, style, [class*="menu"], [class*="sidebar"]'):
                        boilerplate_tag.decompose()

            # 3. Extract clean text from the final selected content.
            if main_content:
                text = main_content.get_text(separator=' ', strip=True)
                page_content[url] = text

           

            for link in soup.find_all('a', href=True):
                href = link['href']
                absolute_url = urljoin(start_url, href)
                parsed_url = urlparse(absolute_url)
                clean_url = parsed_url._replace(query="", fragment="").geturl()

                if urlparse(clean_url).netloc == domain and clean_url not in visited and clean_url not in queue:
                    queue.append(clean_url)

        except requests.RequestException as e:
            print(f"Failed to fetch {url}: {e}")
            skipped_urls.add(url)

    return {"page_content": page_content, "crawled_count": len(page_content), "skipped_count": len(skipped_urls)}