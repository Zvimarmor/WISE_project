import requests
import json
import re
import os

CATEGORIES = [
    "Category:2023 films",
    "Category:2023 establishments in the United States",
    "Category:2024 video games"
]

SAMPLES_PER_CATEGORY = 20
OUTPUT_FILE = "data/custom_temporal/custom_temporal_2023.json"

HEADERS = {
    'User-Agent': 'ResearchBot/1.0 (research@example.com)'
}

def get_category_members(category_name, limit=50):
    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "list": "categorymembers",
        "cmtitle": category_name,
        "cmlimit": limit,
        "format": "json"
    }
    response = requests.get(url, params=params, headers=HEADERS)
    if response.status_code != 200:
        print(f"Error fetching category {category_name}: {response.status_code} - {response.text}")
        return []
        
    try:
        data = response.json()
        if 'query' in data and 'categorymembers' in data['query']:
            return [item['title'] for item in data['query']['categorymembers'] if item['ns'] == 0]
    except Exception as e:
        print(f"JSON Error: {e} - Response: {response.text[:200]}")
    return []

def get_page_summary(title):
    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "prop": "extracts",
        "exintro": True,
        "explaintext": True,
        "titles": title,
        "format": "json"
    }
    response = requests.get(url, params=params, headers=HEADERS)
    if response.status_code == 200:
        try:
            data = response.json()
            pages = data.get('query', {}).get('pages', {})
            for page_id, page_info in pages.items():
                if 'extract' in page_info:
                    return page_info['extract'].strip()
        except:
            pass
    return ""

def clean_summary(text):
    text = re.sub(r'\n+', ' ', text)
    return text.strip()

def main():
    print("Starting Wikipedia Scraper for Custom Temporal Data...")
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    dataset = []
    step = 0

    for category in CATEGORIES:
        print(f"\nScanning {category}...")
        titles = get_category_members(category, limit=SAMPLES_PER_CATEGORY * 3) 
        
        added_count = 0
        for title in titles:
            if added_count >= SAMPLES_PER_CATEGORY:
                break
                
            if title.startswith("List of") or "(disambiguation)" in title:
                continue
                
            summary = get_page_summary(title)
            summary = clean_summary(summary)
            
            if len(summary.split()) < 20 or len(summary.split()) > 200:
                continue
                
            split_idx = summary.find(' is ')
            if split_idx == -1:
                split_idx = summary.find(', ')
                
            if split_idx != -1 and split_idx < len(title) + 20: 
                prompt = summary[:split_idx+3].strip()
                target = summary[split_idx+3:]
                
                if not target.startswith(' '):
                    target = ' ' + target
                    
                entry = {
                    "step": step,
                    "subject": title,
                    "prompt": prompt,
                    "target_new": target
                }
                
                dataset.append(entry)
                step += 1
                added_count += 1
                print(f"  + Added: {title}")
            else:
                prompt = f"{title}"
                target = f" {summary[len(title):].strip()}" if summary.startswith(title) else f" {summary}"
                
                entry = {
                    "step": step,
                    "subject": title,
                    "prompt": prompt,
                    "target_new": target
                }
                dataset.append(entry)
                step += 1
                added_count += 1
                print(f"  * Added (Direct): {title}")

    print(f"\nScraping complete. Total stories extracted: {len(dataset)}")
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=4, ensure_ascii=False)
        
    print(f"Saved dataset to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
