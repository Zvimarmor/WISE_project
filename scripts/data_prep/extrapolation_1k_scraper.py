import wikipedia
import json
import re
import warnings
warnings.filterwarnings("ignore")

wikipedia.set_lang("en")

# Original + extra categories to reach 1000
BASE_CATEGORIES = [
    # Original
    "American novels",
    "films",
    "television series debuts",
    "podcast debuts",
    "restaurants established",
    "protected areas established",
    "museums established",
    "products introduced",
    "food and drink festivals",
    # Extra fallback
    "plays",
    "albums",
    "musicals",
    "comics",
    "animated films",
    "animated series",
    "children's novels",
    "cookbooks",
    "art exhibitions",
    "documentary films",
    "web series",
    "theatre productions",
]

YEARS = [2022, 2023, 2024]
TARGET_COUNT = 1000
OUTPUT_FILE = "data/custom_temporal/extrapolation_1k_dataset.json"

def is_valid_qualitative_summary(summary):
    if len(summary) < 200 or len(summary) > 2000:
        return False
    if "may refer to:" in summary or "disambiguation" in summary.lower():
        return False
    digit_density = len(re.findall(r'\d', summary)) / len(summary)
    if digit_density > 0.025:
        return False
    return True

def clean_subject_name(title):
    return re.sub(r'\s*\(.*?\)\s*', '', title).strip()

def infer_year(text):
    for year in [2024, 2023, 2022]:
        if str(year) in text:
            return year
    return None

def scrape_extrapolation_data():
    import requests
    S = requests.Session()
    S.headers.update({"User-Agent": "WISE_Project_Dataset_Bot/1.0 (zvi.marmor@huji.ac.il) python-requests/2.32.5"})
    URL = "https://en.wikipedia.org/w/api.php"

    # RESUME: load existing data and skip already-seen subjects
    try:
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            dataset = json.load(f)
        seen_subjects = set(item['subject'].lower() for item in dataset)
        print(f"Resuming from {len(dataset)} existing stories. Need {TARGET_COUNT - len(dataset)} more.")
    except FileNotFoundError:
        dataset = []
        seen_subjects = set()
        print("Starting fresh.")

    failed_density = 0

    for year in YEARS:
        for cat in BASE_CATEGORIES:
            if len(dataset) >= TARGET_COUNT:
                break
            category_name = f"Category:{year} {cat}"
            print(f"Scraping {category_name} ({len(dataset)}/{TARGET_COUNT})...")

            PARAMS = {
                "action": "query",
                "list": "categorymembers",
                "cmtitle": category_name,
                "cmlimit": "500",
                "cmtype": "page",
                "format": "json"
            }
            try:
                response = S.get(url=URL, params=PARAMS).json()
                pages = response.get('query', {}).get('categorymembers', [])

                for page in pages:
                    if len(dataset) >= TARGET_COUNT:
                        break
                    title = page['title']
                    clean_sub = clean_subject_name(title)
                    if clean_sub.lower() in seen_subjects:
                        continue
                    try:
                        summary = wikipedia.summary(title, auto_suggest=False)
                        if is_valid_qualitative_summary(summary):
                            if clean_sub.lower() in summary.lower() or title.lower() in summary.lower():
                                prompt_break = -1
                                for keyword in [" is ", " was "]:
                                    idx = summary.find(keyword)
                                    if idx != -1:
                                        prompt_break = idx + len(keyword)
                                        break
                                if prompt_break > 0 and prompt_break < 100:
                                    combined = summary
                                    dataset.append({
                                        "index": len(dataset),
                                        "year": infer_year(combined),
                                        "subject": clean_sub,
                                        "prompt": summary[:prompt_break],
                                        "target_new": summary[prompt_break:]
                                    })
                                    seen_subjects.add(clean_sub.lower())
                                    if len(dataset) % 50 == 0:
                                        print(f"Collected {len(dataset)} / {TARGET_COUNT} valid stories...")
                        else:
                            failed_density += 1
                    except (wikipedia.exceptions.DisambiguationError, wikipedia.exceptions.PageError):
                        pass
                    except Exception:
                        pass
            except Exception as e:
                print(f"Failed to fetch {category_name}: {e}")

        if len(dataset) >= TARGET_COUNT:
            break

    # Re-index everything
    for i, item in enumerate(dataset):
        item['index'] = i

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=4, ensure_ascii=False)

    print(f"\nDone. Total stories: {len(dataset)}. Density-discarded: {failed_density}.")
    print(f"Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    scrape_extrapolation_data()
