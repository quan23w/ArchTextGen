import re
import json
def clean_caption(raw):
    # Remove attribution (e.g., " / Firm Name")
    cleaned = re.sub(r"/\s?[^-]+-", "", raw)

    # Standardize tag formatting
    cleaned = cleaned.strip().lower().replace("photography", "").strip()
    return cleaned

def clean_json_file(input_path, output_path):
    with open(input_path, "r", encoding="utf-8") as f:
        captions = json.load(f)

    cleaned = {filename: clean_caption(caption) for filename, caption in captions.items()}

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(cleaned, f, indent=2, ensure_ascii=False)
        
clean_json_file("caption.json", "captions_cleaned.json")
