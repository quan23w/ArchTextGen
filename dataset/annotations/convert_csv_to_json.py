import csv
import json
import os
def convert_csv_to_json(csv_file, json_file_path):
    dataset = {}
    with open(csv_file, 'r', encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            image = row["filename"]
            caption = row["caption"]
            dataset[image] = caption
    with open(json_file_path, 'w', encoding="utf-8") as json_file:
        json.dump(dataset, json_file, indent=2,ensure_ascii=False)

if __name__ == "__main__":
    convert_csv_to_json("./image_metadata.csv", "./caption.json")
        
            
            