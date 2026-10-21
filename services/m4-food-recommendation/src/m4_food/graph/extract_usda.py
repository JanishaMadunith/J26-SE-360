"""USDA FoodData Central Micronutrient Extractor for M4 Knowledge Graph.

Extracts real scientific measurements of stress-alleviating and cognitive nutrients:
- Magnesium (Nutrient ID 1090) -> Alleviates cortisol and stress
- Potassium (Nutrient ID 1092) -> Electrolyte & blood pressure support
- Protein (Nutrient ID 1003) -> Amino acids for mental stamina
- Energy (Nutrient ID 1008) -> Calories per 100g
- Polyunsaturated fats (Nutrient ID 1293) -> Omega-3 fatty acid proxy
"""

from __future__ import annotations

import csv
import io
import json
import os
import zipfile
from pathlib import Path

DEFAULT_ZIP_PATH = r"c:\y4 s1\Knowledge graph testing\Datasets\Micronutrient & Stress-Relief Dataset\FoodData_Central_sr_legacy_food_csv_2018-04.zip"
OUTPUT_DIR = Path(__file__).parent / "data"
OUTPUT_FILE = OUTPUT_DIR / "usda_micronutrients.json"

TARGET_NUTRIENTS = {
    "1090": "magnesium_mg",
    "1092": "potassium_mg",
    "1003": "protein_g",
    "1008": "calories_kcal",
    "1293": "polyunsaturated_fat_g",
}

SEARCH_KEYWORDS = [
    "spinach", "salmon", "mackerel", "lentil", "chickpea", "oat", "walnut",
    "almond", "banana", "broccoli", "egg", "chicken", "yogurt", "rice",
    "chia", "avocado", "tuna", "blueberry", "tea", "cacao"
]


def extract_usda_nutrients(zip_path: str = DEFAULT_ZIP_PATH) -> dict:
    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"USDA zip not found at: {zip_path}")

    print(f"Opening USDA archive: {zip_path}")
    matched_foods = {}

    with zipfile.ZipFile(zip_path) as z:
        # 1. Find matching food items
        food_csv_name = [n for n in z.namelist() if n.split("/")[-1] == "food.csv"][0]
        with z.open(food_csv_name) as f:
            reader = csv.DictReader(io.TextIOWrapper(f, encoding="utf-8"))
            for row in reader:
                desc = row["description"].lower()
                for kw in SEARCH_KEYWORDS:
                    if kw in desc and not any(junk in desc for junk in ["babyfood", "candy", "fast foods"]):
                        fdc_id = row["fdc_id"]
                        matched_foods[fdc_id] = {
                            "fdc_id": fdc_id,
                            "description": row["description"],
                            "keyword": kw,
                            "nutrients": {},
                        }
                        break

        print(f"Matched {len(matched_foods)} relevant whole foods.")

        # 2. Extract target nutrients for matched foods
        nutrient_csv_name = [n for n in z.namelist() if n.split("/")[-1] == "food_nutrient.csv"][0]
        with z.open(nutrient_csv_name) as f:
            reader = csv.DictReader(io.TextIOWrapper(f, encoding="utf-8"))
            for row in reader:
                fdc_id = row["fdc_id"]
                n_id = row["nutrient_id"]
                if fdc_id in matched_foods and n_id in TARGET_NUTRIENTS:
                    try:
                        amt = float(row["amount"]) if row["amount"] else 0.0
                    except ValueError:
                        amt = 0.0
                    nutrient_name = TARGET_NUTRIENTS[n_id]
                    matched_foods[fdc_id]["nutrients"][nutrient_name] = amt

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
        json.dump(matched_foods, out, indent=2)

    print(f"Successfully exported {len(matched_foods)} enriched foods to: {OUTPUT_FILE}")
    return matched_foods


if __name__ == "__main__":
    extract_usda_nutrients()
