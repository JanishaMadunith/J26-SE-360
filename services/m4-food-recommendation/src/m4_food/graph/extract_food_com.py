"""Food.com Recipe Extractor & Allergen/Nutrient Mapper for M4 Expanded Knowledge Graph.

Streams recipes from RAW_recipes.csv, parses nutrition, tags, and ingredients,
infers meal types and allergens, and cross-references with USDA micronutrients.
"""

from __future__ import annotations

import ast
import csv
import json
from pathlib import Path

DATASETS_DIR = Path(r"c:\y4 s1\Knowledge graph testing\Datasets")
RAW_RECIPES_CSV = DATASETS_DIR / "Food.com" / "RAW_recipes.csv"
OUTPUT_JSON = Path(__file__).resolve().parent / "data" / "expanded_foods.json"

# Common allergen keyword indicators
ALLERGEN_KEYWORDS = {
    "gluten": ["flour", "wheat", "bread", "pasta", "noodles", "barley", "rye"],
    "dairy": ["milk", "butter", "cheese", "cream", "yogurt", "yoghurt", "ghee"],
    "egg": ["egg", "eggs", "mayonnaise"],
    "fish": ["fish", "salmon", "tuna", "cod", "mackerel", "anchovy", "halibut", "tilapia"],
    "shellfish": ["shrimp", "prawn", "crab", "lobster", "clam", "mussel"],
    "peanut": ["peanut", "peanuts", "peanut butter"],
    "nuts": ["almond", "walnut", "cashew", "pecan", "pistachio", "hazelnut"],
    "soy": ["soy", "soya", "tofu", "edamame", "soy sauce"],
    "coconut": ["coconut", "coconut milk", "coconut oil"],
}

# Targeted therapeutic ingredients mapped to nutrients
THERAPEUTIC_INGREDIENTS = {
    "magnesium": ["spinach", "kale", "beans", "chickpeas", "lentils", "oats", "almonds", "cashews", "dark chocolate", "avocado", "banana", "pumpkin seeds"],
    "omega-3": ["salmon", "mackerel", "tuna", "sardines", "walnuts", "flaxseed", "chia"],
    "complex-carb": ["oats", "brown rice", "quinoa", "sweet potato", "lentils", "beans", "whole wheat", "barley"],
    "high-protein": ["chicken", "turkey", "beef", "salmon", "tofu", "eggs", "lentils", "chickpeas", "greek yogurt"],
    "antioxidants": ["blueberries", "strawberries", "green tea", "spinach", "broccoli", "dark chocolate", "tomatoes"],
    "tryptophan": ["turkey", "chicken", "eggs", "cheese", "tofu", "oats", "salmon", "pumpkin seeds"],
}


def extract_curated_recipes(max_items: int = 60) -> list[dict]:
    print(f"Reading from: {RAW_RECIPES_CSV}")
    if not RAW_RECIPES_CSV.exists():
        raise FileNotFoundError(f"Cannot find {RAW_RECIPES_CSV}")

    extracted = []
    seen_names = set()

    with open(RAW_RECIPES_CSV, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            name = row.get("name", "").strip().title()
            if not name or name.lower() in seen_names:
                continue

            # Parse minutes
            try:
                minutes = int(row.get("minutes", 0))
            except ValueError:
                continue
            if minutes <= 0 or minutes > 60:
                continue

            # Parse tags
            raw_tags = row.get("tags", "[]")
            try:
                tags_list = ast.literal_eval(raw_tags)
            except Exception:
                tags_list = []
            tags_lower = [t.lower() for t in tags_list]

            # Parse nutrition: [calories, total_fat, sugar, sodium, protein, saturated_fat, carbohydrates]
            raw_nutr = row.get("nutrition", "[]")
            try:
                nutr_list = ast.literal_eval(raw_nutr)
                kcal = int(round(float(nutr_list[0])))
                fat_pdv = float(nutr_list[1])
                sugar_pdv = float(nutr_list[2])
                sodium_pdv = float(nutr_list[3])
                protein_pdv = float(nutr_list[4])
                carbs_pdv = float(nutr_list[6])
            except Exception:
                continue

            if kcal < 50 or kcal > 850:
                continue

            # Parse ingredients
            raw_ingr = row.get("ingredients", "[]")
            try:
                ingr_list = ast.literal_eval(raw_ingr)
            except Exception:
                ingr_list = []
            ingr_str = " ".join(ingr_list).lower()

            # Determine meal window
            meals = []
            if any(k in tags_lower for k in ("breakfast", "brunch", "morning")) or "oats" in ingr_str or "pancake" in name.lower() or "toast" in name.lower():
                meals.append("breakfast")
            if any(k in tags_lower for k in ("lunch", "sandwiches", "salads", "soups")) or "salad" in name.lower() or "soup" in name.lower():
                meals.append("lunch")
            if any(k in tags_lower for k in ("dinner", "main-dish", "entree")):
                meals.append("dinner")
            if any(k in tags_lower for k in ("snacks", "appetizers")) or kcal < 300:
                meals.append("snack")

            if not meals:
                meals = ["lunch", "dinner"] if kcal >= 350 else ["snack"]

            # Infer dietary tags
            is_vegan = "vegan" in tags_lower
            meat_keywords = ("chicken", "beef", "pork", "fish", "turkey", "bacon", "salmon", "tuna", "sausage", "ham", "lamb", "meat", "shrimp", "seafood", "duck")
            is_veg = is_vegan or "vegetarian" in tags_lower or not any(meat in ingr_str for meat in meat_keywords)
            non_halal_keywords = ("pork", "bacon", "ham", "lard", "wine", "beer", "rum", "sausage")
            is_halal = not any(nh in ingr_str for nh in non_halal_keywords)

            # Infer allergens
            allergens = []
            for alg, kws in ALLERGEN_KEYWORDS.items():
                if any(kw in ingr_str for kw in kws):
                    allergens.append(alg)

            # Infer therapeutic nutrient tags
            nutrient_tags = []
            for nut, kws in THERAPEUTIC_INGREDIENTS.items():
                if any(kw in ingr_str for kw in kws):
                    nutrient_tags.append(nut)

            if protein_pdv >= 25.0 and "high-protein" not in nutrient_tags:
                nutrient_tags.append("high-protein")
            if sugar_pdv > 35.0:
                nutrient_tags.append("high-sugar")
            if sodium_pdv > 30.0:
                nutrient_tags.append("high-sodium")
            if carbs_pdv >= 15.0 and sugar_pdv < 15.0 and "complex-carb" not in nutrient_tags:
                nutrient_tags.append("complex-carb")
            if sugar_pdv < 10.0 and carbs_pdv > 5.0 and "low-GI" not in nutrient_tags:
                nutrient_tags.append("low-GI")
            elif sugar_pdv > 40.0 and "high-GI" not in nutrient_tags:
                nutrient_tags.append("high-GI")

            # Must have at least 1 therapeutic nutrient to be useful for mental health
            if not nutrient_tags:
                continue

            seen_names.add(name.lower())
            extracted.append({
                "name": name,
                "portion": f"1 serving ({minutes} min prep)",
                "calories": kcal,
                "meals": meals,
                "diet_tags": {
                    "vegetarian": is_veg,
                    "vegan": is_vegan,
                    "halal": is_halal,
                },
                "allergens": allergens,
                "nutrients": nutrient_tags,
                "source": "Food.com",
                "ingredients_sample": ingr_list[:5],
            })

            if len(extracted) >= max_items:
                break

    print(f"Extracted {len(extracted)} high-quality global recipes from Food.com.")
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(extracted, f, indent=2)

    print(f"Saved expanded dataset to: {OUTPUT_JSON}")
    return extracted


if __name__ == "__main__":
    extract_curated_recipes(max_items=60)
