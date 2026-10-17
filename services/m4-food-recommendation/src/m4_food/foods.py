"""Small built-in food knowledge table (baseline). Calories are typical single-portion estimates for
planning, not clinical values. Replace/extend with a proper nutrition database (e.g. FAO/INFOODS, USDA
FoodData Central, Sri Lankan food composition tables) in the trained version.

Tags
  diet:      vegetarian, vegan, halal
  nutrition: high-protein, complex-carb, omega-3, magnesium, low-GI, high-GI, high-sodium, fried, high-sugar
  stimulant: caffeine
  allergens: dairy, egg, fish, gluten, nuts, soy, shellfish, peanut, coconut
"""

from __future__ import annotations

from dataclasses import dataclass

from wellness_contracts import MealType

B, L, D, S = MealType.breakfast, MealType.lunch, MealType.dinner, MealType.snack


@dataclass(frozen=True)
class Food:
    name: str
    portion: str
    calories: int
    meals: frozenset[MealType]
    tags: frozenset[str]


def F(name, portion, kcal, meals, tags) -> Food:  # noqa: N802 - short constructor for the table
    return Food(name, portion, kcal, frozenset(meals), frozenset(tags.split()))


FOODS: list[Food] = [
    # ---- Sri Lankan
    F(
        "Red rice with dhal curry and gotukola sambol",
        "1 plate",
        520,
        [L, D],
        "vegetarian vegan halal complex-carb low-GI high-protein magnesium coconut",
    ),
    F(
        "Red rice with fish curry and mallum",
        "1 plate",
        600,
        [L, D],
        "halal complex-carb omega-3 high-protein fish coconut",
    ),
    F("Red rice with chicken curry and beans", "1 plate", 650, [L, D], "halal complex-carb high-protein coconut"),
    F(
        "Mackerel (tin fish) curry with string hoppers",
        "10 string hoppers",
        480,
        [B, D],
        "halal omega-3 high-protein fish gluten",
    ),
    F(
        "String hoppers with kiri hodi and pol sambol",
        "10 string hoppers",
        420,
        [B, D],
        "vegetarian halal complex-carb coconut gluten",
    ),
    F("Kiribath with lunu miris", "2 pieces", 380, [B], "vegetarian vegan halal high-GI coconut"),
    F("Egg hoppers", "2 hoppers", 360, [B, D], "vegetarian halal high-protein egg coconut"),
    F("Pittu with coconut milk and dhal", "1 serving", 450, [B, D], "vegetarian vegan halal complex-carb coconut"),
    F("Kola kenda (herbal porridge)", "1 bowl", 180, [B], "vegetarian vegan halal low-GI magnesium coconut"),
    F(
        "Boiled mung beans (green gram) with coconut",
        "1 cup",
        220,
        [B, S],
        "vegetarian vegan halal high-protein low-GI magnesium coconut",
    ),
    F("Boiled chickpeas (kadala) tempered", "1 cup", 270, [S], "vegetarian vegan halal high-protein low-GI magnesium"),
    F("Jackfruit (polos) curry with red rice", "1 plate", 500, [L, D], "vegetarian vegan halal complex-carb coconut"),
    F("Soya meat curry with red rice", "1 plate", 560, [L, D], "vegetarian vegan halal high-protein complex-carb soy"),
    F(
        "Kankun (water spinach) stir-fry with rice",
        "1 plate",
        450,
        [L, D],
        "vegetarian vegan halal complex-carb magnesium",
    ),
    F("Curd with kithul treacle", "1 cup", 260, [S], "vegetarian halal high-protein high-sugar dairy"),
    F("King coconut water (thambili)", "1 nut", 60, [S], "vegetarian vegan halal coconut"),
    F("Papaya", "1 cup", 60, [B, S], "vegetarian vegan halal low-GI"),
    F("Banana (ambul)", "1 medium", 105, [B, S], "vegetarian vegan halal magnesium"),
    F("Ulundu vadai", "2 pieces", 300, [S], "vegetarian vegan halal fried high-protein"),
    F("Coconut roti with lunu miris", "2 roti", 420, [B, D], "vegetarian vegan halal gluten coconut"),
    # ---- general
    F(
        "Oats porridge with banana and nuts",
        "1 bowl",
        350,
        [B],
        "vegetarian halal complex-carb low-GI magnesium nuts dairy",
    ),
    F(
        "Wholegrain toast with peanut butter",
        "2 slices",
        330,
        [B, S],
        "vegetarian vegan halal high-protein gluten peanut",
    ),
    F("Yoghurt with mixed nuts", "1 cup", 280, [S], "vegetarian halal high-protein magnesium dairy nuts"),
    F("Roasted cashews", "30 g", 170, [S], "vegetarian vegan halal magnesium nuts"),
    F("Dark chocolate (70%+)", "2 squares", 120, [S], "vegetarian halal magnesium caffeine"),
    F("Grilled chicken salad", "1 bowl", 380, [L, D], "halal high-protein low-GI"),
    F(
        "Lentil soup with wholegrain bread",
        "1 bowl",
        400,
        [L, D],
        "vegetarian vegan halal high-protein complex-carb low-GI gluten",
    ),
    F("Baked salmon with vegetables", "1 plate", 520, [L, D], "halal omega-3 high-protein low-GI fish"),
    F("Fruit salad", "1 bowl", 150, [S], "vegetarian vegan halal"),
    F("Green tea", "1 cup", 2, [B, S], "vegetarian vegan halal caffeine"),
    F("Black coffee", "1 cup", 5, [B, S], "vegetarian vegan halal caffeine"),
    F("Milk tea (kiri thé)", "1 cup", 120, [B, S], "vegetarian halal caffeine high-sugar dairy"),
    F("Instant noodles", "1 pack", 380, [L, D, S], "high-sodium high-GI fried gluten"),
]
