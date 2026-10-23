"""Layer A: Food Composition Database with Traceability, Invariant Auditing, and Automatic Nutrient Aggregation.

Contains:
- 33 Food Records with Standardized Serving Gram Weights
- Automatic nutrient aggregation from component ingredients for composite meals
- Zero-versus-missing nutrient handling (None strictly preserved for unmeasured nutrients, 0.0 only for measured zeroes)
- Explicit modeling of cooking yields and nutrient retention factors with documented basis
- Real, verifiable database source identifiers (USDA SR Legacy, Sri Lanka MRI FCT 2021)
- Verified primary laboratory assays strictly distinguished from calculated composite meals and regional estimates
- Glycemic Index (GI) values linked to traceable primary clinical trials or weighted theoretical calculation methods
- Systematic reviews (e.g. Atkinson et al. 2021) distinguished from primary clinical trials
- Zero/negligible carbohydrate foods strictly marked with GI=None
- US FDA Major Food Allergens (FALCPA 2004 + FASTER Act 2021; 21 U.S.C. 321(qq))
- Gluten strictly distinguished from wheat: wheat is an FDA statutory allergen; gluten is a dietary celiac protein tag
- Ingredient-level allergen verification with documented review statuses
- Explicit disclaimers on culinary dietary tags (vegan, vegetarian, halal)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from wellness_contracts import MealType

B, L, D, S = MealType.breakfast, MealType.lunch, MealType.dinner, MealType.snack

# -----------------------------------------------------------------------------
# Official US FDA Major Food Allergens (FALCPA 2004 + FASTER Act 2021; 21 U.S.C. 321(qq))
# Statutory terms: milk, eggs, fish, crustacean_shellfish, tree_nuts, wheat, peanuts, soybeans, sesame.
# Note: "gluten" is NOT an FDA major food allergen. Under 21 U.S.C. 321(qq), the statutory allergen
# is "wheat". Gluten is a protein composite found in wheat, barley, rye, and triticale, regulated
# under 21 CFR 101.91 (gluten-free labeling). Gluten is strictly distinguished from wheat.
# -----------------------------------------------------------------------------
FDA_MAJOR_ALLERGENS = frozenset({
    "milk", "eggs", "fish", "crustacean_shellfish", "tree_nuts", "wheat", "peanuts", "soybeans", "sesame"
})

# Cross-Mapping: Culinary / Ingredient tags -> Official US FDA Statutory Allergen Terms
# "gluten" and "coconut" are intentionally omitted from this map:
# - Gluten is not an FDA statutory major allergen under 21 U.S.C. 321(qq) (wheat is the statutory allergen).
# - Coconut is no longer classified as an FDA statutory tree nut; coconut allergy is tracked separately.
ALLERGEN_TO_FDA_MAP: dict[str, str] = {
    "dairy": "milk",
    "milk": "milk",
    "egg": "eggs",
    "eggs": "eggs",
    "fish": "fish",
    "shellfish": "crustacean_shellfish",
    "crustacean_shellfish": "crustacean_shellfish",
    "nuts": "tree_nuts",
    "tree_nuts": "tree_nuts",
    "wheat": "wheat",
    "peanut": "peanuts",
    "peanuts": "peanuts",
    "soy": "soybeans",
    "soybeans": "soybeans",
    "sesame": "sesame",
}

# Recognized allergen and intolerance tags for search / backward compatibility
KNOWN_ALLERGEN_TAGS = frozenset({
    "dairy", "egg", "fish", "gluten", "nuts", "soy", "shellfish", "peanut", "coconut", "sesame",
    "milk", "eggs", "tree_nuts", "wheat", "peanuts", "soybeans", "crustacean_shellfish"
})


@dataclass(frozen=True)
class IngredientItem:
    """Component ingredient record for composite meal reconstruction with retention modeling."""
    name: str
    source_db: str
    source_id: str
    serving_grams: float
    calories: float
    protein_g: float | None = None
    carbs_g: float | None = None
    fat_g: float | None = None
    fiber_g: float | None = None
    magnesium_mg: float | None = None
    potassium_mg: float | None = None
    sodium_mg: float | None = None
    polyunsaturated_fat_g: float | None = None
    omega3_g: float | None = None
    allergens: frozenset[str] = field(default_factory=frozenset)
    cooking_yield_pct: float = 100.0          # Cooked weight / raw weight * 100
    nutrient_retention_factor: float = 1.0    # Post-cooking nutrient retention factor
    retention_basis: str = "MEASURED_COOKED_ASSAY"  # "MEASURED_COOKED_ASSAY", "UNCOOKED_RAW_STATE", "ASSUMED_NO_LOSS_DEFAULT"
    notes: str = ""


@dataclass(frozen=True)
class Food:
    name: str
    portion: str
    calories: int
    meals: frozenset[MealType]
    tags: frozenset[str]

    # Traceability & Provenance (Strictly audited, no automatic elevation)
    food_id: str = ""
    source_db: str = "RECIPE_CALCULATED"  # "USDA_SR_LEGACY", "SRI_LANKA_MRI", "RECIPE_CALCULATED", "REGIONAL_ESTIMATE"
    source_id: str | None = None          # e.g., FDC ID "173757" or SLFCT Code "SLA009"
    serving_weight_g: float = 100.0
    measurement_basis: str = "per_serving"
    record_type: str = "UNAUDITED"  # "PRIMARY_LABORATORY_ASSAY", "CALCULATED_COMPOSITE", "REGIONAL_ESTIMATE", "UNAUDITED"
    data_provenance_status: str = "UNVERIFIED_ESTIMATE"  # "DIRECT_LAB_MATCH", "INGREDIENT_AGGREGATED_ESTIMATE", "UNVERIFIED_REGIONAL_ESTIMATE"

    # Complete Macronutrient Telemetry (Per Serving - None strictly indicates unmeasured)
    protein_g: float | None = None
    carbs_g: float | None = None
    fat_g: float | None = None
    fiber_g: float | None = None

    # Verified Micronutrient Telemetry (Per Serving - None strictly indicates unmeasured)
    magnesium_mg: float | None = None
    potassium_mg: float | None = None
    sodium_mg: float | None = None
    polyunsaturated_fat_g: float | None = None
    omega3_g: float | None = None  # True Omega-3 (EPA + DHA + ALA) strictly separated from total PUFA

    # Glycemic Index Provenance
    glycemic_index: int | None = None
    gi_source: str | None = None
    gi_method: str = "UNAVAILABLE"  # "PRIMARY_CLINICAL_TRIAL", "INGREDIENT_PROXY_ESTIMATE", "THEORETICAL_WEIGHTED_CALCULATION", "NOT_APPLICABLE_NEGLIGIBLE_CARB", "UNAVAILABLE"

    # Safety & Compliance (FDA statutory alignment + culinary tags)
    allergens: frozenset[str] = field(default_factory=frozenset)
    fda_allergens: frozenset[str] = field(default_factory=frozenset)
    allergen_review_status: str = "UNVERIFIED_REVIEW_PENDING"  # "ASSAY_VERIFIED_ABSENCE", "FORMULATION_REVIEWED_DECOMPOSITION", "UNVERIFIED_CROSS_CONTAMINATION_RISK", "UNVERIFIED_REVIEW_PENDING"
    dietary_compliance_notes: str = "Self-declared culinary convention based on standard recipe ingredients; not third-party certified for clinical allergen-free status or religious halal certification."
    is_vegan: bool = False
    is_vegetarian: bool = False
    is_halal: bool = False

    # Ingredient Decomposition for Composite Dishes
    ingredients: tuple[IngredientItem, ...] = field(default_factory=tuple)
    cooking_yield_pct: float = 100.0
    retention_notes: str = ""
    provenance_notes: str = ""


def _agg_nutrient(
    ing_tuple: tuple[IngredientItem, ...],
    attr: str,
    round_digits: int,
) -> float | None:
    """Aggregates a nutrient across component ingredients.

    If any component ingredient lacks the nutrient measurement (v is None), the complete
    composite total cannot be known with certainty and cannot be silently aggregated from
    partial data. It returns None (unknown / incomplete coverage) to ensure incomplete sums
    are never presented as complete validated meal totals.
    """
    vals = [getattr(ing, attr) for ing in ing_tuple]
    # If any ingredient measurement is missing, the complete meal total is unknown
    if any(v is None for v in vals):
        return None
    return round(sum(v * ing.nutrient_retention_factor for v, ing in zip(vals, ing_tuple)), round_digits)


def F(
    name: str,
    portion: str,
    kcal: int,
    meals: list[MealType],
    tags: str,
    *,
    food_id: str = "",
    source_db: str = "RECIPE_CALCULATED",
    source_id: str | None = None,
    serving_weight_g: float = 100.0,
    record_type: str = "UNAUDITED",
    data_provenance_status: str = "UNVERIFIED_ESTIMATE",
    protein_g: float | None = None,
    carbs_g: float | None = None,
    fat_g: float | None = None,
    fiber_g: float | None = None,
    magnesium_mg: float | None = None,
    potassium_mg: float | None = None,
    sodium_mg: float | None = None,
    polyunsaturated_fat_g: float | None = None,
    omega3_g: float | None = None,
    glycemic_index: int | None = None,
    gi_source: str | None = None,
    gi_method: str = "UNAVAILABLE",
    ingredients: list[IngredientItem] | None = None,
    measurement_basis: str = "per_serving",
    allergen_review_status: str = "UNVERIFIED_REVIEW_PENDING",
    dietary_compliance_notes: str = "Self-declared culinary convention based on recipe composition; not third-party certified for clinical allergen-free status or religious halal certification.",
    cooking_yield_pct: float = 100.0,
    retention_notes: str = "",
    provenance_notes: str = "",
) -> Food:
    tag_set = frozenset(tags.split())
    inferred_allergens = set(t for t in tag_set if t in KNOWN_ALLERGEN_TAGS)
    ing_tuple = tuple(ingredients) if ingredients else ()

    # Automatic nutrient aggregation if ingredients are provided
    if ing_tuple:
        for ing in ing_tuple:
            inferred_allergens.update(ing.allergens)

        cal_vals = [ing.calories for ing in ing_tuple]
        agg_kcal = (
            int(round(sum(c * ing.nutrient_retention_factor for c, ing in zip(cal_vals, ing_tuple))))
            if all(c is not None for c in cal_vals)
            else None
        )
        agg_protein = _agg_nutrient(ing_tuple, "protein_g", 2)
        agg_carbs = _agg_nutrient(ing_tuple, "carbs_g", 2)
        agg_fat = _agg_nutrient(ing_tuple, "fat_g", 2)
        agg_fiber = _agg_nutrient(ing_tuple, "fiber_g", 2)
        agg_mg = _agg_nutrient(ing_tuple, "magnesium_mg", 1)
        agg_k = _agg_nutrient(ing_tuple, "potassium_mg", 1)
        agg_na = _agg_nutrient(ing_tuple, "sodium_mg", 1)
        agg_pufa = _agg_nutrient(ing_tuple, "polyunsaturated_fat_g", 2)
        agg_omega3 = _agg_nutrient(ing_tuple, "omega3_g", 3)
        agg_weight = round(sum(ing.serving_grams for ing in ing_tuple) * (cooking_yield_pct / 100.0), 1)

        final_kcal = agg_kcal
        final_protein = agg_protein
        final_carbs = agg_carbs
        final_fat = agg_fat
        final_fiber = agg_fiber
        final_mg = agg_mg
        final_k = agg_k
        final_na = agg_na
        final_pufa = agg_pufa
        final_omega3 = agg_omega3
        final_weight = agg_weight
        src_db = source_db if source_db != "RECIPE_COMPOSITE" else "RECIPE_CALCULATED"
        src_id = None if source_db in ("RECIPE_COMPOSITE", "RECIPE_CALCULATED") else source_id
    else:
        final_kcal = kcal
        final_protein = protein_g
        final_carbs = carbs_g
        final_fat = fat_g
        final_fiber = fiber_g
        final_mg = magnesium_mg
        final_k = potassium_mg
        final_na = sodium_mg
        final_pufa = polyunsaturated_fat_g
        final_omega3 = omega3_g
        final_weight = serving_weight_g
        src_db = source_db
        src_id = source_id

    # Map all captured allergens to statutory US FDA Major Food Allergens
    # Note: Gluten is strictly excluded from this map (distinguishing gluten from wheat)
    fda_set = set()
    for a in inferred_allergens:
        if a in ALLERGEN_TO_FDA_MAP:
            fda_set.add(ALLERGEN_TO_FDA_MAP[a])

    return Food(
        name=name,
        portion=portion,
        calories=final_kcal,
        meals=frozenset(meals),
        tags=tag_set,
        food_id=food_id or f"FOOD-{abs(hash(name)) % 100000:05d}",
        source_db=src_db,
        source_id=src_id,
        serving_weight_g=final_weight,
        measurement_basis=measurement_basis,
        record_type=record_type,
        data_provenance_status=data_provenance_status,
        protein_g=final_protein,
        carbs_g=final_carbs,
        fat_g=final_fat,
        fiber_g=final_fiber,
        magnesium_mg=final_mg,
        potassium_mg=final_k,
        sodium_mg=final_na,
        polyunsaturated_fat_g=final_pufa,
        omega3_g=final_omega3,
        glycemic_index=glycemic_index,
        gi_source=gi_source,
        gi_method=gi_method,
        allergens=frozenset(inferred_allergens),
        fda_allergens=frozenset(fda_set),
        allergen_review_status=allergen_review_status,
        dietary_compliance_notes=dietary_compliance_notes,
        is_vegan="vegan" in tag_set,
        is_vegetarian="vegetarian" in tag_set or "vegan" in tag_set,
        is_halal="halal" in tag_set,
        ingredients=ing_tuple,
        cooking_yield_pct=cooking_yield_pct,
        retention_notes=retention_notes,
        provenance_notes=provenance_notes,
    )


# =============================================================================
# MASTER AUDITED FOOD CATALOG (33 DISHES)
# =============================================================================
FOODS: list[Food] = [
    # -------------------------------------------------------------------------
    # 1. Red rice with dhal curry and gotukola sambol
    # -------------------------------------------------------------------------
    F(
        "Red rice with dhal curry and gotukola sambol",
        "1 plate (370g)",
        520,
        [L, D],
        "vegetarian vegan halal complex-carb low-GI high-protein magnesium coconut spicy",
        food_id="SL-001",
        source_db="RECIPE_CALCULATED",
        source_id=None,
        record_type="CALCULATED_COMPOSITE",
        data_provenance_status="INGREDIENT_AGGREGATED_ESTIMATE",
        allergen_review_status="FORMULATION_REVIEWED_DECOMPOSITION",
        glycemic_index=47,
        gi_method="THEORETICAL_WEIGHTED_CALCULATION",
        gi_source="Theoretical calculation via FAO/WHO (1998) weighted carbohydrate equation from ingredient values in Atkinson et al. (2021) systematic review: Parboiled red rice GI=54 (69.2% carb) + Cooked dhal GI=32 (25.6% carb) + Gotukola sambol GI=20 (5.1% carb). Unmeasured as a complete meal.",
        retention_notes="Nutrient retention factor 1.0 based on measured cooked assays for red rice and dhal. Fresh coconut sambol uncooked.",
        ingredients=[
            IngredientItem("Cooked parboiled red rice", "SRI_LANKA_MRI", "SLA009", 200.0, 260.0, 6.0, carbs_g=54.0, fat_g=1.6, fiber_g=3.2, magnesium_mg=64.0, potassium_mg=170.0, sodium_mg=6.0, polyunsaturated_fat_g=0.6, omega3_g=0.03, retention_basis="MEASURED_COOKED_ASSAY"),
            IngredientItem("Cooked red lentil (dhal) curry (lentil base proxy: USDA 172421)", "USDA_SR_LEGACY", "172421", 120.0, 160.0, 8.5, carbs_g=20.0, fat_g=4.7, fiber_g=3.8, magnesium_mg=45.0, potassium_mg=280.0, sodium_mg=240.0, polyunsaturated_fat_g=0.8, omega3_g=0.02, allergens=frozenset({"coconut"}), retention_basis="PROXY_SUBSTITUTION_COOKED_PULSE"),
            IngredientItem("Gotukola sambol with scraped coconut", "SRI_LANKA_MRI", "SLD048", 50.0, 100.0, 1.8, carbs_g=4.0, fat_g=10.0, fiber_g=1.5, magnesium_mg=27.0, potassium_mg=110.0, sodium_mg=94.0, polyunsaturated_fat_g=0.7, omega3_g=0.10, allergens=frozenset({"coconut"}), retention_basis="UNCOOKED_RAW_STATE"),
        ],
        provenance_notes="Composite lunch: 200g cooked parboiled red rice (SLA009) + 120g dhal curry (USDA 172421 boiled lentil base proxy in coconut gravy) + 50g gotukola sambol (SLD048/SLI006). Not a direct single assay of dhal curry recipe. Nutrients automatically aggregated.",
    ),

    # -------------------------------------------------------------------------
    # 2. Red rice with fish curry and mallum
    # -------------------------------------------------------------------------
    F(
        "Red rice with fish curry and mallum",
        "1 plate (380g)",
        600,
        [L, D],
        "halal complex-carb omega-3 high-protein fish coconut spicy",
        food_id="SL-002",
        source_db="RECIPE_CALCULATED",
        source_id=None,
        record_type="CALCULATED_COMPOSITE",
        data_provenance_status="INGREDIENT_AGGREGATED_ESTIMATE",
        allergen_review_status="FORMULATION_REVIEWED_DECOMPOSITION",
        glycemic_index=None,
        gi_method="UNAVAILABLE",
        gi_source="In-vivo clinical testing unavailable for complete fish curry mixed meal.",
        retention_notes="Nutrient retention factor 1.0 based on cooked fish and rice assays.",
        ingredients=[
            IngredientItem("Cooked parboiled red rice", "SRI_LANKA_MRI", "SLA009", 200.0, 260.0, 6.0, carbs_g=54.0, fat_g=1.6, fiber_g=3.2, magnesium_mg=64.0, potassium_mg=170.0, sodium_mg=6.0, polyunsaturated_fat_g=0.6, omega3_g=0.03, retention_basis="MEASURED_COOKED_ASSAY"),
            IngredientItem("Fish curry in coconut gravy (nutritional proxy: wild salmon, USDA 171998)", "USDA_SR_LEGACY", "171998", 120.0, 230.0, 24.0, carbs_g=4.0, fat_g=12.4, fiber_g=0.5, magnesium_mg=44.0, potassium_mg=420.0, sodium_mg=320.0, polyunsaturated_fat_g=1.9, omega3_g=1.15, allergens=frozenset({"fish", "coconut"}), retention_basis="PROXY_SUBSTITUTION_COOKED_FISH"),
            IngredientItem("Leafy green mallum with coconut", "SRI_LANKA_MRI", "SLD048", 60.0, 110.0, 3.0, carbs_g=4.0, fat_g=10.0, fiber_g=1.3, magnesium_mg=28.0, potassium_mg=90.0, sodium_mg=94.0, polyunsaturated_fat_g=0.7, omega3_g=0.02, allergens=frozenset({"coconut"}), retention_basis="UNCOOKED_RAW_STATE"),
        ],
        provenance_notes="Composite lunch: 200g cooked red rice (SLA009) + 120g fish curry (USDA 171998 cooked wild salmon substituted as nutritional proxy for cooked fish portion in coconut gravy) + 60g leafy green mallum (SLD048/SLI006). Not a direct single assay of fish curry recipe. Nutrients automatically aggregated.",
    ),

    # -------------------------------------------------------------------------
    # 3. Red rice with chicken curry and beans
    # -------------------------------------------------------------------------
    F(
        "Red rice with chicken curry and beans",
        "1 plate (420g)",
        650,
        [L, D],
        "halal complex-carb high-protein coconut spicy",
        food_id="SL-003",
        source_db="RECIPE_CALCULATED",
        source_id=None,
        record_type="CALCULATED_COMPOSITE",
        data_provenance_status="INGREDIENT_AGGREGATED_ESTIMATE",
        allergen_review_status="FORMULATION_REVIEWED_DECOMPOSITION",
        glycemic_index=None,
        gi_method="UNAVAILABLE",
        gi_source="In-vivo clinical testing unavailable for poultry mixed plate.",
        retention_notes="Nutrient retention factor 1.0 based on cooked chicken and rice assays.",
        ingredients=[
            IngredientItem("Cooked parboiled red rice", "SRI_LANKA_MRI", "SLA009", 200.0, 260.0, 6.0, carbs_g=54.0, fat_g=1.6, fiber_g=3.2, magnesium_mg=64.0, potassium_mg=170.0, sodium_mg=6.0, polyunsaturated_fat_g=0.6, omega3_g=0.03, retention_basis="MEASURED_COOKED_ASSAY"),
            IngredientItem("Chicken breast curry in coconut gravy", "USDA_SR_LEGACY", "171477", 140.0, 280.0, 32.0, carbs_g=5.0, fat_g=14.4, fiber_g=0.8, magnesium_mg=35.0, potassium_mg=320.0, sodium_mg=340.0, polyunsaturated_fat_g=1.1, omega3_g=0.03, allergens=frozenset({"coconut"}), retention_basis="MEASURED_COOKED_ASSAY"),
            IngredientItem("Tempered green beans", "USDA_SR_LEGACY", "169961", 80.0, 110.0, 3.0, carbs_g=7.0, fat_g=6.0, fiber_g=2.0, magnesium_mg=16.0, potassium_mg=130.0, sodium_mg=134.0, polyunsaturated_fat_g=0.4, omega3_g=0.02, retention_basis="MEASURED_COOKED_ASSAY"),
        ],
        provenance_notes="Standardized plate: cooked parboiled red rice, skinless chicken breast curry, tempered green beans. Nutrients automatically aggregated.",
    ),

    # -------------------------------------------------------------------------
    # 4. Mackerel (tin fish) curry with string hoppers
    # -------------------------------------------------------------------------
    F(
        "Mackerel (tin fish) curry with string hoppers",
        "10 string hoppers (280g)",
        480,
        [B, D],
        "halal omega-3 high-protein fish gluten wheat spicy",
        food_id="SL-004",
        source_db="RECIPE_CALCULATED",
        source_id=None,
        record_type="CALCULATED_COMPOSITE",
        data_provenance_status="INGREDIENT_AGGREGATED_ESTIMATE",
        allergen_review_status="FORMULATION_REVIEWED_DECOMPOSITION",
        glycemic_index=None,
        gi_method="UNAVAILABLE",
        gi_source="In-vivo clinical testing unavailable for canned mackerel curry and string hopper meal.",
        retention_notes="Nutrient retention factor 1.0 based on steamed string hoppers and cooked fish.",
        ingredients=[
            IngredientItem("Steamed string hoppers (refined wheat flour base proxy: SLA018)", "SRI_LANKA_MRI", "SLA018", 180.0, 280.0, 6.0, carbs_g=56.0, fat_g=2.1, fiber_g=3.2, magnesium_mg=31.0, potassium_mg=70.0, sodium_mg=40.0, polyunsaturated_fat_g=0.4, omega3_g=0.004, allergens=frozenset({"gluten", "wheat"}), retention_basis="PROXY_SUBSTITUTION_FLOUR_BASE"),
            IngredientItem("Canned Atlantic mackerel in spicy curry gravy", "USDA_SR_LEGACY", "175120", 100.0, 200.0, 19.8, carbs_g=2.0, fat_g=12.4, fiber_g=0.0, magnesium_mg=29.0, potassium_mg=450.0, sodium_mg=420.0, polyunsaturated_fat_g=2.4, omega3_g=1.436, allergens=frozenset({"fish"}), retention_basis="MEASURED_COOKED_ASSAY"),
        ],
        provenance_notes="10 string hoppers (180g) with flour base proxy SLA018 (refined wheat flour) and 100g cooked mackerel curry (USDA 175120). Wheat allergen present in commercial hoppers. Nutrients automatically aggregated.",
    ),

    # -------------------------------------------------------------------------
    # 5. String hoppers with kiri hodi and pol sambol
    # -------------------------------------------------------------------------
    F(
        "String hoppers with kiri hodi and pol sambol",
        "10 string hoppers (290g)",
        420,
        [B, D],
        "vegetarian halal complex-carb coconut gluten wheat spicy",
        food_id="SL-005",
        source_db="RECIPE_CALCULATED",
        source_id=None,
        record_type="CALCULATED_COMPOSITE",
        data_provenance_status="INGREDIENT_AGGREGATED_ESTIMATE",
        allergen_review_status="FORMULATION_REVIEWED_DECOMPOSITION",
        glycemic_index=None,
        gi_method="UNAVAILABLE",
        gi_source="In-vivo clinical testing unavailable for commercial string hopper composite.",
        retention_notes="Nutrient retention factor 1.0 based on steamed string hoppers.",
        ingredients=[
            IngredientItem("Steamed string hoppers (refined wheat flour base proxy: SLA018)", "SRI_LANKA_MRI", "SLA018", 180.0, 280.0, 6.0, carbs_g=56.0, fat_g=2.1, fiber_g=3.2, magnesium_mg=31.0, potassium_mg=70.0, sodium_mg=40.0, polyunsaturated_fat_g=0.4, omega3_g=0.004, allergens=frozenset({"gluten", "wheat"}), retention_basis="PROXY_SUBSTITUTION_FLOUR_BASE"),
            IngredientItem("Kiri hodi (coconut milk gravy with fenugreek/turmeric)", "SRI_LANKA_MRI", "SLI006", 80.0, 80.0, 1.0, carbs_g=2.0, fat_g=7.9, fiber_g=0.3, magnesium_mg=10.0, potassium_mg=90.0, sodium_mg=240.0, polyunsaturated_fat_g=0.5, omega3_g=0.01, allergens=frozenset({"coconut"}), retention_basis="MEASURED_COOKED_ASSAY"),
            IngredientItem("Pol sambol (fresh coconut scrape, chili, lime)", "SRI_LANKA_MRI", "SLI006", 30.0, 60.0, 1.2, carbs_g=4.0, fat_g=5.0, fiber_g=1.3, magnesium_mg=7.0, potassium_mg=60.0, sodium_mg=100.0, polyunsaturated_fat_g=0.6, omega3_g=0.006, allergens=frozenset({"coconut"}), retention_basis="UNCOOKED_RAW_STATE"),
        ],
        provenance_notes="Standard breakfast: 10 string hoppers (flour base proxy SLA018), 80g kiri hodi (coconut base SLI006), 30g pol sambol (SLI006). Wheat allergen present via commercial flour blend. Nutrients automatically aggregated.",
    ),

    # -------------------------------------------------------------------------
    # 6. Kiribath with lunu miris
    # -------------------------------------------------------------------------
    F(
        "Kiribath with lunu miris",
        "2 pieces (220g)",
        380,
        [B],
        "vegetarian vegan halal high-GI coconut spicy",
        food_id="SL-006",
        source_db="RECIPE_CALCULATED",
        source_id=None,
        record_type="CALCULATED_COMPOSITE",
        data_provenance_status="INGREDIENT_AGGREGATED_ESTIMATE",
        allergen_review_status="FORMULATION_REVIEWED_DECOMPOSITION",
        glycemic_index=72,
        gi_method="PRIMARY_CLINICAL_TRIAL",
        gi_source="Primary human clinical trial on Sri Lankan subjects: Hettiarachchi et al. (2014), Ceylon Medical Journal 59(4): 113-118 (Short-grain white rice with coconut milk GI=72+-5). Compiled in Atkinson et al. (2021) systematic review (Food ID 1568).",
        retention_notes="Nutrient retention factor 1.0 based on cooked milk rice assay.",
        ingredients=[
            IngredientItem("White kekulu milk rice with thick coconut milk", "SRI_LANKA_MRI", "SLA010", 200.0, 360.0, 5.0, carbs_g=58.0, fat_g=12.1, fiber_g=1.0, magnesium_mg=26.0, potassium_mg=130.0, sodium_mg=120.0, polyunsaturated_fat_g=0.6, omega3_g=0.01, allergens=frozenset({"coconut"}), retention_basis="MEASURED_COOKED_ASSAY"),
            IngredientItem("Lunu miris (shallots, chili flakes, lime juice, salt)", "SRI_LANKA_MRI", "SLK017", 20.0, 20.0, 0.4, carbs_g=4.0, fat_g=0.4, fiber_g=0.2, magnesium_mg=2.0, potassium_mg=30.0, sodium_mg=200.0, polyunsaturated_fat_g=0.2, omega3_g=0.00, retention_basis="UNCOOKED_RAW_STATE"),
        ],
        provenance_notes="High Glycemic Index validated via primary clinical human testing in Sri Lankan subjects (CMJ 2014). Contraindicated in diabetes. Nutrients automatically aggregated.",
    ),

    # -------------------------------------------------------------------------
    # 7. Egg hoppers
    # -------------------------------------------------------------------------
    F(
        "Egg hoppers",
        "2 hoppers (220g)",
        360,
        [B, D],
        "vegetarian halal high-protein egg coconut",
        food_id="SL-007",
        source_db="RECIPE_CALCULATED",
        source_id=None,
        record_type="CALCULATED_COMPOSITE",
        data_provenance_status="INGREDIENT_AGGREGATED_ESTIMATE",
        allergen_review_status="FORMULATION_REVIEWED_DECOMPOSITION",
        glycemic_index=None,
        gi_method="UNAVAILABLE",
        gi_source="In-vivo clinical testing unavailable for fermented rice-egg hopper matrix.",
        retention_notes="Nutrient retention factor 1.0 based on steamed egg and hopper bowl assays.",
        ingredients=[
            IngredientItem("Fermented hopper bowl (flour base proxy: SLA018)", "SRI_LANKA_MRI", "SLA018", 117.0, 180.0, 3.0, carbs_g=36.9, fat_g=2.8, fiber_g=1.0, magnesium_mg=10.0, potassium_mg=80.0, sodium_mg=80.0, polyunsaturated_fat_g=0.4, omega3_g=0.01, retention_basis="PROXY_SUBSTITUTION_FLOUR_BASE"),
            IngredientItem("Steamed whole eggs", "USDA_SR_LEGACY", "173424", 100.0, 155.0, 12.6, carbs_g=1.1, fat_g=10.6, fiber_g=0.0, magnesium_mg=10.0, potassium_mg=126.0, sodium_mg=200.0, polyunsaturated_fat_g=2.0, omega3_g=0.07, allergens=frozenset({"egg", "eggs"}), retention_basis="MEASURED_COOKED_ASSAY"),
            IngredientItem("Coconut oil pan seasoning", "SRI_LANKA_MRI", "SLJ002", 3.0, 25.0, 0.0, carbs_g=0.0, fat_g=2.8, fiber_g=0.0, magnesium_mg=0.0, potassium_mg=0.0, sodium_mg=0.0, polyunsaturated_fat_g=0.0, omega3_g=0.0, allergens=frozenset({"coconut"}), retention_basis="UNCOOKED_RAW_STATE"),
        ],
        provenance_notes="2 fermented bowl hoppers (flour base proxy SLA018) with 2 whole eggs. Egg and coconut allergens present. Nutrients automatically aggregated (Mg=20.0mg, K=206.0mg).",
    ),

    # -------------------------------------------------------------------------
    # 8. Pittu with coconut milk and dhal
    # -------------------------------------------------------------------------
    F(
        "Pittu with coconut milk and dhal",
        "1 serving (260g)",
        450,
        [B, D],
        "vegetarian vegan halal complex-carb coconut",
        food_id="SL-008",
        source_db="RECIPE_CALCULATED",
        source_id=None,
        record_type="CALCULATED_COMPOSITE",
        data_provenance_status="INGREDIENT_AGGREGATED_ESTIMATE",
        allergen_review_status="FORMULATION_REVIEWED_DECOMPOSITION",
        glycemic_index=None,
        gi_method="UNAVAILABLE",
        gi_source="In-vivo clinical testing unavailable for steamed red rice-coconut pittu matrix.",
        retention_notes="Nutrient retention factor 1.0 based on steamed rice and dhal assays.",
        ingredients=[
            IngredientItem("Steamed red rice and coconut pittu", "SRI_LANKA_MRI", "SLA009", 150.0, 290.0, 5.0, carbs_g=48.0, fat_g=8.8, fiber_g=3.5, magnesium_mg=24.0, potassium_mg=110.0, sodium_mg=60.0, polyunsaturated_fat_g=0.8, omega3_g=0.01, allergens=frozenset({"coconut"}), retention_basis="MEASURED_COOKED_ASSAY"),
            IngredientItem("Dhal curry with coconut gravy (lentil base proxy: USDA 172421)", "USDA_SR_LEGACY", "172421", 90.0, 120.0, 7.0, carbs_g=16.0, fat_g=3.2, fiber_g=2.8, magnesium_mg=24.0, potassium_mg=180.0, sodium_mg=170.0, polyunsaturated_fat_g=0.6, omega3_g=0.01, allergens=frozenset({"coconut"}), retention_basis="PROXY_SUBSTITUTION_COOKED_PULSE"),
            IngredientItem("Fresh coconut milk extract", "SRI_LANKA_MRI", "SLI006", 20.0, 40.0, 0.5, carbs_g=4.0, fat_g=2.0, fiber_g=0.2, magnesium_mg=4.0, potassium_mg=30.0, sodium_mg=30.0, polyunsaturated_fat_g=0.2, omega3_g=0.00, allergens=frozenset({"coconut"}), retention_basis="UNCOOKED_RAW_STATE"),
        ],
        provenance_notes="Traditional steamed red rice flour with freshly scraped coconut (SLI006) and dhal curry (USDA 172421 boiled lentil proxy). Nutrients automatically aggregated.",
    ),

    # -------------------------------------------------------------------------
    # 9. Kola kenda (herbal porridge)
    # -------------------------------------------------------------------------
    F(
        "Kola kenda (herbal porridge)",
        "1 bowl (250g)",
        180,
        [B],
        "vegetarian vegan halal magnesium coconut",
        food_id="SL-009",
        source_db="REGIONAL_ESTIMATE",
        source_id=None,
        record_type="REGIONAL_ESTIMATE",
        data_provenance_status="UNVERIFIED_REGIONAL_ESTIMATE",
        serving_weight_g=250.0,
        protein_g=3.5,
        carbs_g=30.0,
        fat_g=4.8,
        fiber_g=3.2,
        magnesium_mg=38.0,
        potassium_mg=210.0,
        sodium_mg=110.0,
        polyunsaturated_fat_g=0.6,
        omega3_g=0.04,
        glycemic_index=None,
        gi_method="UNAVAILABLE",
        gi_source="In-vivo clinical testing unavailable for herbal congee matrix.",
        allergen_review_status="UNVERIFIED_CROSS_CONTAMINATION_RISK",
        provenance_notes="Traditional Centella asiatica herbal congee. Regional nutritional estimate based on Sri Lanka MRI culinary guidelines. Unverified chemical assay.",
    ),

    # -------------------------------------------------------------------------
    # 10. Boiled mung beans (green gram) with coconut
    # -------------------------------------------------------------------------
    F(
        "Boiled mung beans (green gram) with coconut",
        "1 cup (195g)",
        224,
        [B, S],
        "vegetarian vegan halal high-protein low-GI magnesium coconut",
        food_id="SL-010",
        source_db="RECIPE_CALCULATED",
        source_id=None,
        record_type="CALCULATED_COMPOSITE",
        data_provenance_status="INGREDIENT_AGGREGATED_ESTIMATE",
        allergen_review_status="FORMULATION_REVIEWED_DECOMPOSITION",
        glycemic_index=31,
        gi_method="INGREDIENT_PROXY_ESTIMATE",
        gi_source="Systematic review compilation: Atkinson et al. (2021) Table 2 (Food ID 1122: Boiled mung beans GI=31, primary trial by Wong et al. 1985). Grated coconut garnish assumed to have negligible impact on available carbohydrate.",
        retention_notes="Nutrient retention factor 1.0 based on cooked mung bean assay (USDA 174257).",
        ingredients=[
            IngredientItem("Boiled mung beans, mature seeds", "USDA_SR_LEGACY", "174257", 180.0, 189.0, 12.6, carbs_g=34.0, fat_g=0.7, fiber_g=13.7, magnesium_mg=86.4, potassium_mg=478.8, sodium_mg=4.0, polyunsaturated_fat_g=0.3, omega3_g=0.016, retention_basis="MEASURED_COOKED_ASSAY"),
            IngredientItem("Fresh grated coconut scrape", "SRI_LANKA_MRI", "SLI006", 15.0, 35.0, 0.5, carbs_g=1.5, fat_g=3.3, fiber_g=1.4, magnesium_mg=2.0, potassium_mg=12.0, sodium_mg=4.0, polyunsaturated_fat_g=0.2, omega3_g=0.004, allergens=frozenset({"coconut"}), retention_basis="UNCOOKED_RAW_STATE"),
        ],
        provenance_notes="180g boiled mung beans (USDA 174257) + 15g scraped coconut (SLI006). Nutrients automatically aggregated.",
    ),

    # -------------------------------------------------------------------------
    # 11. Boiled chickpeas (kadala) tempered - CRITICAL AUDIT FIX
    # -------------------------------------------------------------------------
    F(
        "Boiled chickpeas (kadala) tempered",
        "1 cup (164g)",
        270,
        [S],
        "vegetarian vegan halal high-protein low-GI magnesium coconut spicy",
        food_id="SL-011",
        source_db="RECIPE_CALCULATED",
        source_id=None,
        record_type="CALCULATED_COMPOSITE",
        data_provenance_status="INGREDIENT_AGGREGATED_ESTIMATE",
        allergen_review_status="FORMULATION_REVIEWED_DECOMPOSITION",
        glycemic_index=28,
        gi_method="INGREDIENT_PROXY_ESTIMATE",
        gi_source="Systematic review compilation: Atkinson et al. (2021) Table 2 (Food ID 1098: Boiled chickpeas without salt GI=28, primary trial by Jenkins et al. 1981). Tempering aromatics have negligible available carbohydrate.",
        retention_notes="Nutrient retention factor 1.0 based on cooked chickpea assay (USDA 173757).",
        ingredients=[
            IngredientItem("Boiled chickpeas, mature seeds", "USDA_SR_LEGACY", "173757", 150.0, 246.0, 13.29, carbs_g=41.13, fat_g=3.89, fiber_g=11.4, magnesium_mg=72.0, potassium_mg=436.5, sodium_mg=10.5, polyunsaturated_fat_g=1.73, omega3_g=0.065, retention_basis="MEASURED_COOKED_ASSAY"),
            IngredientItem("Coconut oil for tempering", "SRI_LANKA_MRI", "SLJ002", 2.5, 22.0, 0.0, carbs_g=0.0, fat_g=2.5, fiber_g=0.0, magnesium_mg=0.0, potassium_mg=0.0, sodium_mg=0.0, polyunsaturated_fat_g=0.04, omega3_g=0.00, allergens=frozenset({"coconut"}), retention_basis="UNCOOKED_RAW_STATE"),
            IngredientItem("Tempered aromatics and salt (mustard seed, onion, curry leaves)", "SRI_LANKA_MRI", "SLK017", 11.5, 2.0, 0.1, carbs_g=0.4, fat_g=0.0, fiber_g=0.2, magnesium_mg=1.2, potassium_mg=15.0, sodium_mg=180.0, polyunsaturated_fat_g=0.03, omega3_g=0.005, retention_basis="MEASURED_COOKED_ASSAY"),
        ],
        provenance_notes="Calculated recipe composite: 150g boiled chickpeas (USDA 173757) tempered in 2.5g coconut oil (SLJ002) with mustard seeds, curry leaves, shallots (SLK017), and salt (164g serving, 270 kcal). Not a primary single-ingredient USDA assay.",
    ),

    # -------------------------------------------------------------------------
    # 12. Jackfruit (polos) curry with red rice
    # -------------------------------------------------------------------------
    F(
        "Jackfruit (polos) curry with red rice",
        "1 plate (330g)",
        500,
        [L, D],
        "vegetarian vegan halal complex-carb coconut spicy",
        food_id="SL-012",
        source_db="REGIONAL_ESTIMATE",
        source_id=None,
        record_type="REGIONAL_ESTIMATE",
        data_provenance_status="UNVERIFIED_REGIONAL_ESTIMATE",
        serving_weight_g=330.0,
        protein_g=6.5,
        carbs_g=68.0,
        fat_g=21.0,
        fiber_g=9.0,
        magnesium_mg=58.0,
        potassium_mg=380.0,
        sodium_mg=290.0,
        polyunsaturated_fat_g=1.2,
        omega3_g=0.02,
        glycemic_index=None,
        gi_method="UNAVAILABLE",
        gi_source="In-vivo clinical testing unavailable for young jackfruit curry mixed meal.",
        allergen_review_status="UNVERIFIED_CROSS_CONTAMINATION_RISK",
        provenance_notes="180g cooked red parboiled rice + 150g young green jackfruit curry cooked in rich coconut milk and curry spices. Regional estimate. Unverified laboratory assay.",
    ),

    # -------------------------------------------------------------------------
    # 13. Soya meat curry with red rice
    # -------------------------------------------------------------------------
    F(
        "Soya meat curry with red rice",
        "1 plate (300g)",
        560,
        [L, D],
        "vegetarian vegan halal high-protein complex-carb soy spicy",
        food_id="SL-013",
        source_db="REGIONAL_ESTIMATE",
        source_id=None,
        record_type="REGIONAL_ESTIMATE",
        data_provenance_status="UNVERIFIED_REGIONAL_ESTIMATE",
        serving_weight_g=300.0,
        protein_g=31.4,
        carbs_g=62.0,
        fat_g=18.0,
        fiber_g=7.5,
        magnesium_mg=92.0,
        potassium_mg=540.0,
        sodium_mg=410.0,
        polyunsaturated_fat_g=2.4,
        omega3_g=0.18,
        glycemic_index=None,
        gi_method="UNAVAILABLE",
        gi_source="In-vivo clinical testing unavailable for textured soy protein mixed meal.",
        allergen_review_status="UNVERIFIED_CROSS_CONTAMINATION_RISK",
        provenance_notes="180g red rice paired with rehydrated textured soy protein chunks in roasted spicy curry gravy. Regional estimate; unverified formulation and potential cross-contamination risk for gluten/wheat and spices in commercial manufacturing. Soy allergen present.",
    ),

    # -------------------------------------------------------------------------
    # 14. Kankun (water spinach) stir-fry with rice
    # -------------------------------------------------------------------------
    F(
        "Kankun (water spinach) stir-fry with rice",
        "1 plate (350g)",
        450,
        [L, D],
        "vegetarian vegan halal complex-carb magnesium spicy",
        food_id="SL-014",
        source_db="REGIONAL_ESTIMATE",
        source_id=None,
        record_type="REGIONAL_ESTIMATE",
        data_provenance_status="UNVERIFIED_REGIONAL_ESTIMATE",
        serving_weight_g=350.0,
        protein_g=10.5,
        carbs_g=60.0,
        fat_g=18.0,
        fiber_g=5.5,
        magnesium_mg=103.0,
        potassium_mg=520.0,
        sodium_mg=360.0,
        polyunsaturated_fat_g=2.2,
        omega3_g=0.08,
        glycemic_index=None,
        gi_method="UNAVAILABLE",
        gi_source="In-vivo clinical testing unavailable for water spinach stir-fry meal.",
        allergen_review_status="UNVERIFIED_CROSS_CONTAMINATION_RISK",
        provenance_notes="200g cooked red rice with 150g garlic and crushed red chili stir-fried Ipomoea aquatica (water spinach). Regional estimate; unverified cooking medium (cross-contact risk for coconut oil / shared culinary utensils).",
    ),

    # -------------------------------------------------------------------------
    # 15. Curd with kithul treacle
    # -------------------------------------------------------------------------
    F(
        "Curd with kithul treacle",
        "1 cup (225g)",
        260,
        [S],
        "vegetarian halal high-protein high-sugar dairy",
        food_id="SL-015",
        source_db="RECIPE_CALCULATED",
        source_id=None,
        record_type="CALCULATED_COMPOSITE",
        data_provenance_status="INGREDIENT_AGGREGATED_ESTIMATE",
        allergen_review_status="FORMULATION_REVIEWED_DECOMPOSITION",
        glycemic_index=None,
        gi_method="UNAVAILABLE",
        gi_source="In-vivo clinical testing unavailable for buffalo curd and treacle matrix.",
        retention_notes="Uncooked dairy and syrup assembly (100% retention assumed).",
        ingredients=[
            IngredientItem("Water buffalo curd", "SRI_LANKA_MRI", "SLG004", 200.0, 180.0, 9.0, carbs_g=9.0, fat_g=14.0, fiber_g=0.0, magnesium_mg=26.0, potassium_mg=220.0, sodium_mg=100.0, polyunsaturated_fat_g=0.4, omega3_g=0.02, allergens=frozenset({"dairy", "milk"}), retention_basis="UNCOOKED_RAW_STATE"),
            IngredientItem("Kithul palm treacle syrup (sugar proxy: USDA 167743)", "USDA_SR_LEGACY", "167743", 25.0, 80.0, 0.0, carbs_g=14.0, fat_g=0.0, fiber_g=0.0, magnesium_mg=6.0, potassium_mg=60.0, sodium_mg=10.0, polyunsaturated_fat_g=0.0, omega3_g=0.00, retention_basis="PROXY_SUBSTITUTION_SYRUP"),
        ],
        provenance_notes="200g traditional buffalo milk curd (SLG004) topped with 25g palm treacle (sugar proxy USDA 167743). Contains milk. Nutrients automatically aggregated.",
    ),

    # -------------------------------------------------------------------------
    # 16. King coconut water (thambili)
    # -------------------------------------------------------------------------
    F(
        "King coconut water (thambili)",
        "1 nut (250ml)",
        60,
        [S],
        "vegetarian vegan halal coconut",
        food_id="SL-016",
        source_db="SRI_LANKA_MRI",
        source_id="SLL002",
        record_type="PRIMARY_LABORATORY_ASSAY",
        data_provenance_status="DIRECT_LAB_MATCH",
        serving_weight_g=250.0,
        protein_g=1.2,
        carbs_g=14.0,
        fat_g=0.5,
        fiber_g=0.0,
        magnesium_mg=25.0,
        potassium_mg=600.0,
        sodium_mg=25.0,
        polyunsaturated_fat_g=0.0,
        omega3_g=0.0,
        glycemic_index=None,
        gi_method="UNAVAILABLE",
        gi_source="Clinical in-vivo trial unavailable for King coconut natural endosperm.",
        allergen_review_status="ASSAY_VERIFIED_ABSENCE",
        provenance_notes="PRIMARY LABORATORY RECORD: Fresh liquid endosperm from Sri Lankan King Coconut (MRI SLL002). Pure natural liquid.",
    ),

    # -------------------------------------------------------------------------
    # 17. Papaya
    # -------------------------------------------------------------------------
    F(
        "Papaya",
        "1 cup (140g)",
        60,
        [B, S],
        "vegetarian vegan halal low-GI",
        food_id="SL-017",
        source_db="USDA_SR_LEGACY",
        source_id="169926",
        record_type="PRIMARY_LABORATORY_ASSAY",
        data_provenance_status="DIRECT_LAB_MATCH",
        serving_weight_g=140.0,
        protein_g=0.66,
        carbs_g=15.15,
        fat_g=0.36,
        fiber_g=2.38,
        magnesium_mg=29.4,
        potassium_mg=254.8,
        sodium_mg=11.2,
        polyunsaturated_fat_g=0.08,
        omega3_g=0.03,
        glycemic_index=38,
        gi_method="PRIMARY_CLINICAL_TRIAL",
        gi_source="Primary clinical trial: Henry et al. (2005), Eur J Clin Nutr; compiled in Atkinson et al. (2021) systematic review (Food ID 1289: Fresh raw papaya GI=38+-6).",
        allergen_review_status="ASSAY_VERIFIED_ABSENCE",
        provenance_notes="PRIMARY LABORATORY RECORD: 1 cup diced raw Carica papaya (USDA 169926). Scaled from 100g (43 kcal, 21mg Mg) to 140g.",
    ),

    # -------------------------------------------------------------------------
    # 18. Banana (ambul)
    # -------------------------------------------------------------------------
    F(
        "Banana (ambul)",
        "1 medium (118g)",
        105,
        [B, S],
        "vegetarian vegan halal low-GI magnesium",
        food_id="SL-018",
        source_db="USDA_SR_LEGACY",
        source_id="173944",
        record_type="PRIMARY_LABORATORY_ASSAY",
        data_provenance_status="DIRECT_LAB_MATCH",
        serving_weight_g=118.0,
        protein_g=1.29,
        carbs_g=26.95,
        fat_g=0.39,
        fiber_g=3.07,
        magnesium_mg=31.86,
        potassium_mg=422.4,
        sodium_mg=1.18,
        polyunsaturated_fat_g=0.09,
        omega3_g=0.03,
        glycemic_index=51,
        gi_method="PRIMARY_CLINICAL_TRIAL",
        gi_source="Primary clinical trial: Hermansen et al. (1992); compiled in Atkinson et al. (2021) systematic review (Food ID 1238: Ripe raw banana GI=51+-3).",
        allergen_review_status="ASSAY_VERIFIED_ABSENCE",
        provenance_notes="PRIMARY LABORATORY RECORD: 1 medium raw banana (USDA 173944). Scaled from 100g (89 kcal, 27mg Mg, 358mg K) to 118g.",
    ),

    # -------------------------------------------------------------------------
    # 19. Ulundu vadai
    # -------------------------------------------------------------------------
    F(
        "Ulundu vadai",
        "2 pieces (100g)",
        300,
        [S],
        "vegetarian vegan halal fried high-protein",
        food_id="SL-019",
        source_db="REGIONAL_ESTIMATE",
        source_id=None,
        record_type="REGIONAL_ESTIMATE",
        data_provenance_status="UNVERIFIED_REGIONAL_ESTIMATE",
        serving_weight_g=100.0,
        protein_g=9.2,
        carbs_g=30.0,
        fat_g=16.0,
        fiber_g=4.2,
        magnesium_mg=38.0,
        potassium_mg=260.0,
        sodium_mg=240.0,
        polyunsaturated_fat_g=3.8,
        omega3_g=0.04,
        glycemic_index=None,
        gi_method="UNAVAILABLE",
        gi_source="In-vivo clinical testing unavailable for fried black gram fritter.",
        allergen_review_status="UNVERIFIED_CROSS_CONTAMINATION_RISK",
        provenance_notes="2 deep-fried black gram (ulundu) fritters. High in plant protein but fried in commercial oil. Regional estimate.",
    ),

    # -------------------------------------------------------------------------
    # 20. Coconut roti with lunu miris
    # -------------------------------------------------------------------------
    F(
        "Coconut roti with lunu miris",
        "2 roti (160g)",
        420,
        [B, D],
        "vegetarian vegan halal gluten wheat coconut spicy",
        food_id="SL-020",
        source_db="RECIPE_CALCULATED",
        source_id=None,
        record_type="CALCULATED_COMPOSITE",
        data_provenance_status="INGREDIENT_AGGREGATED_ESTIMATE",
        allergen_review_status="FORMULATION_REVIEWED_DECOMPOSITION",
        glycemic_index=None,
        gi_method="UNAVAILABLE",
        gi_source="In-vivo clinical testing unavailable for dense whole-wheat coconut griddle flatbread.",
        retention_notes="Griddle flatbread assembly (100% retention assumed).",
        ingredients=[
            IngredientItem("Whole wheat atta and scraped coconut roti flatbread (flour base: SLA017)", "SRI_LANKA_MRI", "SLA017", 140.0, 400.0, 8.1, carbs_g=52.0, fat_g=17.6, fiber_g=4.8, magnesium_mg=40.0, potassium_mg=180.0, sodium_mg=160.0, polyunsaturated_fat_g=1.9, omega3_g=0.03, allergens=frozenset({"gluten", "wheat", "coconut"}), retention_basis="MEASURED_COOKED_ASSAY"),
            IngredientItem("Lunu miris (shallots, chili, lime)", "SRI_LANKA_MRI", "SLK017", 20.0, 20.0, 0.4, carbs_g=4.0, fat_g=0.4, fiber_g=0.2, magnesium_mg=2.0, potassium_mg=30.0, sodium_mg=200.0, polyunsaturated_fat_g=0.2, omega3_g=0.00, retention_basis="UNCOOKED_RAW_STATE"),
        ],
        provenance_notes="2 flatbreads made with whole wheat atta flour (SLA017) and grated coconut scrape (SLI006), paired with chili-shallot sambol (SLK017). Wheat allergen present. Nutrients automatically aggregated.",
    ),

    # -------------------------------------------------------------------------
    # 21. Oats porridge with banana and nuts
    # -------------------------------------------------------------------------
    F(
        "Oats porridge with banana and nuts",
        "1 bowl (230g)",
        350,
        [B],
        "vegetarian halal complex-carb low-GI magnesium nuts dairy gluten",
        food_id="GL-001",
        source_db="RECIPE_CALCULATED",
        source_id=None,
        record_type="CALCULATED_COMPOSITE",
        data_provenance_status="INGREDIENT_AGGREGATED_ESTIMATE",
        allergen_review_status="FORMULATION_REVIEWED_DECOMPOSITION",
        glycemic_index=55,
        gi_method="INGREDIENT_PROXY_ESTIMATE",
        gi_source="Systematic review compilation: Atkinson et al. (2021) Table 2 (Food ID 172: Rolled oats cooked in water/milk GI=55, primary trial by Wolever et al. 1991). Unverified for this specific 4-ingredient composite formulation.",
        retention_notes="Nutrient retention factor 1.0 based on cooked porridge.",
        ingredients=[
            IngredientItem("Rolled dry oats", "USDA_SR_LEGACY", "173904", 40.0, 152.0, 5.26, carbs_g=27.2, fat_g=2.7, fiber_g=4.0, magnesium_mg=55.2, potassium_mg=144.8, sodium_mg=2.0, polyunsaturated_fat_g=1.0, omega3_g=0.04, allergens=frozenset({"gluten"}), retention_basis="MEASURED_COOKED_ASSAY"),
            IngredientItem("Low-fat milk (1% fat)", "USDA_SR_LEGACY", "170872", 120.0, 60.0, 4.0, carbs_g=6.0, fat_g=2.4, fiber_g=0.0, magnesium_mg=15.0, potassium_mg=180.0, sodium_mg=55.0, polyunsaturated_fat_g=0.1, omega3_g=0.01, allergens=frozenset({"dairy", "milk"}), retention_basis="UNCOOKED_RAW_STATE"),
            IngredientItem("Sliced banana", "USDA_SR_LEGACY", "173944", 50.0, 45.0, 0.55, carbs_g=11.4, fat_g=0.15, fiber_g=1.3, magnesium_mg=13.5, potassium_mg=179.0, sodium_mg=0.5, polyunsaturated_fat_g=0.04, omega3_g=0.01, retention_basis="UNCOOKED_RAW_STATE"),
            IngredientItem("Crushed walnuts & almonds", "USDA_SR_LEGACY", "170187", 20.0, 93.0, 2.3, carbs_g=2.8, fat_g=8.5, fiber_g=1.5, magnesium_mg=18.0, potassium_mg=80.0, sodium_mg=7.5, polyunsaturated_fat_g=3.06, omega3_g=1.04, allergens=frozenset({"nuts", "tree_nuts"}), retention_basis="UNCOOKED_RAW_STATE"),
        ],
        provenance_notes="Composite breakfast: 40g oats (USDA 173904), 120ml low-fat milk (USDA 170872), 50g banana (USDA 173944), 20g walnuts/almonds (USDA 170187). Nutrients automatically aggregated (Mg=101.7mg, K=583.8mg). Gluten present via oats cross-contact (not wheat).",
    ),

    # -------------------------------------------------------------------------
    # 22. Wholegrain toast with peanut butter
    # -------------------------------------------------------------------------
    F(
        "Wholegrain toast with peanut butter",
        "2 slices (90g)",
        329,
        [B, S],
        "vegetarian vegan halal high-protein low-GI gluten wheat peanut",
        food_id="GL-002",
        source_db="RECIPE_CALCULATED",
        source_id=None,
        record_type="CALCULATED_COMPOSITE",
        data_provenance_status="INGREDIENT_AGGREGATED_ESTIMATE",
        allergen_review_status="FORMULATION_REVIEWED_DECOMPOSITION",
        glycemic_index=51,
        gi_method="PRIMARY_CLINICAL_TRIAL",
        gi_source="Primary clinical trial: Foster-Powell et al. (2002), Am J Clin Nutr (Whole wheat bread with peanut butter GI=51+-4); compiled in Atkinson et al. (2021) systematic review (Food ID 54).",
        retention_notes="Toasted bread and raw peanut butter assembly (100% retention).",
        ingredients=[
            IngredientItem("Whole wheat bread toasted (2 slices)", "USDA_SR_LEGACY", "172689", 60.0, 150.0, 6.0, carbs_g=24.0, fat_g=2.0, fiber_g=4.0, magnesium_mg=24.0, potassium_mg=90.0, sodium_mg=180.0, polyunsaturated_fat_g=0.8, omega3_g=0.01, allergens=frozenset({"gluten", "wheat"}), retention_basis="MEASURED_COOKED_ASSAY"),
            IngredientItem("Smooth natural peanut butter (2 tbsp)", "USDA_SR_LEGACY", "172470", 30.0, 179.0, 6.66, carbs_g=6.8, fat_g=15.0, fiber_g=2.0, magnesium_mg=50.4, potassium_mg=167.4, sodium_mg=40.0, polyunsaturated_fat_g=3.3, omega3_g=0.01, allergens=frozenset({"peanut", "peanuts"}), retention_basis="UNCOOKED_RAW_STATE"),
        ],
        provenance_notes="60g whole wheat toast (USDA 172689) with 30g natural peanut butter (USDA 172470). Nutrients automatically aggregated (Mg=74.4mg, K=257.4mg). Wheat and peanut allergens present.",
    ),

    # -------------------------------------------------------------------------
    # 23. Yoghurt with mixed nuts
    # -------------------------------------------------------------------------
    F(
        "Yoghurt with mixed nuts",
        "1 cup (225g)",
        280,
        [S],
        "vegetarian halal high-protein low-GI magnesium dairy nuts",
        food_id="GL-003",
        source_db="RECIPE_CALCULATED",
        source_id=None,
        record_type="CALCULATED_COMPOSITE",
        data_provenance_status="INGREDIENT_AGGREGATED_ESTIMATE",
        allergen_review_status="FORMULATION_REVIEWED_DECOMPOSITION",
        glycemic_index=35,
        gi_method="INGREDIENT_PROXY_ESTIMATE",
        gi_source="Systematic review compilation: Atkinson et al. (2021) Table 2 (Food ID 1042: Low-fat plain yogurt GI=35, primary trial by Wolever et al.). Unverified for composite nut addition.",
        retention_notes="Uncooked yogurt and dry-roasted nut assembly (100% retention).",
        ingredients=[
            IngredientItem("Low-fat plain yogurt", "USDA_SR_LEGACY", "170886", 200.0, 126.0, 10.5, carbs_g=14.0, fat_g=3.1, fiber_g=0.0, magnesium_mg=34.0, potassium_mg=468.0, sodium_mg=140.0, polyunsaturated_fat_g=0.1, omega3_g=0.01, allergens=frozenset({"dairy", "milk"}), retention_basis="UNCOOKED_RAW_STATE"),
            IngredientItem("Roasted cashew nuts & almonds", "USDA_SR_LEGACY", "170571", 25.0, 154.0, 4.2, carbs_g=4.0, fat_g=12.5, fiber_g=1.2, magnesium_mg=68.0, potassium_mg=150.0, sodium_mg=5.0, polyunsaturated_fat_g=2.7, omega3_g=0.14, allergens=frozenset({"nuts", "tree_nuts"}), retention_basis="MEASURED_COOKED_ASSAY"),
        ],
        provenance_notes="200g plain low-fat yogurt (USDA 170886) with 25g dry-roasted cashews and almonds. Nutrients automatically aggregated.",
    ),

    # -------------------------------------------------------------------------
    # 24. Roasted cashews
    # -------------------------------------------------------------------------
    F(
        "Roasted cashews",
        "30 g",
        170,
        [S],
        "vegetarian vegan halal low-GI magnesium nuts",
        food_id="GL-004",
        source_db="USDA_SR_LEGACY",
        source_id="170571",
        record_type="PRIMARY_LABORATORY_ASSAY",
        data_provenance_status="DIRECT_LAB_MATCH",
        serving_weight_g=30.0,
        protein_g=4.59,
        carbs_g=9.81,
        fat_g=13.91,
        fiber_g=0.90,
        magnesium_mg=78.0,
        potassium_mg=169.5,
        sodium_mg=4.8,
        polyunsaturated_fat_g=2.35,
        omega3_g=0.02,
        glycemic_index=25,
        gi_method="PRIMARY_CLINICAL_TRIAL",
        gi_source="Primary clinical trial: Brand-Miller et al. (2003); compiled in Atkinson et al. (2021) systematic review (Food ID 1362: Dry roasted cashews GI=25+-4).",
        allergen_review_status="FORMULATION_REVIEWED_DECOMPOSITION",
        provenance_notes="PRIMARY LABORATORY RECORD: Exactly 30g dry-roasted cashew nuts without added salt (USDA SR Legacy 170571). Nutrients strictly scaled from 100g baseline.",
    ),

    # -------------------------------------------------------------------------
    # 25. Dark chocolate (70%+)
    # -------------------------------------------------------------------------
    F(
        "Dark chocolate (70%+)",
        "2 squares (20g)",
        120,
        [S],
        "vegetarian halal low-GI magnesium caffeine",
        food_id="GL-005",
        source_db="USDA_SR_LEGACY",
        source_id="170273",
        record_type="PRIMARY_LABORATORY_ASSAY",
        data_provenance_status="DIRECT_LAB_MATCH",
        serving_weight_g=20.0,
        protein_g=1.56,
        carbs_g=7.00,  # Available carbohydrate (total carb 9.18g minus 2.18g fiber)
        fat_g=8.53,
        fiber_g=2.18,
        magnesium_mg=45.6,
        potassium_mg=143.0,
        sodium_mg=4.0,
        polyunsaturated_fat_g=0.25,
        omega3_g=0.01,
        glycemic_index=23,
        gi_method="PRIMARY_CLINICAL_TRIAL",
        gi_source="Primary clinical trial: Brand-Miller et al. (2003); compiled in Atkinson et al. (2021) systematic review (Food ID 1405: Dark chocolate 70% cacao solids GI=23+-3).",
        allergen_review_status="FORMULATION_REVIEWED_DECOMPOSITION",
        provenance_notes="PRIMARY LABORATORY RECORD: 20g dark chocolate 70-85% cacao solids (USDA SR Legacy 170273). Contains ~16mg natural caffeine.",
    ),

    # -------------------------------------------------------------------------
    # 26. Grilled chicken salad
    # -------------------------------------------------------------------------
    F(
        "Grilled chicken salad",
        "1 bowl (285g)",
        380,
        [L, D],
        "halal high-protein",
        food_id="GL-006",
        source_db="RECIPE_CALCULATED",
        source_id=None,
        record_type="CALCULATED_COMPOSITE",
        data_provenance_status="INGREDIENT_AGGREGATED_ESTIMATE",
        allergen_review_status="FORMULATION_REVIEWED_DECOMPOSITION",
        glycemic_index=None,
        gi_method="NOT_APPLICABLE_NEGLIGIBLE_CARB",
        gi_source="Not applicable (negligible carbohydrate meal, <10g available carbs; in-vivo GI protocols require 25g/50g carbohydrate).",
        retention_notes="Nutrient retention factor 1.0 based on grilled chicken breast assay (USDA 171477). Fresh raw salad greens.",
        ingredients=[
            IngredientItem("Grilled skinless chicken breast", "USDA_SR_LEGACY", "171477", 120.0, 198.0, 37.2, carbs_g=0.0, fat_g=4.3, fiber_g=0.0, magnesium_mg=34.8, potassium_mg=307.2, sodium_mg=90.0, polyunsaturated_fat_g=0.9, omega3_g=0.04, retention_basis="MEASURED_COOKED_ASSAY"),
            IngredientItem("Mixed leafy greens, cucumber, tomato", "USDA_SR_LEGACY", "168441", 150.0, 47.0, 2.0, carbs_g=8.0, fat_g=0.5, fiber_g=3.5, magnesium_mg=22.0, potassium_mg=240.0, sodium_mg=120.0, polyunsaturated_fat_g=0.2, omega3_g=0.11, retention_basis="UNCOOKED_RAW_STATE"),
            IngredientItem("Extra virgin olive oil herb dressing", "USDA_SR_LEGACY", "171413", 15.0, 135.0, 0.0, carbs_g=0.0, fat_g=15.0, fiber_g=0.0, magnesium_mg=0.0, potassium_mg=0.0, sodium_mg=0.0, polyunsaturated_fat_g=1.0, omega3_g=0.00, retention_basis="UNCOOKED_RAW_STATE"),
        ],
        provenance_notes="Composite lunch: 120g grilled chicken, 150g salad greens, 15ml olive oil dressing. Nutrients automatically aggregated (K=547.2mg, Mg=56.8mg).",
    ),

    # -------------------------------------------------------------------------
    # 27. Lentil soup with wholegrain bread - CRITICAL AUDIT FIX
    # -------------------------------------------------------------------------
    F(
        "Lentil soup with wholegrain bread",
        "1 bowl (360g)",
        400,
        [L, D],
        "vegetarian vegan halal high-protein complex-carb low-GI gluten wheat",
        food_id="GL-007",
        source_db="RECIPE_CALCULATED",
        source_id=None,
        record_type="CALCULATED_COMPOSITE",
        data_provenance_status="INGREDIENT_AGGREGATED_ESTIMATE",
        allergen_review_status="FORMULATION_REVIEWED_DECOMPOSITION",
        glycemic_index=46,  # FIXED: (0.625 * 32) + (0.375 * 69) = 20.0 + 25.88 = 45.88 -> 46
        gi_method="THEORETICAL_WEIGHTED_CALCULATION",
        gi_source="Theoretical calculation via FAO/WHO (1998) weighted carbohydrate equation from ingredient values in Atkinson et al. (2021) systematic review: Boiled lentils GI=32 (62.5% available carbs, 40g) + Whole wheat bread GI=69 (37.5% available carbs, 24g): (0.625 * 32) + (0.375 * 69) = 20.0 + 25.88 = 45.88 -> 46. Unmeasured as a complete meal.",
        retention_notes="Nutrient retention factor 1.0 based on cooked lentil soup and toasted bread assays.",
        ingredients=[
            IngredientItem("Cooked lentil soup (boiled mature lentils)", "USDA_SR_LEGACY", "172421", 300.0, 250.0, 18.0, carbs_g=40.0, fat_g=1.2, fiber_g=15.0, magnesium_mg=72.0, potassium_mg=738.0, sodium_mg=200.0, polyunsaturated_fat_g=0.4, omega3_g=0.11, retention_basis="MEASURED_COOKED_ASSAY"),
            IngredientItem("Whole wheat bread toasted (2 slices)", "USDA_SR_LEGACY", "172689", 60.0, 150.0, 6.0, carbs_g=24.0, fat_g=2.0, fiber_g=4.0, magnesium_mg=24.0, potassium_mg=90.0, sodium_mg=180.0, polyunsaturated_fat_g=0.8, omega3_g=0.01, allergens=frozenset({"gluten", "wheat"}), retention_basis="MEASURED_COOKED_ASSAY"),
        ],
        provenance_notes="300g cooked lentil soup (USDA 172421 boiled mature lentils) paired with 2 slices whole wheat toast (USDA 172689). Nutrients automatically aggregated (Mg=96.0mg, K=828.0mg). Wheat allergen present.",
    ),

    # -------------------------------------------------------------------------
    # 28. Baked salmon with vegetables
    # -------------------------------------------------------------------------
    F(
        "Baked salmon with vegetables",
        "1 plate (325g)",
        520,
        [L, D],
        "halal omega-3 high-protein fish",
        food_id="GL-008",
        source_db="RECIPE_CALCULATED",
        source_id=None,
        record_type="CALCULATED_COMPOSITE",
        data_provenance_status="INGREDIENT_AGGREGATED_ESTIMATE",
        allergen_review_status="FORMULATION_REVIEWED_DECOMPOSITION",
        glycemic_index=None,
        gi_method="NOT_APPLICABLE_NEGLIGIBLE_CARB",
        gi_source="Not applicable (negligible starchy carbohydrate, <15g available carbs).",
        retention_notes="Nutrient retention factor 1.0 based on baked salmon and steamed vegetable assays.",
        ingredients=[
            IngredientItem("Baked wild Atlantic salmon fillet", "USDA_SR_LEGACY", "171998", 160.0, 291.0, 40.7, carbs_g=0.0, fat_g=13.0, fiber_g=0.0, magnesium_mg=59.2, potassium_mg=1004.8, sodium_mg=80.0, polyunsaturated_fat_g=4.21, omega3_g=2.94, allergens=frozenset({"fish"}), retention_basis="MEASURED_COOKED_ASSAY"),
            IngredientItem("Steamed broccoli and carrots", "USDA_SR_LEGACY", "170379", 150.0, 94.0, 4.0, carbs_g=12.0, fat_g=0.5, fiber_g=4.0, magnesium_mg=35.0, potassium_mg=450.0, sodium_mg=140.0, polyunsaturated_fat_g=0.0, omega3_g=0.00, retention_basis="MEASURED_COOKED_ASSAY"),
            IngredientItem("Olive oil lemon glaze", "USDA_SR_LEGACY", "171413", 15.0, 135.0, 0.0, carbs_g=0.0, fat_g=15.0, fiber_g=0.0, magnesium_mg=0.0, potassium_mg=0.0, sodium_mg=0.0, polyunsaturated_fat_g=1.0, omega3_g=0.00, retention_basis="UNCOOKED_RAW_STATE"),
        ],
        provenance_notes="160g cooked wild Atlantic salmon (USDA 171998) + 150g steamed vegetables. True Omega-3 is 2.94g. Nutrients automatically aggregated.",
    ),

    # -------------------------------------------------------------------------
    # 29. Fruit salad
    # -------------------------------------------------------------------------
    F(
        "Fruit salad",
        "1 bowl (250g)",
        150,
        [S],
        "vegetarian vegan halal low-GI",
        food_id="GL-009",
        source_db="RECIPE_CALCULATED",
        source_id=None,
        record_type="CALCULATED_COMPOSITE",
        data_provenance_status="INGREDIENT_AGGREGATED_ESTIMATE",
        allergen_review_status="FORMULATION_REVIEWED_DECOMPOSITION",
        glycemic_index=44,
        gi_method="THEORETICAL_WEIGHTED_CALCULATION",
        gi_source="Theoretical calculation via FAO/WHO (1998) weighted carbohydrate equation from fresh fruit values in Atkinson et al. (2021) systematic review: Watermelon GI=72, Apple GI=36, Papaya GI=38, Banana GI=51. Unmeasured as a mixed bowl.",
        retention_notes="Uncooked fresh raw fruit assembly (100% retention).",
        ingredients=[
            IngredientItem("Diced fresh watermelon & apple", "USDA_SR_LEGACY", "169092", 120.0, 65.0, 0.6, carbs_g=17.0, fat_g=0.2, fiber_g=1.8, magnesium_mg=10.0, potassium_mg=160.0, sodium_mg=3.0, polyunsaturated_fat_g=0.1, omega3_g=0.01, retention_basis="UNCOOKED_RAW_STATE"),
            IngredientItem("Diced fresh papaya & banana", "USDA_SR_LEGACY", "169926", 130.0, 85.0, 0.9, carbs_g=21.0, fat_g=0.3, fiber_g=2.2, magnesium_mg=18.0, potassium_mg=260.0, sodium_mg=5.0, polyunsaturated_fat_g=0.1, omega3_g=0.02, retention_basis="UNCOOKED_RAW_STATE"),
        ],
        provenance_notes="250g mixed fresh cut raw papaya, watermelon, banana, and apple. Nutrients automatically aggregated.",
    ),

    # -------------------------------------------------------------------------
    # 30. Green tea
    # -------------------------------------------------------------------------
    F(
        "Green tea",
        "1 cup (240ml)",
        2,
        [B, S],
        "vegetarian vegan halal caffeine",
        food_id="GL-010",
        source_db="USDA_SR_LEGACY",
        source_id="171917",
        record_type="PRIMARY_LABORATORY_ASSAY",
        data_provenance_status="DIRECT_LAB_MATCH",
        serving_weight_g=240.0,
        protein_g=0.53,
        carbs_g=0.0,
        fat_g=0.0,
        fiber_g=0.0,
        magnesium_mg=2.4,
        potassium_mg=19.2,
        sodium_mg=2.4,
        polyunsaturated_fat_g=0.0,
        omega3_g=0.0,
        glycemic_index=None,
        gi_method="NOT_APPLICABLE_NEGLIGIBLE_CARB",
        gi_source="Not applicable (zero carbohydrate beverage).",
        allergen_review_status="ASSAY_VERIFIED_ABSENCE",
        provenance_notes="PRIMARY LABORATORY RECORD: 8 fl oz brewed unsweetened green tea (USDA SR Legacy 171917). Contains ~30mg natural caffeine and EGCG catechins. Zero carbs (GI not applicable).",
    ),

    # -------------------------------------------------------------------------
    # 31. Black coffee
    # -------------------------------------------------------------------------
    F(
        "Black coffee",
        "1 cup (240ml)",
        2,
        [B, S],
        "vegetarian vegan halal caffeine",
        food_id="GL-011",
        source_db="USDA_SR_LEGACY",
        source_id="171890",
        record_type="PRIMARY_LABORATORY_ASSAY",
        data_provenance_status="DIRECT_LAB_MATCH",
        serving_weight_g=240.0,
        protein_g=0.29,
        carbs_g=0.0,
        fat_g=0.05,
        fiber_g=0.0,
        magnesium_mg=7.2,
        potassium_mg=117.6,
        sodium_mg=4.8,
        polyunsaturated_fat_g=0.002,
        omega3_g=0.0,
        glycemic_index=None,
        gi_method="NOT_APPLICABLE_NEGLIGIBLE_CARB",
        gi_source="Not applicable (zero carbohydrate beverage).",
        allergen_review_status="ASSAY_VERIFIED_ABSENCE",
        provenance_notes="PRIMARY LABORATORY RECORD: 8 fl oz brewed coffee from ground beans with tap water (USDA SR Legacy 171890). Contains ~95mg caffeine. Zero carbs (GI not applicable).",
    ),

    # -------------------------------------------------------------------------
    # 32. Milk tea (kiri thé)
    # -------------------------------------------------------------------------
    F(
        "Milk tea (kiri thé)",
        "1 cup (180ml)",
        130,
        [B, S],
        "vegetarian halal caffeine high-sugar dairy",
        food_id="GL-012",
        source_db="RECIPE_CALCULATED",
        source_id=None,
        record_type="CALCULATED_COMPOSITE",
        data_provenance_status="INGREDIENT_AGGREGATED_ESTIMATE",
        allergen_review_status="FORMULATION_REVIEWED_DECOMPOSITION",
        glycemic_index=None,
        gi_method="UNAVAILABLE",
        gi_source="In-vivo clinical testing unavailable for sweetened commercial tea beverage.",
        retention_notes="Liquid beverage formulation (100% retention).",
        ingredients=[
            IngredientItem("Brewed black tea", "USDA_SR_LEGACY", "173227", 140.0, 2.0, 0.0, carbs_g=0.0, fat_g=0.0, fiber_g=0.0, magnesium_mg=4.0, potassium_mg=40.0, sodium_mg=4.0, polyunsaturated_fat_g=0.0, omega3_g=0.0, retention_basis="MEASURED_COOKED_ASSAY"),
            IngredientItem("Sweetened condensed milk", "USDA_SR_LEGACY", "171275", 40.0, 128.0, 3.1, carbs_g=21.6, fat_g=3.5, fiber_g=0.0, magnesium_mg=10.0, potassium_mg=110.0, sodium_mg=50.0, polyunsaturated_fat_g=0.1, omega3_g=0.01, allergens=frozenset({"dairy", "milk"}), retention_basis="UNCOOKED_RAW_STATE"),
        ],
        provenance_notes="Brewed strong Ceylon black tea (USDA 173227) sweetened with condensed milk (USDA 171275). Nutrients automatically aggregated.",
    ),

    # -------------------------------------------------------------------------
    # 33. Instant noodles
    # -------------------------------------------------------------------------
    F(
        "Instant noodles",
        "1 pack (85g)",
        374,
        [L, D, S],
        "high-sodium high-GI fried gluten wheat",
        food_id="GL-013",
        source_db="USDA_SR_LEGACY",
        source_id="171177",
        record_type="PRIMARY_LABORATORY_ASSAY",
        data_provenance_status="DIRECT_LAB_MATCH",
        serving_weight_g=85.0,
        protein_g=8.64,
        carbs_g=51.22,
        fat_g=14.95,
        fiber_g=2.47,
        magnesium_mg=21.25,
        potassium_mg=153.85,
        sodium_mg=1576.75,
        polyunsaturated_fat_g=1.87,
        omega3_g=0.05,
        glycemic_index=72,
        gi_method="PRIMARY_CLINICAL_TRIAL",
        gi_source="Primary clinical trial: Sugiyama et al. (2003); compiled in Atkinson et al. (2021) systematic review (Food ID 1502: Instant ramen noodles GI=72+-6).",
        allergen_review_status="FORMULATION_REVIEWED_DECOMPOSITION",
        provenance_notes="PRIMARY LABORATORY RECORD: Single commercial pack dry ramen noodle soup (USDA SR Legacy 171177). High sodium (1577mg) and high GI (72). Strictly contraindicated in hypertension and diabetes.",
    ),
]
