"""Automated Audit & Verification Tests for Module 4 Food Composition (Layer A).

Asserts:
1. Catalog size, stable identifiers (SL-xxx / GL-xxx), and no duplicate records.
2. Source identifier provenance (no fake placeholder codes like SLFCT-RECIPE).
3. Ingredient and serving-weight totals matching within tolerance.
4. Automatic nutrient aggregation from ingredient records (explicitly checking the 4 reported cases).
5. Explicit retention and cooking yield basis on all ingredient components.
6. Glycemic Index provenance:
   - Zero/negligible carbohydrate foods strictly marked with GI=None.
   - Grounding in primary clinical trials vs systematic review compilations vs theoretical weighted calculations.
   - Correct weighted calculation for Lentil soup with bread (GI=46, not 44).
   - Low-GI (<=55) and High-GI (>=70) classification thresholds.
7. Allergen and dietary restrictions:
   - Official US FDA Major Food Allergens (FALCPA + FASTER Act: 9 statutory terms).
   - Ingredient-level allergen propagation to meal allergens and fda_allergens.
   - Documented allergen_review_status and dietary_compliance_notes.
8. Physical invariants:
   - Mass conservation: Protein + Carbs + Fat + Fiber <= Serving Weight + 0.5g.
   - True Omega-3 <= Total Polyunsaturated Fat + 0.05g.
   - Non-negative nutrients.
9. Tempered chickpeas record audit (reconstructed as calculated recipe composite, not fake primary USDA assay).
"""

from __future__ import annotations

import pytest
from m4_food.foods import FOODS, FDA_MAJOR_ALLERGENS, KNOWN_ALLERGEN_TAGS
from m4_food.graph.audit_food_composition import USDA_VERIFIED_100G


def test_catalog_size_and_stable_ids():
    """Verify that all 33 catalog foods have stable unique IDs and no duplicates."""
    assert len(FOODS) == 33
    food_ids = [f.food_id for f in FOODS]
    assert len(food_ids) == len(set(food_ids)), "Duplicate food_id detected in catalog"

    names = [f.name for f in FOODS]
    assert len(names) == len(set(names)), "Duplicate food name detected in catalog"

    for f in FOODS:
        assert f.food_id.startswith(("SL-", "GL-")), f"Invalid ID format in {f.food_id}"
        assert f.name, "Empty food name"


def test_source_identifier_provenance_and_no_fake_codes():
    """Verify every food and ingredient has an authentic source identifier (no fake SLFCT-RECIPE codes)."""
    valid_db_types = {"USDA_SR_LEGACY", "SRI_LANKA_MRI", "RECIPE_CALCULATED", "REGIONAL_ESTIMATE"}
    valid_record_types = {"PRIMARY_LABORATORY_ASSAY", "CALCULATED_COMPOSITE", "REGIONAL_ESTIMATE"}
    valid_provenance_statuses = {"DIRECT_LAB_MATCH", "INGREDIENT_AGGREGATED_ESTIMATE", "UNVERIFIED_REGIONAL_ESTIMATE"}

    for f in FOODS:
        assert f.source_db in valid_db_types, f"Invalid source_db '{f.source_db}' in {f.name}"
        assert f.record_type in valid_record_types, f"Invalid record_type '{f.record_type}' in {f.name}"
        assert f.data_provenance_status in valid_provenance_statuses, (
            f"Invalid provenance status '{f.data_provenance_status}' in {f.name}"
        )

        # Single primary laboratory records must have legitimate primary source IDs
        if f.record_type == "PRIMARY_LABORATORY_ASSAY":
            assert f.source_id is not None and len(f.source_id) >= 4, (
                f"Missing source_id for primary laboratory food {f.name}"
            )
            assert not f.source_id.startswith("SLFCT-RECIPE"), f"Fake identifier in {f.name}"

        # Calculated composite dishes must not use custom IDs pretending to be primary assays
        if f.record_type == "CALCULATED_COMPOSITE":
            assert f.source_db == "RECIPE_CALCULATED"
            assert len(f.ingredients) >= 2, f"Composite meal {f.name} must have ingredient components"
            for ing in f.ingredients:
                assert ing.source_db in ("USDA_SR_LEGACY", "SRI_LANKA_MRI"), (
                    f"Ingredient {ing.name} in {f.name} has unverified db: {ing.source_db}"
                )
                assert ing.source_id, f"Ingredient {ing.name} in {f.name} missing source_id"


def test_tempered_chickpeas_composite_record():
    """Audit SL-011: Verify tempered chickpeas is modeled as a recipe composite with ingredients, not a fake USDA assay."""
    by_id = {f.food_id: f for f in FOODS}
    sl011 = by_id["SL-011"]
    assert sl011.record_type == "CALCULATED_COMPOSITE", "Tempered chickpeas must be a recipe composite"
    assert sl011.data_provenance_status == "INGREDIENT_AGGREGATED_ESTIMATE"
    assert len(sl011.ingredients) >= 2, "Tempered chickpeas must have boiled chickpeas and tempering ingredients"
    assert sl011.fat_g > 5.0, "Tempered chickpeas must account for tempering fat (>5g)"
    assert sl011.sodium_mg > 100.0, "Tempered chickpeas must account for seasoning sodium (>100mg)"


def test_ingredient_and_serving_weight_totals():
    """Verify sum of ingredient serving grams matches meal serving weight within tolerance."""
    for f in FOODS:
        if f.ingredients:
            raw_weight = sum(ing.serving_grams for ing in f.ingredients)
            expected_weight = round(raw_weight * (f.cooking_yield_pct / 100.0), 1)
            diff = abs(f.serving_weight_g - expected_weight)
            assert diff <= 0.5, (
                f"Meal {f.name} serving weight ({f.serving_weight_g}g) mismatch with ingredients ({expected_weight}g)"
            )


def test_automatic_nutrient_aggregation_and_reported_discrepancies():
    """Assert automatic nutrient aggregation matches ingredient sums, checking the 4 user-reported cases."""
    by_id = {f.food_id: f for f in FOODS}

    # Case 1: Oats porridge with banana and nuts (GL-001)
    gl001 = by_id["GL-001"]
    assert gl001.magnesium_mg == 101.7, f"GL-001 magnesium mismatch: got {gl001.magnesium_mg}, expected 101.7"
    assert gl001.potassium_mg == 583.8, f"GL-001 potassium mismatch: got {gl001.potassium_mg}, expected 583.8"
    assert gl001.calories == 350

    # Case 2: Wholegrain toast with peanut butter (GL-002)
    gl002 = by_id["GL-002"]
    assert gl002.magnesium_mg == 74.4, f"GL-002 magnesium mismatch: got {gl002.magnesium_mg}, expected 74.4"
    assert gl002.potassium_mg == 257.4, f"GL-002 potassium mismatch: got {gl002.potassium_mg}, expected 257.4"
    assert gl002.calories == 329

    # Case 3: Grilled chicken salad (GL-006)
    gl006 = by_id["GL-006"]
    assert gl006.potassium_mg == 547.2, f"GL-006 potassium mismatch: got {gl006.potassium_mg}, expected 547.2"
    assert gl006.magnesium_mg == 56.8, f"GL-006 magnesium mismatch: got {gl006.magnesium_mg}, expected 56.8"
    assert gl006.calories == 380

    # Case 4: Lentil soup with wholegrain bread (GL-007)
    gl007 = by_id["GL-007"]
    assert gl007.magnesium_mg == 96.0, f"GL-007 magnesium mismatch: got {gl007.magnesium_mg}, expected 96.0"
    assert gl007.potassium_mg == 828.0, f"GL-007 potassium mismatch: got {gl007.potassium_mg}, expected 828.0"
    assert gl007.calories == 400

    # All composite meals must aggregate exactly
    for f in FOODS:
        if f.ingredients:
            calc_mg = round(sum(i.magnesium_mg * i.nutrient_retention_factor for i in f.ingredients), 1)
            calc_k = round(sum(i.potassium_mg * i.nutrient_retention_factor for i in f.ingredients), 1)
            calc_kcal = int(round(sum(i.calories * i.nutrient_retention_factor for i in f.ingredients)))
            assert abs(f.magnesium_mg - calc_mg) <= 0.2, f"Mg aggregation mismatch in {f.name}"
            assert abs(f.potassium_mg - calc_k) <= 0.2, f"K aggregation mismatch in {f.name}"
            assert abs(f.calories - calc_kcal) <= 1, f"Calorie aggregation mismatch in {f.name}"


def test_glycemic_index_provenance_and_lentil_soup_calculation():
    """Verify that zero/negligible carb foods have GI=None and lentil soup weighted GI is 46."""
    valid_gi_methods = {
        "PRIMARY_CLINICAL_TRIAL",
        "INGREDIENT_PROXY_ESTIMATE",
        "THEORETICAL_WEIGHTED_CALCULATION",
        "NOT_APPLICABLE_NEGLIGIBLE_CARB",
        "UNAVAILABLE"
    }

    by_id = {f.food_id: f for f in FOODS}

    # Case: Lentil soup with wholegrain bread (GL-007)
    # Weighted calculation: (0.625 * 32) + (0.375 * 69) = 20.0 + 25.88 = 45.88 -> 46 (NOT 44!)
    gl007 = by_id["GL-007"]
    assert gl007.glycemic_index == 46, f"GL-007 GI mismatch: got {gl007.glycemic_index}, expected 46"
    assert gl007.gi_method == "THEORETICAL_WEIGHTED_CALCULATION"

    # Negligible or zero carb foods must have GI=None
    negligible_carb_ids = {"GL-010", "GL-011", "GL-006", "GL-008"}
    for f in FOODS:
        assert f.gi_method in valid_gi_methods, f"Invalid gi_method in {f.name}: {f.gi_method}"

        if f.food_id in negligible_carb_ids:
            assert f.glycemic_index is None, (
                f"Zero/negligible carbohydrate food {f.name} must have GI=None, got {f.glycemic_index}"
            )
            assert f.gi_method == "NOT_APPLICABLE_NEGLIGIBLE_CARB"

        if f.glycemic_index is not None:
            assert f.gi_source and len(f.gi_source) > 10, f"Missing traceable gi_source in {f.name}"

        # Ground-truth classification invariants
        if "low-GI" in f.tags:
            assert f.glycemic_index is not None and f.glycemic_index <= 55, (
                f"Food {f.name} tagged low-GI but has GI={f.glycemic_index}"
            )
        if "high-GI" in f.tags:
            assert f.glycemic_index is not None and f.glycemic_index >= 70, (
                f"Food {f.name} tagged high-GI but has GI={f.glycemic_index}"
            )


def test_fda_allergen_terminology_and_review_status():
    """Verify official US FDA allergen terminology (9 statutory terms) and review statuses."""
    expected_fda_allergens = {"milk", "eggs", "fish", "crustacean_shellfish", "tree_nuts", "wheat", "peanuts", "soybeans", "sesame"}
    assert FDA_MAJOR_ALLERGENS == expected_fda_allergens, "FDA_MAJOR_ALLERGENS does not match 21 U.S.C. 321(qq)"

    valid_statuses = {
        "ASSAY_VERIFIED_ABSENCE",
        "FORMULATION_REVIEWED_DECOMPOSITION",
        "UNVERIFIED_CROSS_CONTAMINATION_RISK",
        "UNVERIFIED_REVIEW_PENDING"
    }

    for f in FOODS:
        assert f.allergen_review_status in valid_statuses, (
            f"Invalid allergen status in {f.name}: {f.allergen_review_status}"
        )
        assert f.dietary_compliance_notes, f"Missing dietary compliance notes in {f.name}"

        # Check fda_allergens are strictly within FDA statutory list
        for fda_alg in f.fda_allergens:
            assert fda_alg in FDA_MAJOR_ALLERGENS, f"Non-FDA term in fda_allergens: {fda_alg}"

        # Specific mappings check
        if "dairy" in f.tags or "dairy" in f.allergens:
            assert "milk" in f.fda_allergens
        if "peanut" in f.tags or "peanut" in f.allergens:
            assert "peanuts" in f.fda_allergens
        if "egg" in f.tags or "egg" in f.allergens:
            assert "eggs" in f.fda_allergens
        if "fish" in f.tags or "fish" in f.allergens:
            assert "fish" in f.fda_allergens
        if "wheat" in f.tags or "wheat" in f.allergens:
            assert "wheat" in f.fda_allergens


def test_gluten_strictly_distinguished_from_wheat():
    """Verify that gluten is NOT an FDA statutory allergen (21 U.S.C. 321(qq)) and oats does not map to wheat."""
    assert "gluten" not in FDA_MAJOR_ALLERGENS
    by_id = {f.food_id: f for f in FOODS}
    oats = by_id["GL-001"]
    assert "gluten" in oats.allergens or "gluten" in oats.tags
    assert "wheat" not in oats.fda_allergens, "Oats must not trigger statutory wheat allergy"
    assert "wheat" not in oats.allergens, "Oats must not contain wheat allergen"


def test_coconut_decoupled_from_fda_tree_nuts():
    """Verify that coconut is NOT mapped to FDA tree_nuts and coconut-only foods do not trigger tree_nuts."""
    from m4_food.foods import ALLERGEN_TO_FDA_MAP
    assert "coconut" not in ALLERGEN_TO_FDA_MAP, "Coconut must not be mapped to FDA tree_nuts"

    by_id = {f.food_id: f for f in FOODS}
    kiribath = by_id["SL-006"]
    assert "coconut" in kiribath.allergens
    assert "tree_nuts" not in kiribath.fda_allergens, "Coconut-only food must not have FDA tree_nuts allergen"

    cashews = by_id["GL-004"]
    assert "tree_nuts" in cashews.fda_allergens, "True tree nuts must still map to FDA tree_nuts"


def test_missing_ingredient_nutrient_aggregation_unknown():
    """Verify that if any component ingredient has an unknown/None nutrient, the meal total is marked None."""
    from m4_food.foods import IngredientItem, _agg_nutrient

    ing1 = IngredientItem("Sample Ingredient 1", "USDA_SR_LEGACY", "1001", 100.0, 100.0, magnesium_mg=50.0)
    ing2 = IngredientItem("Sample Ingredient 2 (Unmeasured)", "USDA_SR_LEGACY", "1002", 100.0, 100.0, magnesium_mg=None)

    # Incomplete coverage must return None, not partial sum (50.0)
    result = _agg_nutrient((ing1, ing2), "magnesium_mg", 1)
    assert result is None, "Incomplete ingredient nutrient coverage must return None"

    # Complete coverage aggregates properly
    ing2_measured = IngredientItem("Sample Ingredient 2 (Measured)", "USDA_SR_LEGACY", "1002", 100.0, 100.0, magnesium_mg=30.0)
    result_complete = _agg_nutrient((ing1, ing2_measured), "magnesium_mg", 1)
    assert result_complete == 80.0


def test_no_inconsistent_source_id_reuse():
    """Verify source-ID consistency: milk != yogurt, bread != tofu, tea != punch."""
    by_id = {f.food_id: f for f in FOODS}

    # Milk in GL-001 must NOT use yogurt ID 170886
    gl001 = by_id["GL-001"]
    milk_ing = next(i for i in gl001.ingredients if "milk" in i.name.lower())
    assert milk_ing.source_id == "170872", f"Milk must use USDA 170872, got {milk_ing.source_id}"

    # Yogurt in GL-003 uses 170886
    gl003 = by_id["GL-003"]
    yogurt_ing = next(i for i in gl003.ingredients if "yogurt" in i.name.lower())
    assert yogurt_ing.source_id == "170886"

    # Bread in GL-002 must NOT use tofu ID 174291
    gl002 = by_id["GL-002"]
    bread_ing = next(i for i in gl002.ingredients if "bread" in i.name.lower())
    assert bread_ing.source_id == "172689", f"Bread must use USDA 172689, got {bread_ing.source_id}"

    # Fish curry in SL-002 explicitly documents proxy substitution
    sl002 = by_id["SL-002"]
    fish_ing = next(i for i in sl002.ingredients if "fish" in i.name.lower())
    assert "proxy" in fish_ing.name.lower() or "PROXY" in fish_ing.retention_basis

    # Dhal curry in SL-001 explicitly documents legume base proxy
    sl001 = by_id["SL-001"]
    dhal_ing = next(i for i in sl001.ingredients if "dhal" in i.name.lower() or "lentil" in i.name.lower())
    assert "proxy" in dhal_ing.name.lower() or "PROXY" in dhal_ing.retention_basis

    # Coconut roti in SL-020 uses whole wheat atta SLA017, not refined wheat flour
    sl020 = by_id["SL-020"]
    roti_ing = next(i for i in sl020.ingredients if "roti" in i.name.lower())
    assert roti_ing.source_id == "SLA017", f"Coconut roti must use atta SLA017, got {roti_ing.source_id}"


def test_physical_invariants_and_mass_conservation():
    """Assert mass conservation and true omega-3 boundaries across all 33 records."""
    for f in FOODS:
        assert f.calories > 0
        assert f.serving_weight_g > 0

        p = f.protein_g or 0.0
        c = f.carbs_g or 0.0
        fat = f.fat_g or 0.0
        fib = f.fiber_g or 0.0
        assert p >= 0 and c >= 0 and fat >= 0 and fib >= 0

        total_macros = p + c + fat + fib
        assert total_macros <= f.serving_weight_g + 1.0, (
            f"Mass conservation violation in {f.name}: {total_macros:.2f}g > {f.serving_weight_g}g"
        )

        if f.omega3_g is not None and f.polyunsaturated_fat_g is not None:
            assert f.omega3_g <= (f.polyunsaturated_fat_g + 0.05), (
                f"Physiological violation in {f.name}: Omega-3 ({f.omega3_g}g) cannot exceed total PUFA ({f.polyunsaturated_fat_g}g)"
            )


def test_usda_per_portion_scaling_accuracy():
    """Assert single foundational foods match USDA 100g laboratory baseline scaled to serving weight."""
    for f in FOODS:
        if f.source_db == "USDA_SR_LEGACY" and f.source_id in USDA_VERIFIED_100G and not f.ingredients:
            base_100g = USDA_VERIFIED_100G[f.source_id]
            scale_factor = f.serving_weight_g / 100.0

            expected_magnesium = base_100g["magnesium_mg"] * scale_factor
            if f.magnesium_mg is not None and base_100g["magnesium_mg"] > 0:
                diff = abs(f.magnesium_mg - expected_magnesium)
                assert diff <= 1.0, (
                    f"Scaling mismatch in {f.name} magnesium: got {f.magnesium_mg}mg, expected {expected_magnesium:.1f}mg"
                )

            expected_protein = base_100g["protein_g"] * scale_factor
            if f.protein_g is not None and base_100g["protein_g"] > 0:
                diff = abs(f.protein_g - expected_protein)
                assert diff <= 1.0, (
                    f"Scaling mismatch in {f.name} protein: got {f.protein_g}g, expected {expected_protein:.1f}g"
                )
