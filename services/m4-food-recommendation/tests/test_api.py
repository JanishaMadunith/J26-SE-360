import uuid
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from fastapi.testclient import TestClient

from app.main import create_app
from m4_food.foods import FOODS
from wellness_contracts import MealType, Recommendations

COLOMBO = ZoneInfo("Asia/Colombo")
AS_OF = datetime(2026, 10, 19, 6, 30, tzinfo=COLOMBO)
UID = str(uuid.uuid4())
client = TestClient(create_app())


def affective(label):
    probs = dict.fromkeys(("stress", "relaxed", "happy", "sad", "anxiety"), 0.05)
    probs[label] = 0.8
    return {
        "user_id": UID,
        "window_start": (AS_OF - timedelta(hours=1)).isoformat(),
        "window_end": AS_OF.isoformat(),
        "label": label,
        "probabilities": probs,
        "confidence": 0.8,
        "model": {"name": "t", "version": "0"},
    }


def schedule(items):
    return {
        "user_id": UID,
        "generated_at": AS_OF.isoformat(),
        "horizon_start": AS_OF.isoformat(),
        "horizon_end": (AS_OF + timedelta(days=1)).isoformat(),
        "reason": "scheduled",
        "model": {"name": "t", "version": "0"},
        "items": [
            {"task_id": str(uuid.uuid4()), "start": s.isoformat(), "end": e.isoformat(), "completion_probability": 0.7}
            for s, e in items
        ],
    }


def body(**kw):
    b = {"user_id": UID, "as_of": AS_OF.isoformat(), "timezone": "Asia/Colombo"}
    b.update(kw)
    return b


def post(**kw) -> Recommendations:
    r = client.post("/v1/recommendations", json=body(**kw))
    assert r.status_code == 200, r.text
    return Recommendations.model_validate(r.json())


def all_foods(rec):
    return [f.name for m in rec.meals for f in m.foods]


def test_health():
    assert client.get("/health").json()["service"] == "m4-food-recommendation"


def test_minimal_request_gives_four_meals_in_order():
    rec = post()
    assert [m.meal_type for m in rec.meals] == [MealType.breakfast, MealType.lunch, MealType.snack, MealType.dinner]
    assert all(m.foods for m in rec.meals) and rec.suggested_activities == []


def test_meal_moved_out_of_busy_schedule():
    lunch = AS_OF.replace(hour=12, minute=0)
    rec = post(schedule=schedule([(lunch, lunch + timedelta(hours=2))]))
    m = next(m for m in rec.meals if m.meal_type == MealType.lunch)
    assert m.window_end <= lunch or m.window_start >= lunch + timedelta(hours=2)
    assert "adjusted" in m.rationale


def test_vegan_halal_and_allergy_filters():
    rec = post(profile={"vegan": True, "allergies": ["coconut", "gluten"]})
    by_name = {f.name: f for f in FOODS}
    for n in all_foods(rec):
        assert "vegan" in by_name[n].tags and not ({"coconut", "gluten"} & by_name[n].tags)
    rec = post(profile={"halal": True})
    assert all("halal" in by_name[n].tags for n in all_foods(rec))


def test_diabetes_excludes_high_gi():
    by_name = {f.name: f for f in FOODS}
    rec = post(profile={"health_conditions": ["Type 2 diabetes"]})
    assert all("high-GI" not in by_name[n].tags for n in all_foods(rec))


def test_stress_prefers_calming_foods_and_suggests_walk():
    by_name = {f.name: f for f in FOODS}
    rec = post(affective_state=affective("stress"))
    names = all_foods(rec)
    assert not any("caffeine" in by_name[n].tags for n in names)
    assert any({"omega-3", "magnesium"} & by_name[n].tags for n in names)
    assert rec.suggested_activities and rec.suggested_activities[0].category.value == "exercise"


def test_no_caffeine_after_3pm():
    late = AS_OF.replace(hour=15, minute=30)
    by_name = {f.name: f for f in FOODS}
    rec = Recommendations.model_validate(client.post("/v1/recommendations", json=body(as_of=late.isoformat())).json())
    for m in rec.meals:
        if m.window_start.astimezone(COLOMBO).hour >= 15:
            assert not any("caffeine" in by_name[f.name].tags for f in m.foods)


def test_variety_no_repeats_in_a_day():
    names = all_foods(post(affective_state=affective("stress"), profile={"vegetarian": True}))
    assert len(names) == len(set(names))
