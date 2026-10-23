"""Module 4 baseline: rule-based, context-aware meal recommendation.

1. MEAL WINDOWS  For each usual meal time in the next 24 h (breakfast 07:00, lunch 12:30, snack 16:00,
                 dinner 19:30 local) find the nearest free 30-minute gap in the M3 schedule (±90 min);
                 if none, keep the usual time and say the schedule is tight.
2. FILTER        Diet profile: vegetarian / vegan / halal, allergies and dislikes (matched against tags and
                 names); diabetes → no high-GI items; hypertension → no high-sodium items.
3. SCORE         Context rules from the affective state (M1) and the forecast energy at the meal time (M2):
                 stress/anxiety → omega-3, magnesium, complex carbs, avoid caffeine/fried/sugar;
                 sad → omega-3, complex carbs, protein; low energy → protein + complex carbs;
                 caffeine is avoided after 15:00 (sleep). Calories close to the per-meal share of the
                 daily target score higher; foods already suggested today are penalised (variety).
4. ACTIVITIES    Up to two suggestions (walk when stressed, short rest when fatigued, break when overloaded).
                 They are shown to the user only; if accepted they become normal tasks for M3.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from wellness_contracts import (
    AffectiveLabel,
    DietProfile,
    FoodItem,
    MealRecommendation,
    MealType,
    RecommendationRequest,
    ScheduleResult,
    SuggestedActivity,
    TaskCategory,
    WellnessForecast,
)

from .foods import FOODS, Food

USUAL_TIMES = {MealType.breakfast: (7, 0), MealType.lunch: (12, 30), MealType.snack: (16, 0), MealType.dinner: (19, 30)}
MEAL_MINUTES = 30
SEARCH_MINUTES = 90
CALORIE_SHARE = {MealType.breakfast: 0.25, MealType.lunch: 0.35, MealType.dinner: 0.30, MealType.snack: 0.10}
ITEMS_PER_MEAL = {MealType.breakfast: 2, MealType.lunch: 1, MealType.dinner: 1, MealType.snack: 1}
DEFAULT_DAILY_KCAL = 2000
VARIETY_PENALTY = 1.5  # per earlier recommendation of the same food in this 24 h plan

REASONS = {
    "omega-3": "omega-3 fats are linked to better mood regulation",
    "magnesium": "a good source of magnesium, often low during stress",
    "complex-carb": "complex carbohydrates give steady energy",
    "high-protein": "protein helps keep energy and focus stable",
    "low-GI": "low glycaemic index supports gradual carbohydrate absorption",
}


# ------------------------------------------------------------------------------------------ windows
def busy_intervals(schedule: ScheduleResult | None) -> list[tuple[datetime, datetime]]:
    return [] if schedule is None else sorted((i.start, i.end) for i in schedule.items)


def is_free(start: datetime, end: datetime, busy: list[tuple[datetime, datetime]]) -> bool:
    return all(end <= b0 or start >= b1 for b0, b1 in busy)


def meal_windows(as_of: datetime, tz: ZoneInfo, busy) -> list[tuple[MealType, datetime, datetime, bool]]:
    """(meal, start, end, moved_or_tight) for each meal in the next 24 h, in time order."""
    local_now = as_of.astimezone(tz)
    out = []
    for day in (0, 1):
        for meal, (h, m) in USUAL_TIMES.items():
            usual = (local_now + timedelta(days=day)).replace(hour=h, minute=m, second=0, microsecond=0)
            if usual + timedelta(minutes=MEAL_MINUTES) <= as_of or usual >= as_of + timedelta(hours=24):
                continue
            best = None
            for delta in sorted(range(-SEARCH_MINUTES, SEARCH_MINUTES + 1, 15), key=abs):
                s = usual + timedelta(minutes=delta)
                e = s + timedelta(minutes=MEAL_MINUTES)
                if s >= as_of - timedelta(minutes=15) and is_free(s, e, busy):
                    best = (s, e, delta != 0)
                    break
            if best is None:
                # No 30-minute gap near the usual time: take the first free 15-minute gap in the next 3 hours
                # (a quick meal) rather than placing the meal on top of a scheduled task.
                for delta in range(0, 181, 15):
                    s = max(usual, as_of) + timedelta(minutes=delta)
                    e = s + timedelta(minutes=15)
                    if is_free(s, e, busy):
                        best = (s, e, True)
                        break
            if best is None:  # the whole afternoon is booked — keep the usual time and flag it
                best = (max(usual, as_of), max(usual, as_of) + timedelta(minutes=MEAL_MINUTES), True)
            out.append((meal, best[0], best[1], best[2] or not is_free(best[0], best[1], busy)))
    return sorted(out, key=lambda x: x[1])


# ------------------------------------------------------------------------------------------ filtering
def allowed(food: Food, p: DietProfile) -> bool:
    if p.vegan and "vegan" not in food.tags:
        return False
    if p.vegetarian and "vegetarian" not in food.tags:
        return False
    if p.halal and "halal" not in food.tags:
        return False
    name = food.name.lower()
    for a in (x.lower().strip() for x in p.allergies):
        if a and (a in food.tags or a in name):
            return False
    for d in (x.lower().strip() for x in p.dislikes):
        if d and d in name:
            return False
    conditions = " ".join(p.health_conditions).lower()
    if "diabet" in conditions and "high-GI" in food.tags:
        return False
    hypertension = "hypertension" in conditions or "blood pressure" in conditions
    return not (hypertension and "high-sodium" in food.tags)


def energy_at(forecast: WellnessForecast | None, t: datetime) -> float | None:
    if forecast is None:
        return None
    for s in forecast.slots:
        if s.start <= t < s.end:
            return s.energy
    return None


def score(
    food: Food, meal: MealType, label: AffectiveLabel | None, energy: float | None, local_hour: int, kcal_target: float
) -> tuple[float, list[str]]:
    tags, sc, why = food.tags, 0.0, []

    def boost(tag, w):
        nonlocal sc
        if tag in tags:
            sc += w
            if tag in REASONS and REASONS[tag] not in why:
                why.append(REASONS[tag])

    if label in (AffectiveLabel.stress, AffectiveLabel.anxiety):
        boost("omega-3", 1.0)
        boost("magnesium", 0.8)
        boost("complex-carb", 0.6)
        sc -= 1.0 * ("caffeine" in tags) + 0.6 * ("fried" in tags) + 0.5 * ("high-sugar" in tags)
    elif label == AffectiveLabel.sad:
        boost("omega-3", 0.9)
        boost("complex-carb", 0.6)
        boost("high-protein", 0.5)
    if energy is not None and energy < 0.4:
        boost("high-protein", 0.7)
        boost("complex-carb", 0.6)
        sc -= 0.5 * ("high-sugar" in tags)
    boost("low-GI", 0.3)
    sc -= 0.4 * ("high-sodium" in tags) + 0.3 * ("fried" in tags) + 0.2 * ("high-GI" in tags)
    if "caffeine" in tags and local_hour >= 15:
        sc -= 2.0
        why.append("caffeine avoided after 15:00 to protect sleep")
    sc -= 0.6 * abs(food.calories - kcal_target) / max(kcal_target, 1)
    return sc, why


def pick_foods(
    meal, label, energy, local_hour, profile: DietProfile, used: dict[str, int] | None = None
) -> list[FoodItem]:
    """Top foods for one meal. `used` counts foods already recommended today — repeats are penalised for variety."""
    used = used if used is not None else {}
    daily = profile.daily_calorie_target or DEFAULT_DAILY_KCAL
    n = ITEMS_PER_MEAL[meal]
    target = daily * CALORIE_SHARE[meal] / n
    cands = [f for f in FOODS if meal in f.meals and allowed(f, profile)]
    scored = []
    for f in cands:
        sc, why = score(f, meal, label, energy, local_hour, target)
        scored.append(((sc - VARIETY_PENALTY * used.get(f.name, 0), why), f))
    ranked = sorted(scored, key=lambda x: (-x[0][0], x[1].name))
    out = []
    for (_, why), f in ranked[:n]:
        used[f.name] = used.get(f.name, 0) + 1
        reason = "; ".join(why) if why else "fits your profile and the time of day"
        out.append(FoodItem(name=f.name, portion=f.portion, calories=f.calories, reason=reason))
    return out


def rationale(meal, label, energy, moved) -> str:
    parts = []
    if label in (AffectiveLabel.stress, AffectiveLabel.anxiety):
        parts.append("you seem stressed, so calming, steady-energy foods are preferred")
    elif label == AffectiveLabel.sad:
        parts.append("mood-supporting foods are preferred")
    if energy is not None and energy < 0.4:
        parts.append("your energy is forecast to be low around this time")
    if moved:
        parts.append("time adjusted to fit a free gap in your schedule")
    return (
        (f"{meal.value.capitalize()}: " + "; ".join(parts) + ".")
        if parts
        else f"{meal.value.capitalize()}: balanced choice for your profile."
    )


def suggest_activities(req: RecommendationRequest) -> list[SuggestedActivity]:
    out: list[SuggestedActivity] = []
    label = req.affective_state.label if req.affective_state else None
    if label in (AffectiveLabel.stress, AffectiveLabel.anxiety):
        out.append(
            SuggestedActivity(
                title="10-minute walk outside",
                category=TaskCategory.exercise,
                estimated_minutes=10,
                reason="A short walk can help bring stress down.",
            )
        )
    slots = req.wellness.slots[:3] if req.wellness else []
    if slots and sum(s.mental_fatigue for s in slots) / len(slots) > 0.7:
        out.append(
            SuggestedActivity(
                title="20-minute rest break",
                category=TaskCategory.personal,
                estimated_minutes=20,
                reason="Mental fatigue is forecast to be high.",
            )
        )
    if req.schedule is not None and req.schedule.overload_predicted and len(out) < 2:
        out.append(
            SuggestedActivity(
                title="15-minute planning break",
                category=TaskCategory.personal,
                estimated_minutes=15,
                reason="Your schedule looks overloaded — re-prioritise.",
            )
        )
    return out[:2]


def recommend(req: RecommendationRequest) -> tuple[list[MealRecommendation], list[SuggestedActivity]]:
    tz = ZoneInfo(req.timezone)
    label = req.affective_state.label if req.affective_state else None
    meals, used = [], {}
    for meal, start, end, moved in meal_windows(req.as_of, tz, busy_intervals(req.schedule)):
        energy = energy_at(req.wellness, start)
        foods = pick_foods(meal, label, energy, start.astimezone(tz).hour, req.profile, used)
        if not foods:
            continue
        meals.append(
            MealRecommendation(
                meal_type=meal,
                window_start=start,
                window_end=end,
                foods=foods,
                rationale=rationale(meal, label, energy, moved),
            )
        )
    return meals, suggest_activities(req)
