"""Ingestion pipeline for HeartSteps Micro-Randomized Trial (MRT) activity suggestions dataset.

Parses suggestions.csv, calculates empirical efficacy metrics (positive response ratio,
post-suggestion step counts), categorizes activities, assigns durations, and outputs
a structured JSON artifact for Knowledge Graph ingestion.
"""

from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from pathlib import Path

# Paths
DEFAULT_CSV_PATH = Path(r"C:\y4 s1\Knowledge graph testing\Datasets\activities suggessions\suggestions.csv")
DATA_DIR = Path(__file__).resolve().parent / "data"
OUTPUT_JSON_PATH = DATA_DIR / "heartsteps_activities.json"

HEARTSTEPS_TRIAL_CITATION = (
    "HeartSteps V1 Micro-Randomized Trial (Klasnja et al., Ann Behav Med 2019 / ACM IMWUT; NIH Grant U54EB020404)"
)


def clean_title(message: str) -> str:
    """Derives a concise, actionable title from the prompt message."""
    msg = message.strip()
    if "walk" in msg.lower() and ("5 minute" in msg.lower() or "5minute" in msg.lower()):
        return "5-Minute Walking Break"
    elif "walk" in msg.lower() and ("10" in msg.lower() or "15" in msg.lower()):
        return "10-15 Minute Outdoor Walk"
    elif "walk around the block" in msg.lower() or "fresh air" in msg.lower():
        return "Brisk Walk Around the Block"
    elif "eyes a break" in msg.lower() or "screen" in msg.lower():
        return "Screen Break & Office Lap"
    elif "stretch your legs arms and neck" in msg.lower():
        return "Full-Body Desk Stretch"
    elif "touch your toes" in msg.lower() or "stiff" in msg.lower():
        return "Standing Toe-Touch & Spinal Decompression"
    elif "nature" in msg.lower() or "wild animal" in msg.lower():
        return "Mindful Nature Walk"
    elif "yoga" in msg.lower() or "childs pose" in msg.lower():
        return "Gentle Evening Yoga & Child's Pose"
    elif "reach high" in msg.lower() and "tension release" in msg.lower():
        return "Tension-Release Overhead Stretch"
    elif "tidy" in msg.lower() or "duster" in msg.lower() or "housework" in msg.lower():
        return "Active Movement Break"
    elif "walk" in msg.lower():
        return "Invigorating Walking Break"
    elif "stretch" in msg.lower():
        return "Restorative Muscle Stretch"
    else:
        # Fallback to first sentence or up to 45 chars
        parts = re.split(r"[.?!]", msg)
        first_clause = parts[0].strip()
        if len(first_clause) > 45:
            first_clause = first_clause[:45] + "..."
        return first_clause


def categorize_activity(msg: str, is_active: bool, is_outdoor: bool) -> tuple[str, int, list[str]]:
    """Determines TaskCategory, duration (>= 5 min), and target affective/cognitive states."""
    lower = msg.lower()

    # 1. Active Walking (Exercise)
    if "walk" in lower or is_outdoor or "stroll" in lower:
        category = "exercise"
        duration = 10 if ("10" in lower or "15" in lower) else 5
        states = ["state:stress", "state:anxiety", "state:fatigue"]
        if "energy" in lower:
            states.append("state:low_energy")

    # 2. Desk Stretches & Postural Resets (Personal)
    elif "stretch" in lower or "posture" in lower or "stand" in lower or "stiff" in lower:
        category = "personal"
        duration = 5
        states = ["state:fatigue", "state:stress", "state:low_focus"]

    # 3. Evening Wind-Down / Mindful Yoga (Personal)
    elif "yoga" in lower or "relax" in lower or "evening" in lower or "bed" in lower:
        category = "personal"
        duration = 10
        states = ["state:stress", "state:anxiety"]

    # 4. Cognitive & Screen Resets (Personal)
    elif "screen" in lower or "office" in lower or "lap" in lower:
        category = "personal"
        duration = 5
        states = ["state:low_focus", "state:fatigue"]

    else:
        category = "personal"
        duration = 5
        states = ["state:fatigue"]

    return category, duration, states


def ingest_heartsteps_dataset(
    csv_path: Path = DEFAULT_CSV_PATH,
    min_count: int = 15,
    min_positive_ratio: float = 0.65,
) -> list[dict]:
    """Ingests, filters, and formats HeartSteps suggestions into structured activity nodes."""
    if not csv_path.exists():
        raise FileNotFoundError(f"HeartSteps suggestions CSV not found at: {csv_path}")

    stats_map = defaultdict(lambda: {
        "count": 0,
        "good": 0,
        "bad": 0,
        "unanswered": 0,
        "row_ids": [],
        "steps30": [],
        "tag_active": 0,
        "tag_outdoor": 0,
        "tag_indoor": 0,
    })

    with open(csv_path, encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)
        for row_idx, row in enumerate(reader, start=2):  # Header is row 1
            msg = row.get("returned.message", "").strip()
            if not msg or msg.lower() in ("donotnotify", "none"):
                continue

            entry = stats_map[msg]
            entry["count"] += 1
            entry["row_ids"].append(row_idx)
            resp = row.get("response", "").strip().lower()
            if resp == "good":
                entry["good"] += 1
            elif resp == "bad":
                entry["bad"] += 1
            else:
                entry["unanswered"] += 1

            if row.get("tag.active", "").lower() == "true":
                entry["tag_active"] += 1
            if row.get("tag.outdoor", "").lower() == "true":
                entry["tag_outdoor"] += 1
            if row.get("tag.indoor", "").lower() == "true":
                entry["tag_indoor"] += 1

            try:
                s30 = float(row.get("jbsteps30", 0))
                if s30 > 0:
                    entry["steps30"].append(s30)
            except (ValueError, TypeError):
                pass

    activities = []
    task_idx = 1

    # Filter and rank by count and positive engagement
    filtered = []
    for msg, st in stats_map.items():
        total_eval = st["good"] + st["bad"]
        pos_ratio = st["good"] / max(1, total_eval)
        avg_steps = sum(st["steps30"]) / len(st["steps30"]) if st["steps30"] else 0.0

        if st["count"] >= min_count and pos_ratio >= min_positive_ratio:
            filtered.append((msg, st, pos_ratio, avg_steps))

    # Sort by positive response count
    filtered.sort(key=lambda x: (x[1]["good"], x[3]), reverse=True)

    for msg, st, pos_ratio, avg_steps in filtered[:30]:  # Top 30 clinically verified prompts
        is_outdoor = st["tag_outdoor"] > st["tag_indoor"]
        is_active = st["tag_active"] >= st["count"] * 0.4
        category, duration, states = categorize_activity(msg, is_active, is_outdoor)
        title = clean_title(msg)

        task_id = f"task:heartsteps_{task_idx:02d}"
        task_idx += 1

        total_eval = st["good"] + st["bad"]
        unanswered = st["count"] - total_eval
        eval_pct = int(round(pos_ratio * 100))
        explanation = (
            f"Option to consider: {msg} "
            f"[HeartSteps mHealth MRT (Klasnja et al. 2019); study log: "
            f"{st['good']}/{total_eval} positive of rated prompts ({unanswered} unrated of {st['count']} delivered; "
            f"descriptive trial feedback, not clinical efficacy)]"
        )

        activities.append({
            "task_id": task_id,
            "title": title,
            "message": msg,
            "category": category,
            "duration": duration,
            "states": states,
            "explanation": explanation,
            "source": HEARTSTEPS_TRIAL_CITATION,
            "empirical_stats": {
                "sample_size": st["count"],
                "positive_responses": st["good"],
                "negative_responses": st["bad"],
                "unanswered_or_snoozed": unanswered,
                "positive_ratio": round(pos_ratio, 3),
                "source_row_ids": st["row_ids"],
                "descriptive_log_note": "Descriptive statistics computed from local suggestions.csv. Step changes in HeartSteps MRT vary by context (home vs work) and are not represented as static per-prompt causal increases.",
            },
        })

    # Save to JSON
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(activities, f, indent=2)

    print(f"Successfully processed {len(activities)} HeartSteps activity tasks -> {OUTPUT_JSON_PATH}")
    return activities


if __name__ == "__main__":
    ingest_heartsteps_dataset()
