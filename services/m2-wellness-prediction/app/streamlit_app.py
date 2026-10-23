"""Interactive Streamlit demo for the M2 cognitive wellness predictor."""
# ruff: noqa: E402, I001

from __future__ import annotations

import sys
import uuid
from datetime import date, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st

SERVICE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SERVICE_DIR))
sys.path.insert(0, str(SERVICE_DIR / "src"))

from app.predictor import Predictor  # noqa: E402
from wellness_contracts import (  # noqa: E402
    ManualWellnessDay,
    ManualWellnessInputs,
    WellnessForecastRequest,
)


st.set_page_config(
    page_title="Cognitive Wellness Forecast",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def load_predictor() -> Predictor:
    return Predictor()


def feeling_value(feeling: str) -> float:
    return {
        "Exhausted": 0.0,
        "Very tired": 0.25,
        "Okay": 0.5,
        "Good": 0.75,
        "Energetic": 1.0,
    }[feeling]


def daily_inputs(label: str, default_date: date) -> ManualWellnessDay:
    st.markdown(f"#### {label}")
    first, second = st.columns(2)
    with first:
        record_date = st.date_input("Date", value=default_date, key=f"{label}-date")
        sleep_hours = st.number_input(
            "Sleep (hours)",
            min_value=0.0,
            max_value=16.0,
            value=7.0,
            step=0.5,
            key=f"{label}-sleep",
            help="Total sleep during the previous night.",
        )
        wake_hour = st.number_input(
            "Wake-up time",
            min_value=0.0,
            max_value=23.99,
            value=7.0,
            step=0.25,
            key=f"{label}-wake",
            help="Use decimal hours: 7.5 means 07:30.",
        )
        steps = st.number_input(
            "Steps",
            min_value=0,
            max_value=50000,
            value=6000,
            step=500,
            key=f"{label}-steps",
        )
    with second:
        feeling_options = ["Exhausted", "Very tired", "Okay", "Good", "Energetic"]
        feeling = st.select_slider(
            "How do you feel?",
            options=feeling_options,
            value="Okay",
            key=f"{label}-feeling",
            help="Choose the option that best describes your overall energy.",
        )
        energy = feeling_value(feeling)
        tiredness_feeling = st.select_slider(
            "How mentally tired are you?",
            options=feeling_options,
            value="Okay",
            key=f"{label}-tiredness",
        )
        mental_tiredness = 1 - feeling_value(tiredness_feeling)
        focus_feeling = st.select_slider(
            "How focused do you feel?",
            options=feeling_options,
            value="Okay",
            key=f"{label}-focus",
        )
        focus = feeling_value(focus_feeling)
        overwhelmed_feeling = st.select_slider(
            "How overwhelmed do you feel?",
            options=feeling_options,
            value="Okay",
            key=f"{label}-overwhelmed",
        )
        overwhelmed = 1 - feeling_value(overwhelmed_feeling)
        tasks_due = st.number_input(
            "Tasks due in 24 hours",
            min_value=0,
            max_value=100,
            value=2,
            step=1,
            key=f"{label}-due",
        )
        minutes_due = st.number_input(
            "Work minutes due",
            min_value=0,
            max_value=2000,
            value=120,
            step=15,
            key=f"{label}-minutes",
        )
        tasks_overdue = st.number_input(
            "Overdue tasks",
            min_value=0,
            max_value=100,
            value=0,
            step=1,
            key=f"{label}-overdue",
        )
        tasks_pending = st.number_input(
            "Pending tasks",
            min_value=0,
            max_value=100,
            value=2,
            step=1,
            key=f"{label}-pending",
        )
        bedtime_hour = st.number_input(
            "Expected bedtime",
            min_value=0.0,
            max_value=23.99,
            value=23.0,
            step=0.25,
            key=f"{label}-bedtime",
            help="Use decimal hours: 23.5 means 23:30.",
        )
        next_wake_hour = st.number_input(
            "Expected next wake time",
            min_value=0.0,
            max_value=23.99,
            value=7.0,
            step=0.25,
            key=f"{label}-next-wake",
        )

    return ManualWellnessDay(
        date=record_date,
        sleep_hours=sleep_hours,
        wake_hour=wake_hour,
        steps=steps,
        stress=overwhelmed,
        mental_tiredness=mental_tiredness,
        focus=focus,
        energy=energy,
        overwhelmed=overwhelmed,
        bedtime_hour=bedtime_hour,
        next_wake_hour=next_wake_hour,
        tasks_due_24h=tasks_due,
        minutes_due_24h=minutes_due,
        tasks_overdue=tasks_overdue,
        tasks_pending=tasks_pending,
    )


def score_label(value: float) -> str:
    if value < 0.33:
        return "Low"
    if value < 0.67:
        return "Moderate"
    return "High"


def show_results(forecast) -> None:
    st.subheader("Your wellness outlook")
    first_slot = forecast.slots[0]
    cards = [
        ("Energy", first_slot.energy, "Higher is better"),
        ("Focus", first_slot.focus, "Higher is better"),
        ("Mental fatigue", first_slot.mental_fatigue, "Lower is better"),
        ("Burnout risk", first_slot.burnout_risk, "Lower is better"),
    ]
    columns = st.columns(4)
    for column, (name, value, hint) in zip(columns, cards, strict=False):
        with column:
            column.metric(name, f"{value:.0%}", score_label(value))
            st.caption(hint)

    if forecast.sleep_hours_last_night is not None:
        st.info(f"Latest entered sleep: **{forecast.sleep_hours_last_night:.1f} hours**")

    rows = [
        {
            "Time": slot.start,
            "Energy": slot.energy,
            "Focus": slot.focus,
            "Mental fatigue": slot.mental_fatigue,
            "Burnout risk": slot.burnout_risk,
        }
        for slot in forecast.slots
    ]
    chart_data = pd.DataFrame(rows).sort_values("Time").set_index("Time")
    st.subheader("Forecast by time slot")
    st.line_chart(chart_data, y=["Energy", "Focus", "Mental fatigue", "Burnout risk"])

    display_data = chart_data.copy()
    for column in display_data.columns:
        display_data[column] = display_data[column].map(lambda value: f"{value:.0%}")
    st.dataframe(display_data, use_container_width=True)

    st.caption(
        f"Model: {forecast.model.name} v{forecast.model.version} · "
        f"{len(forecast.slots)} slots at {forecast.slot_minutes}-minute intervals"
    )
    st.warning(
        "This is a wellness support estimate, not a medical diagnosis. "
        "Use the trends to plan rest and workload, not as clinical advice."
    )


def main() -> None:
    st.title("🧠 Cognitive Wellness Forecast")
    st.write(
        "Enter the user's current information and up to three previous days. "
        "The M2 model estimates energy, focus, mental fatigue, and burnout risk for the coming hours."
    )

    with st.sidebar:
        st.header("Forecast settings")
        timezone_name = st.selectbox(
            "Timezone",
            ["Asia/Colombo", "Asia/Kolkata", "UTC", "Europe/London", "America/New_York"],
            index=0,
        )
        user_id = st.text_input("User ID", value=str(uuid.uuid4()))
        forecast_date = st.date_input("Prediction date", value=date.today())
        forecast_time = st.time_input("Prediction time", value=time(10, 0))
        horizon_hours = 24
        st.caption("Forecast horizon: 24 hours")
        slot_minutes = st.select_slider("Slot size", options=[15, 30, 60, 120], value=60)

        predictor = load_predictor()
        if predictor.models is not None:
            st.success("Trained model loaded")
        else:
            st.info("Fallback model loaded")

    with st.form("wellness-inputs"):
        st.subheader("Previous days")
        st.caption("Use the last three days to give the model recent context. Older days are optional.")
        history = []
        for offset in (3, 2, 1):
            history.append(daily_inputs(f"Day {offset} ago", forecast_date - timedelta(days=offset)))

        st.divider()
        current = daily_inputs("Current day", forecast_date)
        submitted = st.form_submit_button("Calculate wellness forecast", type="primary", use_container_width=True)

    if submitted:
        try:
            tz = ZoneInfo(timezone_name)
            as_of = datetime.combine(forecast_date, forecast_time).replace(tzinfo=tz)
            request = WellnessForecastRequest(
                user_id=user_id,
                as_of=as_of,
                timezone=timezone_name,
                horizon_hours=horizon_hours,
                slot_minutes=slot_minutes,
                manual_inputs=ManualWellnessInputs(history=history, current=current),
            )
            with st.spinner("Calculating your forecast..."):
                forecast = predictor.predict(request)
            show_results(forecast)
        except (KeyError, TypeError, ValueError) as error:
            st.error(f"Unable to calculate the forecast: {error}")


if __name__ == "__main__":
    main()
