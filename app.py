import streamlit as st
import hashlib
import json
import joblib
import numpy as np
import pandas as pd
import streamlit.components.v1 as components
from pathlib import Path

# -----------------------------
# Page settings
# -----------------------------
st.set_page_config(
    page_title="Academic Progress Predictor",
    page_icon="🎓",
    layout="wide"
)


# -----------------------------
# Final production models trained on all 401 students
# -----------------------------
GRADE_MODEL_BUNDLE_PATH = Path(__file__).with_name(
    "final_grade_model_401.joblib"
)
DURATION_MODEL_BUNDLE_PATH = Path(__file__).with_name(
    "final_duration_model_401.joblib"
)


@st.cache_resource
def load_model_bundles():
    """Load the two final fitted model bundles once per app process."""
    return (
        joblib.load(GRADE_MODEL_BUNDLE_PATH),
        joblib.load(DURATION_MODEL_BUNDLE_PATH),
    )


PERIOD_ORDER = {
    "ΦΕΒΡ": 1,
    "ΙΟΥΝ": 2,
    "ΣΕΠΤ": 3,
}

PERIOD_MONTH = {
    "ΦΕΒΡ": 2,
    "ΙΟΥΝ": 6,
    "ΣΕΠΤ": 9,
}

COURSE_GROUP_BY_UI_CATEGORY = {
    "CORE": "CORE",
    "ΥΚ": "SPECIALIZATION_COMPULSORY",
    "ΕΚ": "SPECIALIZATION_ELECTIVE",
    # Στο training notebook τα ΕΤΑΚ ενώθηκαν με τα επιλογής
    # κατεύθυνσης (OTHER_SPECIALIZATION_ELECTIVE).
    "ΕΤΑΚ": "SPECIALIZATION_ELECTIVE",
}

GROUP_PREFIXES = {
    "CORE": "core",
    "SPECIALIZATION_COMPULSORY": "spec_compulsory",
    "SPECIALIZATION_ELECTIVE": "spec_elective",
    "OTHER": "other",
}


def academic_year_exam_year(academic_year):
    """Return the calendar year of FEB/JUN/SEP for an academic year."""
    return int(str(academic_year).split("-")[1])


def build_model_input(
    course_records,
    enrollment_academic_year,
    enrollment_semester,
    current_academic_year,
    snapshot_period,
):
    """Recreate the snapshot features used in the training notebook."""
    events = pd.DataFrame(course_records).copy()

    if events.empty:
        raise ValueError(
            "Χρειάζεται τουλάχιστον ένα περασμένο ή "
            "αναγνωρισμένο μάθημα."
        )

    events["exam_year"] = events[
        "success_academic_year"
    ].map(academic_year_exam_year)
    events["exam_period_order"] = events[
        "success_period"
    ].map(PERIOD_ORDER)
    events["course_event_index"] = (
        events["exam_year"] * 3
        + events["exam_period_order"]
    )
    events["model_course_group"] = events[
        "curriculum_category"
    ].map(COURSE_GROUP_BY_UI_CATEGORY).fillna("OTHER")
    events["completion_source"] = events[
        "completion_type"
    ].map({
        "graded": "GRADED",
        "recognized": "RECOGNIZED",
    })
    events["course_grade"] = pd.to_numeric(
        events["grade"],
        errors="coerce",
    )
    events["ects"] = pd.to_numeric(
        events["ects"],
        errors="coerce",
    ).fillna(0.0)

    # Η τρέχουσα περίοδος αντιστοιχεί στο δεύτερο έτος
    # του επιλεγμένου ακαδημαϊκού έτους.
    snapshot_year = academic_year_exam_year(
        current_academic_year
    )
    snapshot_index = (
        snapshot_year * 3
        + PERIOD_ORDER[snapshot_period]
    )

    future_events = events.loc[
        events["course_event_index"] > snapshot_index
    ]
    if not future_events.empty:
        raise ValueError(
            "Υπάρχει μάθημα με περίοδο επιτυχίας μεταγενέστερη "
            "από την τρέχουσα περίοδο της πρόβλεψης."
        )

    history_df = events.loc[
        events["course_event_index"] <= snapshot_index
    ].copy()
    current_period_df = history_df.loc[
        history_df["course_event_index"] == snapshot_index
    ]

    graded_df = history_df.loc[
        history_df["completion_source"].eq("GRADED")
    ]
    recognized_df = history_df.loc[
        history_df["completion_source"].eq("RECOGNIZED")
    ]
    # Για τον κανόνα προόδου μετράμε κάθε ολοκληρωμένο μάθημα
    # ακριβώς μία φορά, είτε είναι βαθμολογημένο είτε αναγνωρισμένο.
    graded_course_codes = set(
        graded_df["code"].dropna().astype(str)
    )
    recognized_course_codes = set(
        recognized_df["code"].dropna().astype(str)
    )
    completed_courses_for_progress = len(
        graded_course_codes | recognized_course_codes
    )
    valid_grades = graded_df["course_grade"].dropna()

    current_graded_df = current_period_df.loc[
        current_period_df["completion_source"].eq("GRADED")
    ]
    current_valid_grades = current_graded_df[
        "course_grade"
    ].dropna()

    # Χαρακτηριστικά του νέου μοντέλου βαθμού πτυχίου.
    # Οι GPA υπολογίζονται με βάρη ECTS, όπως στο notebook.
    graded_with_grade_df = graded_df.loc[
        graded_df["course_grade"].notna()
        & graded_df["ects"].gt(0)
    ].copy()
    graded_with_grade_df["grade_points"] = (
        graded_with_grade_df["course_grade"]
        * graded_with_grade_df["ects"]
    )

    graded_ects_for_gpa = float(
        graded_with_grade_df["ects"].sum()
    )
    cumulative_grade_points = float(
        graded_with_grade_df["grade_points"].sum()
    )
    cumulative_gpa = (
        cumulative_grade_points / graded_ects_for_gpa
    )

    period_grade_summary = (
        graded_with_grade_df
        .groupby("course_event_index", as_index=False)
        .agg(
            period_graded_ects=("ects", "sum"),
            period_grade_points=("grade_points", "sum"),
        )
        .sort_values("course_event_index")
    )
    period_grade_summary["period_gpa"] = (
        period_grade_summary["period_grade_points"]
        / period_grade_summary["period_graded_ects"]
    )

    recent_active_periods = period_grade_summary.tail(3)
    recent_3_active_gpa = float(
        recent_active_periods["period_gpa"].mean()
    )
    recent_3_active_gpa_std = float(
        recent_active_periods["period_gpa"].std(ddof=0)
    )
    recent_3_active_graded_ects = float(
        recent_active_periods["period_graded_ects"].sum()
    )

    current_period_summary = period_grade_summary.loc[
        period_grade_summary["course_event_index"].eq(
            snapshot_index
        )
    ]
    if current_period_summary.empty:
        period_gpa_for_model = cumulative_gpa
        gpa_momentum_for_model = 0.0
    else:
        period_gpa_for_model = float(
            current_period_summary["period_gpa"].iloc[0]
        )
        current_period_ects = float(
            current_period_summary["period_graded_ects"].iloc[0]
        )
        current_period_points = float(
            current_period_summary["period_grade_points"].iloc[0]
        )
        prior_ects = graded_ects_for_gpa - current_period_ects
        prior_points = (
            cumulative_grade_points - current_period_points
        )
        gpa_momentum_for_model = (
            period_gpa_for_model - prior_points / prior_ects
            if prior_ects > 0
            else 0.0
        )

    # Στο UI εμφανίζονται τα μαθήματα του προγράμματος σπουδών.
    # Συνεπώς δεν υπάρχουν εγγραφές με το transcript marker (*).
    star_courses_cumulative = 0

    enrollment_start_year = int(
        str(enrollment_academic_year).split("-")[0]
    )
    enrollment_date = pd.Timestamp(
        year=enrollment_start_year,
        month=10,
        day=1,
    )
    snapshot_date = pd.Timestamp(
        year=snapshot_year,
        month=PERIOD_MONTH[snapshot_period],
        day=1,
    )

    # Φεβρουάριος και Ιούνιος ολοκληρώνουν ακαδημαϊκό
    # εξάμηνο. Ο Σεπτέμβριος διατηρεί το ίδιο εξάμηνο.
    checkpoint_dates = pd.date_range(
        start=enrollment_date,
        end=snapshot_date,
        freq="MS",
    )
    elapsed_semesters = max(
        int(checkpoint_dates.month.isin([2, 6]).sum()),
        1,
    )
    # Τα εξάμηνα εισαγωγής αντιστοιχίζονται ανά ακαδημαϊκό
    # έτος: 1–2 → +0, 3–4 → +2 και 5–6 → +4 εξάμηνα.
    entry_semester_offset = 2 * (
        (int(enrollment_semester) - 1) // 2
    )
    program_semester = (
        elapsed_semesters + entry_semester_offset
    )
    program_semester = max(
        int(program_semester),
        int(enrollment_semester),
    )
    elapsed_years = max(
        (snapshot_date - enrollment_date).days / 365.25,
        0.0,
    )

    # ---------------------------------------------------------
    # Final duration-model features
    # ---------------------------------------------------------

    cumulative_recognized_ects = float(
        recognized_df["ects"].sum()
    )
    cumulative_total_earned_ects = (
        graded_ects_for_gpa + cumulative_recognized_ects
    )

    recent_period_start_index = snapshot_index - 2
    recent_3_period_graded_ects = float(
        graded_with_grade_df.loc[
            graded_with_grade_df["course_event_index"].between(
                recent_period_start_index,
                snapshot_index,
                inclusive="both",
            ),
            "ects",
        ].sum()
    )

    avg_ects_per_graded_course = (
        graded_ects_for_gpa
        / graded_with_grade_df["code"].nunique()
        if graded_with_grade_df["code"].nunique() > 0
        else 0.0
    )

    # Period-aware delayed-course ratio. Only graded courses
    # with a curriculum semester participate, exactly as in
    # the training notebook. September adds half a semester.
    delay_events_df = graded_with_grade_df.copy()
    delay_events_df["course_semester"] = pd.to_numeric(
        delay_events_df["curriculum_semester"]
        .astype(str)
        .str.extract(r"(\d+)", expand=False),
        errors="coerce",
    )

    event_exam_year = delay_events_df[
        "success_academic_year"
    ].map(academic_year_exam_year)
    event_month = delay_events_df[
        "success_period"
    ].map(PERIOD_MONTH)

    delay_events_df["completion_date"] = pd.to_datetime({
        "year": event_exam_year,
        "month": event_month,
        "day": 1,
    })
    delay_events_df["entry_date"] = pd.to_datetime({
        "year": pd.Series(
            enrollment_start_year,
            index=delay_events_df.index,
        ),
        "month": 10,
        "day": 1,
    })

    def completed_semesters_at_event(row):
        event_checkpoints = pd.date_range(
            start=row["entry_date"],
            end=row["completion_date"],
            freq="MS",
        )
        return max(
            int(event_checkpoints.month.isin([2, 6]).sum()),
            1,
        )

    delay_events_df["event_elapsed_semesters"] = (
        delay_events_df.apply(
            completed_semesters_at_event,
            axis=1,
        )
    )
    delay_events_df["event_program_semester"] = (
        delay_events_df["event_elapsed_semesters"]
        + entry_semester_offset
    )
    delay_events_df["completion_delay_period_aware"] = (
        delay_events_df["event_program_semester"]
        - delay_events_df["course_semester"]
        + 0.5
        * delay_events_df["success_period"].eq("ΣΕΠΤ").astype(float)
    ).clip(lower=0)

    eligible_delay_events_df = delay_events_df.loc[
        delay_events_df["course_semester"].notna()
    ]
    delayed_course_ratio_period_aware = (
        float(
            eligible_delay_events_df[
                "completion_delay_period_aware"
            ].gt(0).mean()
        )
        if not eligible_delay_events_df.empty
        else 0.0
    )

    # Τα snapshots του training ξεκινούν από την πρώτη
    # καταγεγραμμένη επιτυχία και περιλαμβάνουν όλα τα ενδιάμεσα.
    first_event_index = int(
        history_df["course_event_index"].min()
    )
    snapshot_indices = list(range(
        first_event_index,
        snapshot_index + 1,
    ))
    graded_event_indices = set(
        graded_df["course_event_index"].astype(int)
    )
    no_new_flags = [
        int(index not in graded_event_indices)
        for index in snapshot_indices
    ]

    current_streak = 0
    maximum_streak = 0
    for flag in no_new_flags:
        current_streak = current_streak + 1 if flag else 0
        maximum_streak = max(maximum_streak, current_streak)

    snapshot_number = len(snapshot_indices)
    no_new_cumulative = int(sum(no_new_flags))

    feature_row = {
        "elapsed_years": elapsed_years,
        "elapsed_semesters": int(elapsed_semesters),
        "enrollment_semester": int(enrollment_semester),
        "advanced_entry": int(int(enrollment_semester) > 1),
        "period_gpa_for_model": period_gpa_for_model,
        "cumulative_gpa": cumulative_gpa,
        "graded_ects_for_gpa": graded_ects_for_gpa,
        "recent_3_active_gpa_std": recent_3_active_gpa_std,
        "gpa_momentum_for_model": gpa_momentum_for_model,
        "program_semester": program_semester,
        "is_september_period": int(snapshot_period == "ΣΕΠΤ"),
        "is_september_for_model": int(snapshot_period == "ΣΕΠΤ"),
        "enrollment_semester_for_model": int(enrollment_semester),
        "current_period_gpa": period_gpa_for_model,
        "recent_3_active_periods_gpa": recent_3_active_gpa,
        "cumulative_total_earned_ects": (
            cumulative_total_earned_ects
        ),
        "completed_courses_for_progress": int(
            completed_courses_for_progress
        ),
        "recent_3_period_graded_ects": (
            recent_3_period_graded_ects
        ),
        "delayed_course_ratio_period_aware": (
            delayed_course_ratio_period_aware
        ),
        "avg_ects_per_graded_course": (
            avg_ects_per_graded_course
        ),
        "recent_3_active_graded_ects": (
            recent_3_active_graded_ects
        ),
        "cumulative_graded_courses": int(
            graded_with_grade_df["code"].nunique()
        ),
        "recent_3_active_gpa": recent_3_active_gpa,
        "star_courses_cumulative": star_courses_cumulative,
        "mean_grade": valid_grades.mean(),
        "median_grade": valid_grades.median(),
        "grade_std": valid_grades.std(),
        "minimum_grade": valid_grades.min(),
        "maximum_grade": valid_grades.max(),
        "has_numeric_grade_history": int(len(valid_grades) > 0),
        "new_graded_ects": current_graded_df["ects"].sum(),
        "current_period_mean_grade": current_valid_grades.mean(),
        "no_new_graded_course_period": no_new_flags[-1],
        "no_new_graded_course_periods_cumulative": no_new_cumulative,
        "no_new_graded_course_rate": (
            no_new_cumulative / snapshot_number
        ),
        "current_no_new_graded_course_streak": current_streak,
        "maximum_no_new_graded_course_streak": maximum_streak,
        "snapshot_period": snapshot_period,
    }

    for group_name, prefix in GROUP_PREFIXES.items():
        group_df = history_df.loc[
            history_df["model_course_group"].eq(group_name)
        ]
        group_graded_df = group_df.loc[
            group_df["completion_source"].eq("GRADED")
        ]
        group_recognized_df = group_df.loc[
            group_df["completion_source"].eq("RECOGNIZED")
        ]
        group_grades = group_graded_df[
            "course_grade"
        ].dropna()

        feature_row[f"{prefix}_graded_courses"] = (
            group_graded_df["code"].nunique()
        )
        feature_row[f"{prefix}_graded_ects"] = (
            group_graded_df["ects"].sum()
        )
        feature_row[f"{prefix}_recognized_courses"] = (
            group_recognized_df["code"].nunique()
        )
        feature_row[f"{prefix}_recognized_ects"] = (
            group_recognized_df["ects"].sum()
        )
        feature_row[f"{prefix}_mean_grade"] = (
            group_grades.mean()
        )

    return pd.DataFrame([feature_row])


def apply_duration_postprocessing(
    raw_prediction,
    model_input_df,
    duration_model_bundle,
):
    """Apply temporal consistency and the calibrated slow-progress rule."""
    elapsed_semesters = float(
        model_input_df["elapsed_semesters"].iloc[0]
    )
    completed_duration_years = max(
        (elapsed_semesters - 1) / 2,
        0.0,
    )
    final_prediction = max(
        float(raw_prediction),
        completed_duration_years,
    )

    rule = duration_model_bundle.get("slow_progress_rule", {})
    if not rule.get("enabled", False):
        return final_prediction

    completed_semesters = max(elapsed_semesters - 1, 1.0)
    completed_courses = float(
        model_input_df["completed_courses_for_progress"].iloc[0]
    )
    earned_ects = float(
        model_input_df["cumulative_total_earned_ects"].iloc[0]
    )
    courses_per_semester = completed_courses / completed_semesters
    ects_per_semester = earned_ects / completed_semesters

    is_slow_progress = (
        elapsed_semesters
        >= float(rule["minimum_elapsed_semester"])
        and courses_per_semester
        < float(rule["course_rate_threshold"])
        and ects_per_semester
        < float(rule["ects_rate_threshold"])
    )
    if not is_slow_progress:
        return final_prediction

    remaining_courses = max(
        float(rule["total_required_courses"])
        - completed_courses,
        0.0,
    )
    remaining_ects = max(
        float(rule["total_required_course_ects"])
        - earned_ects,
        0.0,
    )
    duration_from_courses = (
        completed_duration_years
        + remaining_courses
        / float(rule["median_future_courses_per_semester"])
        / 2
    )
    duration_from_ects = (
        completed_duration_years
        + remaining_ects
        / float(rule["median_future_ects_per_semester"])
        / 2
    )

    return max(
        final_prediction,
        duration_from_courses,
        duration_from_ects,
    )

# -----------------------------
# Custom CSS
# -----------------------------
st.markdown("""
<style>
.main {
    background-color: #F7F8FA;
}

.main-title {
    font-size: 38px;
    font-weight: 700;
    color: #2C3E50;
    margin-bottom: 5px;
}

.subtitle {
    font-size: 19px;
    font-weight: 450;
    color: var(--text-color);
    opacity: 1;
    margin-bottom: 30px;
}

.intro-block {
    display: inline-grid;
    width: fit-content;
    max-width: 100%;
}

.intro-block .subtitle {
    width: auto;
    margin-bottom: 0;
}

.intro-block .custom-info {
    width: 100%;
    max-width: 100%;
    box-sizing: border-box;
    background: rgba(37, 99, 235, 0.11);
    color: var(--text-color);
    border-left: 5px solid #2A7DEB;
    border-radius: 10px;
    padding: 12px 16px;
    margin-top: 34px;
    margin-bottom: 16px;
    font-size: 17px;
    font-weight: 450;
    opacity: 1;
}

@media (prefers-color-scheme: dark) {
    .intro-block .custom-info {
        background: rgba(96, 165, 250, 0.15);
    }
}


.card {
    background-color: white;
    padding: 22px;
    border-radius: 16px;
    box-shadow: 0 4px 14px rgba(0,0,0,0.08);
    margin-bottom: 20px;
}

.metric-title {
    font-size: 17px;
    font-weight: 600;
    color: #475569;
}

.metric-value {
    font-size: 30px;
    font-weight: 700;
    color: #4B3869;
}

.section-title {
    font-size: 26px;
    font-weight: 650;
    color: #2C3E50;
    margin-top: 15px;
    margin-bottom: 19px;
}

.small-note {
    font-size: 14px;
    color: #7F8C8D;
}

.course-title {
    font-size: 19px;
    font-weight: 600;
    line-height: 25px;
    color: var(--text-color);
    margin-bottom: 4px;
}

.course-meta {
    font-size: 15px;
    font-weight: 450;
    line-height: 20px;
    color: var(--text-color);
    opacity: 0.65;
}

/* Κατεβάζει το περιεχόμενο των δύο πρώτων στηλών ώστε ο τίτλος
   του μαθήματος να ευθυγραμμίζεται με τα πεδία εισαγωγής. */
.course-status-offset {
    height: 28px;
}

.course-info-offset {
    height: 26px;
}



.course-box {
    background-color: white;
    padding: 15px;
    border-radius: 12px;
    border: 1px solid #E5E7EB;
    margin-bottom: 10px;
}

.semester-timeline {
    display: flex;
    align-items: center;
    justify-content: space-between;
    position: relative;
    margin: 28px 0 18px 0;
    padding: 0 8px;
}

.semester-timeline::before {
    content: "";
    position: absolute;
    top: 18px;
    left: 35px;
    right: 35px;
    height: 4px;
    background: linear-gradient(
        to right,
        #2E9AFE 0%,
        #6C63FF var(--progress-line),
        #2A2D38 var(--progress-line),
        #2A2D38 100%
    );
    border-radius: 999px;
    z-index: 0;
}

.semester-step {
    position: relative;
    z-index: 1;
    display: flex;
    flex-direction: column;
    align-items: center;
    min-width: 58px;
}

.semester-circle {
    width: 38px;
    height: 38px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 800;
    font-size: 15px;
    margin-bottom: 10px;
    border: 3px solid #2A2D38;
    background: #11141C;
    color: #8A90A3;
}

a

.semester-label {
    font-size: 14px;
    color: #9CA3AF;
    font-weight: 600;
}

.semester-step.completed .semester-circle {
    background: linear-gradient(135deg, #2E9AFE, #6C63FF);
    color: white;
    border-color: #6C63FF;
    box-shadow: 0 0 14px rgba(46, 154, 254, 0.45);
}

.semester-step.completed .semester-label {
    color: #374151;
    font-weight: 700;
}

.semester-step.active .semester-circle {
    background: linear-gradient(135deg, #2E9AFE, #6C63FF);
    color: white;
    border: 3px solid #2E9AFE;
    box-shadow: 0 0 18px rgba(46, 154, 254, 0.60);
    transform: scale(1.15);
}

.semester-step.active .semester-label {
    color: #2E9AFE;
    font-weight: 800;
    transform: translateY(2px);
}

.semester-step.upcoming .semester-circle {
    background: #171A23;
    color: #6B7280;
    border-color: #2A2D38;
}

.semester-step.upcoming .semester-label {
    color: #4B5563;
    font-weight: 600;
}

.current-semester-text {
    margin-top: 8px;
    color: #1F2937;
    font-size: 16px;
    font-weight: 700;
}

/* Timeline text colors - works better in both light and dark mode */

.section-title {
    color: #3A506B !important;
    font-weight: 800;
}

.semester-step.completed .semester-label {
    color: #4B5563 !important;
    font-weight: 700;
}

.semester-step.active .semester-label {
    color: #2E9AFE !important;
    font-weight: 800;
}

.semester-step.upcoming .semester-label {
    color: #64748B !important;
    font-weight: 650;
}

.current-semester-text {
     margin-top: 14px;
    color: #3A506B !important;
    font-size: 20px;
    font-weight: 800;
}

.courses-title {
    font-size: 32px !important;
    font-weight: 900 !important;
    margin-top: 28px !important;
    margin-bottom: 22px !important;
}

/* Circular progress card for passed courses */

.donut-card {
    background-color: white;
    padding: 22px;
    border-radius: 16px;
    box-shadow: 0 4px 14px rgba(0,0,0,0.08);
    margin-bottom: 20px;
    display: flex;
    justify-content: center;
    align-items: center;
}

.donut-progress {
    width: 170px;
    height: 170px;
    border-radius: 50%;
    background:
        conic-gradient(
            from 0deg,
            #2E9AFE 0%,
            #6C63FF calc(var(--progress) * 1%),
            #E5E7EB calc(var(--progress) * 1%),
            #E5E7EB 100%
        );
    display: flex;
    align-items: center;
    justify-content: center;
}

.donut-inner {
    width: 125px;
    height: 125px;
    border-radius: 50%;
    background-color: white;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
}

.donut-value {
    font-size: 28px;
    font-weight: 900;
    color: #4B3869;
    line-height: 1.1;
}

.donut-label {
    font-size: 15px;
    font-weight: 700;
    color: #475569;
    margin-top: 5px;
    text-align: center;
}

/* Custom horizontal progress card */

.progress-card {
    background-color: white;
    padding: 22px;
    border-radius: 16px;
    box-shadow: 0 4px 14px rgba(0,0,0,0.08);
    margin-bottom: 20px;
}

.progress-card-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 14px;
}

.progress-card-title {
    color: #475569;
    font-size: 17px;
    font-weight: 600;
}

.progress-card-value {
    color: #4B3869;
    font-size: 26px;
    font-weight: 900;
}

.custom-progress-track {
    width: 100%;
    height: 12px;
    background-color: #E5E7EB;
    border-radius: 999px;
    overflow: hidden;
}

.custom-progress-fill {
    height: 100%;
    width: var(--progress-width);
    background: linear-gradient(90deg, #2E9AFE, #6C63FF);
    border-radius: 999px;
}

.prediction-label {
    font-size: 19px;
    font-weight: 650;
    margin-bottom: 10px;
}

.prediction-value {
    font-size: 38px;
    font-weight: 500;
    line-height: 1.2;
    margin-bottom: 10px;
}

.main-title,
.section-title,
.prediction-label,
.prediction-value {
    color: var(--text-color) !important;
    opacity: 1 !important;
}

/* Πιο ευδιάκριτες κάρτες στο light mode */
.card,
.progress-card,
.donut-card {
    border: 1px solid #E2E8F0;
    box-shadow: 0 4px 16px rgba(15, 23, 42, 0.10);
}

/* Ίδιο μεγάλο μέγεθος σε ενεργούς και ανενεργούς βαθμούς */
div[data-testid="stNumberInput"] input {
    font-size: 18px !important;
    font-weight: 500 !important;
}

/* Συμπαγές πεδίο βαθμού με κοντινά πλήκτρα − / + */
[data-testid="stMain"] div[data-testid="stNumberInput"] {
    max-width: 170px !important;
}

[data-testid="stMain"] div[data-testid="stNumberInput"] button {
    width: 34px !important;
    min-width: 34px !important;
    padding-left: 0 !important;
    padding-right: 0 !important;
}

/* Bold μόνο όταν έχει επιλεγεί το "Περάστηκε" */
div[data-testid="stNumberInput"] input:not(:disabled) {
    font-weight: 650 !important;
    color: var(--text-color) !important;
}

/* Ίδιο μέγεθος σε όλες τις ετικέτες "Βαθμός" */
div[data-testid="stNumberInput"] label p {
    font-size: 17px !important;
    font-weight: 600 !important;
    line-height: 24px !important;
}

/* Bold η ετικέτα όταν ενεργοποιείται ο βαθμός */
div[data-testid="stNumberInput"]:has(input:not(:disabled)) label p {
    font-weight: 650 !important;
    color: var(--text-color) !important;
}

/* Mobile responsiveness */

@media (max-width: 768px) {

    .main-title {
        font-size: 28px !important;
        line-height: 1.2 !important;
    }

    .subtitle {
        font-size: 15px !important;
        margin-bottom: 22px !important;
    }

    .section-title {
        font-size: 26px !important;
        margin-top: 24px !important;
        margin-bottom: 18px !important;
    }

    .semester-timeline {
    overflow-x: hidden;
    justify-content: space-between;
    gap: 0;
    padding: 8px 6px 14px;
    margin-bottom: 10px;
    }

    .semester-timeline::before {
    left: 23px;
    right: 23px;
    width: auto;
    }

    .semester-step {
    min-width: 0;
    flex: 1 1 0;
    }

    .semester-circle {
        width: 34px;
        height: 34px;
        font-size: 13px;
    }

    .semester-label {
        font-size: 13px !important;
    }

    .current-semester-text {
        font-size: 18px !important;
        margin-top: 10px !important;
    }
    

    .card {
        padding: 18px !important;
        border-radius: 14px !important;
    }

    .metric-value {
        font-size: 26px !important;
    }

    .donut-progress {
        width: 145px !important;
        height: 145px !important;
    }

    .donut-inner {
        width: 105px !important;
        height: 105px !important;
    }

    .donut-value {
        font-size: 23px !important;
    }

    .donut-label {
        font-size: 12px !important;
    }

    .progress-card {
        padding: 18px !important;
    }

    .progress-card-value {
        font-size: 22px !important;
    }

    .custom-progress-track {
        height: 10px !important;
    }

    .card {
        padding: 14px !important;
    }

    .ects-row {
        gap: 5px !important;
    }

    .ects-label {
        font-size: 14px !important;
    }

    .ects-value {
        font-size: 18px !important;
    }

    .semester-step.active .semester-circle,
    .semester-step.completed .semester-circle {
        box-shadow: none !important;
    }

}

/* Διάταξη μαθημάτων μόνο σε οριζόντιο κινητό */
@media (min-width: 481px) and (max-width: 768px) and (orientation: landscape) {

    [data-testid="stHorizontalBlock"]
    > [data-testid="stColumn"]:first-child
    [data-testid="stHorizontalBlock"]
    > [data-testid="stColumn"]:nth-child(1) {
        flex: 1.6 1 0% !important;
        width: 0 !important;
        min-width: 0 !important;
    }

    [data-testid="stHorizontalBlock"]
    > [data-testid="stColumn"]:first-child
    [data-testid="stHorizontalBlock"]
    > [data-testid="stColumn"]:nth-child(2) {
        flex: 4 1 0% !important;
        width: 0 !important;
        min-width: 0 !important;
    }

    [data-testid="stHorizontalBlock"]
    > [data-testid="stColumn"]:first-child
    [data-testid="stHorizontalBlock"]
    > [data-testid="stColumn"]:nth-child(3) {
        flex: 1 1 0% !important;
        width: 0 !important;
        min-width: 0 !important;
    }

    div[data-testid="stCheckbox"] label p {
        font-size: 15px !important;
        white-space: nowrap !important;
    }
}


/* Κινητό σε κάθετη προβολή */
@media (max-width: 480px) {

    html,
    body,
    [data-testid="stAppViewContainer"],
    [data-testid="stMain"],
    .main {
        max-width: 100% !important;
        overflow-x: hidden !important;
    }

    [data-testid="stMainBlockContainer"] {
        max-width: 100% !important;
        overflow-x: hidden !important;
    }

    .intro-block {
        display: block !important;
        width: 100% !important;
        max-width: 100% !important;
    }

    .semester-timeline {
        width: 100% !important;
        max-width: 100% !important;
        box-sizing: border-box !important;

        justify-content: flex-start !important;
        gap: 14px !important;

        padding: 6px 4px 14px !important;
        margin: 20px 0 8px !important;

        overflow-x: auto !important;
        overflow-y: hidden !important;
        overscroll-behavior-x: contain;
        -webkit-overflow-scrolling: touch;
        scrollbar-width: none;
    }

    .semester-timeline::-webkit-scrollbar {
        display: none;
    }

    .semester-timeline::before {
    left: 28px !important;
    right: auto !important;
    width: 558px !important;
    top: 19px !important;
    height: 3px !important;
    }

    .semester-step {
        min-width: 48px !important;
        flex: 0 0 48px !important;
    }

    .semester-circle {
        width: 30px !important;
        height: 30px !important;
        font-size: 11px !important;
        border-width: 2px !important;
    }

    .semester-step.active .semester-circle {
        transform: scale(1.12) !important;
    }

    .semester-label {
        font-size: 11px !important;
        margin-top: 5px !important;
    }

    .current-semester-text {
        font-size: 16px !important;
        margin-top: 8px !important;
    }

    /* Συμπαγής κάρτα μαθήματος στο κατακόρυφο κινητό.
       Οι πέντε desktop στήλες γίνονται δύο λειτουργικές στήλες,
       χωρίς να αλλάζει το θέμα dark/light της συσκευής. */
    [data-testid="stMain"]
    [data-testid="stHorizontalBlock"]:has(
        > [data-testid="stColumn"]:nth-child(5)
    ):not(:has(> [data-testid="stColumn"]:nth-child(6))) {
        display: grid !important;
        grid-template-columns: repeat(2, minmax(0, 1fr)) !important;
        column-gap: 12px !important;
        row-gap: 10px !important;
        align-items: start !important;
        margin-bottom: 14px !important;
        padding: 14px !important;
        border: 1px solid rgba(122, 132, 148, 0.22);
        border-radius: 14px;
        background: rgba(122, 132, 148, 0.06);
    }

    [data-testid="stMain"]
    [data-testid="stHorizontalBlock"]:has(
        > [data-testid="stColumn"]:nth-child(5)
    ):not(:has(> [data-testid="stColumn"]:nth-child(6)))
    > [data-testid="stColumn"] {
        width: 100% !important;
        min-width: 0 !important;
        flex: none !important;
    }

    [data-testid="stMain"]
    [data-testid="stHorizontalBlock"]:has(
        > [data-testid="stColumn"]:nth-child(5)
    ) > [data-testid="stColumn"]:nth-child(1) {
        grid-column: 1 / -1;
        grid-row: 2;
    }

    [data-testid="stMain"]
    [data-testid="stHorizontalBlock"]:has(
        > [data-testid="stColumn"]:nth-child(5)
    ) > [data-testid="stColumn"]:nth-child(2) {
        grid-column: 1 / -1;
        grid-row: 1;
    }

    [data-testid="stMain"]
    [data-testid="stHorizontalBlock"]:has(
        > [data-testid="stColumn"]:nth-child(5)
    ) > [data-testid="stColumn"]:nth-child(3) {
        grid-column: 1 / -1;
        grid-row: 3;
    }

    [data-testid="stMain"]
    [data-testid="stHorizontalBlock"]:has(
        > [data-testid="stColumn"]:nth-child(5)
    ) > [data-testid="stColumn"]:nth-child(4) {
        grid-column: 1;
        grid-row: 4;
    }

    [data-testid="stMain"]
    [data-testid="stHorizontalBlock"]:has(
        > [data-testid="stColumn"]:nth-child(5)
    ) > [data-testid="stColumn"]:nth-child(5) {
        grid-column: 2;
        grid-row: 4;
    }

    [data-testid="stMain"]
    [data-testid="stHorizontalBlock"]:has(
        > [data-testid="stColumn"]:nth-child(5)
    ) > [data-testid="stColumn"]:nth-child(1)
    > [data-testid="stVerticalBlock"] {
        display: flex !important;
        flex-direction: row !important;
        flex-wrap: wrap !important;
        gap: 8px 18px !important;
        align-items: center !important;
    }

    .course-status-offset,
    .course-info-offset {
        display: none !important;
    }

    [data-testid="stMain"] div[data-testid="stCheckbox"] label p {
        font-size: 14px !important;
        white-space: nowrap !important;
    }

    [data-testid="stMain"] .course-title {
        font-size: 16px !important;
        line-height: 1.3 !important;
    }

    [data-testid="stMain"] .course-meta,
    [data-testid="stMain"] div[data-testid="stNumberInput"] label p,
    [data-testid="stMain"] div[data-testid="stSelectbox"] label p {
        font-size: 13px !important;
        line-height: 1.25 !important;
    }

    [data-testid="stMain"] div[data-testid="stNumberInput"] {
        max-width: none !important;
    }
}




.current-semester-text,
.current-semester-text strong {
    color: var(--text-color) !important;
    opacity: 1 !important;
    font-weight: 650 !important;
}

/* Κουμπί πρόβλεψης */
div.stButton {
    display: block;
    width: 100%;
    margin: 0;
}

div.stButton > button {
    width: 100% !important;
    max-width: none !important;
    height: 56px !important;

    border: none !important;
    border-radius: 15px !important;

    background: linear-gradient(
        90deg,
        #2E9AFE 0%,
        #6C63FF 100%
    ) !important;

    color: white !important;
    font-size: 19px !important;
    font-weight: 650 !important;

    box-shadow: 0 4px 14px rgba(79, 103, 255, 0.28);

    transition:
        transform 0.15s ease,
        box-shadow 0.15s ease,
        filter 0.15s ease;
}

div.stButton > button p {
    color: white !important;
    font-size: 19px !important;
    font-weight: 650 !important;
}

div.stButton > button:hover:not(:disabled) {
    background: linear-gradient(
        90deg,
        #2E9AFE 0%,
        #6C63FF 100%
    ) !important;
    transform: translateY(-1px);
    filter: brightness(1.05);
    box-shadow: 0 6px 18px rgba(79, 103, 255, 0.35);
}

div.stButton > button:disabled {
    background: linear-gradient(
        90deg,
        #2E9AFE 0%,
        #6C63FF 100%
    ) !important;
    opacity: 0.45;
    cursor: not-allowed;
    box-shadow: none;
}

/* Τίτλος αποτελεσμάτων στη δεξιά στήλη */
.prediction-results-title {
    color: var(--text-color);
    font-size: 21px;
    font-weight: 700;
    margin-top: 22px;
    margin-bottom: 12px;
}

/* Κάρτες αποτελεσμάτων */
.prediction-result-card {
    background-color: white;
    padding: 20px 22px;
    border: 1px solid #E2E8F0;
    border-radius: 16px;
    box-shadow: 0 4px 16px rgba(15, 23, 42, 0.10);
    margin-bottom: 14px;
}

.prediction-result-label {
    color: #475569;
    font-size: 16px;
    font-weight: 600;
    line-height: 1.35;
    margin-bottom: 8px;
}

.prediction-result-value {
    color: #4B3869;
    font-size: 32px;
    font-weight: 800;
    line-height: 1.2;
}

.duration-value-line {
    display: block;
    white-space: nowrap;
}


.prediction-disclaimer {
    color: var(--text-color);
    font-size: 13px;
    font-weight: 400;
    line-height: 1.45;
    opacity: 0.72;
    margin: 4px 2px 20px;
}

.ects-card-title {
    color: #475569;
    font-size: 17px;
    font-weight: 650;
    margin-bottom: 14px;
}

.ects-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 12px;
}

.ects-label {
    color: #334155;
    font-size: 16px;
    font-weight: 600;
}

.ects-value {
    color: #4B3869;
    font-size: 22px;
    font-weight: 750;
    white-space: nowrap;
}

.ects-divider {
    height: 1.5px;
    background-color: #CBD5E1;
    margin: 12px 0;
}

.metric-title,
.progress-card-title,
.donut-label,
.ects-card-title,
.ects-label,
.prediction-result-label {
    color: #334155;
}

/* Κείμενο "Περάστηκε" */
div[data-testid="stCheckbox"] label p {
    font-size: 16px !important;
    font-weight: 600 !important;
    line-height: 24px !important;
}

/* Ενιαία τυπογραφία στα μικρά πεδία της γραμμής μαθήματος */
[data-testid="stMain"] div[data-testid="stSelectbox"] label p {
    font-size: 17px !important;
    font-weight: 600 !important;
    line-height: 24px !important;
}

[data-testid="stMain"] div[data-testid="stSelectbox"]
div[data-baseweb="select"] {
    font-size: 17px !important;
    font-weight: 500 !important;
}

/* Ελαφρώς μεγαλύτερο checkbox */
div[data-testid="stCheckbox"] label > span:first-child {
    transform: scale(1);
    transform-origin: center;
}

/* Χρώμα στα βελάκια ανοίγματος και κλεισίματος sidebar */
[data-testid="stSidebarCollapseButton"] button,
[data-testid="stExpandSidebarButton"] button {
    color: #2A7DEB !important;
    background: transparent !important;
    box-shadow: none !important;
}

/* Material icon του Streamlit */
[data-testid="stSidebarCollapseButton"] span,
[data-testid="stExpandSidebarButton"] span {
    color: #2A7DEB !important;
    font-size: 28px !important;
}

/* Εφεδρικά, αν αποδοθεί ως SVG */
[data-testid="stSidebarCollapseButton"] svg,
[data-testid="stExpandSidebarButton"] svg {
    color: #2A7DEB !important;
    stroke: #2A7DEB!important;
    line-height: 1 !important;
}

/* Τίτλοι των επιλογών στο sidebar */
[data-testid="stSidebar"]
div[data-testid="stSelectbox"] label p {
    font-size: 17px !important;
    font-weight: 450 !important;
}

/* Μπλε μόνο το κουτάκι, όχι το κείμενο */
[data-testid="stCheckbox"]
label:has(input:checked) > span:first-child {
    background-color: #2A7DEB !important;
    border-color: #2A7DEB !important;
}

/* Ενεργή καρτέλα εξαμήνου */
button[data-baseweb="tab"][aria-selected="true"] {
    color: #2A7DEB !important;
}

/* Γραμμή κάτω από την ενεργή καρτέλα */
div[data-baseweb="tab-highlight"] {
    background-color: #2A7DEB !important;
}

.info-tooltip {
    position: relative;
    display: inline-block;
    margin-left: 8px;
    color: #7A8494;
    font-size: 17px;
    cursor: help;
    vertical-align: middle;
}

.tooltip-text {
    visibility: hidden;
    opacity: 0;
    position: absolute;
    left: 50%;
    top: 30px;
    transform: translateX(-50%);
    width: 290px;
    padding: 10px 12px;
    border-radius: 8px;
    background: #252832;
    color: #FFFFFF;
    font-size: 14px;
    font-weight: 400;
    line-height: 1.4;
    text-align: left;
    z-index: 100;
    transition: opacity 0.15s ease;
    box-shadow: 0 5px 18px rgba(0, 0, 0, 0.18);
}

.info-tooltip:hover .tooltip-text {
    visibility: visible;
    opacity: 1;
}

</style>
""", unsafe_allow_html=True)

# On touch devices, Streamlit's searchable selectboxes focus their internal
# text input and open the software keyboard. Keep the dropdown fully tappable,
# but make that internal search field read-only only on mobile/touch screens.
components.html(
    """
    <script>
        (function () {
            const parentWindow = window.parent;
            const parentDocument = parentWindow.document;
            const isTouchDevice =
                parentWindow.matchMedia("(pointer: coarse)").matches ||
                parentWindow.matchMedia("(max-width: 768px)").matches;

            if (!isTouchDevice) {
                return;
            }

            function disableSelectboxKeyboard() {
                parentDocument
                    .querySelectorAll('[data-baseweb="select"] input')
                    .forEach(function (input) {
                        input.readOnly = true;
                        input.setAttribute("inputmode", "none");
                        input.setAttribute("autocomplete", "off");
                    });
            }

            disableSelectboxKeyboard();

            const observer = new MutationObserver(
                disableSelectboxKeyboard
            );
            observer.observe(parentDocument.body, {
                childList: true,
                subtree: true
            });
        })();
    </script>
    """,
    height=0,
)

# -----------------------------
# Example curriculum
# Later we will replace this with a real curriculum file
# -----------------------------
courses_by_semester = {
   
    "1ο Εξάμηνο": [
        {"code": "101", "title": "Λογισμός Ι (Μαθηματικά Ι)", "ects": 4},
        {"code": "102", "title": "Φυσική", "ects": 6},
        {"code": "103", "title": "Μηχανολογικό Σχέδιο Ι", "ects": 6},
        {"code": "104", "title": "Επιστήμη και Τεχνολογία Υλικών Ι", "ects": 6},
        {"code": "131", "title": "Γραμμική Άλγεβρα", "ects": 4},
        {"code": "132", "title": "Χημεία", "ects": 4},
    ],

    "2ο Εξάμηνο": [
        {"code": "105", "title": "Πληροφορική", "ects": 6},
        {"code": "106", "title": "Λογισμός ΙΙ (Μαθηματικά ΙΙ)", "ects": 6},
        {"code": "108", "title": "Στατική", "ects": 6},
        {"code": "109", "title": "Μηχανολογικό Σχέδιο ΙΙ", "ects": 6},
        {"code": "115", "title": "Ηλεκτροτεχνία - Ηλεκτρικές Μηχανές", "ects": 6},
    ],

    "3ο Εξάμηνο": [
        {"code": "111", "title": "Διαφορικές Εξισώσεις (Μαθηματικά ΙΙΙ)", "ects": 6},
        {"code": "112", "title": "Αντοχή Υλικών", "ects": 6},
        {"code": "113", "title": "Θερμοδυναμική Ι", "ects": 6},
        {"code": "114", "title": "Στατιστική", "ects": 6},
        {"code": "123", "title": "Μορφοποιήσεις με Αφαίρεση Υλικού", "ects": 6},
    ],

    "4ο Εξάμηνο": [
        {"code": "116", "title": "Δυναμική", "ects": 6},
        {"code": "117", "title": "Επιστήμη και Τεχνολογία Υλικών ΙΙ", "ects": 6},
        {"code": "118", "title": "Μηχανική Ρευστών Ι", "ects": 6},
        {"code": "120", "title": "Αριθμητική Ανάλυση", "ects": 6},
        {"code": "126", "title": "Ηλεκτρονική", "ects": 6},
    ],

    "5ο Εξάμηνο": [
        {"code": "119", "title": "Στοιχεία Μηχανών Ι", "ects": 6},
        {"code": "122", "title": "Μετάδοση Θερμότητας", "ects": 6},
        {"code": "124", "title": "Ταλαντώσεις και Δυναμική Μηχανών", "ects": 6},
        {"code": "125", "title": "Επιχειρησιακή Έρευνα Ι", "ects": 6},
        {"code": "216", "title": "Θερμοδυναμική ΙΙ", "ects": 6},
    ],

    "6ο Εξάμηνο": [
        {"code": "121", "title": "Στοιχεία Μηχανών ΙΙ", "ects": 6},
        {"code": "128", "title": "Συστήματα Μετατροπής Ενέργειας", "ects": 6},
        {"code": "129", "title": "Αυτόματος Έλεγχος", "ects": 6},
        {"code": "130", "title": "Διοίκηση Συστημάτων Παραγωγής και Υπηρεσιών", "ects": 6},
        {"code": "351", "title": "Μηχανική Ρευστών ΙΙ", "ects": 6},
    ],

}

direction_courses_by_semester = {
    "Ενεργειακός": {
        "7ο Εξάμηνο": [
    {"code": "205", "title": "Αεροδυναμική", "ects": 5, "category": "ΥΚ"},
    {"code": "207", "title": "Στροβιλομηχανές", "ects": 5, "category": "ΥΚ"},
    {"code": "219", "title": "Μηχανές Εσωτερικής Καύσης Ι", "ects": 5, "category": "ΥΚ"},
    {"code": "349", "title": "Ανανεώσιμες Πηγές Ενέργειας", "ects": 5, "category": "ΥΚ"},

    {"code": "229", "title": "Μετάδοση Θερμότητας ΙΙ", "ects": 5, "category": "ΕΚ"},

    {"code": "202", "title": "Μηχανολογικός Σχεδιασμός & Ανάπτυξη Προϊόντων (Στοιχεία Μηχανών ΙΙΙ)", "ects": 5, "category": "ΕΤΑΚ"},
    {"code": "203", "title": "Ευέλικτα Συστήματα Μηχανουργικών Μορφοποιήσεων Υποστηριζόμενα από Η/Υ", "ects": 5, "category": "ΕΤΑΚ"},
    {"code": "208", "title": "Στατιστική Πολλών Μεταβλητών", "ects": 5, "category": "ΕΤΑΚ"},
    {"code": "209", "title": "Οργάνωση και Διοίκηση Εργοστασίων", "ects": 5, "category": "ΕΤΑΚ"},
    {"code": "210", "title": "Βάσεις Δεδομένων και Συστήματα Πληροφοριών Διοίκησης", "ects": 5, "category": "ΕΤΑΚ"},
    {"code": "215", "title": "Πειραματική Αντοχή Υλικών", "ects": 5, "category": "ΕΤΑΚ"},
    {"code": "221", "title": "Επιχειρησιακή Έρευνα ΙΙ", "ects": 5, "category": "ΕΤΑΚ"},
    {"code": "305", "title": "Μέθοδος Πεπερασμένων Στοιχείων", "ects": 5, "category": "ΕΤΑΚ"},
    {"code": "327", "title": "Διαχείριση Αποθεμάτων", "ects": 5, "category": "ΕΤΑΚ"},
    {"code": "379", "title": "Αυτόματος και Ψηφιακός Έλεγχος Δυναμικών Συστημάτων", "ects": 5, "category": "ΕΤΑΚ"},
],
        "8ο Εξάμηνο": [
    {"code": "217", "title": "Θέρμανση – Ψύξη – Κλιματισμός", "ects": 5, "category": "ΥΚ"},
    {"code": "218", "title": "Μηχανική Αειφορίας (Τεχνική Προστασίας Περιβάλλοντος – Αρχές Αειφορίας)", "ects": 5, "category": "ΥΚ"},
    {"code": "387", "title": "Αεροπορικοί Κινητήρες", "ects": 5, "category": "ΥΚ"},

    {"code": "206", "title": "Τεχνική Φυσικών Διεργασιών", "ects": 5, "category": "ΕΚ"},
    {"code": "226", "title": "Φαινόμενα Μεταφοράς Μάζας και Ενέργειας", "ects": 5, "category": "ΕΚ"},

    {"code": "204", "title": "Επιλογή και Αστοχία Υλικών", "ects": 5, "category": "ΕΤΑΚ"},
    {"code": "211", "title": "Επιχειρηματικό Περιβάλλον", "ects": 5, "category": "ΕΤΑΚ"},
    {"code": "212", "title": "Μορφοποιήσεις με Πλαστική Παραμόρφωση Υλικού", "ects": 5, "category": "ΕΤΑΚ"},
    {"code": "213", "title": "Ανάλυση και Σύνθεση Μηχανισμών", "ects": 5, "category": "ΕΤΑΚ"},
    {"code": "214", "title": "Δυναμική Κατασκευών", "ects": 5, "category": "ΕΤΑΚ"},
    {"code": "222", "title": "Προγραμματισμός και Έλεγχος Παραγωγής", "ects": 5, "category": "ΕΤΑΚ"},
    {"code": "223", "title": "Τεχνικές και Οργάνωση Ελέγχου Ποιότητας", "ects": 5, "category": "ΕΤΑΚ"},
    {"code": "227", "title": "Εφαρμοσμένα Μη-Γραμμικά Δυναμικά Συστήματα", "ects": 5, "category": "ΕΤΑΚ"},
    {"code": "228", "title": "Ανάλυση Δεδομένων", "ects": 5, "category": "ΕΤΑΚ"},
    {"code": "333", "title": "Διάβρωση και Προστασία Υλικών", "ects": 5, "category": "ΕΤΑΚ"},
    {"code": "365", "title": "Διαχείριση Εφοδιαστικής Αλυσίδας", "ects": 5, "category": "ΕΤΑΚ"},
    {"code": "369", "title": "Μηχανική Συμπεριφορά και Κόπωση", "ects": 5, "category": "ΕΤΑΚ"},
],

"9ο Εξάμηνο": [
    {"code": "354", "title": "Τεχνολογία Επεξεργασίας Νερού και Υγρών Αποβλήτων", "ects": 5, "category": "ΥΚ"},

    {"code": "314", "title": "Διαχείριση Ενεργειακών Πόρων", "ects": 5, "category": "ΕΚ"},
    {"code": "315", "title": "Θερμικές Στροβιλομηχανές", "ects": 5, "category": "ΕΚ"},
    {"code": "319", "title": "Εκτίμηση Περιβαλλοντικών Επιπτώσεων", "ects": 5, "category": "ΕΚ"},
    {"code": "323", "title": "Αριθμητική Βελτιστοποίηση σε Μηχανολογικές Κατασκευές και Διεργασίες", "ects": 5, "category": "ΕΚ"},
    {"code": "347", "title": "Καύση – Ηλεκτροχημεία (Καύση)", "ects": 5, "category": "ΕΚ"},
    {"code": "352", "title": "Ατμοσφαιρική Ρύπανση", "ects": 5, "category": "ΕΚ"},
    {"code": "355", "title": "Υπολογιστική Ρευστομηχανική", "ects": 5, "category": "ΕΚ"},
    {"code": "360", "title": "Αριθμητικές Μέθοδοι σε Ταλαντώσεις Μηχανολογικών Συστημάτων", "ects": 5, "category": "ΕΚ"},
    {"code": "363", "title": "Ενεργειακή και Περιβαλλοντική Απόδοση Κτιρίων", "ects": 5, "category": "ΕΚ"},
    {"code": "366", "title": "Συστήματα Θέρμανσης", "ects": 5, "category": "ΕΚ"},
    {"code": "370", "title": "Εισαγωγή στη Μηχανοηλεκτρονική", "ects": 5, "category": "ΕΚ"},
    {"code": "378", "title": "Περιβαλλοντική Πληροφορική", "ects": 5, "category": "ΕΚ"},
    {"code": "398", "title": "Βιομηχανική Ψύξη", "ects": 5, "category": "ΕΚ"},
],

"10ο Εξάμηνο": [
    {"code": "313", "title": "Θερμικές Διεργασίες και Εφαρμογές τους", "ects": 5, "category": "ΕΚ"},
    {"code": "316", "title": "Μηχανές Εσωτερικής Καύσης ΙΙ", "ects": 5, "category": "ΕΚ"},
    {"code": "318", "title": "Τεχνολογία Αντιρρύπανσης (Τεχνολογία Αντιρρύπανσης Σταθερών Πηγών)", "ects": 5, "category": "ΕΚ"},
    {"code": "321", "title": "Τεχνολογίες Μείωσης Αερίων Θερμοκηπίου στις Μεταφορές (Πηγές)", "ects": 5, "category": "ΕΚ"},
    {"code": "325", "title": "Διαχείριση Αποβλήτων και Κυκλική Οικονομία", "ects": 5, "category": "ΕΚ"},
    {"code": "326", "title": "Πειραματική Μηχανική Ρευστών", "ects": 5, "category": "ΕΚ"},
    {"code": "348", "title": "Ευφυή Ενεργειακά Συστήματα και Δίκτυα", "ects": 5, "category": "ΕΚ"},
    {"code": "357", "title": "Τεχνικές Προβλέψεων", "ects": 5, "category": "ΕΚ"},
    {"code": "367", "title": "Κλιματισμός", "ects": 5, "category": "ΕΚ"},
    {"code": "377", "title": "Βέλτιστος Έλεγχος Δυναμικών Συστημάτων", "ects": 5, "category": "ΕΚ"},
    {"code": "389", "title": "Αεροδυναμικός Σχεδιασμός και Έλεγχος Αεροσκαφών", "ects": 5, "category": "ΕΚ"},
    {"code": "396", "title": "Υπολογιστική Δυναμική Παραμορφώσιμων Σωμάτων", "ects": 5, "category": "ΕΚ"},
    {"code": "399", "title": "Ανανεώσιμες Πηγές Ενέργειας ΙΙ", "ects": 5, "category": "ΕΚ"},
],
    },

    "Κατασκευαστικός": {
        "7ο Εξάμηνο": [
    {"code": "202", "title": "Μηχανολογικός Σχεδιασμός & Ανάπτυξη Προϊόντων (Στοιχεία Μηχανών ΙΙΙ)", "ects": 5, "category": "ΥΚ"},
    {"code": "203", "title": "Ευέλικτα Συστήματα Μηχανουργικών Μορφοποιήσεων Υποστηριζόμενα από Η/Υ", "ects": 5, "category": "ΥΚ"},
    {"code": "379", "title": "Αυτόματος και Ψηφιακός Έλεγχος Δυναμικών Συστημάτων", "ects": 5, "category": "ΥΚ"},

    {"code": "215", "title": "Πειραματική Αντοχή Υλικών", "ects": 5, "category": "ΕΚ"},
    {"code": "305", "title": "Μέθοδος Πεπερασμένων Στοιχείων", "ects": 5, "category": "ΕΚ"},

    {"code": "205", "title": "Αεροδυναμική", "ects": 5, "category": "ΕΤΑΚ"},
    {"code": "207", "title": "Στροβιλομηχανές", "ects": 5, "category": "ΕΤΑΚ"},
    {"code": "208", "title": "Στατιστική Πολλών Μεταβλητών", "ects": 5, "category": "ΕΤΑΚ"},
    {"code": "209", "title": "Οργάνωση και Διοίκηση Εργοστασίων", "ects": 5, "category": "ΕΤΑΚ"},
    {"code": "210", "title": "Βάσεις Δεδομένων και Συστήματα Πληροφοριών Διοίκησης", "ects": 5, "category": "ΕΤΑΚ"},
    {"code": "219", "title": "Μηχανές Εσωτερικής Καύσης Ι", "ects": 5, "category": "ΕΤΑΚ"},
    {"code": "221", "title": "Επιχειρησιακή Έρευνα ΙΙ", "ects": 5, "category": "ΕΤΑΚ"},
    {"code": "229", "title": "Μετάδοση Θερμότητας ΙΙ", "ects": 5, "category": "ΕΤΑΚ"},
    {"code": "327", "title": "Διαχείριση Αποθεμάτων", "ects": 5, "category": "ΕΤΑΚ"},
    {"code": "349", "title": "Ανανεώσιμες Πηγές Ενέργειας", "ects": 5, "category": "ΕΤΑΚ"},
],
        "8ο Εξάμηνο": [
    {"code": "204", "title": "Επιλογή και Αστοχία Υλικών", "ects": 5, "category": "ΥΚ"},
    {"code": "212", "title": "Μορφοποιήσεις με Πλαστική Παραμόρφωση Υλικού", "ects": 5, "category": "ΥΚ"},
    {"code": "214", "title": "Δυναμική Κατασκευών", "ects": 5, "category": "ΥΚ"},
    {"code": "333", "title": "Διάβρωση και Προστασία Υλικών", "ects": 5, "category": "ΥΚ"},
    {"code": "369", "title": "Μηχανική Συμπεριφορά και Κόπωση", "ects": 5, "category": "ΥΚ"},

    {"code": "213", "title": "Ανάλυση και Σύνθεση Μηχανισμών", "ects": 5, "category": "ΕΚ"},
    {"code": "227", "title": "Εφαρμοσμένα Μη-Γραμμικά Δυναμικά Συστήματα", "ects": 5, "category": "ΕΚ"},

    {"code": "206", "title": "Τεχνική Φυσικών Διεργασιών", "ects": 5, "category": "ΕΤΑΚ"},
    {"code": "211", "title": "Επιχειρηματικό Περιβάλλον", "ects": 5, "category": "ΕΤΑΚ"},
    {"code": "217", "title": "Θέρμανση – Ψύξη – Κλιματισμός", "ects": 5, "category": "ΕΤΑΚ"},
    {"code": "218", "title": "Μηχανική Αειφορίας (Τεχνική Προστασίας Περιβάλλοντος – Αρχές Αειφορίας)", "ects": 5, "category": "ΕΤΑΚ"},
    {"code": "222", "title": "Προγραμματισμός και Έλεγχος Παραγωγής", "ects": 5, "category": "ΕΤΑΚ"},
    {"code": "223", "title": "Τεχνικές και Οργάνωση Ελέγχου Ποιότητας", "ects": 5, "category": "ΕΤΑΚ"},
    {"code": "226", "title": "Φαινόμενα Μεταφοράς Μάζας και Ενέργειας", "ects": 5, "category": "ΕΤΑΚ"},
    {"code": "228", "title": "Ανάλυση Δεδομένων", "ects": 5, "category": "ΕΤΑΚ"},
    {"code": "365", "title": "Διαχείριση Εφοδιαστικής Αλυσίδας", "ects": 5, "category": "ΕΤΑΚ"},
    {"code": "387", "title": "Αεροπορικοί Κινητήρες", "ects": 5, "category": "ΕΤΑΚ"},
],

"9ο Εξάμηνο": [
    {"code": "301", "title": "Επιστήμη και Τεχνολογία Συγκολλήσεων (Χύτευση – Συγκολλήσεις)", "ects": 5, "category": "ΥΚ"},
    {"code": "309", "title": "Μετροτεχνία Ποιοτικού Ελέγχου", "ects": 5, "category": "ΥΚ"},
    {"code": "311", "title": "Αριθμητικές Μέθοδοι για την Προσομοίωση Μεθόδων Μορφοποίησης", "ects": 5, "category": "ΥΚ"},
    {"code": "323", "title": "Αριθμητική Βελτιστοποίηση σε Μηχανολογικές Κατασκευές και Διεργασίες", "ects": 5, "category": "ΥΚ"},
    {"code": "336", "title": "Τεχνολογία Προηγμένων Υλικών", "ects": 5, "category": "ΥΚ"},
    {"code": "360", "title": "Αριθμητικές Μέθοδοι σε Ταλαντώσεις Μηχανολογικών Συστημάτων", "ects": 5, "category": "ΥΚ"},
    {"code": "393", "title": "Σύγχρονες Μέθοδοι Υπολογισμού Διάρκειας Ζωής Μηχανολογικών Κατασκευών", "ects": 5, "category": "ΥΚ"},
    {"code": "394", "title": "CAE – Προσομοίωση Μηχανολογικών Κατασκευών", "ects": 5, "category": "ΥΚ"},

    {"code": "302", "title": "Πειραματικές Μέθοδοι Μελέτης των Υλικών", "ects": 5, "category": "ΕΚ"},
    {"code": "308", "title": "Εργαλειομηχανές", "ects": 5, "category": "ΕΚ"},
    {"code": "310", "title": "Μηχανουργικές Μορφοποιήσεις με Ψηφιακή Καθοδήγηση", "ects": 5, "category": "ΕΚ"},
    {"code": "315", "title": "Θερμικές Στροβιλομηχανές", "ects": 5, "category": "ΕΚ"},
    {"code": "345", "title": "Διαγνωστικός Έλεγχος Εργαλειομηχανών", "ects": 5, "category": "ΕΚ"},
    {"code": "346", "title": "Ειδικά Κεφάλαια Ευέλικτων Συστημάτων Μηχανουργικών Μορφοποιήσεων", "ects": 5, "category": "ΕΚ"},
    {"code": "347", "title": "Καύση – Ηλεκτροχημεία (Καύση)", "ects": 5, "category": "ΕΚ"},
    {"code": "352", "title": "Ατμοσφαιρική Ρύπανση", "ects": 5, "category": "ΕΚ"},
    {"code": "355", "title": "Υπολογιστική Ρευστομηχανική", "ects": 5, "category": "ΕΚ"},
    {"code": "370", "title": "Εισαγωγή στη Μηχανοηλεκτρονική", "ects": 5, "category": "ΕΚ"},
    {"code": "373", "title": "Μελέτη και Σχεδιασμός Οχημάτων", "ects": 5, "category": "ΕΚ"},
    {"code": "374", "title": "Εμβιομηχανική", "ects": 5, "category": "ΕΚ"},
    {"code": "376", "title": "Υλικά και Περιβάλλον", "ects": 5, "category": "ΕΚ"},
    {"code": "378", "title": "Περιβαλλοντική Πληροφορική", "ects": 5, "category": "ΕΚ"},
    {"code": "391", "title": "Στερεοποίηση – Χύτευση", "ects": 5, "category": "ΕΚ"},

],

"10ο Εξάμηνο": [
    {"code": "334", "title": "Θερμικές Κατεργασίες και Μετασχηματισμοί Φάσεων", "ects": 5, "category": "ΥΚ"},
    {"code": "337", "title": "Εφαρμογές Επιφανειακών Επικαλύψεων στις Μορφοποιήσεις", "ects": 5, "category": "ΥΚ"},
    {"code": "342", "title": "Ειδικά Κεφάλαια Πεπερασμένων Στοιχείων", "ects": 5, "category": "ΥΚ"},
    {"code": "368", "title": "Αντίστροφη Μηχανολογία και Μέθοδοι Ταχείας Πρωτυποποίησης", "ects": 5, "category": "ΥΚ"},
    {"code": "375", "title": "Έξυπνα Υλικά – Νανοτεχνολογία", "ects": 5, "category": "ΥΚ"},
    {"code": "377", "title": "Βέλτιστος Έλεγχος Δυναμικών Συστημάτων", "ects": 5, "category": "ΥΚ"},
    {"code": "395", "title": "Ανάλυση Συγκολλητών Κατασκευών", "ects": 5, "category": "ΥΚ"},
    {"code": "396", "title": "Υπολογιστική Δυναμική Παραμορφώσιμων Σωμάτων", "ects": 5, "category": "ΥΚ"},

    {"code": "303", "title": "Τριβολογία", "ects": 5, "category": "ΕΚ"},
    {"code": "312", "title": "Χωρικοί Μηχανισμοί – Βιομηχανικά Ρομπότ", "ects": 5, "category": "ΕΚ"},
    {"code": "316", "title": "Μηχανές Εσωτερικής Καύσης ΙΙ", "ects": 5, "category": "ΕΚ"},
    {"code": "329", "title": "Δυναμική Συστημάτων", "ects": 5, "category": "ΕΚ"},
    {"code": "338", "title": "Ανυψωτικές και Μεταφορικές Μηχανές", "ects": 5, "category": "ΕΚ"},
    {"code": "340", "title": "Ελαφρές Κατασκευές", "ects": 5, "category": "ΕΚ"},
    {"code": "344", "title": "Μηχανικές και Θερμικές Κατεργασίες Μορφοποίησης Οδοντώσεων", "ects": 5, "category": "ΕΚ"},
    {"code": "361", "title": "Πειραματικές Μέθοδοι στη Δυναμική Μηχανών", "ects": 5, "category": "ΕΚ"},
    {"code": "364", "title": "Προσομοίωση", "ects": 5, "category": "ΕΚ"},
    {"code": "389", "title": "Αεροδυναμικός Σχεδιασμός και Έλεγχος Αεροσκαφών", "ects": 5, "category": "ΕΚ"},
    {"code": "392", "title": "Σύνθετα Υλικά", "ects": 5, "category": "ΕΚ"},
],
    },

    "Βιομηχανικής Διοίκησης": {
        "7ο Εξάμηνο": [
    {"code": "208", "title": "Στατιστική Πολλών Μεταβλητών", "ects": 5, "category": "ΥΚ"},
    {"code": "327", "title": "Διαχείριση Αποθεμάτων", "ects": 5, "category": "ΥΚ"},

    {"code": "209", "title": "Οργάνωση και Διοίκηση Εργοστασίων", "ects": 5, "category": "ΕΚ"},
    {"code": "210", "title": "Βάσεις Δεδομένων και Συστήματα Πληροφοριών Διοίκησης", "ects": 5, "category": "ΕΚ"},
    {"code": "221", "title": "Επιχειρησιακή Έρευνα ΙΙ", "ects": 5, "category": "ΕΚ"},

    {"code": "202", "title": "Μηχανολογικός Σχεδιασμός & Ανάπτυξη Προϊόντων (Στοιχεία Μηχανών ΙΙΙ)", "ects": 5, "category": "ΕΤΑΚ"},
    {"code": "203", "title": "Ευέλικτα Συστήματα Μηχανουργικών Μορφοποιήσεων Υποστηριζόμενα από Η/Υ", "ects": 5, "category": "ΕΤΑΚ"},
    {"code": "205", "title": "Αεροδυναμική", "ects": 5, "category": "ΕΤΑΚ"},
    {"code": "207", "title": "Στροβιλομηχανές", "ects": 5, "category": "ΕΤΑΚ"},
    {"code": "215", "title": "Πειραματική Αντοχή Υλικών", "ects": 5, "category": "ΕΤΑΚ"},
    {"code": "219", "title": "Μηχανές Εσωτερικής Καύσης Ι", "ects": 5, "category": "ΕΤΑΚ"},
    {"code": "229", "title": "Μετάδοση Θερμότητας ΙΙ", "ects": 5, "category": "ΕΤΑΚ"},
    {"code": "305", "title": "Μέθοδος Πεπερασμένων Στοιχείων", "ects": 5, "category": "ΕΤΑΚ"},
    {"code": "349", "title": "Ανανεώσιμες Πηγές Ενέργειας", "ects": 5, "category": "ΕΤΑΚ"},
    {"code": "379", "title": "Αυτόματος και Ψηφιακός Έλεγχος Δυναμικών Συστημάτων", "ects": 5, "category": "ΕΤΑΚ"},
],
       "8ο Εξάμηνο": [
    {"code": "223", "title": "Τεχνικές και Οργάνωση Ελέγχου Ποιότητας", "ects": 5, "category": "ΥΚ"},
    {"code": "228", "title": "Ανάλυση Δεδομένων", "ects": 5, "category": "ΥΚ"},
    {"code": "365", "title": "Διαχείριση Εφοδιαστικής Αλυσίδας", "ects": 5, "category": "ΥΚ"},

    {"code": "211", "title": "Επιχειρηματικό Περιβάλλον", "ects": 5, "category": "ΕΚ"},
    {"code": "222", "title": "Προγραμματισμός και Έλεγχος Παραγωγής", "ects": 5, "category": "ΕΚ"},

    {"code": "204", "title": "Επιλογή και Αστοχία Υλικών", "ects": 5, "category": "ΕΤΑΚ"},
    {"code": "206", "title": "Τεχνική Φυσικών Διεργασιών", "ects": 5, "category": "ΕΤΑΚ"},
    {"code": "212", "title": "Μορφοποιήσεις με Πλαστική Παραμόρφωση Υλικού", "ects": 5, "category": "ΕΤΑΚ"},
    {"code": "213", "title": "Ανάλυση και Σύνθεση Μηχανισμών", "ects": 5, "category": "ΕΤΑΚ"},
    {"code": "214", "title": "Δυναμική Κατασκευών", "ects": 5, "category": "ΕΤΑΚ"},
    {"code": "217", "title": "Θέρμανση – Ψύξη – Κλιματισμός", "ects": 5, "category": "ΕΤΑΚ"},
    {"code": "218", "title": "Μηχανική Αειφορίας (Τεχνική Προστασίας Περιβάλλοντος – Αρχές Αειφορίας)", "ects": 5, "category": "ΕΤΑΚ"},
    {"code": "226", "title": "Φαινόμενα Μεταφοράς Μάζας και Ενέργειας", "ects": 5, "category": "ΕΤΑΚ"},
    {"code": "227", "title": "Εφαρμοσμένα Μη-Γραμμικά Δυναμικά Συστήματα", "ects": 5, "category": "ΕΤΑΚ"},
    {"code": "333", "title": "Διάβρωση και Προστασία Υλικών", "ects": 5, "category": "ΕΤΑΚ"},
    {"code": "369", "title": "Μηχανική Συμπεριφορά και Κόπωση", "ects": 5, "category": "ΕΤΑΚ"},
    {"code": "387", "title": "Αεροπορικοί Κινητήρες", "ects": 5, "category": "ΕΤΑΚ"},
],

"9ο Εξάμηνο": [
    {"code": "356", "title": "Οργάνωση και Διοίκηση Έργων", "ects": 5, "category": "ΥΚ"},
    {"code": "359", "title": "Βιομηχανική Πληροφορική", "ects": 5, "category": "ΥΚ"},
    {"code": "364", "title": "Προσομοίωση", "ects": 5, "category": "ΥΚ"},

    {"code": "314", "title": "Διαχείριση Ενεργειακών Πόρων", "ects": 5, "category": "ΕΚ"},
    {"code": "319", "title": "Εκτίμηση Περιβαλλοντικών Επιπτώσεων", "ects": 5, "category": "ΕΚ"},
    {"code": "323", "title": "Αριθμητική Βελτιστοποίηση σε Μηχανολογικές Κατασκευές και Διεργασίες", "ects": 5, "category": "ΕΚ"},
    {"code": "331", "title": "Επιχειρηματικότητα και Καινοτομία", "ects": 5, "category": "ΕΚ"},
    {"code": "378", "title": "Περιβαλλοντική Πληροφορική", "ects": 5, "category": "ΕΚ"},

    
],

"10ο Εξάμηνο": [
    {"code": "224", "title": "Οικονομική των Επιχειρήσεων", "ects": 5, "category": "ΕΚ"},
    {"code": "325", "title": "Διαχείριση Αποβλήτων και Κυκλική Οικονομία", "ects": 5, "category": "ΕΚ"},
    {"code": "328", "title": "Αξιοπιστία και Συντήρηση", "ects": 5, "category": "ΕΚ"},
    {"code": "329", "title": "Δυναμική Συστημάτων", "ects": 5, "category": "ΕΚ"},
    {"code": "330", "title": "Διοίκηση Ανθρώπινου Δυναμικού", "ects": 5, "category": "ΕΚ"},
    {"code": "357", "title": "Τεχνικές Προβλέψεων", "ects": 5, "category": "ΕΚ"},
    {"code": "358", "title": "Μάρκετινγκ και Επικοινωνία", "ects": 5, "category": "ΕΚ"},

    
],
    },
}


TOTAL_COURSES = 49
COURSE_ECTS = 270
THESIS_ECTS = 30
TOTAL_DEGREE_ECTS = COURSE_ECTS + THESIS_ECTS




# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("Το προφίλ μου")

PROFILE_STATE_KEYS = {
    "enrollment_academic_year",
    "enrollment_semester",
    "direction",
    "thesis_status",
    "has_recognized_courses",
}
COURSE_STATE_PREFIXES = (
    "graded_",
    "recognized_",
    "grade_",
    "success_year_",
    "success_period_",
)


def state_aware_selectbox(
    container,
    label,
    options,
    *,
    key,
    default_index=0,
    **kwargs,
):
    """Avoid passing a default when a restored Session State value exists."""
    selectbox_kwargs = {
        "key": key,
        **kwargs,
    }
    if key not in st.session_state:
        selectbox_kwargs["index"] = default_index
    return container.selectbox(
        label,
        options,
        **selectbox_kwargs,
    )


def progress_state_for_export():
    """Return only the profile and course fields entered by the user."""
    saved_state = {}
    for key, value in st.session_state.items():
        if (
            key in PROFILE_STATE_KEYS
            or key.startswith(COURSE_STATE_PREFIXES)
        ):
            saved_state[key] = value

    return {
        "format_version": 1,
        "application": "academic-progress-predictor",
        "state": saved_state,
    }


uploaded_progress = st.sidebar.file_uploader(
    "Φόρτωση αποθηκευμένης προόδου",
    type=["json"],
    key="progress_file",
    help=(
        "Επίλεξε ένα αρχείο προόδου που είχε δημιουργηθεί "
        "από την εφαρμογή."
    ),
)

if uploaded_progress is not None:
    uploaded_bytes = uploaded_progress.getvalue()
    uploaded_fingerprint = hashlib.sha256(uploaded_bytes).hexdigest()

    if st.session_state.get("loaded_progress_fingerprint") != uploaded_fingerprint:
        try:
            progress_payload = json.loads(uploaded_bytes.decode("utf-8"))
            restored_state = progress_payload.get("state")

            if not isinstance(restored_state, dict):
                raise ValueError("Μη έγκυρη δομή αρχείου προόδου.")

            for key, value in restored_state.items():
                if (
                    key in PROFILE_STATE_KEYS
                    or key.startswith(COURSE_STATE_PREFIXES)
                ):
                    st.session_state[key] = value

            st.session_state["loaded_progress_fingerprint"] = (
                uploaded_fingerprint
            )
            st.session_state["progress_loaded_message"] = True
            st.rerun()
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
            st.sidebar.error(
                f"Δεν ήταν δυνατό να φορτωθεί το αρχείο: {error}"
            )

if st.session_state.pop("progress_loaded_message", False):
    st.sidebar.success("Η αποθηκευμένη πρόοδος φορτώθηκε.")

# Χρονικά στοιχεία του snapshot.
# Η εφαρμογή υποστηρίζει ενεργές ακαδημαϊκές πορείες
# με έτος εισαγωγής από το 2016-2017 και μετά.
athens_now = pd.Timestamp.now(tz="Europe/Athens")
current_calendar_year = athens_now.year
current_month = athens_now.month
current_year_start = (
    current_calendar_year
    if current_month >= 9
    else current_calendar_year - 1
)
current_academic_year = (
    f"{current_year_start}-{current_year_start + 1}"
)

academic_years = [
    f"{year}-{year + 1}"
    for year in range(current_year_start, 2015, -1)
]

enrollment_academic_year = state_aware_selectbox(
    st.sidebar,
    "Ακαδημαϊκό έτος εισαγωγής",
    academic_years,
    key="enrollment_academic_year",
    default_index=0,
)

enrollment_semester = state_aware_selectbox(
    st.sidebar,
    "Εξάμηνο εισαγωγής",
    list(range(1, 7)),
    key="enrollment_semester",
    default_index=0,
    help=(
        "Δήλωσε το εξάμηνο στο οποίο πραγματοποιήθηκε "
        "η εισαγωγή σου."
    ),
)

# Αυτόματος προσδιορισμός της πιο πρόσφατης ολοκληρωμένης
# εξεταστικής περιόδου από τον τρέχοντα ημερολογιακό μήνα.
if 1 <= current_month <= 5:
    snapshot_period = "ΦΕΒΡ"
    snapshot_academic_year = current_academic_year
elif 6 <= current_month <= 8:
    snapshot_period = "ΙΟΥΝ"
    snapshot_academic_year = current_academic_year
else:
    snapshot_period = "ΣΕΠΤ"
    snapshot_academic_year = (
        f"{current_year_start - 1}-{current_year_start}"
    )

enrollment_year_start = int(
    enrollment_academic_year.split("-")[0]
)
if current_year_start < enrollment_year_start:
    st.sidebar.error(
        "Το τρέχον ακαδημαϊκό έτος δεν μπορεί "
        "να προηγείται της πρώτης εγγραφής."
    )

advanced_entry = int(enrollment_semester > 1)

# Επιτρεπόμενα χρονικά στοιχεία επιτυχίας μαθημάτων.
snapshot_year_start = int(
    snapshot_academic_year.split("-")[0]
)
available_success_years = [
    academic_year
    for academic_year in academic_years
    if enrollment_year_start
    <= int(academic_year.split("-")[0])
    <= snapshot_year_start
]
if not available_success_years:
    available_success_years = [enrollment_academic_year]

exam_periods = ["ΦΕΒΡ", "ΙΟΥΝ", "ΣΕΠΤ"]

semester_in_current_year = (
    1 if current_month in {9, 10, 11, 12, 1} else 2
)
entry_semester_offset = 2 * (
    (int(enrollment_semester) - 1) // 2
)
current_semester = (
    1
    + 2 * (current_year_start - enrollment_year_start)
    + semester_in_current_year
    - 1
    + entry_semester_offset
)
current_semester = max(
    int(current_semester),
    int(enrollment_semester),
    1,
)

direction = st.sidebar.selectbox(
    "Τομέας",
    [
        "Δεν έχει επιλεγεί",
        "Ενεργειακός",
        "Κατασκευαστικός",
        "Βιομηχανικής Διοίκησης"
    ],
    key="direction",
)

thesis_status_legacy_map = {
    "Δεν έχει ξεκινήσει": "Δεν ξεκίνησε",
    "Έχει ξεκινήσει": "Σε εξέλιξη",
    "Έχει ολοκληρωθεί": "Ολοκληρώθηκε",
}

stored_thesis_status = st.session_state.get(
    "thesis_status",
    "Δεν ξεκίνησε"
)
st.session_state["thesis_status"] = thesis_status_legacy_map.get(
    stored_thesis_status,
    stored_thesis_status
)

if current_semester >= 9:
    thesis_status = st.sidebar.selectbox(
        "Διπλωματική εργασία",
        [
            "Δεν ξεκίνησε",
            "Σε εξέλιξη",
            "Ολοκληρώθηκε",
        ],
        key="thesis_status",
    )
else:
    st.session_state["thesis_status"] = "Δεν ξεκίνησε"
    thesis_status = "Δεν ξεκίνησε"

has_recognized_courses = st.sidebar.checkbox(
    "Έχω αναγνωρισμένα μαθήματα",
    value=False,
    key="has_recognized_courses",
)

all_courses_by_semester = courses_by_semester.copy()

if direction != "Δεν έχει επιλεγεί":
    all_courses_by_semester.update(
        direction_courses_by_semester[direction]
    )

display_courses_by_semester = {
    semester_name: semester_courses
    for semester_name, semester_courses in all_courses_by_semester.items()
    if int(semester_name.split("ο")[0]) <= current_semester
}


# -----------------------------
# Header
# -----------------------------
st.markdown('<div class="main-title">🎓 Πρόβλεψη Ακαδημαϊκής Πορείας</div>', unsafe_allow_html=True)

st.markdown(
    """
    <div class="intro-block">
        <div class="subtitle">
            Εκτίμησε τον βαθμό πτυχίου και τη διάρκεια φοίτησης με βάση την έως τώρα πρόοδό σου.
        </div>
        <div class="custom-info">
            Συμπλήρωσε το προφίλ σου από το sidebar.
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


# -----------------------------
# Progress step
# -----------------------------
st.markdown('<div class="section-title">Τρέχουσα πορεία</div>', unsafe_allow_html=True)

semester_steps_html = ""
semester_progress_percent = ((current_semester - 1) / 9) * 100

for sem in range(1, 11):
    if sem < current_semester:
        step_class = "completed"
        circle_content = "✓"
    elif sem == current_semester:
        step_class = "active"
        circle_content = ""
    else:
        step_class = "upcoming"
        circle_content = ""

    semester_steps_html += f"""
    <div class="semester-step {step_class}">
        <div class="semester-circle">{circle_content}</div>
        <div class="semester-label">{sem}ο</div>
    </div>
    """

st.markdown(
    f"""
    <div class="semester-timeline" style="--progress-line: {semester_progress_percent}%;">
        {semester_steps_html}
    </div>
    <div class="current-semester-text">
        Τρέχον εξάμηνο: <strong>{current_semester}ο</strong>
    </div>
    """,
    unsafe_allow_html=True
)



# -----------------------------
# Main layout
# -----------------------------
left_col, right_col = st.columns([3, 1])

passed_courses = []
recognized_courses = []
passed_course_records = []
total_ects = 0
grades = []


def select_course_completion(selected_key, other_key):
    """Keep passed and recognized mutually exclusive with one click."""
    if st.session_state.get(selected_key, False):
        st.session_state[other_key] = False

with left_col:
    st.markdown(
    """
    <div class="section-title courses-title">
        Τα μαθήματά μου
        <span class="info-tooltip">
            ⓘ
            <span class="tooltip-text">
                Επίλεξε τα μαθήματα που έχεις περάσει και καταχώρισε τον βαθμό σου.
            </span>
        </span>
    </div>
    """,
    unsafe_allow_html=True
)
    
    tabs = st.tabs(list(display_courses_by_semester.keys()))


category_labels = {
    "ΥΚ": "Υποχρεωτικά κατεύθυνσης",
    "ΕΚ": "Επιλογής κατεύθυνσης",
    "ΕΤΑΚ": "Επιλογής άλλης κατεύθυνσης"
}

for tab, (semester_name, semester_courses) in zip(tabs, display_courses_by_semester.items()):
    semester_number = int(semester_name.split("ο")[0])
    default_success_period = (
        "ΦΕΒΡ" if semester_number % 2 == 1 else "ΙΟΥΝ"
    )
    default_success_year_start = (
        enrollment_year_start + (semester_number - 1) // 2
    )
    default_success_year = (
        f"{default_success_year_start}-"
        f"{default_success_year_start + 1}"
    )
    if default_success_year not in available_success_years:
        default_success_year = available_success_years[-1]
    default_success_year_index = available_success_years.index(
        default_success_year
    )

    with tab:
        st.subheader(semester_name)

        # Αν το εξάμηνο έχει κατηγορίες ΥΚ / ΕΚ / ΕΤΑΚ
        if semester_courses and "category" in semester_courses[0]:

            for category_code, category_title in category_labels.items():
                category_courses = [
                    course for course in semester_courses
                    if course.get("category") == category_code
                ]

                if category_courses:
                    with st.expander(category_title, expanded=True):

                        for course in category_courses:
                            (
                                col1,
                                col2,
                                col3,
                                col4,
                                col5,
                            ) = st.columns(
                                [1.65, 2.65, 1.45, 1.65, 1.15],
                                vertical_alignment="top"
                            )

                            with col1:
                                st.markdown(
                                    '<div class="course-status-offset"></div>',
                                    unsafe_allow_html=True
                                )
                                graded_key = (
                                    f"graded_{semester_name}_{category_code}_"
                                    f"{course['code']}"
                                )
                                recognized_key = (
                                    f"recognized_{semester_name}_{category_code}_"
                                    f"{course['code']}"
                                )

                                passed = st.checkbox(
                                    "Περάστηκε",
                                    key=graded_key,
                                    on_change=select_course_completion,
                                    args=(graded_key, recognized_key),
                                )

                                if has_recognized_courses:
                                    recognized = st.checkbox(
                                        "Αναγνωρίστηκε",
                                        key=recognized_key,
                                        on_change=select_course_completion,
                                        args=(recognized_key, graded_key),
                                    )
                                else:
                                    recognized = False

                                completed = passed or recognized

                        

                            with col2:
                                st.markdown(
                                    '<div class="course-info-offset"></div>',
                                    unsafe_allow_html=True
                                )
                                st.markdown(
                                        f'<div class="course-title">{course["title"]}</div>',
                                        unsafe_allow_html=True
                                    )
                                st.markdown(
                                        f'<div class="course-meta">'
                                        f'Κωδικός: {course["code"]} | ECTS: {course["ects"]}'
                                        f'</div>',
                                        unsafe_allow_html=True
                                    )

                            with col3:
                                grade = st.number_input(
                                    "Βαθμός",
                                    min_value=5.0,
                                    max_value=10.0,
                                    step=0.5,
                                    key=f"grade_{semester_name}_{category_code}_{course['code']}",
                                    disabled=not passed
                                )

                            with col4:
                                success_year_key = (
                                    f"success_year_{semester_name}_"
                                    f"{category_code}_{course['code']}"
                                )
                                success_year = state_aware_selectbox(
                                    st,
                                    (
                                        "Έτος αναγνώρισης"
                                        if recognized
                                        else "Έτος επιτυχίας"
                                    ),
                                    available_success_years,
                                    default_index=default_success_year_index,
                                    key=success_year_key,
                                    disabled=not completed
                                )

                            # Η εξεταστική επιτυχίας δηλώνεται ανεξάρτητα
                            # από την τρέχουσα περίοδο του sidebar.
                            allowed_periods = exam_periods

                            with col5:
                                success_period_key = (
                                    f"success_period_{semester_name}_"
                                    f"{category_code}_{course['code']}"
                                )
                                success_period = state_aware_selectbox(
                                    st,
                                    "Εξεταστική",
                                    allowed_periods,
                                    default_index=allowed_periods.index(
                                        default_success_period
                                    ),
                                    key=success_period_key,
                                    disabled=not completed
                                )

                            if completed:
                                total_ects += course["ects"]
                                if passed:
                                    passed_courses.append(course)
                                    grades.append(grade)
                                elif recognized:
                                    recognized_courses.append(course)

                                passed_course_records.append({
                                    "code": course["code"],
                                    "title": course["title"],
                                    "ects": course["ects"],
                                    "curriculum_semester": semester_name,
                                    "curriculum_category": category_code,
                                    "completion_type": (
                                        "graded" if passed else "recognized"
                                    ),
                                    "grade": grade if passed else None,
                                    "success_academic_year": success_year,
                                    "success_period": success_period
                                })

        # Αν είναι απλό εξάμηνο κορμού, χωρίς κατηγορίες
        else:

            for course in semester_courses:
                (
                    col1,
                    col2,
                    col3,
                    col4,
                    col5,
                ) = st.columns(
                    [1.65, 2.65, 1.45, 1.65, 1.15],
                    vertical_alignment="top"
                )

                with col1:
                    st.markdown(
                        '<div class="course-status-offset"></div>',
                        unsafe_allow_html=True
                    )
                    graded_key = (
                        f"graded_{semester_name}_{course['code']}"
                    )
                    recognized_key = (
                        f"recognized_{semester_name}_{course['code']}"
                    )

                    passed = st.checkbox(
                        "Περάστηκε",
                        key=graded_key,
                        on_change=select_course_completion,
                        args=(graded_key, recognized_key),
                    )

                    if has_recognized_courses:
                        recognized = st.checkbox(
                            "Αναγνωρίστηκε",
                            key=recognized_key,
                            on_change=select_course_completion,
                            args=(recognized_key, graded_key),
                        )
                    else:
                        recognized = False

                    completed = passed or recognized

                
                with col2:
                        st.markdown(
                            '<div class="course-info-offset"></div>',
                            unsafe_allow_html=True
                        )
                        st.markdown(
                            f'<div class="course-title">{course["title"]}</div>',
                                unsafe_allow_html=True
                            )
                        st.markdown(
                            f'<div class="course-meta">'
                            f'Κωδικός: {course["code"]} | ECTS: {course["ects"]}'
                            f'</div>',
                                unsafe_allow_html=True
                            )

                with col3:
                    grade = st.number_input(
                        "Βαθμός",
                        min_value=5.0,
                        max_value=10.0,
                        step=0.5,
                        key=f"grade_{semester_name}_{course['code']}",
                        disabled=not passed
                    )

                with col4:
                    success_year_key = (
                        f"success_year_{semester_name}_{course['code']}"
                    )
                    success_year = state_aware_selectbox(
                        st,
                        (
                            "Έτος αναγνώρισης"
                            if recognized
                            else "Έτος επιτυχίας"
                        ),
                        available_success_years,
                        default_index=default_success_year_index,
                        key=success_year_key,
                        disabled=not completed
                    )

                # Η εξεταστική επιτυχίας δηλώνεται ανεξάρτητα
                # από την τρέχουσα περίοδο του sidebar.
                allowed_periods = exam_periods

                with col5:
                    success_period_key = (
                        f"success_period_{semester_name}_{course['code']}"
                    )
                    success_period = state_aware_selectbox(
                        st,
                        "Εξεταστική",
                        allowed_periods,
                        default_index=allowed_periods.index(
                            default_success_period
                        ),
                        key=success_period_key,
                        disabled=not completed
                    )

                if completed:
                    total_ects += course["ects"]
                    if passed:
                        passed_courses.append(course)
                        grades.append(grade)
                    elif recognized:
                        recognized_courses.append(course)

                    passed_course_records.append({
                        "code": course["code"],
                        "title": course["title"],
                        "ects": course["ects"],
                        "curriculum_semester": semester_name,
                        "curriculum_category": "CORE",
                        "completion_type": (
                            "graded" if passed else "recognized"
                        ),
                        "grade": grade if passed else None,
                        "success_academic_year": success_year,
                        "success_period": success_period
                    })

progress_json = json.dumps(
    progress_state_for_export(),
    ensure_ascii=False,
    indent=2,
).encode("utf-8")

st.sidebar.download_button(
    "Αποθήκευση προόδου",
    data=progress_json,
    file_name="akadimaiki_proodos.json",
    mime="application/json",
    use_container_width=True,
    help=(
        "Κατεβάζει τις επιλογές μαθημάτων, τους βαθμούς και "
        "τα στοιχεία του προφίλ σου για μελλοντική φόρτωση."
    ),
)

with right_col:
    st.markdown('<div class="section-title courses-title">Σύνοψη</div>', unsafe_allow_html=True)

    passed_count = len(passed_courses) + len(recognized_courses)
    avg_grade = sum(grades) / len(grades) if grades else 0

    thesis_ects_completed = THESIS_ECTS if thesis_status == "Ολοκληρώθηκε" else 0
    total_degree_ects = total_ects + thesis_ects_completed

    progress_courses = passed_count / TOTAL_COURSES
    progress_course_ects = total_ects / COURSE_ECTS
    progress_degree_ects = total_degree_ects / TOTAL_DEGREE_ECTS

    courses_progress_percent = progress_courses * 100

    st.markdown(
    f"""
    <div class="donut-card">
        <div class="donut-progress" style="--progress: {courses_progress_percent};">
            <div class="donut-inner">
                <div class="donut-value">{passed_count}/{TOTAL_COURSES}</div>
                <div class="donut-label">Περασμένα<br>μαθήματα</div>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

    st.markdown(
        f"""
        <div class="card ects-summary-card">
            <div class="ects-card-title">ECTS</div>
            <div class="ects-row">
            <span class="ects-label">Μαθημάτων</span>
            <span class="ects-value">{total_ects}/{COURSE_ECTS}</span>
            </div>
            <div class="ects-divider"></div>
            <div class="ects-row">
            <span class="ects-label">Πτυχίου</span>
            <span class="ects-value">{total_degree_ects}/{TOTAL_DEGREE_ECTS}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    

    degree_progress_percent = progress_degree_ects * 100

    st.markdown(
    f"""
    <div class="progress-card">
        <div class="progress-card-header">
            <div class="progress-card-title">Πρόοδος πτυχίου</div>
            <div class="progress-card-value">{degree_progress_percent:.1f}%</div>
        </div>
        <div class="custom-progress-track">
            <div class="custom-progress-fill" style="--progress-width: {degree_progress_percent}%;"></div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)
    st.markdown(
    f"""
    <div class="card">
        <div class="metric-title">Μέσος όρος</div>
        <div class="metric-value">{avg_grade:.2f}</div>
    </div>
    """,
    unsafe_allow_html=True
)

    predict_clicked = st.button(
        "Πρόβλεψη",
        type="primary",
        use_container_width=True,
        disabled=(passed_count == 0)
    )

    if predict_clicked:
        try:
            (
                grade_model_bundle,
                duration_model_bundle,
            ) = load_model_bundles()

            model_input_df = build_model_input(
                course_records=passed_course_records,
                enrollment_academic_year=(
                    enrollment_academic_year
                ),
                enrollment_semester=enrollment_semester,
                current_academic_year=snapshot_academic_year,
                snapshot_period=snapshot_period,
            )

            duration_features = duration_model_bundle[
                "features"
            ]
            grade_features = grade_model_bundle[
                "features"
            ]

            missing_duration_features = [
                feature
                for feature in duration_features
                if feature not in model_input_df.columns
            ]
            missing_grade_features = [
                feature
                for feature in grade_features
                if feature not in model_input_df.columns
            ]

            if (
                missing_duration_features
                or missing_grade_features
            ):
                raise RuntimeError(
                    "Δεν δημιουργήθηκαν όλα τα χαρακτηριστικά "
                    "που απαιτούν τα μοντέλα."
                )

            raw_predicted_duration = float(
                duration_model_bundle["model"].predict(
                    model_input_df[duration_features]
                )[0]
            )
            estimated_degree = float(
                grade_model_bundle["model"].predict(
                    model_input_df[grade_features]
                )[0]
            )

            predicted_duration = apply_duration_postprocessing(
                raw_predicted_duration,
                model_input_df,
                duration_model_bundle,
            )

            predicted_duration_semesters = round(
                predicted_duration * 2
            )
            duration_display = (
                f'<span class="duration-value-line">'
                f"{predicted_duration:.2f} έτη</span>"
                f'<span class="duration-value-line">'
                f"(~{predicted_duration_semesters} εξάμηνα)</span>"
            )


        except FileNotFoundError:
            st.error(
                "Δεν βρέθηκε το αρχείο των εκπαιδευμένων "
                "μοντέλων."
            )
            st.stop()
        except (ValueError, RuntimeError) as error:
            st.error(str(error))
            st.stop()

        st.markdown(
            f"""
            <div id="prediction-results" class="prediction-result-card">
                <div class="prediction-result-label">
                    Εκτιμώμενος βαθμός πτυχίου
                </div>
                <div class="prediction-result-value">
                    {estimated_degree:.2f}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            f"""
            <div id="prediction-results-end" class="prediction-result-card">
                <div class="prediction-result-label">
                    Εκτιμώμενη συνολική διάρκεια φοίτησης
                </div>
                <div class="prediction-result-value">
                    {duration_display}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            """
            <div id="prediction-results-note" class="prediction-disclaimer">
                Οι προβλέψεις αποτελούν στατιστικές εκτιμήσεις
                και δεν εγγυώνται το τελικό αποτέλεσμα.
            </div>
            <div id="prediction-scroll-end" style="height: 18px;"></div>
            """,
            unsafe_allow_html=True
        )

        components.html(
            """
            <script>
                setTimeout(function () {
                    const target =
                        window.parent.document.getElementById(
                            "prediction-scroll-end"
                        );

                    if (target) {
                        target.scrollIntoView({
                            behavior: "smooth",
                            block: "end"
                        });
                    }
                }, 200);
            </script>
            """,
            height=0
        )
