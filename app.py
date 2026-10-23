
import csv
import json
import re
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from src.analysis import analyze_video


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="ClassLens | 8x8 Class Quality",
    page_icon="🎓",
    layout="wide",
)

st.title("🎓 ClassLens")
st.caption("AI-assisted analysis of online class recordings")

st.info(
    "ClassLens analyses speech, audio activity and visual changes, "
    "and provides a structured teaching-quality assessment. "
    "Automated signals alone cannot objectively determine teaching quality."
)


# ============================================================
# ASSESSMENT RUBRIC
# ============================================================

RUBRIC = {
    "Rapport and introduction": {
        "description": (
            "Welcomes students, builds rapport and gives new joiners "
            "an opportunity to introduce themselves."
        ),
        "evidence": (
            "Review the opening of the lesson for a welcoming introduction, "
            "rapport-building and opportunities for student introductions."
        ),
    },
    "Student interaction": {
        "description": (
            "Addresses students individually and asks questions "
            "during the lesson."
        ),
        "evidence": (
            "Look for individual student involvement and questions used "
            "to engage learners or check understanding."
        ),
    },
    "Communication": {
        "description": (
            "Communicates clearly, energetically and pleasantly."
        ),
        "evidence": (
            "Review the recording for clarity, energy and pleasant "
            "interaction. Transcript text alone cannot assess tone."
        ),
    },
    "Visual presentation": {
        "description": (
            "Uses the 8x8 background appropriately, with the teacher's "
            "face visible and adequate lighting."
        ),
        "evidence": (
            "Inspect the video to check background use, teacher visibility "
            "and lighting. Pixel variation alone cannot establish these."
        ),
    },
    "Learning support and structure": {
        "description": (
            "Uses an appropriate lesson introduction and conclusion, "
            "and provides relevant resources or homework where appropriate."
        ),
        "evidence": (
            "Review lesson structure and whether relevant videos, resources "
            "or homework are provided when appropriate."
        ),
    },
}

GRADE_OPTIONS = ["Poor", "Average", "Good", "Excellent"]

SCORE_TO_GRADE = {
    1: "Poor",
    2: "Average",
    3: "Good",
    4: "Excellent",
}


# ============================================================
# REFERENCE FEEDBACK
# ============================================================

def find_reference_feedback(filename):
    """
    Match the recording to feedback using class ID and date.

    CSV location:
        data/reference_feedback.csv

    If no matching date exists, all feedback for the class ID
    is returned rather than silently discarding useful entries.
    """

    csv_path = (
        Path(__file__).resolve().parent
        / "data"
        / "reference_feedback.csv"
    )

    if not csv_path.exists():
        return []

    class_match = re.search(
        r"B\d{3}",
        str(filename),
        re.IGNORECASE,
    )

    if not class_match:
        return []

    class_id = class_match.group(0).upper()

    date_match = re.search(
        r"(\d{4})-(\d{2})-(\d{2})",
        str(filename),
    )

    try:
        with csv_path.open(
            "r",
            encoding="utf-8-sig",
            newline="",
        ) as file:
            rows = list(csv.DictReader(file))

    except (OSError, csv.Error) as exc:
        st.warning(f"Could not read reference feedback: {exc}")
        return []

    matches = [
        row
        for row in rows
        if row.get("class_id", "").strip().upper() == class_id
    ]

    if date_match:
        wanted_date = "-".join(date_match.groups())

        exact_matches = [
            row
            for row in matches
            if row.get("date", "").strip() == wanted_date
        ]

        if exact_matches:
            matches = exact_matches

    return matches


# ============================================================
# MAP REFERENCE NOTES TO RUBRIC CRITERIA
# ============================================================

def identify_relevant_criteria(feedback):
    """
    Highlight rubric areas mentioned by reference feedback.

    This is keyword-based matching, not semantic AI inference.
    It identifies areas to review; it does not assign scores.
    """

    text = feedback.lower()
    relevant = []

    keyword_groups = {
        "Rapport and introduction": [
            "rapport",
            "introduction",
            "intro",
            "new joiners",
            "self-introduction",
            "conclusion",
            "welcoming",
        ],
        "Student interaction": [
            "interaction",
            "individually",
            "one-to-one",
            "questions",
            "consider everyone",
        ],
        "Communication": [
            "communication",
            "pleasant",
            "energetic",
            "happy interaction",
            "clarity",
        ],
        "Visual presentation": [
            "background",
            "face",
            "camera",
            "light",
            "lighting",
            "visible",
        ],
        "Learning support and structure": [
            "homework",
            "hws",
            "videos",
            "resources",
            "lesson",
            "conclusion",
        ],
    }

    for criterion, keywords in keyword_groups.items():
        if any(keyword in text for keyword in keywords):
            relevant.append(criterion)

    return relevant


# ============================================================
# RUBRIC SCORING
# ============================================================

def suggest_grade(scores):
    """Calculate a proposed grade from the reviewer-entered ratings."""

    if not scores:
        return None

    mean_score = sum(scores.values()) / len(scores)

    if mean_score >= 3.5:
        return "Excellent"
    elif mean_score >= 2.5:
        return "Good"
    elif mean_score >= 1.5:
        return "Average"
    else:
        return "Poor"


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.header("Analysis settings")

    model_size = st.selectbox(
        "Speech recognition model",
        ["base", "small", "tiny"],
        index=0,
        help=(
            "Larger models may improve transcription but use "
            "more processing resources."
        ),
    )

    sample_every = st.slider(
        "Video frame sampling interval (seconds)",
        min_value=5,
        max_value=30,
        value=10,
    )

    st.caption(
        "The first run may download a speech-recognition model."
    )


# ============================================================
# VIDEO UPLOAD
# ============================================================

uploaded = st.file_uploader(
    "Upload an online class recording",
    type=["mp4", "mov", "mkv", "avi", "webm"],
)

if uploaded:
    st.video(uploaded)

    if st.button("Analyse class", type="primary"):

        suffix = Path(uploaded.name).suffix or ".mp4"

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix,
        ) as tmp:
            tmp.write(uploaded.getbuffer())
            video_path = tmp.name

        try:
            with st.spinner(
                "Analysing audio, video and speech. "
                "This may take a few minutes..."
            ):
                result = analyze_video(
                    video_path,
                    model_size=model_size,
                    frame_interval=sample_every,
                )

            st.session_state["analysis_result"] = result
            st.session_state["uploaded_name"] = uploaded.name

            # Reset reviewer choices when a new video is analysed.
            st.session_state.pop("rubric_scores", None)
            st.session_state.pop("rubric_notes", None)
            st.session_state.pop("reviewer_grade", None)
            st.session_state.pop("quality_issue", None)

        except Exception as exc:
            st.error(f"Analysis failed: {exc}")
            st.exception(exc)

        finally:
            try:
                Path(video_path).unlink(missing_ok=True)
            except OSError:
                pass


# ============================================================
# RESULTS
# ============================================================

if "analysis_result" in st.session_state:

    result = st.session_state["analysis_result"]
    features = result["features"]

    recording_name = st.session_state.get(
        "uploaded_name",
        "uploaded video",
    )

    st.divider()
    st.subheader("Class recording analysis")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Recording duration",
        f"{features.get('duration_seconds', 0) / 60:.1f} min",
    )

    col2.metric(
        "Transcript words",
        f"{features.get('word_count', 0):,}",
    )

    col3.metric(
        "Transcript segments",
        f"{features.get('segment_count', 0):,}",
    )


    # --------------------------------------------------------
    # OBSERVATIONS
    # --------------------------------------------------------

    st.subheader("Evidence and observations")

    for item in result.get("observations", []):

        with st.container(border=True):

            st.markdown(f"**{item['title']}**")
            st.write(item["detail"])

            if item.get("timestamp") is not None:
                seconds = max(0, int(item["timestamp"]))

                st.caption(
                    "Approximate timestamp: "
                    f"{seconds // 60:02d}:{seconds % 60:02d}"
                )


    # --------------------------------------------------------
    # MEASURED SIGNALS
    # --------------------------------------------------------

    st.subheader("Measured signals")

    descriptions = {
        "duration_seconds": "Duration (seconds)",
        "speech_ratio": "Estimated audio-active ratio",
        "mean_rms": "Mean audio energy",
        "rms_variability": "Audio energy variability",
        "silence_ratio": "Estimated low-energy ratio",
        "clip_ratio": "Potential clipped-sample ratio",
        "word_count": "Transcript word count",
        "words_per_minute": "Transcript words per minute",
        "question_count": "Recognised question marks",
        "exclamation_count": "Recognised exclamation marks",
        "visual_change": "Sampled-frame pixel variation",
        "frame_count": "Frames sampled",
        "mean_frame_brightness": "Mean frame brightness",
        "segment_count": "Transcript segments",
    }

    rows = [
        {
            "Signal": descriptions.get(key, key),
            "Value": (
                round(float(value), 4)
                if isinstance(value, (int, float))
                else str(value)
            ),
        }
        for key, value in features.items()
        if key in descriptions
    ]

    st.dataframe(
        pd.DataFrame(rows),
        hide_index=True,
        use_container_width=True,
    )


    # --------------------------------------------------------
    # TRANSCRIPT
    # --------------------------------------------------------

    with st.expander("View transcript"):
        st.text(
            result.get("transcript")
            or "No transcript was produced."
        )


    # --------------------------------------------------------
    # REFERENCE FEEDBACK
    # --------------------------------------------------------

    st.divider()
    st.subheader("Matching 8x8 reference feedback")

    reference_matches = find_reference_feedback(recording_name)

    relevant_criteria = set()

    if reference_matches:

        for reference in reference_matches:

            category = reference.get(
                "category",
                "Unlabelled",
            ).strip()

            feedback = reference.get(
                "feedback",
                "",
            ).strip()

            relevant_criteria.update(
                identify_relevant_criteria(feedback)
            )

            with st.container(border=True):

                st.markdown(
                    f"**{reference.get('class_id', 'Unknown')} "
                    f"— {category}**"
                )

                st.caption(
                    "Reference date: "
                    f"{reference.get('date', 'Not specified')}"
                )

                st.write(feedback)

                if category.lower() == "quality issue":
                    st.warning(
                        "This reference is flagged for a separate "
                        "quality review, not a fifth grade."
                    )

                elif category.lower() == "weak":
                    st.caption(
                        "The source category is Weak. Poor is the "
                        "proposed equivalent in this app."
                    )

        if relevant_criteria:
            st.markdown("**Rubric areas mentioned in the reference notes**")

            for criterion in RUBRIC:
                if criterion in relevant_criteria:
                    st.markdown(f"- {criterion}")

        st.caption(
            "Reference feedback is historical context. It is not proof "
            "that the uploaded recording has the same issues."
        )

    else:
        st.info(
            "No matching reference feedback was found. "
            "Check the class ID in the filename and the reference CSV."
        )


    # --------------------------------------------------------
    # REVIEWER RUBRIC
    # --------------------------------------------------------

    st.divider()
    st.header("Teaching-quality assessment")

    st.write(
        "Rate each criterion after reviewing the recording. "
        "Reference notes may help focus your review but do not "
        "automatically change the ratings."
    )

    if "rubric_scores" not in st.session_state:
        st.session_state["rubric_scores"] = {}

    if "rubric_notes" not in st.session_state:
        st.session_state["rubric_notes"] = {}


    # --------------------------------------------------------
    # FIVE CRITERIA
    # --------------------------------------------------------

    for criterion, details in RUBRIC.items():

        with st.container(border=True):

            st.markdown(f"**{criterion}**")
            st.write(details["description"])
            st.caption(details["evidence"])

            if criterion in relevant_criteria:
                st.info(
                    "This area is mentioned in the matching reference "
                    "feedback. Review the recording before assigning a score."
                )

            score = st.select_slider(
                "Reviewer rating",
                options=[1, 2, 3, 4],
                value=st.session_state["rubric_scores"].get(
                    criterion,
                    2,
                ),
                format_func=lambda value: SCORE_TO_GRADE[value],
                key=f"score_{criterion}",
            )

            st.session_state["rubric_scores"][criterion] = score

            note = st.text_area(
                "Evidence or reviewer notes",
                value=st.session_state["rubric_notes"].get(
                    criterion,
                    "",
                ),
                placeholder=(
                    "Describe what you observed and add a timestamp "
                    "if possible."
                ),
                key=f"note_{criterion}",
            )

            st.session_state["rubric_notes"][criterion] = note


    # --------------------------------------------------------
    # SUGGESTED GRADE
    # --------------------------------------------------------

    scores = st.session_state["rubric_scores"]

    suggested_grade = (
        suggest_grade(scores)
        if len(scores) == len(RUBRIC)
        else None
    )

    st.subheader("Rubric result")

    if suggested_grade:

        mean_score = sum(scores.values()) / len(scores)

        col1, col2 = st.columns(2)

        col1.metric(
            "Mean rubric score",
            f"{mean_score:.2f} / 4",
        )

        col2.metric(
            "Suggested grade",
            suggested_grade,
        )

        st.caption(
            "Proposed thresholds: Poor below 1.5; Average from 1.5; "
            "Good from 2.5; Excellent from 3.5. These are project rules, "
            "not official 8x8 thresholds."
        )

    else:
        st.warning(
            "Complete all five criteria to calculate the suggested grade."
        )


    # --------------------------------------------------------
    # FINAL GRADE AND QUALITY FLAG
    # --------------------------------------------------------

    reviewer_grade = st.selectbox(
        "Final reviewer grade",
        options=["Use rubric suggestion"] + GRADE_OPTIONS,
        index=0,
        key="reviewer_grade",
    )

    quality_issue = st.checkbox(
        "Flag this recording for a separate quality review",
        value=st.session_state.get("quality_issue", False),
        key="quality_issue",
        help=(
            "Use this for possible background, visibility or other "
            "quality concerns. It is not a fifth grade."
        ),
    )

    if reviewer_grade == "Use rubric suggestion":
        final_grade = suggested_grade
    else:
        final_grade = reviewer_grade

    if final_grade:
        st.markdown(f"### Final grade: {final_grade}")

    if quality_issue:
        st.warning(
            "Quality review flagged. Investigate the concern separately "
            "from the overall teaching-quality grade."
        )


    # --------------------------------------------------------
    # DOWNLOADABLE REPORT
    # --------------------------------------------------------

    report = {
        "recording": recording_name,
        "assessment_status": (
            "Reviewer rubric completed"
            if suggested_grade
            else "Reviewer assessment incomplete"
        ),
        "suggested_grade": suggested_grade,
        "final_grade": final_grade,
        "quality_review_flag": quality_issue,
        "reference_feedback": reference_matches,
        "rubric": {
            criterion: {
                "score": scores.get(criterion),
                "rating": SCORE_TO_GRADE.get(scores.get(criterion)),
                "reference_feedback_relevance": (
                    criterion in relevant_criteria
                ),
                "reviewer_notes": st.session_state[
                    "rubric_notes"
                ].get(criterion, ""),
            }
            for criterion in RUBRIC
        },
        "features": features,
        "observations": result.get("observations", []),
        "transcript": result.get("transcript", ""),
    }

    st.download_button(
        "Download assessment report (JSON)",
        data=json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        ),
        file_name="classlens_assessment.json",
        mime="application/json",
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Privacy: analysis runs locally by default. The speech-recognition "
    "model may be downloaded on first use. No external LLM API is used."
)
