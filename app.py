import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Student Outcome Predictor",
    page_icon="🎓",
    layout="centered",
)

# ── Load model artifacts ─────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    model    = joblib.load("best_model.pkl")
    scaler   = joblib.load("scaler.pkl")
    features = joblib.load("features.pkl")
    return model, scaler, features

model, scaler, feature_names = load_artifacts()

LABEL_MAP = {0: "Dropout", 1: "Graduate"}

COURSE_NAMES = {
    1:  "Biofuel Production Technologies",
    2:  "Animation and Multimedia Design",
    3:  "Social Service (evening)",
    4:  "Agronomy",
    5:  "Communication Design",
    6:  "Veterinary Nursing",
    7:  "Informatics Engineering",
    8:  "Equinculture",
    9:  "Management",
    10: "Social Service",
    11: "Tourism",
    12: "Nursing",
    13: "Oral Hygiene",
    14: "Advertising and Marketing",
    15: "Journalism and Communication",
    16: "Basic Education",
    17: "Management (evening)",
}

DIFFICULTY_MAP = {"Easy": 0.15, "Medium": 0.35, "Hard": 0.55}

# ── UI ───────────────────────────────────────────────────────────────────────
st.title("🎓 Student Outcome Predictor")
st.caption("Predict whether a student will **Graduate** or **Drop out**.")
st.divider()

# ── Academic info ────────────────────────────────────────────────────────────
st.subheader("📚 Academic Information")

course_name = st.selectbox(
    "Course",
    list(COURSE_NAMES.values()),
    index=11,
    help="The academic program the student is enrolled in."
)
course_id = [k for k, v in COURSE_NAMES.items() if v == course_name][0]

avg_grade = st.slider(
    "Average grade (0 – 20)", 0.0, 20.0, 12.0, 0.5,
    help="Student's average grade across both semesters. The single most important predictor — grades above 12 strongly indicate graduation."
)

difficulty = st.radio(
    "Course difficulty", ["Easy", "Medium", "Hard"],
    index=1, horizontal=True,
    help="How demanding the course is. Harder courses have higher fail rates which increases dropout risk."
)

credits = st.slider(
    "Credited units (transfer credits)", 0, 20, 0,
    help="Units transferred from a previous institution. More credits suggest prior academic experience, a mild positive signal."
)

st.divider()

# ── Personal info ────────────────────────────────────────────────────────────
st.subheader("👤 Personal Information")

col1, col2 = st.columns(2)

age = col1.number_input(
    "Age at enrollment", min_value=17, max_value=70, value=20, step=1,
    help="Age when the student first enrolled. Older students (25+) have significantly higher dropout rates due to work and family responsibilities competing with studies."
)

attendance = col2.radio(
    "Attendance time", ["Daytime", "Evening"],
    index=0,
    help="When the student attends. Evening students drop out more because they typically work during the day, leaving less time and energy for studies."
)

st.divider()

# ── Financial & support ──────────────────────────────────────────────────────
st.subheader("💰 Financial & Support")

col3, col4, col5 = st.columns(3)
tuition_ok  = col3.toggle("Tuition up to date",   value=True,  help="Whether tuition is paid. Unpaid tuition is one of the strongest dropout signals in the dataset.")
scholarship = col4.toggle("Scholarship holder",   value=False, help="Scholarship students tend to graduate more — they have financial security and higher motivation.")
debtor      = col5.toggle("Has outstanding debt", value=False, help="Owing money to the institution raises dropout risk, especially when combined with unpaid tuition.")

st.divider()

# ── Predict ──────────────────────────────────────────────────────────────────
if st.button("🔮  Predict Outcome", use_container_width=True, type="primary"):

    # Engineered features
    fail_rate      = DIFFICULTY_MAP[difficulty]
    success_rate   = avg_grade / 20.0
    total_enrolled = 12
    total_approved = round(success_rate * total_enrolled)
    total_evals    = round(total_enrolled * (1 - fail_rate * 0.5))
    total_failed   = max(total_evals - total_approved, 0)
    fail_rate_feat = total_failed / (total_evals + 1e-5)
    financial_risk = int(debtor and not tuition_ok)
    attendance_bin = 1 if attendance == "Evening" else 0

    input_data = {
        "Marital status":             1,
        "Application mode":           1,
        "Application order":          1,
        "Course":                     course_id,
        "Daytime/evening attendance": attendance_bin,
        "Previous qualification":     1,
        "Displaced":                  0,
        "Debtor":                     int(debtor),
        "Tuition fees up to date":    int(tuition_ok),
        "Gender":                     1,
        "Scholarship holder":         int(scholarship),
        "Age at enrollment":          int(age),
        "Unemployment rate":          10.8,
        "Inflation rate":             1.4,
        "GDP":                        1.74,
        "Total_approved":             total_approved,
        "Total_enrolled":             total_enrolled,
        "Success_rate":               success_rate,
        "Avg_grade":                  avg_grade,
        "Total_evaluations":          total_evals,
        "Total_failed":               total_failed,
        "Fail_rate":                  fail_rate_feat,
        "Grade_improvement":          0.0,
        "Approval_improvement":       0,
        "Financial_risk":             financial_risk,
        "Total_credited":             credits,
    }

    input_df     = pd.DataFrame([input_data])[feature_names]
    input_scaled = scaler.transform(input_df.values)

    prediction    = model.predict(input_scaled)[0]
    probabilities = model.predict_proba(input_scaled)[0]
    label         = LABEL_MAP[prediction]

    # ── Result ───────────────────────────────────────────────────────────────
    st.divider()

    if label == "Graduate":
        st.success("## ✅  Graduate")
    else:
        st.error("## ⚠️  Dropout")

    st.write("#### Confidence breakdown")
    for i in np.argsort(probabilities)[::-1]:
        lbl  = LABEL_MAP[i]
        prob = round(float(probabilities[i]) * 100, 1)
        st.progress(int(prob), text=f"{lbl}:  {prob}%")

    st.write("#### Key factors")
    c1, c2, c3 = st.columns(3)
    c1.metric("Average Grade",    f"{avg_grade:.1f} / 20")
    c2.metric("Success Rate",     f"{round(success_rate * 100)}%")
    c3.metric("Fail Rate",        f"{round(fail_rate_feat * 100)}%")

    c4, c5, c6 = st.columns(3)
    c4.metric("Age at Enrollment", age)
    c5.metric("Attendance",        attendance)
    c6.metric("Financial Risk",    "🔴 High" if financial_risk else "🟢 Low")

st.divider()
st.caption("Model: SVM (RBF kernel) · Binary classifier: Dropout vs Graduate · Trained on student academic dataset")
