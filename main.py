import pandas as pd
import pickle
import streamlit as st

# --- 1. Load Models ---
@st.cache_resource
def load_models():
    with open("classification_model.pkl", "rb") as f:
        class_artifacts = pickle.load(f)
    with open("Regression_model.pkl", "rb") as f:
        reg_artifacts = pickle.load(f)
    return class_artifacts, reg_artifacts

class_artifacts, reg_artifacts = load_models()

# --- 2. Preprocessing Function ---
def preprocess_input(input_dict, model_artifacts):
    df = pd.DataFrame([input_dict])
    
    if "Post_Semester_GPA" in df.columns and "Pre_Semester_GPA" in df.columns:
        df["GPA_Delta"] = df["Post_Semester_GPA"] - df["Pre_Semester_GPA"]
    
    label_map = model_artifacts["label_encoder"]
    scaler = model_artifacts["scaler"]
    scaled_cols = model_artifacts["scaled_cols"]
    feature_order = model_artifacts["feature_order"]
    bool_cols = model_artifacts["bool_cols"]
    
    for col, mapping in label_map.items():
        if col in df.columns:
            df[col] = df[col].map(mapping)
            
    cols_to_dummy = [col for col in ["Major_Category", "Primary_Use_Case"] if col in df.columns]
    if cols_to_dummy:
        df = pd.get_dummies(df, columns=cols_to_dummy)
        
    for col in feature_order:
        if col not in df.columns:
            df[col] = False if col in bool_cols else 0
            
    df = df[feature_order]
    df[scaled_cols] = scaler.transform(df[scaled_cols])
    
    return df

# --- 3. Streamlit UI Design ---
st.title("AI Impact on Students Predictor 🎓")

prediction_task = st.radio(
    "What would you like to predict?",
    ["Predict Burnout Risk", "Predict Post-Semester GPA"],
    horizontal=True
)

st.divider()

col1, col2 = st.columns(2)

with col1:
    st.subheader("Academic & Study Data")
    pre_gpa = st.number_input("Pre-Semester GPA", min_value=0.0, max_value=4.0, value=3.5, step=0.1)
    
    if prediction_task == "Predict Burnout Risk":
        post_gpa = st.number_input("Post-Semester GPA", min_value=0.0, max_value=4.0, value=3.5, step=0.1)
        burnout_input = "Low" 
    else:
        post_gpa = 0.0 
        burnout_input = st.selectbox("Burnout Risk Level", ["Low", "Medium", "High"])
        
    study_hours = st.number_input("Traditional Study Hours (Weekly)", min_value=0, value=15)
    anxiety = st.slider("Anxiety Level During Exams", min_value=1, max_value=10, value=5)
    
    # Corrected Major Categories based on your index
    major = st.selectbox("Major Category", ["Arts", "Business", "Humanities", "Medical", "STEM"])
    
    year = st.selectbox("Year of Study", ["Freshman", "Sophomore", "Junior", "Senior", "Graduate"])
    skill_retention = st.number_input("Skill Retention Score", min_value=0.0, max_value=100.0, value=85.0, step=1.0)

with col2:
    st.subheader("AI Usage Data")
    genai_hours = st.number_input("Weekly GenAI Hours", min_value=0, value=5)
    ai_dependency = st.slider("Perceived AI Dependency", min_value=1, max_value=10, value=3)
    prompt_skill = st.selectbox("Prompt Engineering Skill", ["Beginner", "Intermediate", "Advanced"])
    policy = st.selectbox("Institutional Policy", ["Strict_Ban", "Allowed_With_Citation", "Actively_Encouraged"])
    
    # Corrected Use Cases based on your index
    use_case = st.selectbox(
        "Primary Use Case", 
        [
            "Copywriting/Drafting", 
            "Debugging/Troubleshooting", 
            "Direct_Answer_Generation", 
            "Ideation", 
            "Summarizing_Reading"
        ]
    )
    
    tool_diversity = st.number_input("Tool Diversity (Count of AI tools used)", min_value=1, value=2, step=1)
    paid_sub = st.checkbox("Paid AI Subscription?")

# Compile dictionary with exact column names matched to your index
user_input = {
    "Pre_Semester_GPA": pre_gpa,
    "Post_Semester_GPA": post_gpa,
    "Burnout_Risk_Level": burnout_input,
    "Weekly_GenAI_Hours": genai_hours,
    "Traditional_Study_Hours": study_hours,
    "Anxiety_Level_During_Exams": anxiety,
    "Year_of_Study": year,
    "Prompt_Engineering_Skill": prompt_skill,
    "Institutional_Policy": policy,
    "Major_Category": major,
    "Primary_Use_Case": use_case,
    "Tool_Diversity": tool_diversity,
    "Paid_Subscription": paid_sub,
    "Skill_Retention_Score": skill_retention
}
# --- 4. Prediction Logic ---
st.divider()

if st.button("Generate Prediction", type="primary", use_container_width=True):
    
    if prediction_task == "Predict Burnout Risk":
        X_class = preprocess_input(user_input, class_artifacts)
        burnout_pred = class_artifacts["model"].predict(X_class)[0]
        
        burnout_reverse_map = {v: k for k, v in class_artifacts["label_encoder"]["Burnout_Risk_Level"].items()}
        burnout_label = burnout_reverse_map.get(burnout_pred, "Unknown")
        
        st.success("Classification successful!")
        st.metric(label="Predicted Burnout Risk", value=burnout_label)

    elif prediction_task == "Predict Post-Semester GPA":
        X_reg = preprocess_input(user_input, reg_artifacts)
        gpa_pred = reg_artifacts["model"].predict(X_reg)[0]
        
        st.success("Regression successful!")
        st.metric(label="Predicted Post-Semester GPA", value=f"{gpa_pred:.2f}")