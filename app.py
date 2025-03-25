import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

# --- Load the saved model, scaler, and feature names ---
rf_model = joblib.load('rf_model_employee.sav')
scaler = joblib.load('scaler_employee.sav')
feature_names = joblib.load('feature_names_employee.pkl')  # Correct feature order

# --- Department Mapping ---
department_mapping = {
    0: 'IT',
    1: 'RandD',
    2: 'accounting',
    3: 'hr',
    4: 'management',
    5: 'marketing',
    6: 'product_mng',
    7: 'sales',
    8: 'support',
    9: 'technical'
}

# Reverse the mapping to convert back to numeric values
reverse_department_mapping = {v: k for k, v in department_mapping.items()}

# --- Streamlit App Title and Description ---
st.markdown("<h1 style='text-align: center;'>📊 Employee Job Satisfaction Prediction</h1>", unsafe_allow_html=True)
st.markdown("<h5 style='text-align: center;'>Predict if an employee is likely to leave and get retention suggestions.</h5>", unsafe_allow_html=True)

# --- Left Sidebar for Employee Details (Inputs) ---
with st.sidebar:
    st.header("📥 Employee Details")
    satisfaction_level = st.slider('Satisfaction Level (0-1)', 0.0, 1.0, 0.5)
    last_evaluation = st.slider('Last Evaluation (0-1)', 0.0, 1.0, 0.7)
    number_project = st.number_input('Number of Projects', 1, 10, 4)
    average_monthly_hours = st.number_input('Average Monthly Hours', 100, 300, 160)
    time_spend_company = st.number_input('Time Spent in Company (Years)', 1, 10, 3)
    work_accident = st.selectbox('Work Accident', [0, 1])
    promotion_last_5years = st.selectbox('Promotion in Last 5 Years', [0, 1])
    
    # --- Department Dropdown with Labels ---
    selected_department_name = st.selectbox('Department', list(department_mapping.values()))
    department = reverse_department_mapping[selected_department_name]  # Convert name back to numeric

    salary = st.selectbox('Salary Level (0 = Low, 1 = Medium, 2 = High)', [0, 1, 2])

    # --- Create a DataFrame for Input Features ---
    input_data = pd.DataFrame({
        'satisfaction_level': [satisfaction_level],
        'last_evaluation': [last_evaluation],
        'number_project': [number_project],
        'average_montly_hours': [average_monthly_hours],
        'time_spend_company': [time_spend_company],
        'Work_accident': [work_accident],
        'promotion_last_5years': [promotion_last_5years],
        'salary': [salary],
        'Department': [department]  # Numeric value passed to the model
    })

    # --- Match Feature Order Correctly Before Scaling ---
    input_data = input_data[feature_names]

    # --- Scale the Input Data ---
    input_data_scaled = scaler.transform(input_data)

# --- Center Section for Prediction and Suggestions ---
st.markdown("<div style='text-align: center; margin-top: 50px;'>", unsafe_allow_html=True)

if st.button('🔍 Predict'):
    # --- Make Predictions ---
    prediction = rf_model.predict(input_data_scaled)

    if prediction[0] == 1:
        st.markdown("<h2 style='color: red; text-align: center;'>❌ The employee is likely to leave the company.</h2>", unsafe_allow_html=True)
        
            # HR Suggestions
    st.header("🛠 HR Recommendations")

    if prediction[0] == 1:
        st.markdown("""
        <div style="text-align: left;">
        ✅ Conduct 1-on-1 meetings to understand concerns.<br>
        ✅ Provide opportunities for career growth or promotions.<br>
        ✅ Re-evaluate work-life balance and workload.<br>
        ✅ Offer competitive salary and benefits.<br>
        ✅ Recognize and reward outstanding performance.<br>
        </div>
        """, unsafe_allow_html=True)
    
    else:
        st.markdown("""
        <div style="text-align: left;">
        ✅ Employee is satisfied. Keep up the good work!<br>
        ✅ Recognize contributions regularly.<br>
        ✅ Continue providing career growth opportunities.<br>
        ✅ Ensure a balanced workload.<br>
        </div>
        """, unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)
