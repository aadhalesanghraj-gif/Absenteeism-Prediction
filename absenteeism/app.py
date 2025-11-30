import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Absenteeism Predictor",
    page_icon="🏢",
    layout="wide"
)

# Load the saved model and scaler
@st.cache_resource
def load_models():
    import os
    try:
        # Get the directory where app.py is located
        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base_dir, 'models', 'logistic_model.pkl')
        scaler_path = os.path.join(base_dir, 'models', 'scaler.pkl')
        
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        with open(scaler_path, 'rb') as file:
            scaler = pickle.load(file)
        return model, scaler
    except FileNotFoundError:
        st.error("⚠️ Model files not found! Please ensure logistic_model.pkl and scaler.pkl are in the 'models' folder.")
        st.stop()  

model, scaler = load_models()

# App title and description
st.title("🏢 Employee Absenteeism Prediction System")
st.markdown("### Predict the likelihood of employee absenteeism based on key factors")
st.markdown("---")

# Create two columns for layout
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📊 Employee Information")
    
    # Create tabs for better organization
    tab1, tab2, tab3 = st.tabs(["Personal Info", "Work Details", "Absence Reasons"])
    
    with tab1:
        col_a, col_b = st.columns(2)
        with col_a:
            age = st.slider("Age", 18, 65, 35, help="Employee's age in years")
            body_mass_index = st.slider("Body Mass Index (BMI)", 15, 50, 25, help="BMI value")
            children = st.number_input("Number of Children", 0, 5, 0, help="Number of children")
        with col_b:
            education = st.selectbox("Education Level", [0, 1, 2, 3, 4], 
                                    format_func=lambda x: ["High School", "Bachelor's", "Master's", "PhD", "Other"][x],
                                    help="Highest education level completed")
            pets = st.number_input("Number of Pets", 0, 5, 0, help="Number of pets owned")
            month_value = st.slider("Month", 1, 12, 7, help="Month of the year (1=Jan, 12=Dec)")
    
    with tab2:
        col_c, col_d = st.columns(2)
        with col_c:
            distance_to_work = st.slider("Distance to Work (km)", 1, 100, 20, help="Distance from home to workplace")
            daily_work_load = st.slider("Daily Work Load Average", 100.0, 400.0, 239.5, 
                                       help="Average daily workload hours")
        with col_d:
            transportation_expense = st.slider("Transportation Expense", 50, 500, 200, 
                                              help="Monthly transportation cost")
            day_of_week = st.selectbox("Day of the Week", [1, 2, 3, 4, 5], 
                                      format_func=lambda x: ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"][x-1],
                                      help="Day of the week")
    
    with tab3:
        st.markdown("**Select reasons for potential absence:**")
        col_e, col_f = st.columns(2)
        with col_e:
            reason_1 = st.checkbox("Reason 1: Medical Consultation", value=False)
            reason_2 = st.checkbox("Reason 2: Dental Consultation", value=False)
        with col_f:
            reason_3 = st.checkbox("Reason 3: Medical Examination", value=False)
            reason_4 = st.checkbox("Reason 4: Laboratory Examination", value=False)

with col2:
    st.subheader("🎯 Quick Summary")
    st.info(f"""
    **Employee Profile:**
    - Age: {age} years
    - Education: {["High School", "Bachelor's", "Master's", "PhD", "Other"][education]}
    - Distance: {distance_to_work} km
    - Children: {children}
    - Pets: {pets}
    """)
    
    st.markdown("---")
    st.subheader("ℹ️ Model Info")
    st.success("""
    **Algorithm:** Logistic Regression
    
    **Accuracy:** 78%
    
    **Features:** 14 employee factors
    """)

# Predict button
st.markdown("---")
predict_col1, predict_col2, predict_col3 = st.columns([1, 1, 1])

with predict_col2:
    predict_button = st.button("🔮 PREDICT ABSENTEEISM RISK", type="primary", use_container_width=True)

if predict_button:
    # Create input dataframe matching training data columns
    input_data = pd.DataFrame({
        'reason_1': [int(reason_1)],
        'reason_2': [int(reason_2)],
        'reason_3': [int(reason_3)],
        'reason_4': [int(reason_4)],
        'Transportation Expense': [transportation_expense],
        'Distance to Work': [distance_to_work],
        'Age': [age],
        'Daily Work Load Average': [daily_work_load],
        'Body Mass Index': [body_mass_index],
        'Education': [education],
        'Children': [children],
        'Pets': [pets],
        'Month Value': [month_value],
        'Day of the week': [day_of_week]
    })
    
    # Scale the input data
    input_scaled = scaler.transform(input_data)
    
    # Make prediction
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0]
    
    # Display results
    st.markdown("---")
    st.markdown("## 📋 Prediction Results")
    
    # Create three columns for results
    res_col1, res_col2, res_col3 = st.columns([1, 2, 1])
    
    with res_col2:
        if prediction == 1:
            st.error("### ⚠️ HIGH RISK OF ABSENTEEISM")
            st.markdown(f"**Absence Probability: {probability[1]*100:.1f}%**")
        else:
            st.success("### ✅ LOW RISK OF ABSENTEEISM")
            st.markdown(f"**Absence Probability: {probability[1]*100:.1f}%**")
        
        # Probability visualization
        st.markdown("#### Risk Level Indicator")
        st.progress(float(probability[1]))
        
        # Risk interpretation
        if probability[1] > 0.7:
            st.warning("🔴 **Very High Risk** - Immediate attention recommended")
        elif probability[1] > 0.5:
            st.warning("🟡 **Moderate Risk** - Monitor closely")
        else:
            st.info("🟢 **Low Risk** - Standard monitoring")
    
    # Show detailed breakdown
    st.markdown("---")
    with st.expander("📊 View Detailed Input Data"):
        st.dataframe(input_data, use_container_width=True)
    
    with st.expander("💡 Interpretation Guide"):
        st.markdown("""
        **How to interpret the results:**
        - **Probability > 70%:** High likelihood of absenteeism - consider proactive measures
        - **Probability 50-70%:** Moderate risk - monitor and support employee
        - **Probability < 50%:** Low risk - standard management approach
        
        **Key factors typically affecting absenteeism:**
        - Distance to work and transportation costs
        - Medical/health-related reasons
        - Work load and schedule
        - Personal circumstances (children, pets)
        """)

else:
    st.info("👆 Fill in the employee information above and click 'PREDICT ABSENTEEISM RISK' to see results")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>Built with Streamlit | Logistic Regression Model | 78% Accuracy</p>
    <p>For demonstration purposes - Always consult HR professionals for actual decisions</p>
</div>
""", unsafe_allow_html=True)


