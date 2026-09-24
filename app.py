import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import os

# Set page config
st.set_page_config(page_title="Diabetes Detection", layout="centered", initial_sidebar_state="expanded")

st.title("🏥 Diabetes Detection using Neural Network")
st.write("Enter patient health metrics to predict diabetes risk")

# Load the pre-trained model
@st.cache_resource
def load_trained_model():
    model_path = os.path.join(os.path.dirname(__file__), 'Models', 'model.h5')
    try:
        # Load model without compiling to avoid compatibility issues
        model = load_model(model_path, compile=False)
        # Recompile with current TensorFlow version
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        st.success("✅ Model loaded successfully!")
        return model
    except FileNotFoundError:
        st.error(f"❌ Model file not found at: {model_path}")
        st.info("Make sure model.h5 exists in the Models folder")
        return None
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.info(f"Details: {str(e)}")
        return None

model = load_trained_model()

if model:
    st.divider()
    
    # Create input fields in two columns
    col1, col2 = st.columns(2)
    
    with col1:
        pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=0, step=1)
        glucose = st.number_input("Glucose Level (mg/dL)", min_value=0, max_value=200, value=100, step=1)
        blood_pressure = st.number_input("Blood Pressure (mmHg)", min_value=0, max_value=150, value=70, step=1)
        skin_thickness = st.number_input("Skin Thickness (mm)", min_value=0, max_value=100, value=20, step=1)
    
    with col2:
        insulin = st.number_input("Insulin (mu U/ml)", min_value=0, max_value=900, value=80, step=1)
        bmi = st.number_input("BMI (kg/m²)", min_value=0.0, max_value=70.0, value=25.0, step=0.1)
        dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5, step=0.01)
        age = st.number_input("Age (years)", min_value=1, max_value=120, value=30, step=1)
    
    st.divider()
    
    # Prediction button
    if st.button("🔍 Predict Diabetes Risk", width='stretch', type="primary"):
        try:
            # Prepare input data
            input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, 
                                   insulin, bmi, dpf, age]])
            
            # Reshape for sequence models (add time dimension)
            input_data = np.reshape(input_data, (input_data.shape[0], 1, input_data.shape[1]))
            
            # Make prediction
            prediction = model.predict(input_data, verbose=0)
            # Extract scalar value from prediction
            risk_score = float(prediction.flatten()[0])
            
            # Display results
            st.divider()
            st.subheader("📊 Prediction Results")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Risk Score", f"{risk_score:.4f}")
            
            with col2:
                if risk_score > 0.5:
                    st.metric("Status", "⚠️ HIGH RISK")
                else:
                    st.metric("Status", "✅ LOW RISK")
            
            with col3:
                confidence = (1 - abs(0.5 - risk_score)) * 100
                st.metric("Confidence", f"{confidence:.1f}%")
            
            # Detailed message
            st.divider()
            if risk_score > 0.5:
                st.warning(f"⚠️ **High Diabetes Risk Detected** ({risk_score*100:.2f}%)\n\nPlease consult a healthcare professional for further evaluation and guidance.")
            else:
                st.success(f"✅ **Low Diabetes Risk** ({(1-risk_score)*100:.2f}% confidence)\n\nMaintain healthy lifestyle habits and regular check-ups.")
            
            # Display input summary
            st.divider()
            st.subheader("📋 Input Summary")
            summary_df = pd.DataFrame({
                'Metric': ['Pregnancies', 'Glucose', 'Blood Pressure', 'Skin Thickness', 'Insulin', 'BMI', 'Diabetes Pedigree Function', 'Age'],
                'Value': [pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]
            })
            st.dataframe(summary_df, width='stretch', hide_index=True)
            
        except Exception as e:
            st.error(f"❌ Error during prediction: {str(e)}")
else:
    st.error("❌ Cannot start app without model file.")
