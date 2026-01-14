import pandas as pd
import datetime
import xgboost as xgb
import streamlit as st

def main():
    html_temp = """
     <div style="background-color:lightblue;padding:16px">
     <h2 style="color:black;text-align:center;">Car Price Prediction Using ML</h2>
     </div>
     """
     
    try:
        # Load the model
        model = xgb.Booster()
        model.load_model('xgb_model.json')

        # Convert Booster to XGBRegressor for predictions
        model_regressor = xgb.XGBRegressor()
        model_regressor._Booster = model

    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return
    
    st.markdown(html_temp, unsafe_allow_html=True)
    st.write('')
    st.markdown("##### Thinking of selling your car? Let's find out what it's worth. Our powerful predictor will give you a quick and accurate evaluation.")
     
    # User Inputs
    p1 = st.number_input("What is the current ex-showroom price of the car (In lakhs)?", 2.5, step=1.0)
    p2 = st.number_input("What is the distance completed by the car in kms?", 100, 500000, step=200)
    s1 = st.selectbox("What is the fuel type of the car?", ('Petrol', 'Diesel', 'CNG'))
    p3 = {'Petrol': 0, 'Diesel': 1, 'CNG': 2}[s1]
    s2 = st.selectbox("Are you a dealer or an individual?", ('Dealer', 'Individual'))
    p4 = {'Dealer': 0, 'Individual': 1}[s2]
    s3 = st.selectbox("What is the transmission type?", ('Manual', 'Automatic'))
    p5 = {'Manual': 0, 'Automatic': 1}[s3]
    p6 = st.slider("Number of owners the car previously had?", 0, 3)
    date_time = datetime.datetime.now()
    years = st.number_input("In which year car was purchased?", 1990, date_time.year)
    p7 = date_time.year - years
    
    # Data Preparation
    data_new = pd.DataFrame({
        'Present_Price': [p1],
        'Kms_Driven': [p2],
        'Fuel_Type': [p3],
        'Seller_Type': [p4],
        'Transmission': [p5],
        'Owner': [p6],
        'Age': [p7]
    })
    
    try:
        if st.button('Predict'):
            pred = model_regressor.predict(data_new)
            if pred[0] > 0:
                st.balloons()
                st.success(f"You can sell your car for ₹{pred[0]:.2f} lakhs.")
            else:
                st.warning("You might not be able to sell this car, because the value is too low")
    except Exception as e:
        st.warning(f"Prediction failed: {e}")

if __name__ == '__main__':
    main()



