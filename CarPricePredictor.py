import streamlit as st
import pandas as pd
import datetime
import os

st.set_page_config(page_title="CarWorth", layout="centered")

st.title("🚗 CarWorth")

# --- Lazy import & model loading ---
@st.cache_resource
def load_model():
    import xgboost as xgb
    model = xgb.Booster()
    model.load_model("xgb_model.json")
    return model, xgb


def main():
    html_temp = """
    <div style="background-color:lightblue;padding:17px">
        <h2 style="color:black;text-align:center;">
            CarWorth: Know your car’s true price with AI
        </h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    st.markdown(
        "##### Selling your car? Get a fair, data-backed price in seconds."
    )

    # --- Inputs ---
    p1 = st.number_input("Ex-showroom price (₹ lakhs)", 2.5, step=1.0)
    p2 = st.number_input("Distance driven (km)", 100, 500000, step=200)

    s1 = st.selectbox("Fuel type", ("Petrol", "Diesel", "CNG"))
    p3 = {"Petrol": 0, "Diesel": 1, "CNG": 2}[s1]

    s2 = st.selectbox("Seller type", ("Dealer", "Individual"))
    p4 = {"Dealer": 0, "Individual": 1}[s2]

    s3 = st.selectbox("Transmission", ("Manual", "Automatic"))
    p5 = {"Manual": 0, "Automatic": 1}[s3]

    p6 = st.slider("Number of previous owners", 0, 3)

    year_now = datetime.datetime.now().year
    year_bought = st.number_input("Year of purchase", 1990, year_now)
    p7 = year_now - year_bought

    data_new = pd.DataFrame({
        "Present_Price": [p1],
        "Kms_Driven": [p2],
        "Fuel_Type": [p3],
        "Seller_Type": [p4],
        "Transmission": [p5],
        "Owner": [p6],
        "Age": [p7],
    })

    if st.button("Predict Price"):
        if not os.path.exists("xgb_model.json"):
            st.error("Model file not found")
            return

        try:
            model, xgb = load_model()
            dmatrix = xgb.DMatrix(data_new)
            pred = model.predict(dmatrix)

            if pred[0] > 0:
                st.balloons()
                st.success(f"Estimated selling price: ₹{pred[0]:.2f} lakhs")
            else:
                st.warning("Predicted value is very low.")
        except Exception as e:
            st.error(f"Prediction failed: {e}")


if __name__ == "__main__":
    main()


