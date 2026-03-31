import streamlit as st
import joblib as j
import numpy as np
import pandas as pd
import itertools

st.title("💻 Laptop Price & Recommendation System")

# Step 1: Top-level choice
role = st.radio("Who are you?", ("Company", "User"))

# Load resources once
@st.cache_resource
def load_resources():
    pipe = j.load('pipe.joblib')
    df = j.load('df.joblib')
    return pipe, df

pipe, df = load_resources()

# -------------------------------
# Company Dashboard
# -------------------------------
if role == "Company":
    st.header("Company Dashboard")
    company_choice = st.radio("Select an option:", 
                              ("Predict Laptop Price", "Recommend Laptops Within Budget"))

    if company_choice == "Predict Laptop Price":
        st.subheader("Predict Laptop Price Based on Specs")
        company = st.selectbox('Brand', df['Company'].unique())
        type_ = st.selectbox('Type', df['TypeName'].unique())
        ram = st.selectbox('RAM(in GB)', [2,4,6,8,12,16,24,32,64])
        weight = st.number_input('Weight of the Laptop')
        screen_size = st.slider('Screen size in inches', 10.0, 18.0, 13.0)
        resolution = st.selectbox('Screen Resolution', [
            '1920x1080','1366x768','1600x900','3840x2160',
            '3200x1800','2880x1800','2560x1600','2560x1440','2304x1440'
        ])
        cpu = st.selectbox('CPU', df['Cpu brand'].unique())
        hdd = st.selectbox('HDD(in GB)', [0,128,256,512,1024,2048])
        ssd = st.selectbox('SSD(in GB)', [0,8,128,256,512,1024])
        gpu = st.selectbox('GPU', df['Gpu Brand'].unique())
        os = st.selectbox('OS', df['Os Brand'].unique())

        if st.button('Predict Price'):
            with st.spinner("🔄 Predicting..."):
                X_res, Y_res = map(int, resolution.split('x'))
                ppi = ((X_res**2 + Y_res**2)**0.5) / screen_size
                query_df = pd.DataFrame({
                    'Company': [company],
                    'TypeName': [type_],
                    'Ram': [ram],
                    'Weight': [weight],
                    'Touchscreen': [0],
                    'Ips': [0],
                    'Cpu brand': [cpu],
                    'HDD': [hdd],
                    'SSD': [ssd],
                    'ppi': [ppi],
                    'Gpu Brand': [gpu],
                    'Os Brand': [os]
                })
                predicted_price = int(np.exp(pipe.predict(query_df)[0]))
            st.success(f"Predicted Price: ₹{predicted_price}")

    elif company_choice == "Recommend Laptops Within Budget":
        st.subheader("Get Recommendations Based on Your Budget")
        budget = st.number_input("Enter your budget (₹)", min_value=10000, max_value=200000, step=1000)
        company = st.selectbox('Brand', df['Company'].unique())
        type_ = st.selectbox('Type', df['TypeName'].unique())
        weight = st.number_input('Weight of the Laptop')
        screen_size = st.slider('Screen size in inches', 10.0, 18.0, 13.0)
        resolution = st.selectbox('Screen Resolution', [
            '1920x1080','1366x768','1600x900','3840x2160',
            '3200x1800','2880x1800','2560x1600','2560x1440','2304x1440'
        ])
        X_res, Y_res = map(int, resolution.split('x'))
        ppi = ((X_res**2 + Y_res**2)**0.5) / screen_size

        if st.button("Recommend Laptops"):
            with st.spinner("🔄 Generating recommendations..."):
                rams = [4, 8, 16]
                hdds = [0, 512]
                ssds = [128, 256, 512]
                cpus = df['Cpu brand'].unique()
                gpus = df['Gpu Brand'].unique()
                oss = df['Os Brand'].unique()
                ips_options = [0, 1]
                touchscreen_options = [0, 1]

                combos = list(itertools.product(rams, hdds, ssds, cpus, gpus, oss, ips_options, touchscreen_options))
                results = []
                progress = st.progress(0)

                for i, combo in enumerate(combos):
                    ram, hdd, ssd, cpu, gpu, os, ips, touchscreen = combo
                    query_df = pd.DataFrame({
                        'Company': [company],
                        'TypeName': [type_],
                        'Ram': [ram],
                        'Weight': [weight],
                        'Touchscreen': [touchscreen],
                        'Ips': [ips],
                        'Cpu brand': [cpu],
                        'HDD': [hdd],
                        'SSD': [ssd],
                        'ppi': [ppi],
                        'Gpu Brand': [gpu],
                        'Os Brand': [os]
                    })
                    predicted_price = int(np.exp(pipe.predict(query_df)[0]))
                    if predicted_price <= budget:
                        results.append((predicted_price, query_df))
                    progress.progress(int((i+1)/len(combos)*100))

                results.sort(key=lambda x: x[0], reverse=True)
            if results:
                for price, specs in results[:5]:
                    st.write(f"💰 Predicted Price: ₹{price}")
                    st.write(specs)
            else:
                st.warning("No laptops found within this budget.")

# -------------------------------
# User Dashboard
# -------------------------------
elif role == "User":
    st.header("User Dashboard")
    st.subheader("Get Recommendations Based on Your Budget")

    budget = st.number_input("Enter your budget (₹)", min_value=10000, max_value=200000, step=1000)
    company = st.selectbox('Brand', df['Company'].unique())
    type_ = st.selectbox('Type', df['TypeName'].unique())
    weight = st.number_input('Weight of the Laptop')
    screen_size = st.slider('Screen size in inches', 10.0, 18.0, 13.0)
    resolution = st.selectbox('Screen Resolution', [
        '1920x1080','1366x768','1600x900','3840x2160',
        '3200x1800','2880x1800','2560x1600','2560x1440'
    ])
    X_res, Y_res = map(int, resolution.split('x'))
    ppi = ((X_res**2 + Y_res**2)**0.5) / screen_size

    if st.button("Recommend Laptops"):
        with st.spinner("🔄 Generating recommendations..."):
            rams = [4, 8, 16]
            hdds = [0, 512]
            ssds = [128, 256, 512]
            cpus = df['Cpu brand'].unique()[:3]
            gpus = df['Gpu Brand'].unique()[:2]
            oss = df['Os Brand'].unique()[:2]
            ips_options = [0, 1]
            touchscreen_options = [0, 1]

            combos = list(itertools.product(rams, hdds, ssds, cpus, gpus, oss, ips_options, touchscreen_options))
            query_df = pd.DataFrame([{
                'Company': company,
                'TypeName': type_,
                'Ram': ram,
                'Weight': weight,
                'Touchscreen': touchscreen,
                'Ips': ips,
                'Cpu brand': cpu,
                'HDD': hdd,
                'SSD': ssd,
                'ppi': ppi,
                'Gpu Brand': gpu,
                'Os Brand': os
            } for ram, hdd, ssd, cpu, gpu, os, ips, touchscreen in combos])

            predicted_prices = np.exp(pipe.predict(query_df))
            query_df['Predicted Price'] = predicted_prices.astype(int)
            results = query_df[query_df['Predicted Price'] <= budget].sort_values('Predicted Price', ascending=False).head(5)

        if not results.empty:
            st.write("Top Recommendations:")
            st.dataframe(results)
        else:
            st.warning("No laptops found within this budget.")
