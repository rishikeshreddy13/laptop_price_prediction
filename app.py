import streamlit as st
import joblib as j
import numpy as np
import pandas as pd
import itertools

st.title("💻 Laptop Price & Recommendation System")

# Step 1: Show only two options
choice = st.radio(
    "Select what you want to do:",
    ("Predict Laptop Price", "Recommend Laptops Within Budget")
)

# Step 2: Load model only after choice
pipe = j.load('pipe.joblib')
df = j.load('df.joblib')

# Step 3: Show interface based on choice
if choice == "Predict Laptop Price":
    st.header("Predict Laptop Price Based on Specs")

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
        with st.spinner("🔄 Loading model and predicting price..."):
            X_res, Y_res = map(int, resolution.split('x'))
            ppi = ((X_res**2 + Y_res**2)**0.5) / screen_size

            query_data = {
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
            }

            query_df = pd.DataFrame(query_data)
            predicted_price = int(np.exp(pipe.predict(query_df)[0]))
        st.success(f"Predicted Price: ₹{predicted_price}")

elif choice == "Recommend Laptops Within Budget":
    st.header("Get Recommendations Based on Your Budget")

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
    
    def recommend_within_budget(budget, df, pipe):
        rams = [4, 8, 16]
        hdds = [0, 512]
        ssds = [128, 256, 512]
        cpus = df['Cpu brand'].unique()
        gpus = df['Gpu Brand'].unique()
        oss = df['Os Brand'].unique()
        ips_options = [0, 1]
        touchscreen_options = [0, 1]

        results = []
        for combo in itertools.product(rams, hdds, ssds, cpus, gpus, oss, ips_options, touchscreen_options):
            ram, hdd, ssd, cpu, gpu, os, ips, touchscreen = combo

            

            query_data = {
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
            }

            query_df = pd.DataFrame(query_data)
            predicted_price = int(np.exp(pipe.predict(query_df)[0]))

            if predicted_price <= budget:
                results.append((predicted_price, query_data))

        results.sort(key=lambda x: x[0], reverse=True)
        return results[:5]

    if st.button("Recommend Laptops"):
        with st.spinner("🔄 Generating recommendations..."):
            recommendations = recommend_within_budget(budget, df, pipe)
        if recommendations:
            for price, specs in recommendations:
                st.write(f"💰 Predicted Price: ₹{price}")
                st.write(pd.DataFrame(specs))
        else:
            st.warning("No laptops found within this budget.")
