import streamlit as st
import joblib as j
import numpy as np
import pandas as pd
import itertools

st.set_page_config(page_title="Laptop Predictor", layout="wide")

st.title("💻 Laptop Price & Recommendation System")

# -------------------------------
# Load resources
# -------------------------------
@st.cache_resource
def load_resources():
    pipe = j.load('pipe.joblib')
    df = j.load('df.joblib')
    return pipe, df

pipe, df = load_resources()

# -------------------------------
# Helper functions
# -------------------------------

def get_os_options(company):
    if company.lower() == "apple":
        return [x for x in df['Os Brand'].unique() if 'mac' in x.lower()]
    else:
        return [x for x in df['Os Brand'].unique() if 'mac' not in x.lower()]


def calculate_ppi(resolution, screen_size):
    X_res, Y_res = map(int, resolution.split('x'))
    return ((X_res**2 + Y_res**2) ** 0.5) / screen_size


def predict_price(input_df):
    return int(np.exp(pipe.predict(input_df)[0]))


# -------------------------------
# Mode Selection
# -------------------------------
mode = st.radio(
    "Choose Mode:",
    ("Predict Laptop Price", "Recommend Laptops")
)

# -------------------------------
# COMMON INPUTS (Sidebar)
# -------------------------------
st.sidebar.header("⚙️ Laptop Specs")

company = st.sidebar.selectbox('Brand', df['Company'].unique())
type_ = st.sidebar.selectbox('Type', df['TypeName'].unique())
weight = st.sidebar.number_input('Weight (kg)', min_value=0.5, max_value=5.0, step=0.1)
screen_size = st.sidebar.slider('Screen Size (inches)', 10.0, 18.0, 13.0)

# ✅ SINGLE resolution input (fixed)
resolution = st.sidebar.selectbox(
    'Resolution',
    [
        '1366x768',
        '1600x900',
        '1920x1080',
        '2560x1440',
        '3840x2160'
    ]
)

ppi = calculate_ppi(resolution, screen_size)

os_options = get_os_options(company)

# -------------------------------
# PREDICTION MODE
# -------------------------------
if mode == "Predict Laptop Price":

    st.header("💰 Price Prediction")

    ram = st.selectbox('RAM (GB)', sorted(df['Ram'].unique()))
    cpu = st.selectbox('CPU', df['Cpu brand'].unique())
    hdd = st.selectbox('HDD', sorted(df['HDD'].unique()))
    ssd = st.selectbox('SSD', sorted(df['SSD'].unique()))
    gpu = st.selectbox('GPU', df['Gpu Brand'].unique())
    os = st.selectbox('OS', os_options)

    if st.button("Predict Price"):

        query = pd.DataFrame([{
            'Company': company,
            'TypeName': type_,
            'Ram': ram,
            'Weight': weight,
            'Touchscreen': 0,
            'Ips': 0,
            'Cpu brand': cpu,
            'HDD': hdd,
            'SSD': ssd,
            'ppi': ppi,
            'Gpu Brand': gpu,
            'Os Brand': os
        }])

        price = predict_price(query)

        st.success(f"💰 Estimated Price: ₹{price:,}")


# -------------------------------
# RECOMMENDATION MODE
# -------------------------------
elif mode == "Recommend Laptops":

    st.header("🤖 Smart Recommendations")

    budget = st.number_input("Budget (₹)", 10000, 200000, step=1000)

    storage_pref = st.selectbox(
        "Storage Preference",
        ["Any", "SSD Only", "SSD + HDD"]
    )

    if st.button("Generate Recommendations"):

        with st.spinner("Generating best configs..."):

            rams = [8, 16, 32]
            ssds = [256, 512, 1024]
            hdds = [0, 512, 1024]

            cpus = df['Cpu brand'].value_counts().index[:4]
            gpus = df['Gpu Brand'].value_counts().index[:3]
            oss = os_options

            combos = list(itertools.product(
                rams, hdds, ssds, cpus, gpus, oss, [0,1], [0,1]
            ))

            data = []

            for ram, hdd, ssd, cpu, gpu, os, ips, touch in combos:
                data.append({
                    'Company': company,
                    'TypeName': type_,
                    'Ram': ram,
                    'Weight': weight,
                    'Touchscreen': touch,
                    'Ips': ips,
                    'Cpu brand': cpu,
                    'HDD': hdd,
                    'SSD': ssd,
                    'ppi': ppi,
                    'Gpu Brand': gpu,
                    'Os Brand': os
                })

            df_test = pd.DataFrame(data)

            # Predict
            df_test['Price'] = np.exp(pipe.predict(df_test)).astype(int)

            # Budget filter
            lower, upper = budget * 0.8, budget * 1.2
            results = df_test[(df_test['Price'] >= lower) & (df_test['Price'] <= upper)]

            # Storage filter
            if storage_pref == "SSD Only":
                results = results[results['HDD'] == 0]
            elif storage_pref == "SSD + HDD":
                results = results[results['HDD'] > 0]

            # Remove unrealistic combos safely
            results = results[~(
                (results['Gpu Brand'].str.lower() == 'amd') &
                (results['Os Brand'].str.lower().str.contains('mac'))
            )]

            # Remove duplicates
            results = results.drop_duplicates(
                subset=['Ram','SSD','HDD','Cpu brand','Gpu Brand']
            )

            # Scoring
            results['score'] = (
                results['Ram'] * 4 +
                results['SSD'] * 0.03 +
                results['HDD'] * 0.01 +
                results['Cpu brand'].str.contains('i7', case=False) * 30 +
                results['Cpu brand'].str.contains('i5', case=False) * 20 +
                (results['Gpu Brand'].str.lower() == 'nvidia') * 25 +
                (results['Os Brand'].str.contains('Windows', case=False)) * 10
            )

            results = results.sort_values(by='score', ascending=False).head(5)

        # OUTPUT
        if not results.empty:

            st.subheader("📊 Insights")

            col1, col2, col3 = st.columns(3)
            col1.metric("Avg Price", f"₹{int(results['Price'].mean()):,}")
            col2.metric("Top RAM", f"{results['Ram'].mode()[0]} GB")
            col3.metric("Top CPU", results['Cpu brand'].mode()[0])

            st.subheader("💻 Best Configurations")

            for _, row in results.iterrows():
                st.markdown(f"""
                ### 💻 {row['Company']} {row['TypeName']}
                - **CPU:** {row['Cpu brand']}
                - **RAM:** {row['Ram']} GB
                - **SSD:** {row['SSD']} GB
                - **HDD:** {row['HDD']} GB
                - **GPU:** {row['Gpu Brand']}
                - **OS:** {row['Os Brand']}
                - **💰 Price:** ₹{row['Price']:,}
                """)

        else:
            st.warning("No suitable configs found 😢")
