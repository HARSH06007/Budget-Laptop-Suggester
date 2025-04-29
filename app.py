import streamlit as st
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import pandas as pd

# Load the saved model
with open("laptop_recommender.pkl", "rb") as f:
    model_data = pickle.load(f)

feature_matrix = model_data["feature_matrix"]
scaler = model_data["scaler"]
df = model_data["df"]
processor_options = model_data["processor_options"]
os_options = model_data["os_options"]

# Load original dataset for images
original_df = pd.read_csv("laptop refined.csv")

# Extract unique spec options from the dataset and round appropriately
ram_options = sorted(np.round(df["ram"] * scaler.data_range_[0] + scaler.data_min_[0]).astype(int).unique())
ssd_options = sorted(np.round(df["ssd"] * scaler.data_range_[1] + scaler.data_min_[1]).astype(int).unique())
hdd_options = sorted(np.round(df["hdd"] * scaler.data_range_[2] + scaler.data_min_[2]).astype(int).unique())
display_options = sorted(np.round(df["display"] * scaler.data_range_[3] + scaler.data_min_[3], 1).unique())

# Streamlit interface
st.title("Laptop Recommender System")
st.markdown("Find the perfect laptop for your needs! Adjust the specs below or use a preset.")

# Sidebar for inputs
with st.sidebar:
    st.header("Your Preferences")
    
    # Budget filter
    budget = st.slider("Max Budget (₹)", min_value=15000, max_value=400000, value=100000, step=5000,
                       help="Set your maximum budget to filter affordable options.")
    
    # Preset options
    preset = st.selectbox("Choose a Use Case Preset", ["Custom", "Gaming", "Work", "Budget"],
                          help="Select a preset to auto-fill specs.")
    if preset == "Gaming":
        ram = max([r for r in ram_options if r <= 16], default=16)
        ssd = max([s for s in ssd_options if s <= 512], default=512)
        hdd = 0
        display = max([d for d in display_options if d <= 15.6], default=15.6)
        processor = "intel_core_i7"
        os = "windows"
    elif preset == "Work":
        ram = max([r for r in ram_options if r <= 8], default=8)
        ssd = max([s for s in ssd_options if s <= 512], default=512)
        hdd = 0
        display = max([d for d in display_options if d <= 14], default=14)
        processor = "intel_core_i5"
        os = "windows"
    elif preset == "Budget":
        ram = max([r for r in ram_options if r <= 4], default=4)
        ssd = max([s for s in ssd_options if s <= 256], default=256)
        hdd = 0
        display = max([d for d in display_options if d <= 15.6], default=15.6)
        processor = "intel_core_i3"
        os = "windows"
    else:
        ram = st.selectbox("RAM (GB)", ram_options, 
                           index=ram_options.index(max([r for r in ram_options if r <= 8], default=8)),
                           help="More RAM improves multitasking.")
        ssd = st.selectbox("SSD Storage (GB)", ssd_options, 
                           index=ssd_options.index(max([s for s in ssd_options if s <= 512], default=512)),
                           help="SSD offers faster performance.")
        hdd = st.selectbox("HDD Storage (GB)", hdd_options, index=0,
                           help="HDD provides more storage at a lower cost.")
        display = st.selectbox("Screen Size (inches)", display_options, 
                               index=display_options.index(max([d for d in display_options if d <= 15.6], default=15.6)),
                               help="Larger screens are better for productivity.")
        processor = st.selectbox("Processor", options=processor_options,
                                 help="Higher-tier processors (e.g., i7, Ryzen 7) are more powerful.")
        os = st.selectbox("Operating System", options=os_options,
                          help="Choose your preferred OS.")

    # Reset button
    if st.button("Reset"):
        st.session_state.clear()

# Prepare user input vector
user_numerical = scaler.transform([[ram, ssd, hdd, display]])
user_processor = np.array([2 if p == processor else 0 for p in processor_options])  # Weighted processor
user_os = np.array([1 if o == os else 0 for o in os_options])
user_vector = np.concatenate([user_numerical[0], user_processor, user_os])

# Calculate similarity with progress bar
with st.spinner("Finding your perfect laptop..."):
    similarities = cosine_similarity([user_vector], feature_matrix)
    top_indices = similarities.argsort()[0][::-1][:10]  # Top 10 for filtering
    top_similarities = similarities[0][top_indices]
    top_df = df.iloc[top_indices].copy()
    top_df["similarity"] = top_similarities

    # Filter by budget and sort by similarity then rating
    top_df = top_df[top_df["price"] <= budget]
    if top_df.empty:
        st.warning("No laptops match your specs and budget. Try adjusting your preferences!")
    else:
        top_df = top_df.sort_values(by=["similarity", "rating"], ascending=[False, False])
        recommendation = top_df.iloc[0]
        other_options = top_df.iloc[1:3]

        # Main recommendation
        st.subheader("Your Top Recommendation")
        col1, col2 = st.columns([1, 3])
        with col1:
            img_link = original_df.iloc[top_indices[0]]["img_link"]
            st.image(img_link, width=150, caption="Laptop Image")
        with col2:
            st.write(f"**Brand**: {recommendation['brand'].capitalize()}")
            st.write(f"**Name**: {recommendation['name']}")
            st.write(f"**Price**: ₹{recommendation['price']}")
            # Use input values directly for display to avoid denormalization issues
            st.write(f"**Specs**: {processor.replace('_', ' ').capitalize()}, {int(ram)} GB RAM, "
                     f"SSD: {int(ssd)} GB, HDD: {int(hdd)} GB, {display:.1f}\" Screen, OS: {os.capitalize()}")
            st.write(f"**Rating**: {recommendation['rating']:.1f}/5")
            st.write(f"*Why this?* Matches your {processor.replace('_', ' ')} preference and budget.")

        # Comparison table for top 3
        st.subheader("Compare Top Options")
        comparison_df = top_df[["brand", "name", "price", "ram", "ssd", "hdd", "display", "rating"]].head(3)
        # Denormalize and round for display
        comparison_df["ram"] = np.round(comparison_df["ram"] * scaler.data_range_[0] + scaler.data_min_[0]).astype(int)
        comparison_df["ssd"] = np.round(comparison_df["ssd"] * scaler.data_range_[1] + scaler.data_min_[1]).astype(int)
        comparison_df["hdd"] = np.round(comparison_df["hdd"] * scaler.data_range_[2] + scaler.data_min_[2]).astype(int)
        comparison_df["display"] = np.round(comparison_df["display"] * scaler.data_range_[3] + scaler.data_min_[3], 1)
        comparison_df["brand"] = comparison_df["brand"].str.capitalize()
        comparison_df["Specs"] = (comparison_df["ram"].astype(str) + " GB RAM, " +
                                 comparison_df["ssd"].astype(str) + " GB SSD, " +
                                 comparison_df["hdd"].astype(str) + " GB HDD, " +
                                 comparison_df["display"].astype(str) + '" Screen')
        st.table(comparison_df[["brand", "name", "price", "Specs", "rating"]].rename(columns={
            "brand": "Brand", "name": "Name", "price": "Price (₹)", "rating": "Rating"
        }))

        # Other options
        st.subheader("More Suggestions")
        for _, alt in other_options.iterrows():
            alt_ram = int(np.round(alt["ram"] * scaler.data_range_[0] + scaler.data_min_[0]))
            alt_ssd = int(np.round(alt["ssd"] * scaler.data_range_[1] + scaler.data_min_[1]))
            alt_hdd = int(np.round(alt["hdd"] * scaler.data_range_[2] + scaler.data_min_[2]))
            alt_display = np.round(alt["display"] * scaler.data_range_[3] + scaler.data_min_[3], 1)
            st.write(f"- {alt['brand'].capitalize()} {alt['name']} - ₹{alt['price']} "
                     f"(Rating: {alt['rating']:.1f})")