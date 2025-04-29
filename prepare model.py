import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pickle
import re
import os

# Verify file existence and load dataset
input_file = "laptop refined.csv"  # Adjusted to match your naming
if not os.path.exists(input_file):
    raise FileNotFoundError(f"Input file '{input_file}' not found. Please check the file name or path.")
df = pd.read_csv(input_file)

# Check for required columns
required_cols = ["name", "price(in Rs.)", "processor", "ram", "os", "storage", "display(in inch)"]
missing_cols = [col for col in required_cols if col not in df.columns]
if missing_cols:
    raise ValueError(f"Missing required columns: {missing_cols}")

# Extract brand from name
df["brand"] = df["name"].apply(lambda x: x.split()[0].lower() if pd.notna(x) else "unknown")

# Simplify column names
df = df.rename(columns={
    "price(in Rs.)": "price",
    "display(in inch)": "display"
})

# Processor simplification with broader matching
def simplify_processor(proc):
    if pd.isna(proc):
        return "other"
    proc = proc.lower().strip()
    if "intel" in proc:
        if "core i3" in proc or "i3" in proc: return "intel_core_i3"
        elif "core i5" in proc or "i5" in proc: return "intel_core_i5"
        elif "core i7" in proc or "i7" in proc: return "intel_core_i7"
        elif "core i9" in proc or "i9" in proc: return "intel_core_i9"
        elif "pentium" in proc: return "intel_pentium"
        elif "celeron" in proc: return "intel_celeron"
        else: return "intel_other"
    elif "amd" in proc:
        if "ryzen 3" in proc or "r3" in proc: return "amd_ryzen_3"
        elif "ryzen 5" in proc or "r5" in proc: return "amd_ryzen_5"
        elif "ryzen 7" in proc or "r7" in proc: return "amd_ryzen_7"
        elif "ryzen 9" in proc or "r9" in proc: return "amd_ryzen_9"
        elif "athlon" in proc: return "amd_athlon"
        else: return "amd_other"
    elif "apple" in proc:
        if "m1" in proc: return "apple_m1"
        elif "m2" in proc: return "apple_m2"
        else: return "apple_other"
    elif "qualcomm" in proc: return "qualcomm_snapdragon"
    elif "mediatek" in proc: return "mediatek_kompanio"
    else: return "other"

df["processor"] = df["processor"].apply(simplify_processor)

# Extract RAM with error handling
def extract_ram(ram_str):
    try:
        match = re.search(r"(\d+)", str(ram_str))
        return int(match.group(1)) if match else 0
    except (ValueError, AttributeError):
        return 0  # Default to 0 if parsing fails

df["ram"] = df["ram"].apply(extract_ram)

# Extract Storage with robust parsing
def extract_storage(stor):
    if pd.isna(stor):
        return 0, 0
    stor = stor.lower().strip()
    ssd, hdd = 0, 0
    parts = stor.split("|") if "|" in stor else [stor]
    for part in parts:
        match = re.search(r"(\d+)", part)
        if not match:
            continue
        gb = int(match.group(1))
        gb = gb * 1024 if "tb" in part else gb
        if "ssd" in part:
            ssd = gb
        elif "hdd" in part:
            hdd = gb
    return ssd, hdd

df[["ssd", "hdd"]] = pd.DataFrame(df["storage"].apply(extract_storage).tolist(), index=df.index)

# Handle display with validation
df["display"] = pd.to_numeric(df["display"], errors="coerce")
median_display = df["display"].median()
if pd.isna(median_display):
    median_display = 15.6  # Default fallback if all displays are missing
df["display"].fillna(median_display, inplace=True)

# Simplify OS with error handling
def simplify_os(os):
    if pd.isna(os):
        return "other"
    os = os.lower().strip()
    if "windows" in os: return "windows"
    elif "mac" in os: return "mac"
    elif "chrome" in os: return "chrome"
    elif "dos" in os: return "dos"
    else: return "other"

df["os"] = df["os"].apply(simplify_os)

# Handle ratings with validation
df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
median_rating = df["rating"].median()
if pd.isna(median_rating):
    median_rating = 4.0  # Default fallback if all ratings are missing
df["rating"].fillna(median_rating, inplace=True)

# Select relevant columns
columns = ["brand", "name", "price", "processor", "ram", "ssd", "hdd", "display", "os", "rating"]
df = df[columns]

# Remove rows with critical missing values (e.g., price)
df = df.dropna(subset=["price", "processor", "ram", "ssd", "hdd", "display", "os"])

# Normalize numerical features
scaler = MinMaxScaler()
numerical = ["ram", "ssd", "hdd", "display"]
df[numerical] = scaler.fit_transform(df[numerical])

# One-hot encode categorical features
processor_dummies = pd.get_dummies(df["processor"]) * 2  # Weight processor by 2
os_dummies = pd.get_dummies(df["os"])

# Create feature matrix
feature_matrix = pd.concat([df[numerical], processor_dummies, os_dummies], axis=1).values

# Save outputs
df.to_csv("refined_laptop_dataset.csv", index=False)
model_data = {
    "feature_matrix": feature_matrix,
    "scaler": scaler,
    "df": df,
    "processor_options": processor_dummies.columns.tolist(),
    "os_options": os_dummies.columns.tolist()
}
with open("laptop_recommender.pkl", "wb") as f:
    pickle.dump(model_data, f)

print(f"Dataset processed successfully. Saved refined dataset as 'refined_laptop_dataset.csv' and model as 'laptop_recommender.pkl'")
print(f"Processed {len(df)} laptops with {feature_matrix.shape[1]} features.")