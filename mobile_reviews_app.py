import streamlit as st
import joblib
import pandas as pd


# Load saved files

kmeans = joblib.load("mobile_kmeans.pkl")
cluster_scaler = joblib.load("mobile_cluster_scaler.pkl")
product_df = joblib.load("mobile_products.pkl")
recommend_scaler = joblib.load("mobile_recommendation_scaler.pkl")
similarity_matrix = joblib.load("mobile_similarity_matrix.pkl")
cluster_features = joblib.load("mobile_cluster_features.pkl")
recommend_features = joblib.load("mobile_recommend_features.pkl")

st.title("Mobile Product Segmentation & Recommendation System")


st.subheader("Select a Mobile Product")

selected_mobile = st.selectbox(
    "Choose a mobile model:",
    product_df['model'].tolist()
)

st.write("Selected mobile:", selected_mobile)
selected_details = product_df[
    product_df['model'] == selected_mobile
]


# Get selected mobile index
selected_index = product_df[
    product_df['model'] == selected_mobile
].index[0]

# Get similarity scores
similarity_scores = list(
    enumerate(similarity_matrix[selected_index])
)

# Sort from highest to lowest
similarity_scores = sorted(
    similarity_scores,
    key=lambda x: x[1],
    reverse=True
)

# Exclude the selected mobile itself
similarity_scores = similarity_scores[1:6]

# Get recommended indexes
recommended_indices = [
    i[0] for i in similarity_scores
]

# Get recommended products
recommendations = product_df.loc[
    recommended_indices,
    [
        'brand',
        'model',
        'price_usd',
        'rating',
        'battery_life_rating',
        'camera_rating',
        'performance_rating',
        'design_rating',
        'display_rating'
    ]
].copy()

# Add similarity score
recommendations['similarity_score'] = [
    score for _, score in similarity_scores
]

# Format selected mobile details
selected_display = selected_details[
    [
        'brand',
        'model',
        'price_usd',
        'rating',
        'battery_life_rating',
        'camera_rating',
        'performance_rating',
        'design_rating',
        'display_rating'
    ]
].copy()

selected_display['price_usd'] = selected_display['price_usd'].round(2)
selected_display['rating'] = selected_display['rating'].round(2)

recommendations['price_usd'] = recommendations['price_usd'].round(2)
recommendations['rating'] = recommendations['rating'].round(2)
recommendations['similarity_score'] = recommendations['similarity_score'].round(3)
st.subheader("Selected Mobile Details")
st.dataframe(selected_display, use_container_width=True)

st.subheader("Top 5 Recommended Mobiles")
st.dataframe(recommendations, use_container_width=True)

st.subheader("Product Segmentation")

# Prepare selected product features for clustering
selected_cluster_data = selected_details[cluster_features]

# Scale the selected product
selected_scaled = cluster_scaler.transform(selected_cluster_data)

# Predict cluster
predicted_cluster = kmeans.predict(selected_scaled)[0]

# Convert cluster number into readable label
# Convert cluster number into readable label
segment_labels = {
    0: "Low Rated",
    1: "High Rated"
}

# Get segment name
segment_name = segment_labels.get(predicted_cluster, "Unknown")

st.info(f"Predicted Cluster: {predicted_cluster}")

if segment_name == "High Rated":
    st.success(f"Product Segment: {segment_name}")
else:
    st.warning(f"Product Segment: {segment_name}")

st.subheader("Key Insights")

st.write("• Products were segmented into 2 clusters based on K-Means clustering.")
st.write("• Cluster 0 represents lower-rated products.")
st.write("• Cluster 1 represents higher-rated products.")
st.write("• Product price showed very little correlation with customer ratings.")
st.write("• Battery, camera, performance, design, and display ratings were strongly related to overall customer rating.")
st.write("• Positive sentiment was the most common review sentiment.")