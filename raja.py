import cv2
import numpy as np
from sklearn.cluster import KMeans
import streamlit as st

def segment_image(image, k):
    pixel_values = image.reshape((-1, 3))
    pixel_values = np.float32(pixel_values)

    kmeans = KMeans(n_clusters=k, random_state=0)
    labels = kmeans.fit_predict(pixel_values)
    centers = kmeans.cluster_centers_

    centers = np.uint8(centers)
    segmented_image = centers[labels.flatten()]
    segmented_image = segmented_image.reshape(image.shape)

    return segmented_image, labels, centers

def calculate_color_percentages(labels, centers):
    total_pixels = len(labels)
    percentages = []

    for i in range(len(centers)):
        count = np.sum(labels == i)
        percentage = count / total_pixels
        percentages.append(percentage)

    return percentages

def display_color_percentages(percentages, centers):
    for i, percentage in enumerate(percentages):
        color = centers[i].astype(int)
        color_hex = '#{:02x}{:02x}{:02x}'.format(color[0], color[1], color[2])
        st.write(f'Color {i}: {color_hex}, Percentage: {percentage:.2%}')

uploaded_file = st.file_uploader("Upload gambar", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1)
    st.image(image, caption='Gambar asli.', use_column_width=True)

    k = st.slider('Jumlah klaster (k)', 1, 10, 3)

    if st.button("Segmentasi Gambar"):
        segmented_image, labels, centers = segment_image(image, k)
        st.image(segmented_image, caption='Gambar yang telah disegmentasi.', use_column_width=True)
        
        percentages = calculate_color_percentages(labels, centers)
        display_color_percentages(percentages, centers)
