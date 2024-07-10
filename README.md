| Nama  |  Nim | Kelas |
| ------------- | ------------- |------------- |
| Raja Happyanto  | 312210235 | TI 22 A2 |

## UAS Proyek Segmentasi Gambar Menggunakan K-Means dan Streamlit

### Penjelasan
Aplikasi ini Di tujukan untuk uas yang dimana berguna untuk melakukan segmentasi terhadap gambar, akan menunjukkan persentasi warna pada sebuah gambar

### Kode
```
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
```

### Langkah-Langkah Menjalankan Aplikasi

1. **Clone atau Download Repository ini, lalu buka terminal dan jalankan perintah berikut:**
   ```bash
   git clone https://github.com/Hapiyansyah/UAS-PengolahanCitra.git
   ```
3. **Instal dependensi yang dibutuhkan dengan menjalankan perintah berikut di terminal:**
    ```bash
    pip install streamlit scikit-learn opencv-python pillow scipy
    ```

4. **Jalankan aplikasi Streamlit dengan perintah berikut:**
    ```bash
    streamlit run raja.py
    ```
(sesuaikan nama file dan arahkan ke penyimpanan terlebih dahulu sebelum melakukan run)

5. **Aplikasi Streamlit akan terbuka di browser Anda. Anda dapat mengunggah gambar, memilih jumlah cluster, dan melihat hasil segmentasi serta warna-warna yang ada beserta persentase.**

### Tampilan Aplikasi

1. **Foto Bahan**
![foto](https://github.com/RajaHappy/Pengolahan-Citra-UAS/blob/main/foto/eee.jpg)

2. **Tampilan Halaman Depan**
![foto](https://github.com/RajaHappy/Pengolahan-Citra-UAS/blob/main/foto/Screenshot%20(190).png)

3. **Tampilan Pemilihan jumlah klaster**
![foto](https://github.com/RajaHappy/Pengolahan-Citra-UAS/blob/main/foto/Screenshot%20(191).png)

4. **Tampilan gambar tersekmentasi**
![foto](https://github.com/RajaHappy/Pengolahan-Citra-UAS/blob/main/foto/Screenshot%20(192).png)

5. **Tampilan Persentase**
![foto](https://github.com/RajaHappy/Pengolahan-Citra-UAS/blob/main/foto/Screenshot%20(194).png)

5. **Tampilan kode editor <_>**
![foto](https://github.com/RajaHappy/Pengolahan-Citra-UAS/blob/main/foto/Screenshot%20(189).png)

# Thankyou

