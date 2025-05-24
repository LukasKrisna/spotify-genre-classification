# Laporan Proyek Machine Learning - Klasifikasi Genre Musik Spotify

## Domain Proyek

Industri musik digital telah mengalami pertumbuhan eksponensial, dengan platform streaming seperti Spotify menyediakan akses ke puluhan juta lagu. Dalam katalog musik yang begitu luas, genre musik berfungsi sebagai deskriptor fundamental yang membantu dalam organisasi, penemuan, dan personalisasi rekomendasi musik. Klasifikasi genre musik secara otomatis menjadi krusial untuk mengelola basis data musik skala besar ini secara efisien dan meningkatkan pengalaman pengguna. Proses ini melibatkan analisis fitur audio yang diekstrak dari lagu untuk melatih model _machine learning_ yang dapat memprediksi genre sebuah lagu.

Secara tradisional, penentuan genre musik bisa bersifat subjektif dan membutuhkan pendengaran manual oleh ahli, yang tidak praktis untuk volume musik yang terus bertambah. _Machine learning_ menawarkan pendekatan yang objektif dan dapat diskalakan untuk mengklasifikasikan genre berdasarkan karakteristik audio intrinsik dari musik tersebut. Proyek ini bertujuan untuk mengembangkan dan mengevaluasi model _machine learning_ untuk klasifikasi genre musik menggunakan dataset lagu dari Spotify, dengan fokus pada fitur-fitur audio yang disediakan. Pentingnya analisis fitur audio untuk memahami karakteristik musik juga ditekankan dalam penelitian terkait, seperti analisis fitur akustik untuk prediksi popularitas lagu.

## Business Understanding

Nilai bisnis utama dari proyek klasifikasi genre musik terletak pada peningkatan platform musik (seperti Spotify, Apple Music), layanan rekomendasi, dan industri terkait musik lainnya. Dengan klasifikasi genre yang akurat, platform dapat meningkatkan personalisasi, memfasilitasi penemuan musik bagi pengguna, dan membantu dalam penargetan konten atau iklan.

### Problem Statements

Berdasarkan latar belakang tersebut, rincian permasalahan yang akan dibahas pada proyek ini adalah:

1.  Bagaimana cara layanan _streaming_ musik dapat secara otomatis dan akurat mengkategorikan koleksi lagu yang terus berkembang ke dalam genre masing-masing menggunakan fitur audio?
2.  Fitur audio spesifik mana (`danceability`, `energy`, `instrumentalness`, dsb) yang paling diskriminatif untuk membedakan antar genre musik yang berbeda?
3.  Pendekatan _machine learning_ mana (Random Forest vs. Neural Network) yang paling efektif untuk membangun sistem klasifikasi genre yang tangguh?

### Goals

Berdasarkan _problem statements_, berikut adalah tujuan yang ingin dicapai dalam proyek ini:

1.  Mengembangkan model prediktif yang mampu mengklasifikasikan genre musik dengan akurasi tinggi menggunakan fitur audio lagu.
2.  Mengidentifikasi dan memahami tingkat kepentingan berbagai fitur audio dalam proses klasifikasi genre.
3.  Membandingkan kinerja model Random Forest dan Neural Network untuk memilih algoritma yang sesuai untuk tugas klasifikasi ini.

### Solution Statement

Untuk mencapai tujuan-tujuan di atas, solusi yang diajukan adalah sebagai berikut:

1.  Melakukan pra-pemrosesan pada dataset lagu Spotify yang mencakup pembersihan data, _feature engineering_ (pembuatan fitur interaksi seperti `energy_danceability`), dan penskalaan fitur.
2.  Mengembangkan dua model klasifikasi yang berbeda: sebuah _Random Forest classifier_ dan sebuah _Neural Network_ (Jaringan Saraf Tiruan) _deep learning_.
3.  Melatih dan mengevaluasi kedua model menggunakan metrik seperti akurasi, F1-_score_, dan _confusion matrix_ untuk menentukan model yang paling efektif untuk klasifikasi genre musik.

## Data Understanding

Dataset yang digunakan dalam proyek ini adalah "Dataset of Songs in Spotify" yang diperoleh dari Kaggle. Dataset ini berisi berbagai fitur audio untuk sejumlah besar lagu.

- **Sumber Data:** [https://www.kaggle.com/datasets/mrmorj/dataset-of-songs-in-spotify/data](https://www.kaggle.com/datasets/mrmorj/dataset-of-songs-in-spotify/data)
- **Informasi Awal Dataset:**
  - Dataset awal `genres_v2.csv` memiliki 42305 baris dan 22 kolom.
  - Fitur-fitur mencakup aspek numerik dari audio dan informasi kategorikal tentang lagu dan genre.

### Deskripsi Variabel

Setelah pembersihan awal dan pemilihan fitur, variabel-variabel utama yang digunakan untuk pemodelan adalah sebagai berikut:

| Variabel           | Keterangan                                                                                                                                                                                                                                                                                |
| ------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `danceability`     | Menggambarkan seberapa cocok sebuah lagu untuk menari berdasarkan kombinasi elemen musik termasuk tempo, stabilitas ritme, kekuatan ketukan, dan keteraturan keseluruhan. Nilai 0.0 paling tidak bisa ditarikan dan 1.0 paling bisa ditarikan.                                            |
| `energy`           | Merupakan ukuran persepsi dari intensitas dan aktivitas. Biasanya, trek yang energik terasa cepat, keras, dan berisik. Nilai dari 0.0 hingga 1.0.                                                                                                                                         |
| `key`              | Kunci keseluruhan dari trek. Direpresentasikan dalam notasi Pitch Class standar (misalnya 0 = C, 1 = C♯/D♭, 2 = D, dst.).                                                                                                                                                                 |
| `loudness`         | Kekerasan keseluruhan sebuah trek dalam desibel (dB). Nilai biasanya berkisar antara -60 dan 0 db.                                                                                                                                                                                        |
| `mode`             | Mode (mayor atau minor) dari sebuah trek. Mayor direpresentasikan oleh 1 dan minor oleh 0.                                                                                                                                                                                                |
| `speechiness`      | Mendeteksi keberadaan kata-kata yang diucapkan dalam sebuah trek. Semakin eksklusif rekaman itu mirip ucapan (misalnya acara bincang-bincang, buku audio, puisi), semakin mendekati 1.0 nilai atributnya.                                                                                 |
| `acousticness`     | Ukuran kepercayaan dari 0.0 hingga 1.0 apakah trek tersebut akustik.                                                                                                                                                                                                                      |
| `instrumentalness` | Memprediksi apakah sebuah trek tidak mengandung vokal. Suara "Ooh" dan "aah" dianggap instrumental dalam konteks ini. Semakin mendekati 1.0 nilai instrumentalness, semakin besar kemungkinan trek tersebut tidak mengandung konten vokal.                                                |
| `liveness`         | Mendeteksi keberadaan audiens dalam rekaman. Nilai liveness yang lebih tinggi menunjukkan kemungkinan yang meningkat bahwa trek tersebut dibawakan secara langsung.                                                                                                                       |
| `valence`          | Ukuran dari 0.0 hingga 1.0 yang menggambarkan positivitas musik yang disampaikan oleh sebuah trek. Trek dengan valensi tinggi terdengar lebih positif (misalnya bahagia, ceria, euforia), sedangkan trek dengan valensi rendah terdengar lebih negatif (misalnya sedih, tertekan, marah). |
| `tempo`            | Perkiraan tempo keseluruhan sebuah trek dalam ketukan per menit (BPM).                                                                                                                                                                                                                    |
| `genre`            | Genre musik dari lagu tersebut (variabel target).                                                                                                                                                                                                                                         |

### Exploratory Data Analysis (EDA)

Tahapan EDA yang dilakukan meliputi:

1.  **Pemeriksaan Missing Values:** Dataset diperiksa untuk nilai null, dan tidak ditemukan nilai null yang signifikan yang memerlukan imputasi kompleks setelah penghapusan kolom awal.

```python
data.isnull().sum()
```

2.  **Penghapusan Kolom Tidak Relevan:** Kolom seperti `type`, `id`, `uri`, `track_href`, `analysis_url`, `song_name`, `Unnamed: 0`, `title`, `duration_ms`, dan `time_signature` dihapus karena tidak relevan untuk pemodelan fitur audio.

```python
columns_to_drop = ["type", "id", "uri", "track_href", "analysis_url",
                  "song_name", "Unnamed: 0", "title", "duration_ms", "time_signature"]
df = data.drop(columns_to_drop, axis=1)
```

3.  **Penanganan Ketidakseimbangan Kelas dan Konsolidasi Genre:**
    - Genre "Pop" dihapus karena jumlahnya yang dominan dapat menyebabkan bias pada model dan sifatnya yang terlalu luas dapat mengganggu pembelajaran fitur pembeda genre lain yang lebih spesifik.
    - Beberapa sub-genre seperti "Trap Metal", "Underground Rap", "Emo", "RnB", "Hiphop", dan "Dark Trap" dipetakan ke dalam genre "Rap" yang lebih umum untuk menyederhanakan dan mengkonsolidasikan kelas.
    - Dilakukan _downsampling_ pada genre "Rap" dengan mengambil sampel acak sebanyak 3000 lagu untuk menyeimbangkan distribusi kelas dengan genre lainnya.

```python
df = df[df['genre'] != "Pop"].reset_index(drop=True)
```

```python
genre_mapping = {
    "Trap Metal": "Rap",
    "Underground Rap": "Rap",
    "Emo": "Rap",
    "RnB": "Rap",
    "Hiphop": "Rap",
    "Dark Trap": "Rap"
}
df['genre'] = df['genre'].replace(genre_mapping)
```

```python
rap_genre = df[df['genre'] == 'Rap'].sample(3000, random_state=42)
other_genre = df[df['genre'] != 'Rap']
df = pd.concat([rap_genre, other_genre]).reset_index(drop=True)
```

4.  **Feature Engineering:** Empat fitur baru dibuat dengan mengalikan pasangan fitur yang sudah ada untuk menangkap interaksi potensial:

```python
df['energy_danceability'] = df['energy'] * df['danceability']
df['acousticness_instrumentalness'] = df['acousticness'] * df['instrumentalness']
df['loudness_speechiness'] = df['loudness'] * df['speechiness']
df['tempo_energy'] = df['tempo'] * df['energy']
```

5.  **Visualisasi Distribusi Fitur:**

    - Histogram digunakan untuk melihat distribusi masing-masing fitur numerik. Banyak fitur seperti `speechiness`, `acousticness`, dan `instrumentalness` menunjukkan distribusi yang condong ke kanan (\*right-skewed\*).
      ![histogram](https://github.com/user-attachments/assets/8faea733-d84a-40db-985d-dfc60f234d29)

    - Distribusi genre setelah penyesuaian menunjukkan jumlah sampel yang relatif seimbang per genre.

    ![distribution](https://github.com/user-attachments/assets/3477e03b-2b4b-4b8a-909e-b111add641ad)

    - _Box plot_ digunakan untuk membandingkan distribusi fitur audio antar genre, menunjukkan pola yang berbeda untuk fitur seperti `instrumentalness` dan `speechiness`.

    ![boxplot](https://github.com/user-attachments/assets/b8ef4f03-ad13-4d33-a4bc-3d17597e0c8b)

6.  **Analisis Korelasi:**

    - _Pair plot_ digunakan untuk memvisualisasikan hubungan antar fitur kunci (`danceability`, `energy`, `loudness`, `speechiness`) dan bagaimana mereka berinteraksi per genre. `speechiness` tampak menjadi pembeda kuat untuk genre Rap dan Trap.
      ![pair-plot](https://github.com/user-attachments/assets/c2d802a2-4361-4cab-b8cd-9b9857393cfd)

    - _Heatmap_ korelasi menunjukkan hubungan linear antar semua fitur numerik.

    ![heatmap](https://github.com/user-attachments/assets/66a6eb2d-0c79-43eb-8f75-a3568b9ff718)

    - Diagram batang korelasi fitur dengan variabel target (genre yang telah di-encode) menunjukkan `tempo` memiliki korelasi negatif terkuat, sementara `energy` memiliki korelasi positif terkuat.

    ![diagram](https://github.com/user-attachments/assets/288ae65b-3ff5-4f60-8992-804f4be55c59)

## Data Preparation

Setelah EDA, beberapa langkah persiapan data dilakukan untuk menyiapkan data untuk pemodelan:

1.  **Pemisahan Fitur dan Target:** Dataset dibagi menjadi matriks fitur `X` (semua kolom kecuali `genre`) dan vektor target `y` (kolom `genre`).

```python
X = df.drop('genre', axis=1)
y = df['genre']
```

2.  **Pembagian Data Latih dan Uji (_Train-Test Split_):** Data `X` dan `y` dibagi menjadi set pelatihan (80%) dan set pengujian (20%) menggunakan `train_test_split` dengan `random_state=42` untuk reproduktifitas.
    - Alasan: Pembagian ini penting untuk mengevaluasi kinerja model pada data yang tidak terlihat selama pelatihan, sehingga memberikan estimasi yang lebih realistis tentang bagaimana model akan berkinerja pada data baru.

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

3.  **Penskalaan Fitur (_Feature Scaling_):** Fitur numerik pada data latih dan uji diskalakan menggunakan `MinMaxScaler`. Scaler ini di-_fit_ hanya pada data latih (`X_train_scaled`) dan kemudian digunakan untuk mentransformasi data latih dan data uji (`X_test_scaled`).
    - Alasan: Penskalaan membawa semua fitur ke rentang yang sama (0 hingga 1), yang penting untuk algoritma yang sensitif terhadap skala fitur seperti Neural Networks (terutama dengan regularisasi atau optimasi berbasis gradien) dan juga dapat membantu kinerja Random Forest dalam beberapa kasus. Ini mencegah fitur dengan nilai absolut yang lebih besar mendominasi proses pembelajaran.

```python
scaler = MinMaxScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)
```

4.  **Encoding Label Target (_Label Encoding_):** Label genre tekstual pada variabel target `y_train` dan `y_test` dikonversi menjadi representasi numerik menggunakan `LabelEncoder`. Encoder di-_fit_ pada `y_train` dan digunakan untuk mentransformasi `y_train` dan `y_test`.
    - Alasan: Algoritma _machine learning_ umumnya memerlukan input numerik, sehingga label kategori perlu diubah menjadi angka.

```python
le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)
y_test_encoded = le.transform(y_test)
```

## Modeling

Dua model _machine learning_ dikembangkan dan dievaluasi untuk tugas klasifikasi genre musik ini.

### 1. Model Random Forest

Random Forest adalah algoritma _ensemble learning_ yang membangun banyak pohon keputusan selama pelatihan dan mengeluarkan kelas yang merupakan modus dari kelas-kelas (klasifikasi) atau prediksi rata-rata (regresi) dari masing-masing pohon.

- **Tahapan dan Parameter:**
  - Model diinisialisasi menggunakan `RandomForestClassifier` dari `sklearn.ensemble`.
  - Parameter yang digunakan:
    - `n_estimators=200`: Jumlah pohon keputusan dalam hutan. Nilai yang lebih tinggi umumnya meningkatkan kinerja tetapi juga waktu pelatihan.
    - `max_depth=20`: Kedalaman maksimum setiap pohon. Mengontrol kompleksitas pohon individu dan membantu mencegah _overfitting_.
    - `random_state=42`: Untuk reproduktifitas hasil.
  - Model dilatih menggunakan `X_train_scaled` dan `y_train_encoded`.
- **Kelebihan:**
  - Cukup robust terhadap _outliers_ dan data non-linear.
  - Kurang rentan terhadap _overfitting_ dibandingkan metode _Decision Tree_.
  - Dapat memberikan estimasi pentingnya fitur.
  - Efektif pada dataset dengan dimensi tinggi.
- **Kekurangan:**
  - Bisa menjadi _black box_; lebih sulit diinterpretasikan dibandingkan pohon keputusan tunggal.
  - Membutuhkan lebih banyak sumber daya komputasi (waktu dan memori) untuk pelatihan dibandingkan algoritma yang lebih sederhana, terutama dengan banyak pohon.

### 2. Model Neural Network (Jaringan Saraf Tiruan)

Neural Network adalah model yang terinspirasi oleh struktur otak manusia, terdiri dari lapisan-lapisan neuron yang saling terhubung. Model ini sangat baik dalam menangkap pola non-linear yang kompleks dalam data.

- **Tahapan dan Parameter:**
  - Model sekuensial dibangun menggunakan `keras.Sequential`.
  - Arsitektur model:
    - `InputLayer`: Bentuk input sesuai dengan jumlah fitur pada `X_train_scaled`.
    - `Dense(512, activation="relu")`, diikuti `BatchNormalization()`, `Dropout(0.3)`
    - `Dense(256, activation="relu")`, diikuti `BatchNormalization()`, `Dropout(0.3)`
    - `Dense(128, activation="relu")`, diikuti `BatchNormalization()`, `Dropout(0.2)`
    - `Dense(len(np.unique(y_train_encoded)), activation="softmax")`: Lapisan output dengan aktivasi softmax untuk klasifikasi multi-kelas.
  - _Optimizer_: `keras.optimizers.Adam(learning_rate=0.0003)`.
  - _Loss Function_: `"sparse_categorical_crossentropy"` (sesuai untuk target integer).
  - _Metrics_: `["accuracy"]`.
  - _Callbacks_: Dua `EarlyStopping` callback digunakan, satu memantau `val_loss` dan satu lagi `val_accuracy`, keduanya dengan `patience=15` dan `restore_best_weights=True`.
  - Model dilatih menggunakan `X_train_scaled` dan `y_train_encoded` dengan `epochs=150` dan `batch_size=64`, serta data validasi `(X_test_scaled, y_test_encoded)`.
- **Kelebihan:**
  - Kemampuan untuk mempelajari representasi fitur yang kompleks dan hubungan non-linear.
  - Dapat mencapai kinerja _state-of-the-art_ pada banyak tugas jika data cukup dan arsitektur dirancang dengan baik.
  - Fleksibel dalam hal arsitektur.
- **Kekurangan:**
  - Membutuhkan dataset yang besar untuk kinerja yang baik dan menghindari _overfitting_.
  - Secara komputasi mahal untuk dilatih, terutama untuk jaringan yang dalam dan besar.
  - Sering dianggap sebagai "black box" karena sulit untuk menginterpretasikan keputusan internalnya.
  - Sensitif terhadap pemilihan arsitektur dan _hyperparameter_.

![neural-network](https://github.com/user-attachments/assets/17d24624-bce5-4860-bd1a-662438c6547c)

> model.summary()

## Evaluation

Evaluasi model dilakukan untuk membandingkan kinerja Random Forest dan Neural Network dalam mengklasifikasikan genre musik. Metrik utama yang digunakan adalah Akurasi dan F1-Score (tertimbang/weighted).

### Penjelasan Metrik Evaluasi

- **Akurasi (_Accuracy_):** Mengukur proporsi prediksi yang benar dari total prediksi. Dihitung sebagai:
  <br>
  <br>
  <br>
  $$\text{Accuracy} = \frac{\text{TP} + \text{TN}}{\text{TP} + \text{TN} + \text{FP} + \text{FN}}$$
  <br>
  <br>
  <br>
  Dimana TP = True Positives, TN = True Negatives, FP = False Positives, FN = False Negatives. Metrik ini baik ketika kelas-kelas seimbang.
- **F1-Score:** Merupakan rata-rata harmonik dari _Precision_ dan _Recall_. Ini adalah metrik yang baik ketika ada ketidakseimbangan kelas atau ketika penting untuk menyeimbangkan antara _Precision_ dan _Recall_. F1-score tertimbang menghitung metrik untuk setiap label, dan menemukan rata-rata mereka yang ditimbang oleh dukungan (jumlah instance sebenarnya untuk setiap label).
  <br>
  <br>
  <br>
  $$\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}$$
  $$\text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}$$
  $$\text{F1-Score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$
  <br>
  <br>
  <br>
- **Confusion Matrix:** Tabel yang memvisualisasikan kinerja model klasifikasi. Baris mewakili kelas aktual dan kolom mewakili kelas yang diprediksi. Ini memberikan detail tentang kesalahan klasifikasi antar kelas.

### Hasil Evaluasi Model

Berdasarkan eksekusi kode:

**1. Random Forest Results:**

- **Accuracy:** Sekitar 0.8949 (atau 89.49%)
- **F1 Score (weighted):** Sekitar 0.8947
- **Classification Report:** Menunjukkan presisi, recall, dan f1-score untuk setiap kelas. Model ini secara umum menunjukkan kinerja yang baik di sebagian besar kelas.
- **Confusion Matrix:** Diagonal pada _confusion matrix_ menunjukkan jumlah prediksi yang benar untuk setiap kelas. Random Forest menunjukkan nilai diagonal yang umumnya tinggi.

![rf](https://github.com/user-attachments/assets/00cee93d-1f1d-4fcc-b5a7-fcbbed06eeef)

> Classification Report Random Forest

**2. Neural Network Results:**

- **Accuracy:** Sekitar 0.8814 (atau 88.14%)
- **F1 Score (weighted):** Sekitar 0.8810
- **Classification Report:** Juga memberikan detail per kelas.
- **Confusion Matrix:** Neural Network juga menunjukkan performa yang baik, meskipun sedikit di bawah Random Forest pada metrik keseluruhan. Beberapa kelas mungkin menunjukkan lebih banyak misklasifikasi dibandingkan dengan Random Forest.
- **Training History Plots (Accuracy & Loss):** Plot akurasi pelatihan dan validasi menunjukkan bagaimana akurasi model berubah selama epoch. Idealnya, akurasi validasi akan meningkat dan stabil. Plot kerugian (_loss_) pelatihan dan validasi menunjukkan bagaimana kesalahan model berkurang. _Early stopping_ membantu mencegah _overfitting_ dengan menghentikan pelatihan ketika metrik validasi tidak lagi membaik.

![nn](https://github.com/user-attachments/assets/251a6397-e26b-4d70-914e-b45e8d6c2395)

> Classification Report Neural Network

![training-plot](https://github.com/user-attachments/assets/96647297-b33c-487e-9079-ca92564fcc82)

> Training History Plot

![cf-matrix](https://github.com/user-attachments/assets/d3426e33-a1d2-40ba-b3bf-bf96883d643c)

> Confusion Matrix

### Perbandingan Model dan Pemilihan Model Terbaik

Berdasarkan ringkasan perbandingan model yang disediakan dalam kode:

| Model          | Accuracy | F1 Score |
| -------------- | -------- | -------- |
| Random Forest  | 0.894913 | 0.894704 |
| Neural Network | 0.881431 | 0.881024 |

- **Random Forest** menunjukkan kinerja yang sedikit lebih baik daripada Neural Network dalam hal Akurasi dan F1-Score pada dataset ini.

- Analisis _confusion matrix_ juga menunjukkan bahwa Random Forest memiliki jumlah prediksi benar yang umumnya lebih tinggi dan lebih sedikit kesalahan klasifikasi antar kelas dibandingkan Neural Network untuk dataset ini. **Oleh karena itu, model Random Forest dipilih sebagai model terbaik** untuk tugas klasifikasi genre musik ini berdasarkan metrik evaluasi yang lebih unggul dan konsistensi klasifikasi di berbagai kelas.

## Kesimpulan

Proyek ini berhasil mengembangkan dan mengevaluasi dua model _machine learning_ untuk klasifikasi genre musik Spotify berdasarkan fitur audio.

1. Proses _Exploratory Data Analysis_ (EDA) dan _Data Preparation_ sangat penting, termasuk penanganan ketidakseimbangan kelas, _feature engineering_, dan penskalaan fitur, untuk menyiapkan data yang optimal untuk pemodelan.
2. Model Random Forest mencapai akurasi sekitar 89.5% dan F1-score sekitar 89.5%.
3. Model Neural Network mencapai akurasi sekitar 88.1% dan F1-score sekitar 88.1%.
4. Berdasarkan perbandingan metrik evaluasi (Akurasi dan F1-Score) serta analisis _confusion matrix_, model **Random Forest** menunjukkan kinerja yang sedikit lebih unggul dan lebih konsisten dibandingkan Neural Network untuk dataset dan konfigurasi yang digunakan dalam proyek ini.
5. Fitur audio seperti `speechiness`, `instrumentalness`, `energy`, dan `tempo` terbukti memiliki peran penting dalam membedakan genre musik, seperti yang terindikasi selama EDA.

Proyek ini menunjukkan kelayakan penggunaan _machine learning_ untuk klasifikasi genre musik secara otomatis, yang dapat memberikan manfaat signifikan bagi platform musik dan pengguna. Pengembangan lebih lanjut dapat mencakup eksplorasi fitur yang lebih canggih (misalnya, dari analisis spektral), _hyperparameter tuning_ yang lebih ekstensif, atau penggunaan arsitektur Neural Network yang lebih kompleks.

## Referensi

1.  Rizqi, M. F., & Nabila, S. A. (2023). Aplikasi Data Mining untuk Pengelompokan Pola Manipulasi Akuntansi Menggunakan Algoritma K-Means Clustering (Studi Kasus pada Perusahaan Manufaktur yang Terdaftar di Bursa Efek Indonesia Periode 2016-2018). _Indonesian Journal on Advanced Technology and Industrial System (IJATIS)_, _5_(1), 1-11. Diperoleh dari [https://journal.irpi.or.id/index.php/ijatis/article/view/1123/921](https://journal.irpi.or.id/index.php/ijatis/article/view/1123/921)
2.  Sijbesma, D. (2024). _The Sound of Success? Predicting a song's popularity using acoustic features and artist familiarity from the Spotify API_ (Master's Thesis). Utrecht University, Department of Information and Computing Sciences. Diperoleh dari [https://studenttheses.uu.nl/bitstream/handle/20.500.12932/47974/Thesis%20-%20ADE%20-%20David%20Sijbesma.pdf?sequence=1](https://studenttheses.uu.nl/bitstream/handle/20.500.12932/47974/Thesis%20-%20ADE%20-%20David%20Sijbesma.pdf?sequence=1)
3.  Dicoding Indonesia. (n.d.). _Machine Learning Terapan_. Dicoding Academy. Diakses pada 24 Mei 2025, dari [https://www.dicoding.com/academies/319-machine-learning-terapan](https://www.dicoding.com/academies/319-machine-learning-terapan)
