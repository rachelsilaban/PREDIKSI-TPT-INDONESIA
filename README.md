Prediksi Tingkat Pengangguran Terbuka (TPT) Provinsi Indonesia

Proyek ini merupakan aplikasi berbasis web yang digunakan untuk memprediksi Tingkat Pengangguran Terbuka (TPT) di berbagai provinsi di Indonesia menggunakan metode Long Short-Term Memory (LSTM). Aplikasi ini memanfaatkan data historis TPT yang bersumber dari Badan Pusat Statistik (BPS) Indonesia untuk melakukan prediksi beberapa tahun ke depan (multi-step forecasting). Hasil prediksi disajikan dalam bentuk visualisasi interaktif serta dilengkapi dengan pendekatan Explainable Artificial Intelligence (XAI) untuk meningkatkan transparansi model.

Fitur Utama:
1. Prediksi Multi-Step Menggunakan LSTM
   - Menggunakan model Long Short-Term Memory (LSTM) 
     untuk memprediksi nilai TPT beberapa tahun ke depan secara sekaligus (multi-step forecasting).
   - Memungkinkan analisis tren TPT jangka panjang 
     berdasarkan data historis.

2. Explainable Artificial Intelligence (XAI) Menggunakan SHAP
   - Menjelaskan hasil prediksi model LSTM menggunakan metode SHAP.
   - Menunjukkan kontribusi data TPT periode sebelumnya terhadap hasil ]
     prediksi.
   - Membantu meningkatkan transparansi dan pemahaman terhadap hasil 
     model.
   - Mengurangi sifat black box pada model deep learning.

Dataset
Nama file: dataset.csv
Sumber data: Badan Pusat Statistik (BPS) Indonesia
Dataset berisi data historis Tingkat Pengangguran Terbuka (TPT) per provinsi di Indonesia yang digunakan sebagai dasar pemodelan dan prediksi.

Menjalankan Aplikasi
1. Jalankan Flask:
   python app.py
2. Buka browser:
   http://127.0.0.1:5000
3. Pilih provinsi, masukkan jumlah tahun ke depan (multi-step), pilih fitur, lalu klik Submit.
4. Hasil prediksi TPT dan chart akan ditampilkan di halaman web.

Teknologi yang Digunakan
1. Python
2. Flask
3. TensorFlow / Keras (LSTM)
4. Pandas & NumPy
5. Matplotlib
6. SHAP