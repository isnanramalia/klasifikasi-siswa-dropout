# Klasifikasi Faktor Siswa Keluar

## Deskripsi Proyek
Proyek ini adalah eksperimen untuk mengklasifikasikan faktor-faktor yang menyebabkan siswa keluar dari SMK Askhabul Kahfi Semarang. Saya menggunakan tiga teknik klasifikasi berbeda, yaitu Decision Tree, K-Nearest Neighbors (KNN), dan Naive Bayes, untuk mengidentifikasi faktor-faktor tersebut.

## Deployment
[smkaskha.streamlit.app](https://smkaskha.streamlit.app/)

## Eksperimen

### Decision Tree
Dalam eksperimen kami dengan Decision Tree, kami melakukan langkah-langkah berikut:
1. **Persiapan Data**: Data yang digunakan adalah dataset yang berisi atribut-atribut siswa dan label "Keluar Karena."
2. **Pelatihan Model**: Kami melatih model Decision Tree menggunakan data pelatihan.
3. **Evaluasi Model**: Setelah melatih model, kami menguji performa model menggunakan data pengujian.
4. **Hasil**: Model Decision Tree mencapai akurasi sekitar 70% dalam mengklasifikasikan faktor siswa keluar.

### K-Nearest Neighbors (KNN) dan Naive Bayes
Saya juga melakukan eksperimen dengan teknik KNN dan Naive Bayes, meskipun hasil utama didasarkan pada eksperimen dengan Decision Tree.

## Cara Menjalankan Kode
Anda dapat menemukan kode eksperimen di repositori ini. Untuk menjalankan kode, Anda memerlukan lingkungan Python yang sesuai dengan pustaka-pustaka seperti scikit-learn.

1. Klon repository ini ke komputer Anda.
2. Instal pustaka yang diperlukan dengan menjalankan `pip install -r requirements.txt`.
3. Jalankan kode sesuai dengan instruksi yang ada dalam masing-masing direktori eksperimen.


## Kontribusi
Anda tidak dipersilahkan dan sangat dilarang menggunakan dataset atau semua file didalam repository ini untuk kepentingan publik yang dapat menyebabkan data SMK Askhabul Kahfi tersebar.
