from sklearn import tree
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
import seaborn as sns
import numpy as np
from io import BytesIO
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# Fungsi Persiapan Data


def prepare_data():
    # Load data
    data = pd.read_csv("data_numerik.csv")
    dataFinal = pd.read_csv("data_smk_askha_final.csv")

    # Split data menjadi data latih dan data uji
    x = data[["JK", "Tempat Lahir", "Asal Kecamatan", "Jenis Tinggal", "Alat Transportasi", "Jenjang Pendidikan Ayh", "Pekerjaan Ayh", "Penghasilan Ayh", "Pekerjaan Ibu", "Penghasilan Ibu"]]
    y = data["Keluar Karena"]
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42)

    return data, dataFinal, x_train, x_test, y_train, y_test

# Fungsi Membuat Model
    
def create_model(x_train, y_train):
    dt_clf = DecisionTreeClassifier(max_depth=4, criterion="entropy")
    dt_clf = dt_clf.fit(x_train, y_train)
    return dt_clf

# Fungsi Evaluasi Performa Model


def evaluate_model(y_test, y_pred):
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    return accuracy, f1, precision, recall

# Fungsi plotting decision tree


def plot_decision_tree(dt_clf, feature_names, class_names):
    fig, ax = plt.subplots(figsize=(80, 50))
    tree.plot_tree(dt_clf,
                   feature_names=feature_names,
                   class_names=class_names,
                   filled=True,
                   ax=ax)
    return fig

# Fungsi decision tree


def display_decision_tree(dt_clf, feature_names, class_names):
    st.subheader("Decision Tree")
    st.pyplot(plot_decision_tree(dt_clf, feature_names, class_names))
    from sklearn.tree import export_text
    
    with st.expander(label='Lihat Penjelasan'):
        # Menyimpan aturan-aturan dari decision tree dalam sebuah variabel
        rules = export_text(dt_clf, feature_names=feature_names, spacing=3, decimals=1, show_weights=True, max_depth=4) 

        # Membersihkan dan memformat aturan-aturan ke dalam list yang lebih terstruktur
        cleaned_rules = [line.strip() for line in rules.split("\n")]  # Memisahkan setiap baris aturan dan membersihkan spasi

        # Menampilkan aturan-aturan yang telah dirapikan menggunakan st.write di Streamlit
        st.write("### Aturan-aturan dari Decision Tree")
        st.write(cleaned_rules)


# Fungsi diagram batang


def plot_diagram(dataFinal):
    # Visualisasi variabel JK
    ax1 = sns.displot(dataFinal, x="JK", height=3, aspect=2)
    # Visualisasi variabel Tempat Lahir
    ax2 = sns.displot(dataFinal, x="Templat Lahir", height=3, aspect=2)
    # Visualisasi variabel Asal Kecamatan
    ax3 = sns.displot(dataFinal, x="Asal Kecamatan", height=3, aspect=2)
    # Visualisasi variabel Jenis Tinggal
    ax4 = sns.displot(dataFinal, x="Jenis Tinggal", height=3, aspect=2)
    # Visualisasi variabel Alat Transportasi
    ax5 = sns.displot(dataFinal, x="Alat Transportasi", height=3, aspect=2)
    # Visualisasi variabel Jenjang Pendidikan Ayh
    ax6 = sns.displot(dataFinal, x="Jenjang Pendidikan Ayh", height=3, aspect=2)
    # Visualisasi variabel Pekerjaan Ayh
    ax7 = sns.displot(dataFinal, x="Pekerjaan Ayh", height=3, aspect=2)
    # Visualisasi variabel Penghasilan Ayh
    ax8 = sns.displot(dataFinal, x="Penghasilan Ayh", height=3, aspect=2)
    # Visualisasi variabel Pekerjaan Ibu
    ax9 = sns.displot(dataFinal, x="Pekerjaan Ibu", height=3, aspect=2)
    # Visualisasi variabel Penghasilan Ibu
    ax10 = sns.displot(dataFinal, x="Penghasilan Ibu", height=3, aspect=2)
    return ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10

# Fungsi pie chart


def plot_pie(dataFinal, column):
    fig, ax = plt.subplots(figsize=(10, 10))
    dataFinal[column].value_counts().plot.pie(autopct='%1.1f%%', ax=ax)
    return fig


# Fungsi untuk Persebaran Data dengan Diagram Batang
def display_bar_chart(dataFinal):
    option = st.selectbox("Pilih Variabel:", list(dataFinal.columns))
    if option:
        ax = sns.displot(dataFinal, x=option, height=3, aspect=2)
        st.pyplot(ax)

# Fungsi untuk Persebaran Data dengan Pie Chart


def display_pie_chart(dataFinal):
    option = st.selectbox("Pilih Variabel:", list(dataFinal.columns))
    if option:
        fig, ax = plt.subplots(figsize=(10, 10))
        dataFinal[option].value_counts().plot.pie(autopct='%1.1f%%', ax=ax)
        st.pyplot(fig)

# Fungsi untuk Persebaran Data dengan Scatter Plot


def display_scatter_plot(dataFinal):
    option_x = st.selectbox(
        "Pilih Variabel untuk sumbu X:", list(dataFinal.columns))
    option_y = st.selectbox(
        "Pilih Variabel untuk sumbu Y:", list(dataFinal.columns))

    if option_x and option_y and option_x != option_y:
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=option_x, y=option_y, data=dataFinal)
        plt.xlabel(option_x)
        plt.ylabel(option_y)
        plt.title(f"Scatter Plot: {option_x} vs {option_y}")
        st.pyplot()
    else:
        st.write("Pilih dua variabel yang berbeda untuk sumbu X dan Y.")

# Fungsi untuk Korelasi Data


def display_correlation(data):
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.subheader("Korelasi antar variabel:")
    plt.figure(figsize=(15, 15))
    numerical_data = data.select_dtypes(include=[np.number])
    sns.heatmap(numerical_data.corr(), annot=True)
    st.pyplot()
    with st.expander(label='Lihat Penjelasan'):
        # Judul
        st.write("### Cara Membaca Heatmap")

        # Penjelasan Sumbu
        st.write("#### 1. Sumbu X dan Y:")
        st.write("- Sumbu X dan Y menunjukkan label atau nama variabel pada masing-masing dimensi.")
        st.write("- Nilai-nilai pada sumbu X dan Y menunjukkan kategori atau rentang dari data yang diamati.")

        # Penjelasan Warna
        st.write("#### 2. Warna:")
        st.write("- Setiap sel dalam heatmap direpresentasikan oleh warna tertentu.")
        st.write("- Warna-warna yang lebih gelap atau terang menggambarkan nilai yang lebih rendah atau tinggi, semakin terang warnanya maka semakin tinggi korelasinya.")
        
# Fungsi untuk Seluruh Data dengan Tabel


def display_full_data(dataFinal):
    st.write("Seluruh Data:")
    st.write(dataFinal)

# Fungsi untuk Seluruh Data Numerik dengan Tabel


def display_numeric_data(data):
    st.write("Seluruh Data Numerik:")
    st.write(data)

# Fungsi faktor penyebab mutasi


def display_faktor():
    st.subheader("Faktor Penyebab Mutasi")
    st.write("""
        1. Jenis Kelamin
        2. Jenjang Pendidikan Ayah
        3. Pekerjaan Ayah
        4. Penghasilan Ayah
        5. Pekerjaan Ibu
        6. Penghasilan Ibu            
             """)

# Fungsi prediksi siswa mutasi


def display_prediksi(dt_clf, data):
    st.subheader("Prediksi Siswa Mutasi")
    st.write("Masukkan data siswa yang ingin diprediksi:")

    jk_map = {0: 'L', 1: 'P'}
    tempatLahir_map = {0: 'Semarang', 1: 'Tegal', 2: 'Kendal', 3: 'Demak', 4: 'Bekasi', 5: 'Cirebon', 6: 'Brebes', 7: 'Purworejo', 8: 'Pemalang', 9: '(Tidak Diisi)', 10: 'Grobogan', 11: 'Cilacap', 12: 'Wonosobo', 13: 'Kudus', 14: 'Jakarta', 15: 'Palangka Raya', 16: 'Magelang', 17: 'Tata Karya', 18: 'Kotawaringin Barat', 19: 'Kebumen', 20: 'Sukoharjo', 21: 'Gunungkidul', 22: 'Batam', 23: 'Telaga Dua', 24: 'Bumi Dipasena Agung', 25: 'Kantan Muara', 26: 'Babat Terawas', 27: 'Depok', 28: 'Klaten', 29: 'Banjarnegara', 30: 'Batam, Kota Batam', 31: 'Kertosono', 32: 'Temanggung', 33: 'Purwosari', 34: 'Batu', 35: 'Sumaja Makmur', 36: 'Panta Dewa', 37: 'Batang Malas', 38: 'Simpang Pesak', 39: 'Bengkulu', 40: 'Tangerang', 41: 'Seruyan', 42: 'Batang', 43: 'Boyolali', 44: 'Pati', 45: 'Lamandau', 46: 'Timber Riau', 47: 'Bangun Jaya', 48: 'Bogor', 49: 'Jepara', 50: 'Indramayu', 51: 'Curup', 52: 'Kota Cirebon', 53: 'Banyumas', 54: 'Sragen', 55: 'Maumere', 56: 'Sukamulya', 57: 'Pangkalan Lada', 58: 'Wonogiri', 59: 'Pamekasan', 60: 'Bangkalan', 61: 'Kediri', 62: 'Nabire', 63: 'Kualasimpang', 64: 'Kotabaru', 65: 'Makkah', 66: 'Lubuk Makmur', 67: 'Kupang', 68: 'Lahat', 69: 'Bukit Makmur', 70: 'Mambulau', 71: 'Sukamandang', 72: 'Lubuk Mandarsah', 73: 'Suban', 74: 'Pekalongan', 75: 'Rembang', 76: 'Purbalingga', 77: 'Ketapang', 78: 'Babu Salam, D.I.Aceh', 79: 'Manokwari', 80: 'Kuwaron'}
    asalKec_map = {0: 'Cilacap Tengah', 1: 'Mijen', 2: 'Ngaliyan', 3: 'Balapulang', 4: 'Semarang', 5: 'Gunung Pati', 6: 'Kaliwungu', 7: 'Boja', 8: 'Singorojo', 9: 'Bonang', 10: 'Bergas', 11: 'Cakung', 12: 'Patean', 13: 'Plumbon', 14: 'Pedurungan', 15: 'Nabire', 16: 'Ungaran Timur', 17: 'Gajah Mungkur', 18: 'Pasigitan', 19: 'Sijeruk', 20: 'Karangjati', 21: 'Pakintelan', 22: 'Podorejo', 23: 'Kalisidi', 24: 'Bener', 25: 'Bantarbolang', 26: 'Kaliwungu Selatan', 27: 'Arut Utara', 28: 'Semarang Utara', 29: 'Tembalang', 30: 'Mranggen', 31: 'Cening', 32: 'Penawangan', 33: 'Ungaran Barat', 34: 'Limbangan', 35: 'Bandungan', 36: 'Karang Tengah', 37: 'Kendal', 38: 'Kebonagung', 39: 'Kebonbatur', 40: 'Pakis', 41: 'Tugu', 42: 'Polaman', 43: 'Warureja', 44: 'Kesugihan', 45: 'Kota Kendal', 46: 'Kejajar', 47: 'Jambu', 48: 'Kaliputih', 49: 'Dawe', 50: 'Pedurungan Kidul', 51: 'Semarang Barat', 52: 'Merbuh', 53: 'Sayung', 54: 'Tuntang', 55: 'Cepiring', 56: 'Patebon', 57: 'Maliku', 58: 'Nanggewer', 59: 'Banyumanik', 60: 'Seputih Agung', 61: 'Kliris', 62: 'Mentobi Raya', 63: 'Petanahan', 64: 'Srondol Kulon', 65: 'Pungangan', 66: 'Ngawen', 67: 'Bancak', 68: 'Losari', 69: 'Batu Ampar', 70: 'Sungai Bengkuang', 71: 'Sawangan', 72: 'Piyanggang', 73: 'Sintang', 74: 'Manyaran', 75: 'Rawajitu Timur', 76: 'Binjai Hulu', 77: 'Kertosari', 78: 'Talang Kelapa', 79: 'Rowosari', 80: 'Margadana', 81: 'Tapos', 82: 'Semarang Tengah', 83: 'Sumowono', 84: 'Danau Seluluk', 85: 'Genuk', 86: 'Tlogosari Kulon', 87: 'Pesantren', 88: 'Kalirejo', 89: 'Penusupan', 90: 'Gonoharjo', 91: 'Sagulung Kota', 92: 'Kebogadung', 93: 'Gayamsari', 94: 'Kesambi', 95: 'Jayaloka', 96: 'Cilacap Selatan', 97: 'Pringapus', 98: 'Ngareanak', 99: 'Candisari', 100: 'Bejen', 101: 'Leyangan', 102: 'Tembok Kidul', 103: 'Karanganyar', 104: 'Bawen', 105: 'Tandang', 106: 'Betung', 107: 'Beji', 108: 'Bayat', 109: 'Kebumen', 110: 'Pondok Labu', 111: 'Bulu Lor', 112: 'Manokwari Barat', 113: 'Brangsong', 114: 'Cangkiran', 115: 'Kedungpani', 116: 'Gunung Megang', 117: 'Talang Ubi', 118: 'Tebing Tinggi Barat', 119: 'Pulosari', 120: 'Pondok Aren', 121: 'Bringin', 122: 'Petarukan', 123: 'Bebengan', 124: 'Suka Maju', 125: 'Pranten', 126: 'Bandar', 127: 'Nusawungu', 128: 'Trisari', 129: 'Semarang Selatan', 130: 'Campurejo', 131: 'Cileunyi', 132: 'Sempu', 133: 'Sukamara', 134: 'Cakung Barat', 135: 'Wonosobo', 136: 'Tlogowungu', 137: 'Gempolsewu', 138: 'Wonorejo', 139: 'Bulik Timur', 140: 'Gebugan', 141: 'Penawar Aji', 142: 'Suradadi', 143: 'Wates', 144: 'Andong', 145: 'Tambusai Utara', 146: 'Pabuaran', 147: 'Metesih', 148: 'Wedung', 149: 'Jatingaleh', 150: 'Kangkung', 151: 'Lohbener', 152: 'Ulujami', 153: 'Arahan', 154: 'Bawang', 155: 'Curup Selatan', 156: 'Seluas', 157: 'Nolokerto', 158: 'Pandansari', 159: 'Bendan Ngisor', 160: 'Weru', 161: 'Gedanganak', 162: 'Kaliangkrik', 163: 'Nogosari', 164: 'Kramat Jati', 165: 'Plantaran', 166: 'Alok Timur', 167: 'Air Sugihan', 168: 'Pangkalan Lada', 169: 'Ngadirojo', 170: 'Pegandon', 171: 'Jimbaran', 172: 'Secang', 173: 'Palengaan', 174: 'Bekasi Utara', 175: 'Babelan', 176: 'Sidomakmur', 177: 'Weleri', 178: 'Duren Jaya', 179: 'Gubug', 180: 'Jlegong', 181: 'Gajah', 182: 'Tengaran', 183: 'Tegal Barat', 184: 'Protomulyo', 185: 'Tampingan', 186: 'Wringin Putih', 187: 'Pondokgede', 188: 'Bongsari', 189: 'Toroh', 190: 'Kota Lintang', 191: 'Banyubiru', 192: 'Jatibarang', 193: 'Candirejo', 194: 'Purwoyoso', 195: 'Bangsri', 196: 'Pedurungan Tengah', 197: 'Gisikdrono', 198: 'Kelumpang Tengah', 199: 'Sambeng', 200: 'Sengon', 201: 'Banguntapan', 202: 'Lebaksiu', 203: 'Lempuing', 204: 'Ngempon', 205: 'Pangkalan Banteng', 206: 'Pakis Aji', 207: 'Tanjungmojo', 208: 'Parakan', 209: 'Kandangan', 210: 'Kedu', 211: 'Lasiana', 212: 'Semarang Timur', 213: 'Karangpakis', 214: 'Jangli', 215: 'Sendangmulyo', 216: 'Purwosari', 217: 'Randudongkal', 218: 'Bekasi Timur', 219: 'Lahat', 220: 'Plantungan', 221: 'Tieng', 222: 'Kebon Jeruk', 223: 'Boyolali', 224: 'Lerep', 225: 'Sungai Bahar', 226: 'Karanggede', 227: 'Seruyan Tengah', 228: 'Ngadikusuman', 229: 'Getas', 230: 'Mangkang Kulon', 231: 'Kawengen', 232: 'Tengah Ilir', 233: 'Banjarsari', 234: 'Tlogomulyo', 235: 'Pabelan', 236: 'Merbau Mataram', 237: 'Sragi', 238: 'Juwangi', 239: 'Pandu Sanjaya', 240: 'Kradenan', 241: 'Pancur', 242: 'Klepu', 243: 'Sumberejo', 244: 'Runtu', 245: 'Rembang', 246: 'Kendawangan', 247: 'Blimbing', 248: 'Patemon', 249: 'Sungai Lilin', 250: 'Gunungpati', 251: 'Teluk Belengkong', 252: 'Muka Kuning', 253: 'Penggaron Kidul', 254: 'Banjarejo', 255: 'Kalibareng', 256: 'Kaloran', 257: 'Bojong', 258: 'Kemuning', 259: 'Bulakamba', 260: 'Ngampel', 261: 'Kaligading'}
    jenisTinggal_map = {0: 'Bersama Orang Tua', 1: 'Panti Asuhan', 2: 'Asrama', 3: 'Rumah Tinggal', 4: 'Pesantren', 5: 'Lainnya', 6: 'Wali'}
    alatTransportasi_map = {0: 'Sepeda', 1: 'Jalan Kaki', 2: 'Ojek', 3: 'Sepeda Motor', 4: 'Angkutan Umum/Bus/Pete-Pete', 5: 'Mobil/Bus Antar Jemput', 6: 'Lainnya'}
    jenjangAyah_map = {0: 'Smp / Sederajat', 1: 'Sma / Sederajat', 2: 'Sd / Sederajat', 3: 'Putus Sd', 4: 'Tidak Sekolah', 5: 'S1', 6: 'D3', 7: 'D1', 8: 'Lainnya', 9: 'D4', 10: 'Tk / Sederajat', 11: 'D2'}
    pekerjaanAyah_map = {0: 'Wiraswasta', 1: 'Karyawan Swasta', 2: 'Buruh', 3: 'Petani', 4: 'Pensiunan', 5: 'Rp. 1,000,000 - Rp. 1,999,999', 6: 'Sudah Meninggal', 7: 'Pedagang Kecil', 8: 'Nelayan', 9: 'Peternak', 10: 'Tidak Bekerja', 11: 'Tidak Dapat Diterapkan', 12: 'Lainnya', 13: 'Pns/Tni/Polri', 14: 'Wirausaha', 15: 'Pedagang Besar'}
    penghasilanAyah_map = {0: 'Rp. 5,000,000 - Rp. 20,000,000', 1: 'Rp. 500,000 - Rp. 999,999', 2: 'Rp. 1,000,000 - Rp. 1,999,999', 3: 'Rp. 2,000,000 - Rp. 4,999,999', 4: 'Kurang Dari Rp. 500,000', 5: 'Tidak Berpenghasilan', 6: 'Kurang Dari Rp 1.000.000'}
    # jenjangIbu_map= {0: 'D3', 1: 'Smp / Sederajat', 2: 'Sd / Sederajat', 3: 'Putus Sd', 4: 'Sma / Sederajat', 5: 'Tidak Sekolah', 6: 'D2', 7: 'S2', 8: 'S1', 9: 'D1', 10: 'D4', 11: 'Lainnya'}
    pekerjaanIbu_map= {0: 'Karyawan Swasta', 1: 'Tidak Bekerja', 2: 'Buruh', 3: 'Petani', 4: 'Wiraswasta', 5: 'Pedagang Kecil', 6: 'Lainnya', 7: 'Pedagang Besar', 8: 'Pns/Tni/Polri', 9: 'Wirausaha', 10: 'Peternak', 11: 'Sudah Meninggal', 12: 'Nelayan', 13: 'Tidak Dapat Diterapkan'}
    penghasilanIbu_map = {0: 'Rp. 2,000,000 - Rp. 4,999,999', 1: 'Tidak Berpenghasilan', 2: 'Rp. 1,000,000 - Rp. 1,999,999', 3: 'Rp. 500,000 - Rp. 999,999', 4: 'Kurang Dari Rp. 500,000', 5: 'Kurang Dari Rp 1.000.000'}


    # Mengonversi teks kembali ke nilai numerik sebelum prediksi
    jk_options = list(jk_map.values())
    tempat_lahir_options = list(tempatLahir_map.values())
    asal_kec_options = list(asalKec_map.values())
    jenis_tinggal_options = list(jenisTinggal_map.values())
    alat_transportasi_options = list(alatTransportasi_map.values())
    jenjang_ayah_options = list(jenjangAyah_map.values())
    pekerjaan_ayah_options = list(pekerjaanAyah_map.values())
    penghasilan_ayah_options = list(penghasilanAyah_map.values())
    # jenjang_ibu_options = list(jenjangIbu_map.values())
    pekerjaan_ibu_options = list(pekerjaanIbu_map.values())
    penghasilan_ibu_options = list(penghasilanIbu_map.values())


    # Mendapatkan nilai teks yang dipilih oleh pengguna
    jk_selected_text = st.selectbox("Jenis Kelamin", jk_options)
    tempatLahir_selected_text = st.selectbox("Tempat Lahir", tempat_lahir_options)
    asalKec_selected_text = st.selectbox("Asal Kecamatan", asal_kec_options)
    jenisTinggal_selected_text = st.selectbox("Jenis Tinggal", jenis_tinggal_options)
    alatTransportasi_selected_text = st.selectbox("Alat Transportasi", alat_transportasi_options)
    jenjangAyah_selected_text = st.selectbox("Jenjang Ayah", jenjang_ayah_options)
    pekerjaanAyah_selected_text = st.selectbox("Pekerjaan Ayah", pekerjaan_ayah_options)
    penghasilanAyah_selected_text = st.selectbox("Penghasilan Ayah", penghasilan_ayah_options)
    # jenjangIbu_selected_text = st.selectbox("Jenjang Ibu", jenjang_ibu_options)
    pekerjaanIbu_selected_text = st.selectbox("Pekerjaan Ibu", pekerjaan_ibu_options)
    penghasilanIbu_selected_text = st.selectbox("Penghasilan Ibu", penghasilan_ibu_options)


    # Mengonversi nilai teks yang dipilih kembali ke nilai numerik
    jk_selected_value = list(jk_map.keys())[list(jk_map.values()).index(jk_selected_text)]
    tempatLahir_selected_value = list(tempatLahir_map.keys())[list(tempatLahir_map.values()).index(tempatLahir_selected_text)]
    asalKec_selected_value = list(asalKec_map.keys())[list(asalKec_map.values()).index(asalKec_selected_text)]
    jenisTinggal_selected_value = list(jenisTinggal_map.keys())[list(jenisTinggal_map.values()).index(jenisTinggal_selected_text)]
    alatTransportasi_selected_value = list(alatTransportasi_map.keys())[list(alatTransportasi_map.values()).index(alatTransportasi_selected_text)]
    jenjangAyah_selected_value = list(jenjangAyah_map.keys())[list(jenjangAyah_map.values()).index(jenjangAyah_selected_text)]
    pekerjaanAyah_selected_value = list(pekerjaanAyah_map.keys())[list(pekerjaanAyah_map.values()).index(pekerjaanAyah_selected_text)]
    penghasilanAyah_selected_value = list(penghasilanAyah_map.keys())[list(penghasilanAyah_map.values()).index(penghasilanAyah_selected_text)]
    # jenjangIbu_selected_value = list(jenjangIbu_map.keys())[list(jenjangIbu_map.values()).index(jenjangIbu_selected_text)]
    pekerjaanIbu_selected_value = list(pekerjaanIbu_map.keys())[list(pekerjaanIbu_map.values()).index(pekerjaanIbu_selected_text)]
    penghasilanIbu_selected_value = list(penghasilanIbu_map.keys())[list(penghasilanIbu_map.values()).index(penghasilanIbu_selected_text)]

    # Gunakan nilai numerik yang sudah dikonversi kembali dalam prediksi
    input_data = {
        "JK": jk_selected_value,
        "Tempat Lahir": tempatLahir_selected_value,
        "Asal Kecamatan": asalKec_selected_value,
        "Jenis Tinggal":  jenisTinggal_selected_value,
        "Alat Transportasi": alatTransportasi_selected_value,
        "Jenjang Pendidikan Ayh": jenjangAyah_selected_value,
        "Pekerjaan Ayh": pekerjaanAyah_selected_value,
        "Penghasilan Ayh": penghasilanAyah_selected_value,
        # "Jenjang Pendidikan Ibu": jenjangIbu_selected_value,
        "Pekerjaan Ibu": pekerjaanIbu_selected_value,
        "Penghasilan Ibu": penghasilanIbu_selected_value
    }

    input_df = pd.DataFrame(input_data, index=[0])
    prediction = dt_clf.predict(input_df)

    # Prediksi berdasarkan input
    if st.button("Prediksi"):
        if len(prediction) > 0:
            if prediction[0] == 1:
                st.markdown('<p style="color:green">Siswa berpotensi untuk mutasi.</p>', unsafe_allow_html=True)
            elif prediction[0] == 2:
                st.markdown('<p style="color:green">Siswa berpotensi untuk dikeluarkan.</p>', unsafe_allow_html=True)
            elif prediction[0] == 3:
                st.markdown('<p style="color:green">Siswa berpotensi untuk lainnya.</p>', unsafe_allow_html=True)
            elif prediction[0] == 4:
                st.markdown('<p style="color:green">Siswa berpotensi untuk mengundurkan diri.</p>', unsafe_allow_html=True)
            elif prediction[0] == 0:
                st.markdown('<p style="color:green">Siswa diperkirakan akan lulus.</p>', unsafe_allow_html=True)
            else:
                st.markdown('<p style="color:green">Hasil prediksi tidak jelas.</p>', unsafe_allow_html=True)
        else:
            st.markdown('<p style="color:green">Tidak ada prediksi yang dihasilkan.</p>', unsafe_allow_html=True)


# Fungsi prediksi siswa mutasi dgn upload file

def display_prediksiCSV(uploaded_file, dt_clf, data):
    if uploaded_file is None:
        st.warning("Harap unggah file CSV untuk melakukan prediksi.")
        return

    if uploaded_file.type == 'text/csv':
        input_data = pd.read_csv(uploaded_file)

        if 'Nama' in input_data.columns:  # Periksa apakah kolom 'Nama' ada
            # Lakukan prediksi berdasarkan data yang diunggah
            prediction = dt_clf.predict(input_data.drop(columns=['Nama'], errors='ignore'))

            # Filter hasil prediksi untuk hanya menampilkan nilai yang bukan 0
            filtered_indices = np.where(prediction != 0)[0]

            if len(filtered_indices) > 0:
                # Ambil data siswa yang memiliki kemungkinan mutasi dari indeks hasil prediksi
                filtered_data = input_data.iloc[filtered_indices]

                # Tampilkan nama siswa beserta kemungkinan mutasinya
                filtered_data['Kemungkinan'] = prediction[filtered_indices]
                filtered_data['Kemungkinan'].replace({1: "Mutasi", 2: "Dikeluarkan"}, inplace=True)
                st.subheader("Data Siswa Berpotensi Mutasi")
                st.write(filtered_data[['Nama', 'Kemungkinan']])  # Menampilkan nama siswa dan kemungkinan mutasi
            else:
                st.info("Tidak ada siswa yang berpotensi mutasi.")
        else:
            st.error("Kolom 'Nama' tidak ditemukan dalam file CSV yang diunggah.")

  
# Fungsi template CSV
def create_templateCSV():
    # Buat DataFrame kosong dengan kolom yang diperlukan
    template_columns = ["Nama", "JK", "Tempat Lahir", "Asal Kecamatan", "Jenis Tinggal", "Alat Transportasi",
                        "Jenjang Pendidikan Ayh", "Pekerjaan Ayh", "Penghasilan Ayh", "Pekerjaan Ibu", "Penghasilan Ibu"]
    template_data = pd.DataFrame(columns=template_columns)

    # Simpan sebagai file CSV
    csv = template_data.to_csv(index=False)

    # Buat file CSV dalam bentuk BytesIO
    csv_bytes = BytesIO(csv.encode())

    return csv_bytes


# Fungsi Main App


def main():
    # Persiapan Data
    data, dataFinal, x_train, x_test, y_train, y_test = prepare_data()

    # Buat Model
    dt_clf = create_model(x_train, y_train)

    # Prediksi menggunakan Model
    y_pred = dt_clf.predict(x_test)

    # Evaluasi Performa Model
    accuracy, f1, precision, recall = evaluate_model(y_test, y_pred)
    
    # Uploaden File
    uploaded_file = None

    st.title('Klasifikasi Siswa SMK Askhabul Kahfi :school:')

    # Sidebar untuk Memilih Fitur
    st.sidebar.image('logo_smk.png', width=150)

    st.sidebar.title('Pilihan Fitur')
    selected_feature = st.sidebar.selectbox('Pilih Untuk Menampilkan:', [
                                            'Persebaran Data', 'Dataset', 'Analisis Data', 'Akurasi Model'])

    if selected_feature == 'Persebaran Data':
        st.sidebar.subheader('Pilihan Visualisasi Data')
        display_option = st.sidebar.radio(
            'Pilih Visualisasi:', ['Diagram Batang', 'Pie Chart', 'Scatter Plot', 'Korelasi Data'])

        if display_option == 'Diagram Batang':
            display_bar_chart(dataFinal)
        elif display_option == 'Pie Chart':
            display_pie_chart(dataFinal)
        elif display_option == 'Scatter Plot':
            display_scatter_plot(dataFinal)
            st.set_option('deprecation.showPyplotGlobalUse', False)
        elif display_option == 'Korelasi Data':
            display_correlation(data)

    elif selected_feature == 'Dataset':
        st.sidebar.subheader('Pilihan Tampilan Data')
        display_option = st.sidebar.radio(
            'Pilih Data:', ['Seluruh Data', 'Seluruh Data Numerik'])

        if display_option == 'Seluruh Data':
            display_full_data(dataFinal)
        elif display_option == 'Seluruh Data Numerik':
            display_numeric_data(data)

    elif selected_feature == 'Analisis Data':
        st.sidebar.subheader('Pilihan Analisis Data')
        display_option = st.sidebar.radio(
            'Pilih Data:', ['Faktor Penyebab Mutasi', 'Prediksi Siswa Mutasi', 'Prediksi Siswa Mutasi dengan Upload File'])

        if display_option == 'Faktor Penyebab Mutasi':
            display_faktor()
        elif display_option == 'Prediksi Siswa Mutasi':
            display_prediksi(dt_clf, data)
        elif display_option == 'Prediksi Siswa Mutasi dengan Upload File':
            file_status = st.radio("Apakah sudah memiliki file CSV?", ("Sudah", "Belum"))
            if file_status == "Sudah":
                uploaded_file = st.file_uploader("Ungguah File CSV", type=["csv"])
            else:
                st.write("Buat Template CSV")
                if st.button("Unduh Template CSV"):
                    csv = create_templateCSV()
                    st.download_button(
                        label="Unduh Template Prediksi",
                        data=csv,
                        file_name='template_prediksi.csv',
                        mime='text/csv'
                    )
            display_prediksiCSV(uploaded_file, dt_clf, data)

    elif selected_feature == 'Akurasi Model':
        st.sidebar.subheader('Pilihan Tampilan Data')
        display_option = st.sidebar.radio(
            'Pilih Data:', ['Decision Tree', 'Akurasi Model'])

        if display_option == 'Decision Tree':
            feature_names = ["JK", "Tempat Lahir", "Asal Kecamatan", "Jenis Tinggal", "Alat Transportasi",
                             "Jenjang Pendidikan Ayh", "Pekerjaan Ayh", "Penghasilan Ayh", "Pekerjaan Ibu", "Penghasilan Ibu"]
            class_names = ["Lulus", "Mutasi", "Dikeluarkan",
                           "Mengundurkan Diri", "Lainnya"]
            display_decision_tree(dt_clf, feature_names, class_names)
        elif display_option == 'Akurasi Model':
            # Outputkan informasi performa model
            st.write(f'Accuracy: `{accuracy}`')
            st.write(f'F1 Score: `{f1}`')
            st.write(f'Precision: `{precision}`')
            st.write(f'Recall: `{recall}`')


if __name__ == "__main__":
    main()
