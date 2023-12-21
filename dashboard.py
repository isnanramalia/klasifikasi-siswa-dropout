from sklearn import tree
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# Fungsi Persiapan Data


def prepare_data():
    # Load data
    data = pd.read_csv("data_numerik.csv")
    dataasli = pd.read_csv("data_smk_askha_final.csv")

    # Split data menjadi data latih dan data uji
    x = data[["JK", "Jenjang Pendidikan Ayh", "Pekerjaan Ayh",
              "Penghasilan Ayh", "Pekerjaan Ibu", "Penghasilan Ibu"]]
    y = data["Keluar Karena"]
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42)

    return data, dataasli, x_train, x_test, y_train, y_test

# Fungsi Membuat Model


def create_model(x_train, y_train):
    dt_clf = DecisionTreeClassifier(max_depth=3, criterion="entropy")
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
    fig, ax = plt.subplots(figsize=(50, 50))
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


# Fungsi diagram batang


def plot_diagram(dataasli):
    # Visualisasi variabel JK
    ax1 = sns.displot(dataasli, x="JK", height=3, aspect=2)
    # Visualisasi variabel Keluar Karena
    ax2 = sns.displot(dataasli, x="Keluar Karena", height=3, aspect=2)
    # Visualisasi variabel Penghasilan Ayah
    ax3 = sns.displot(dataasli, x="Penghasilan Ayh", height=3, aspect=7)
    # Visualisasi variabel Penghasilan Ibu
    ax4 = sns.displot(dataasli, x="Penghasilan Ibu", height=3, aspect=7)
    return ax1, ax2, ax3, ax4

# Fungsi pie chart


def plot_pie(dataasli, column):
    fig, ax = plt.subplots(figsize=(10, 10))
    dataasli[column].value_counts().plot.pie(autopct='%1.1f%%', ax=ax)
    return fig


# Fungsi untuk Persebaran Data dengan Diagram Batang
def display_bar_chart(dataasli):
    option = st.selectbox("Pilih Variabel:", list(dataasli.columns))
    if option:
        ax = sns.displot(dataasli, x=option, height=3, aspect=2)
        st.pyplot(ax)

# Fungsi untuk Persebaran Data dengan Pie Chart


def display_pie_chart(dataasli):
    option = st.selectbox("Pilih Variabel:", list(dataasli.columns))
    if option:
        fig, ax = plt.subplots(figsize=(10, 10))
        dataasli[option].value_counts().plot.pie(autopct='%1.1f%%', ax=ax)
        st.pyplot(fig)

# Fungsi untuk Persebaran Data dengan Scatter Plot


def display_scatter_plot(dataasli):
    option_x = st.selectbox(
        "Pilih Variabel untuk sumbu X:", list(dataasli.columns))
    option_y = st.selectbox(
        "Pilih Variabel untuk sumbu Y:", list(dataasli.columns))

    if option_x and option_y and option_x != option_y:
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=option_x, y=option_y, data=dataasli)
        plt.xlabel(option_x)
        plt.ylabel(option_y)
        plt.title(f"Scatter Plot: {option_x} vs {option_y}")
        st.pyplot()
    else:
        st.write("Pilih dua variabel yang berbeda untuk sumbu X dan Y.")

# Fungsi untuk Korelasi Data


def display_correlation(data):
    st.subheader("Korelasi antar variabel:")
    plt.figure(figsize=(15, 15))
    numerical_data = data.select_dtypes(include=[np.number])
    sns.heatmap(numerical_data.corr(), annot=True)
    st.pyplot()
    with st.expander(label='Lihat Penjelasan'):
        st.write("""
            Kesimpulan dari diagram diatas yaitu data tidak ada yang berkorelasi kuat, namun ada beberapa yang berkorelasi sedang seperti: JK dan Keluar Karena, JK dan Jenjang Pendidikan Ibu, JK dan Pekerjaan Ayh.
                 Maka dari itu, saya memilih variabel JK untuk dijadikan variabel prediktor dan Keluar Karena sebagai variabel target
        """)

# Fungsi untuk Seluruh Data dengan Tabel


def display_full_data(dataasli):
    st.write("Seluruh Data:")
    st.write(dataasli)

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

    # Kamus pemetaan dari teks ke nilai numerik
    jk_map = {0: 'L', 1: 'P'}
    jenjang_ayah_map = {0: 'Smp / Sederajat', 1: 'Sma / Sederajat', 2: 'Sd / Sederajat', 3: 'Putus Sd', 4: 'Tidak Sekolah', 5: 'S1', 6: 'D3', 7: 'D1', 8: 'Lainnya', 9: 'D4', 10: 'Tk / Sederajat', 11: 'D2'}
    pekerjaan_ayah_map = {0: 'Wiraswasta', 1: 'Karyawan Swasta', 2: 'Buruh', 3: 'Petani', 4: 'Pensiunan', 5: 'Rp. 1,000,000 - Rp. 1,999,999', 6: 'Sudah Meninggal', 7: 'Pedagang Kecil', 8: 'Nelayan', 9: 'Peternak', 10: 'Tidak Bekerja', 11: 'Tidak Dapat Diterapkan', 12: 'Lainnya', 13: 'Pns/Tni/Polri', 14: 'Wirausaha', 15: 'Pedagang Besar'}
    penghasilan_ayah_map = {0: 'Rp. 5,000,000 - Rp. 20,000,000', 1: 'Rp. 500,000 - Rp. 999,999', 2: 'Rp. 1,000,000 - Rp. 1,999,999', 3: 'Rp. 2,000,000 - Rp. 4,999,999', 4: 'Kurang Dari Rp. 500,000', 5: 'Tidak Berpenghasilan', 6: 'Kurang Dari Rp 1.000.000'}
    # jenjang_ibu_map = {0: 'D3', 1: 'Smp / Sederajat', 2: 'Sd / Sederajat', 3: 'Putus Sd', 4: 'Sma / Sederajat', 5: 'Tidak Sekolah', 6: 'D2', 7: 'S2', 8: 'S1', 9: 'D1', 10: 'D4', 11: 'Lainnya'}
    pekerjaan_ibu_map = {0: 'Karyawan Swasta', 1: 'Tidak Bekerja', 2: 'Buruh', 3: 'Petani', 4: 'Wiraswasta', 5: 'Pedagang Kecil', 6: 'Lainnya', 7: 'Pedagang Besar', 8: 'Pns/Tni/Polri', 9: 'Wirausaha', 10: 'Peternak', 11: 'Sudah Meninggal', 12: 'Nelayan', 13: 'Tidak Dapat Diterapkan'}
    penghasilan_ibu_map = {0: 'Rp. 2,000,000 - Rp. 4,999,999', 1: 'Tidak Berpenghasilan', 2: 'Rp. 1,000,000 - Rp. 1,999,999', 3: 'Rp. 500,000 - Rp. 999,999', 4: 'Kurang Dari Rp. 500,000', 5: 'Kurang Dari Rp 1.000.000'}

    # Mengonversi teks kembali ke nilai numerik sebelum prediksi
    jk_options = list(jk_map.values())
    jenjang_ayah_options = list(jenjang_ayah_map.values())
    pekerjaan_ayah_options = list(pekerjaan_ayah_map.values())
    penghasilan_ayah_options = list(penghasilan_ayah_map.values())
    # jenjang_ibu_options = list(jenjang_ibu_map.values())
    pekerjaan_ibu_options = list(pekerjaan_ibu_map.values())
    penghasilan_ibu_options = list(penghasilan_ibu_map.values())

    # Mendapatkan nilai teks yang dipilih oleh pengguna
    jk_selected_text = st.selectbox("Jenis Kelamin", jk_options)
    jenjang_ayah_selected_text = st.selectbox("Jenjang Pendidikan Ayah", jenjang_ayah_options)
    pekerjaan_ayah_selected_text = st.selectbox("Pekerjaan Ayah", pekerjaan_ayah_options)
    penghasilan_ayah_selected_text = st.selectbox("Penghasilan Ayah", penghasilan_ayah_options)
    # jenjang_ibu_selected_text = st.selectbox("Jenjang Pendidikan Ibu", jenjang_ibu_options)
    pekerjaan_ibu_selected_text = st.selectbox("Pekerjaan Ibu", pekerjaan_ibu_options)
    penghasilan_ibu_selected_text = st.selectbox("Penghasilan Ibu", penghasilan_ibu_options)

    # Mengonversi nilai teks yang dipilih kembali ke nilai numerik
    jk_selected_value = list(jk_map.keys())[list(jk_map.values()).index(jk_selected_text)]
    jenjang_ayah_selected_value = list(jenjang_ayah_map.keys())[list(jenjang_ayah_map.values()).index(jenjang_ayah_selected_text)]
    pekerjaan_ayah_selected_value = list(pekerjaan_ayah_map.keys())[list(pekerjaan_ayah_map.values()).index(pekerjaan_ayah_selected_text)]
    penghasilan_ayah_selected_value = list(penghasilan_ayah_map.keys())[list(penghasilan_ayah_map.values()).index(penghasilan_ayah_selected_text)]
    # jenjang_ibu_selected_value = list(jenjang_ibu_map.keys())[list(jenjang_ibu_map.values()).index(jenjang_ibu_selected_text)]
    pekerjaan_ibu_selected_value = list(pekerjaan_ibu_map.keys())[list(pekerjaan_ibu_map.values()).index(pekerjaan_ibu_selected_text)]
    penghasilan_ibu_selected_value = list(penghasilan_ibu_map.keys())[list(penghasilan_ibu_map.values()).index(penghasilan_ibu_selected_text)]

    # Gunakan nilai numerik yang sudah dikonversi kembali dalam prediksi
    input_data = {
        "JK": jk_selected_value,
        "Jenjang Pendidikan Ayh": jenjang_ayah_selected_value,
        "Pekerjaan Ayh": pekerjaan_ayah_selected_value,
        "Penghasilan Ayh": penghasilan_ayah_selected_value,
        # "Jenjang Pendidikan Ibu": jenjang_ibu_selected_value,
        "Pekerjaan Ibu": pekerjaan_ibu_selected_value,
        "Penghasilan Ibu": penghasilan_ibu_selected_value
    }

    input_df = pd.DataFrame(input_data, index=[0])
    prediction = dt_clf.predict(input_df)

    # Prediksi berdasarkan input
    if st.button("Prediksi"):
        if prediction[0] == 'Mutasi':
            st.write("Siswa berpotensi untuk mutasi.")
        else:
            st.write("Siswa tidak berpotensi untuk mutasi.")


# Fungsi Main App


def main():
    # Persiapan Data
    data, dataasli, x_train, x_test, y_train, y_test = prepare_data()

    # Buat Model
    dt_clf = create_model(x_train, y_train)

    # Prediksi menggunakan Model
    y_pred = dt_clf.predict(x_test)

    # Evaluasi Performa Model
    accuracy, f1, precision, recall = evaluate_model(y_test, y_pred)

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
            display_bar_chart(dataasli)
        elif display_option == 'Pie Chart':
            display_pie_chart(dataasli)
        elif display_option == 'Scatter Plot':
            display_scatter_plot(dataasli)
            st.set_option('deprecation.showPyplotGlobalUse', False)
        elif display_option == 'Korelasi Data':
            display_correlation(data)

    elif selected_feature == 'Dataset':
        st.sidebar.subheader('Pilihan Tampilan Data')
        display_option = st.sidebar.radio(
            'Pilih Data:', ['Seluruh Data', 'Seluruh Data Numerik'])

        if display_option == 'Seluruh Data':
            display_full_data(dataasli)
        elif display_option == 'Seluruh Data Numerik':
            display_numeric_data(data)

    elif selected_feature == 'Analisis Data':
        st.sidebar.subheader('Pilihan Analisis Data')
        display_option = st.sidebar.radio(
            'Pilih Data:', ['Faktor Penyebab Mutasi', 'Prediksi Siswa Mutasi'])

        if display_option == 'Faktor Penyebab Mutasi':
            display_faktor()
        elif display_option == 'Prediksi Siswa Mutasi':
            display_prediksi(dt_clf, data)

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
