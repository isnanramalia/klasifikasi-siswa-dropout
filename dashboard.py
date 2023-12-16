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
    data = pd.read_csv("data_numerik_1.csv")
    dataasli = pd.read_csv("data_smk_askha_clean.csv")

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

    # Variabel dari data asli untuk pengisian selecbox
    jk_options = data['JK'].unique()
    jenjang_ayah_options = data['Jenjang Pendidikan Ayh'].unique()
    pekerjaan_ayah_options = data['Pekerjaan Ayh'].unique()
    penghasilan_ayah_options = data['Penghasilan Ayh'].unique()
    pekerjaan_ibu_options = data['Pekerjaan Ibu'].unique()
    penghasilan_ibu_options = data['Penghasilan Ibu'].unique()

    # Kolom input untuk variabel
    jk = st.selectbox("Jenis Kelamin", jk_options)
    jenjang_ayah = st.selectbox(
        "Jenjang Pendidikan Ayah", jenjang_ayah_options)
    pekerjaan_ayah = st.selectbox("Pekerjaan Ayah", pekerjaan_ayah_options)
    penghasilan_ayah = st.selectbox(
        "Penghasilan Ayah", penghasilan_ayah_options)
    pekerjaan_ibu = st.selectbox("Pekerjaan Ibu", pekerjaan_ibu_options)
    penghasilan_ibu = st.selectbox("Penghasilan Ibu", penghasilan_ibu_options)

    # Prediksi berdasarkan input
    if st.button("Prediksi"):
        input_data = {
            "JK": jk,
            "Jenjang Pendidikan Ayh": jenjang_ayah,
            "Pekerjaan Ayh": pekerjaan_ayah,
            "Penghasilan Ayh": penghasilan_ayah,
            "Pekerjaan Ibu": pekerjaan_ibu,
            "Penghasilan Ibu": penghasilan_ibu
        }

        input_df = pd.DataFrame(input_data, index=[0])
        prediction = dt_clf.predict(input_df)

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
