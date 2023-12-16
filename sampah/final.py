from sklearn import tree
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# TODO 1: Persiapan data
# Load data
data = pd.read_csv("data_numerik_2.csv")
dataasli = pd.read_csv("data_smk_askha_clean.csv")

# Split data menjadi data latih dan data uji
x = data[["JK", "Tempat Lahir", "Asal Kecamatan", "Jenis Tinggal", "Alat Transportasi",
          "Jenjang Pendidikan Ayh", "Pekerjaan Ayh", "Penghasilan Ayh", "Pekerjaan Ibu", "Penghasilan Ibu"]]
y = data["Keluar Karena"]
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42)

# Buat model
dt_clf = DecisionTreeClassifier(max_depth=3, criterion="entropy")
dt_clf = dt_clf.fit(x_train, y_train)

# Prediksi menggunakan model
y_pred = dt_clf.predict(x_test)

# Evaluasi performa model
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')

# Fungsi plotting decision tree


def plot_decision_tree(dt_clf, feature_names, class_names):
    fig, ax = plt.subplots(figsize=(50, 50))
    tree.plot_tree(dt_clf,
                   feature_names=feature_names,
                   class_names=class_names,
                   filled=True,
                   ax=ax)
    return fig

# Fungsi diagram batang


import seaborn as sns

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
    


# TODO 2: Buat aplikasi web
# Judul aplikasi
st.title('Klasifikasi Siswa Keluar dari Sekolah :school:')

# Tampilkan informasi performa model
st.write(f'Accuracy: {accuracy}')
st.write(f'F1 Score: {f1}')
st.write(f'Precision: {precision}')
st.write(f'Recall: {recall}')

# Tampilkan fungsi decision tree plotting
feature_names = ["JK", "Tempat Lahir", "Asal Kecamatan", "Jenis Tinggal", "Alat Transportasi",
                 "Jenjang Pendidikan Ayh", "Pekerjaan Ayh", "Penghasilan Ayh", "Pekerjaan Ibu", "Penghasilan Ibu"]
class_names = ["Lulus", "Mutasi", "Dikeluarkan",
               "Mengundurkan Diri", "Lainnya"]

fig = plot_decision_tree(dt_clf, feature_names, class_names)
st.pyplot(fig)

# Tampilkan fungsi diagram batang
ax1, ax2, ax3, ax4 = plot_diagram(dataasli)
st.pyplot(ax1.figure)
st.pyplot(ax2.figure)
st.pyplot(ax3.figure)
st.pyplot(ax4.figure)

# Tampilkan fungsi pie chart
st.write("Diagram Pie Jenis Kelamin") 
fig = plot_pie(dataasli, "JK")
st.pyplot(fig)

fig = plot_pie(dataasli, "Keluar Karena")
st.pyplot(fig)

fig = plot_pie(dataasli, "Pekerjaan Ayh")
st.pyplot(fig)

fig = plot_pie(dataasli, "Pekerjaan Ibu")
st.pyplot(fig)

fig = plot_pie(dataasli, "Jenis Tinggal")
st.pyplot(fig)

fig = plot_pie(dataasli, "Alat Transportasi")
st.pyplot(fig)

fig = plot_pie(dataasli, "Jenjang Pendidikan Ayh")
st.pyplot(fig)

# Outputkan data

# Tambahkan fitur input untuk pengguna jika ingin melakukan prediksi
with st.sidebar:
    st.image('logo_smk.png', width=100)

    st.title('Pilihan Fitur')
