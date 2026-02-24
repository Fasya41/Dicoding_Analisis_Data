# 📊 Proyek Analisis Data E-Commerce

Proyek ini merupakan analisis data E-Commerce yang bertujuan untuk mengeksplorasi pola transaksi, perilaku pelanggan, performa pembayaran, serta tren revenue berdasarkan dataset transaksi.

Project ini terdiri dari:
- Notebook analisis data
- Dataset mentah
- Dashboard interaktif menggunakan Streamlit

---

## 📁 Struktur Project

```
Submisson/
│
├── dashboard/
│   ├── Dashboard.py
│   └── all_df.csv
│
├── data/
│   ├── customers_dataset.csv
│   ├── orders_dataset.csv
│   ├── order_items_dataset.csv
│   ├── order_payments_dataset.csv
│   ├── order_reviews_dataset.csv
│   └── products_dataset.csv
│
├── Dicoding_Proyek_Analisis_Data.ipynb
├── requirements.txt
└── url.txt
```

---

## 📌 Deskripsi File

### 📓 Dicoding_Proyek_Analisis_Data.ipynb
Notebook utama yang berisi:
- Proses Data Wrangling
- Exploratory Data Analysis (EDA)
- Visualisasi data
- Insight bisnis berdasarkan hasil analisis

### 📊 Folder `data/`
Berisi dataset mentah yang digunakan dalam analisis:

- `customers_dataset.csv` → Data pelanggan  
- `orders_dataset.csv` → Data pesanan  
- `order_items_dataset.csv` → Detail produk dalam pesanan  
- `order_payments_dataset.csv` → Informasi metode pembayaran  
- `order_reviews_dataset.csv` → Rating dan ulasan pelanggan  
- `products_dataset.csv` → Informasi produk  

### 📈 Folder `dashboard/`
Berisi file untuk menjalankan dashboard interaktif:

- `Dashboard.py` → Script utama Streamlit  
- `all_df.csv` → Dataset hasil pengolahan yang digunakan untuk dashboard  

### 📦 requirements.txt
Berisi daftar library Python yang dibutuhkan untuk menjalankan project.

### 🔗 url.txt
Berisi link terkait project (misalnya link deployment dashboard atau sumber dataset).

---

## 🎯 Tujuan Analisis

Beberapa fokus analisis dalam project ini:

1. Analisis perkembangan revenue bulanan  
2. Analisis distribusi rating pelanggan  
3. Analisis metode pembayaran  
4. Analisis performa pengiriman  
5. Analisis perilaku pembelian pelanggan  

---

## ⚙️ Cara Menjalankan Project

### 1️⃣ Clone Repository

```bash
git clone <repository-url>
cd Submisson
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Menjalankan Notebook

Buka file berikut menggunakan Jupyter Notebook atau Google Colab:

```
Dicoding_Proyek_Analisis_Data.ipynb
```

### 4️⃣ Menjalankan Dashboard

```bash
streamlit run dashboard/Dashboard.py
```

Dashboard akan terbuka di browser secara otomatis.

---

## 🛠️ Teknologi yang Digunakan

- Python  
- Pandas  
- NumPy  
- Matplotlib  
- Seaborn  
- Streamlit  
- Jupyter Notebook  

---

## 📌 Output Project

- Notebook analisis lengkap  
- Insight bisnis berbasis data  
- Dashboard interaktif untuk eksplorasi data  

---

## 👤 Author

Nama: (faXyZa)  
Project: Analisis Data E-Commerce  
Tahun: 2026  
