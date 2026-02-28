# =========================
# IMPORT LIBRARY
# =========================
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from babel.numbers import format_currency

sns.set_style("whitegrid")

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="E-Commerce Analytics Dashboard",
    layout="wide"
)

# =========================
# LOAD DATA
# =========================
@st.cache_data
def load_data():
    df = pd.read_csv("all_df.csv")

    df["order_purchase_timestamp"] = pd.to_datetime(df["order_purchase_timestamp"])
    df["order_delivered_customer_date"] = pd.to_datetime(df["order_delivered_customer_date"])
    df["order_estimated_delivery_date"] = pd.to_datetime(df["order_estimated_delivery_date"])

    df["is_late"] = (
        df["order_delivered_customer_date"] >
        df["order_estimated_delivery_date"]
    )

    df["year_month"] = df["order_purchase_timestamp"].dt.to_period("M").astype(str)

    return df

all_df = load_data()

# =========================
# SIDEBAR FILTER (SYNC DENGAN DATA)
# =========================
with st.sidebar:
    st.title("Filter Waktu")

    min_date = all_df["order_purchase_timestamp"].min().date()
    max_date = all_df["order_purchase_timestamp"].max().date()

    start_date, end_date = st.date_input(
        "Rentang Tanggal",
        min_value=min_date,
        max_value=max_date,
        value=(min_date, max_date)
    )

    st.caption(f"Data tersedia dari {min_date} sampai {max_date}")

# =========================
# FILTER DATA
# =========================
main_df = all_df[
    (all_df["order_purchase_timestamp"] >= pd.to_datetime(start_date)) &
    (all_df["order_purchase_timestamp"] <= pd.to_datetime(end_date))
]

# =========================
# KPI & BASIC METRICS
# =========================
daily_orders = (
    main_df
    .set_index("order_purchase_timestamp")
    .resample("D")
    .agg(
        total_orders=("order_id", "nunique"),
        revenue=("payment_value", "sum")
    )
    .reset_index()
)

orders_per_customer = main_df.groupby("customer_unique_id")["order_id"].nunique()
repeat_rate = (orders_per_customer > 1).mean() * 100

# =========================
# RFM ANALYSIS
# =========================
snapshot_date = main_df["order_purchase_timestamp"].max()

rfm = (
    main_df.groupby("customer_unique_id")
    .agg(
        recency=("order_purchase_timestamp",
                 lambda x: (snapshot_date - x.max()).days),
        frequency=("order_id", "nunique"),
        monetary=("payment_value", "sum")
    )
    .reset_index()
)

rfm["R"] = pd.qcut(rfm["recency"], 4, labels=[4, 3, 2, 1])
rfm["F"] = pd.qcut(rfm["frequency"].rank(method="first"), 4, labels=[1, 2, 3, 4])
rfm["M"] = pd.qcut(rfm["monetary"], 4, labels=[1, 2, 3, 4])

rfm["rfm_score"] = rfm[["R", "F", "M"]].astype(int).sum(axis=1)

rfm["segment"] = pd.cut(
    rfm["rfm_score"],
    bins=[0, 5, 8, 12],
    labels=["Low Value", "Mid Value", "High Value"]
)

# =========================
# TITLE
# =========================
st.markdown(
    "<h1 style='text-align:center;'>📊 E-Commerce Analytics Dashboard</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align:center;'>Analisis Performa Bisnis & Perilaku Pelanggan</p>",
    unsafe_allow_html=True
)
st.divider()

# =========================
# KPI (CENTERED)
# =========================
_, c1, c2, c3, _ = st.columns([1, 2, 2, 2, 1])

c1.metric("Total Orders", daily_orders["total_orders"].sum())
c2.metric(
    "Total Revenue",
    format_currency(daily_orders["revenue"].sum(), "BRL", locale="pt_BR")
)
c3.metric("Repeat Customer (%)", f"{repeat_rate:.2f}%")

st.divider()

# =========================
# 1️⃣ TREN ORDER HARIAN
# =========================
st.subheader("📈 Tren Order Harian")

_, center, _ = st.columns([1, 6, 1])
with center:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(daily_orders["order_purchase_timestamp"], daily_orders["total_orders"])
    ax.set_xlabel("Tanggal")
    ax.set_ylabel("Jumlah Order")
    st.pyplot(fig)

# =========================
# 2️⃣ ORDER PER BULAN
# =========================
st.subheader("📅 Jumlah Order per Bulan")

monthly_orders = (
    main_df.groupby("year_month")["order_id"]
    .nunique()
    .reset_index()
)

_, center, _ = st.columns([1, 6, 1])
with center:
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.barplot(data=monthly_orders, x="year_month", y="order_id", ax=ax)
    plt.xticks(rotation=45)
    ax.set_xlabel("Bulan")
    ax.set_ylabel("Jumlah Order")
    st.pyplot(fig)

# =========================
# 3️⃣ TOP KATEGORI PRODUK
# =========================
st.subheader("🛒 Top 10 Kategori Produk")

top_category = (
    main_df["product_category_name"]
    .value_counts()
    .head(10)
    .reset_index()
)
top_category.columns = ["Kategori", "Jumlah Produk"]

_, center, _ = st.columns([1, 6, 1])
with center:
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.barplot(data=top_category, y="Kategori", x="Jumlah Produk", ax=ax)
    st.pyplot(fig)

# =========================
# 4️⃣ LOGISTIK vs REVIEW
# =========================
st.subheader("🚚 Keterlambatan Pengiriman vs Review Score")

_, center, _ = st.columns([1, 6, 1])
with center:
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.boxplot(data=main_df, x="is_late", y="review_score", ax=ax)
    ax.set_xlabel("Terlambat")
    ax.set_ylabel("Review Score")
    st.pyplot(fig)

# =========================
# 5️⃣ DISTRIBUSI RFM
# =========================
st.subheader("📊 Distribusi RFM")

_, center, _ = st.columns([1, 8, 1])
with center:
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    sns.histplot(rfm["recency"], ax=ax[0]); ax[0].set_title("Recency")
    sns.histplot(rfm["frequency"], ax=ax[1]); ax[1].set_title("Frequency")
    sns.histplot(rfm["monetary"], ax=ax[2]); ax[2].set_title("Monetary")
    st.pyplot(fig)

# =========================
# 6️⃣ PAYMENT
# =========================
st.subheader("💳 Rata-rata Nilai Transaksi per Metode Pembayaran")

fig, ax = plt.subplots(figsize=(8,5))
sns.barplot(
    data=all_df,
    x="payment_value",
    y="payment_type",
    ax=ax
)
ax.set_xlabel("Rata-rata Payment Value")
ax.set_ylabel("Metode Pembayaran")
st.pyplot(fig)

# =========================
# SEGMENTASI CUSTOMER (RFM)
# =========================
st.subheader("🔁 Segmentasi Customer berdasarkan RFM")

segment_count = rfm["segment"].value_counts().reset_index()
segment_count.columns = ["Segment", "Jumlah Customer"]

_, center, _ = st.columns([1, 6, 1])
with center:
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(data=segment_count, x="Segment", y="Jumlah Customer", ax=ax)

    st.pyplot(fig)


