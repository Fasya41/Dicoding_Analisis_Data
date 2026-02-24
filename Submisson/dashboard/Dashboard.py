# =========================
# IMPORT LIBRARY
# =========================
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from babel.numbers import format_currency

sns.set(style="darkgrid")

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

    return df

all_df = load_data()

# =========================
# SIDEBAR FILTER
# =========================
with st.sidebar:
    st.title("Filter Waktu")

    min_date = all_df["order_purchase_timestamp"].min()
    max_date = all_df["order_purchase_timestamp"].max()

    start_date, end_date = st.date_input(
        "Rentang Tanggal",
        value=[min_date, max_date],
        min_value=min_date,
        max_value=max_date
    )

# =========================
# FILTER DATA
# =========================
main_df = all_df[
    (all_df["order_purchase_timestamp"] >= pd.to_datetime(start_date)) &
    (all_df["order_purchase_timestamp"] <= pd.to_datetime(end_date))
]

# =========================
# PREPARE DATA
# =========================

# 1. Daily Orders
daily_orders = (
    main_df
    .resample("D", on="order_purchase_timestamp")
    .agg(
        total_orders=("order_id", "nunique"),
        revenue=("payment_value", "sum")
    )
    .reset_index()
)

# 2. Repeat Customer
order_per_customer = main_df.groupby("customer_unique_id")["order_id"].nunique()
repeat_rate = (order_per_customer > 1).mean() * 100

# 3. Payment
payment_df = (
    main_df.groupby("payment_type")["payment_value"]
    .mean()
    .reset_index()
)

# 4. RFM
snapshot_date = main_df["order_purchase_timestamp"].max()

rfm = (
    main_df.groupby("customer_unique_id")
    .agg(
        recency=("order_purchase_timestamp", lambda x: (snapshot_date - x.max()).days),
        frequency=("order_id", "nunique"),
        monetary=("payment_value", "sum")
    )
    .reset_index()
)

rfm["R"] = pd.qcut(rfm["recency"], 4, labels=[4,3,2,1])
rfm["F"] = pd.qcut(rfm["frequency"].rank(method="first"), 4, labels=[1,2,3,4])
rfm["M"] = pd.qcut(rfm["monetary"], 4, labels=[1,2,3,4])

rfm["rfm_score"] = rfm[["R","F","M"]].astype(int).sum(axis=1)

rfm["segment"] = pd.cut(
    rfm["rfm_score"],
    bins=[0,5,8,12],
    labels=["Low Value","Mid Value","High Value"]
)

repeat_segment = (
    order_per_customer.reset_index()
    .rename(columns={"order_id":"total_orders"})
    .merge(rfm[["customer_unique_id","segment"]], on="customer_unique_id")
)

repeat_segment["is_repeat"] = repeat_segment["total_orders"] > 1

repeat_segment_df = (
    repeat_segment.groupby("segment")["is_repeat"]
    .mean()
    .reset_index()
)

# =========================
# DASHBOARD
# =========================
st.title("📊 E-Commerce Analytics Dashboard")

# KPI
col1, col2, col3 = st.columns(3)

col1.metric("Total Orders", daily_orders["total_orders"].sum())
col2.metric(
    "Total Revenue",
    format_currency(daily_orders["revenue"].sum(), "BRL", locale="pt_BR")
)
col3.metric("Repeat Customer (%)", f"{repeat_rate:.2f}%")

# =========================
# 1️⃣ TREND ORDER
# =========================
st.subheader("📈 Tren Order Harian")

fig, ax = plt.subplots(figsize=(14,5))
ax.plot(daily_orders["order_purchase_timestamp"], daily_orders["total_orders"])
ax.set_xlabel("Tanggal")
ax.set_ylabel("Jumlah Order")
st.pyplot(fig)

# =========================
# 2️⃣ LOGISTICS vs REVIEW
# =========================
st.subheader("🚚 Keterlambatan Pengiriman vs Review Score")

fig, ax = plt.subplots(figsize=(8,5))
sns.boxplot(
    data=main_df,
    x="is_late",
    y="review_score",
    ax=ax
)
ax.set_xlabel("Terlambat")
ax.set_ylabel("Review Score")
st.pyplot(fig)

# =========================
# 3️⃣ PAYMENT
# =========================
st.subheader("💳 Rata-rata Nilai Transaksi per Metode Pembayaran")

fig, ax = plt.subplots(figsize=(8,5))
sns.barplot(
    data=payment_df,
    x="payment_value",
    y="payment_type",
    ax=ax
)
ax.set_xlabel("Rata-rata Payment Value")
ax.set_ylabel("Metode Pembayaran")
st.pyplot(fig)

# =========================
# 4️⃣ RFM DISTRIBUTION
# =========================
st.subheader("📊 Distribusi RFM")

fig, ax = plt.subplots(1,3, figsize=(18,5))
sns.histplot(rfm["recency"], ax=ax[0]); ax[0].set_title("Recency")
sns.histplot(rfm["frequency"], ax=ax[1]); ax[1].set_title("Frequency")
sns.histplot(rfm["monetary"], ax=ax[2]); ax[2].set_title("Monetary")
st.pyplot(fig)

# =========================
# 5️⃣ REPEAT vs SEGMENT
# =========================
st.subheader("🔁 Repeat Order berdasarkan Segmentasi RFM")

fig, ax = plt.subplots(figsize=(8,5))
sns.barplot(
    data=repeat_segment_df,
    x="segment",
    y="is_repeat",
    ax=ax
)
ax.set_ylabel("Repeat Rate")
ax.set_xlabel("Customer Segment")
st.pyplot(fig)