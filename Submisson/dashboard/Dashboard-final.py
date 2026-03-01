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

def rfm_segment(row):
    if row["R"] == 4 and row["F"] == 4 and row["M"] == 4:
        return "Best Customer"
    elif row["F"] == 4 and row["M"] >= 3:
        return "Loyal Customer"
    elif row["R"] >= 3 and row["F"] <= 2:
        return "Potential Customer"
    elif row["R"] == 1 and row["F"] >= 3:
        return "At Risk Customer"
    else:
        return "Low Value Customer"

rfm["segment"] = rfm.apply(rfm_segment, axis=1)


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
# 1️⃣ & 2️⃣ TREN ORDER (SIDE BY SIDE)
# =========================
col1, col2 = st.columns(2)

with col1:
    st.markdown("##### 📈 Perkembangan Revenue Bulanan")

    monthly_revenue = (
        main_df
        .assign(order_month=main_df["order_purchase_timestamp"].dt.to_period("M"))
        .groupby("order_month")["payment_value"]
        .sum()
        .reset_index()
    )

    monthly_revenue["order_month"] = monthly_revenue["order_month"].astype(str)

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(
        monthly_revenue["order_month"],
        monthly_revenue["payment_value"],
        marker="o"
    )

    ax.set_xlabel("Bulan", fontsize=9)
    ax.set_ylabel("Total Revenue", fontsize=9)
    ax.tick_params(axis="x", rotation=45, labelsize=6)
    ax.tick_params(axis="y", labelsize=6)
    ax.grid(True)

    st.pyplot(fig, use_container_width=False)


# --- Jumlah Order per Bulan ---
with col2:
    st.markdown("##### 📅 Jumlah Order per Bulan")  # judul lebih kecil

    monthly_orders = (
        main_df.groupby("year_month")["order_id"]
        .nunique()
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(5, 3))
    sns.barplot(
        data=monthly_orders,
        x="year_month",
        y="order_id",
        ax=ax
    )

    ax.set_xlabel("Bulan", fontsize=9)
    ax.set_ylabel("Jumlah Order", fontsize=9)

    ax.tick_params(axis="x", rotation=45, labelsize=5)
    ax.tick_params(axis="y", labelsize=6)

    st.pyplot(fig, use_container_width=False)

# =========================
# 7️⃣ DISTRIBUSI REVIEW & STATUS PENGIRIMAN
# =========================
col1, col2 = st.columns(2)

# --- Pie Chart Review Score ---
with col1:
    st.markdown("##### ⭐ Distribusi Review Score (1–5)") 
    
    review_counts = (
        all_df["review_score"]
        .value_counts()
        .sort_index()
    )
    review_percent = review_counts / review_counts.sum() * 100

    fig, ax = plt.subplots(figsize=(2.5, 2.5))
    ax.pie(
        review_percent,
        labels=review_percent.index,
        autopct="%1.1f%%",
        startangle=90,
        textprops={"fontsize": 5}
    )
    ax.axis("equal")

    st.pyplot(fig, use_container_width=False)


# --- Pie Chart Status Pengiriman ---
with col2:
    st.markdown("##### Rata-rata Review Score vs Keterlambatan Pengiriman") 

    fig, ax = plt.subplots(figsize=(4, 3))

    sns.barplot(
        data=all_df,
        x="is_late",
        y="review_score",
        estimator="mean",
        ax=ax
    )

    ax.set_xlabel("Terlambat (True = Ya)", fontsize=8)
    ax.set_ylabel("Rata-rata Review Score", fontsize=8)
    ax.tick_params(axis="x", labelsize=7)
    ax.tick_params(axis="y", labelsize=7)

    st.pyplot(fig, use_container_width=False)
    

st.markdown("##### ⭐ Review Score per Kategori (Tepat Waktu vs Terlambat)")

# spacer kiri - konten - spacer kanan
_, center_col, _ = st.columns([1, 3, 1])

with center_col:
    top_category = (
        all_df["product_category_name"]
        .value_counts()
        .head(10)
        .index
    )

    fig, ax = plt.subplots(figsize=(5, 3))

    sns.barplot(
        data=all_df[all_df["product_category_name"].isin(top_category)],
        x="product_category_name",
        y="review_score",
        hue="is_late",
        ax=ax
    )

    ax.set_xlabel("Kategori Produk", fontsize=9)
    ax.set_ylabel("Rata-rata Review Score", fontsize=7)
    ax.tick_params(axis="x", rotation=45, labelsize=6)
    ax.tick_params(axis="y", labelsize=6)
    ax.legend(title="Terlambat", fontsize=7, title_fontsize=7)

    fig.tight_layout(pad=1)

    st.pyplot(fig, use_container_width=False)

# =========================
# 3️⃣ TOP KATEGORI PRODUK
# =========================
st.markdown("##### Distribusi Total Belanja Customer (≤ P95)")

_, center_col, _ = st.columns([1, 3, 1])

with center_col:
    customer_spending = (
        all_df
        .groupby("customer_unique_id")["payment_value"]
        .sum()
        .reset_index()
        .rename(columns={"payment_value": "total_spending"})
    )

    upper_limit = customer_spending["total_spending"].quantile(0.95)

    fig, ax = plt.subplots(figsize=(5, 3))

    ax.hist(
        customer_spending.loc[
            customer_spending["total_spending"] <= upper_limit,
            "total_spending"
        ],
        bins=40
    )

    ax.set_xlabel("Total Belanja", fontsize=9)
    ax.set_ylabel("Jumlah Customer", fontsize=9)
    ax.tick_params(axis="x", labelsize=8)
    ax.tick_params(axis="y", labelsize=8)

    fig.tight_layout(pad=1)

    st.pyplot(fig, use_container_width=False)


col1, col2 = st.columns(2)

# =========================
# SEGMENT CUSTOMER
# =========================
with col1:
    st.markdown("##### Jumlah Customer per Segment")

    q60 = customer_spending["total_spending"].quantile(0.6)
    q90 = customer_spending["total_spending"].quantile(0.9)

    def segment_customer(x):
        if x <= q60:
            return "Low Value"
        elif x <= q90:
            return "Mid Value"
        else:
            return "High Value"

    customer_spending["segment"] = customer_spending["total_spending"].apply(segment_customer)

    segment_count = customer_spending["segment"].value_counts()

    fig, ax = plt.subplots(figsize=(5,4))
    ax.bar(segment_count.index, segment_count.values)
    ax.set_ylabel("Jumlah Customer")

    st.pyplot(fig)


# --- Review Score per Kategori berdasarkan Keterlambatan ---
with col2:
    st.markdown("##### Kontribusi Revenue per Segment")

    segment_revenue = (
        customer_spending
        .groupby("segment")["total_spending"]
        .sum()
    )

    fig, ax = plt.subplots(figsize=(5,4))
    ax.bar(segment_revenue.index, segment_revenue.values)
    ax.set_ylabel("Total Revenue")

    st.pyplot(fig)
# =========================
# 4️⃣ LOGISTIK vs REVIEW
# =========================

# =========================
# PAYMENT SUMMARY
# =========================
# =========================
# PAYMENT SUMMARY
# =========================
st.markdown("#### 💳 Perbandingan Metode Pembayaran")

payment_summary = (
    all_df
    .groupby("payment_type")
    .agg(
        avg_transaction_value=("payment_value", "mean"),
        total_orders=("order_id", "nunique"),
        total_transactions=("payment_value", "count")
    )
    .reset_index()
    .sort_values(by="avg_transaction_value", ascending=False)
)

col1, col2 = st.columns(2)

# =========================
# CHART 1: Rata-rata Nilai Transaksi
# =========================
with col1:
    fig, ax = plt.subplots(figsize=(5, 4))

    ax.bar(
        payment_summary["payment_type"],
        payment_summary["avg_transaction_value"]
    )

    ax.set_title("Rata-rata Nilai Transaksi", fontsize=10)
    ax.set_xlabel("Metode Pembayaran", fontsize=9)
    ax.set_ylabel("Rata-rata Nilai Transaksi", fontsize=9)
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    ax.tick_params(axis="y", labelsize=8)

    for i, v in enumerate(payment_summary["avg_transaction_value"]):
        ax.text(i, v, f"{v:.0f}", ha="center", va="bottom", fontsize=7)

    fig.tight_layout()
    st.pyplot(fig)


# =========================
# CHART 2: Frekuensi Pembelian
# =========================
with col2:
    fig, ax = plt.subplots(figsize=(5, 4))

    ax.bar(
        payment_summary["payment_type"],
        payment_summary["total_orders"]
    )

    ax.set_title("Frekuensi Pembelian", fontsize=10)
    ax.set_xlabel("Metode Pembayaran", fontsize=9)
    ax.set_ylabel("Jumlah Order", fontsize=9)
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    ax.tick_params(axis="y", labelsize=8)

    for i, v in enumerate(payment_summary["total_orders"]):
        ax.text(i, v, f"{v}", ha="center", va="bottom", fontsize=7)

    fig.tight_layout()
    st.pyplot(fig)

col1, col2 = st.columns(2)

# =========================
# LOGISTIK vs REVIEW
# =========================
with col1:
    st.markdown("##### 🚚 Keterlambatan Pengiriman vs Rata-rata Review Score")

    late_review = (
        main_df
        .groupby("is_late")["review_score"]
        .mean()
        .reset_index()
    )

    late_review["Status Pengiriman"] = late_review["is_late"].map({
        False: "Tepat Waktu",
        True: "Terlambat"
    })

    fig, ax = plt.subplots(figsize=(6,4))
    sns.barplot(
        data=late_review,
        x="Status Pengiriman",
        y="review_score",
        ax=ax
    )

    ax.set_ylabel("Rata-rata Review Score")

    st.pyplot(fig)


# =========================
# PAYMENT
# =========================
with col2:
    st.markdown("##### Tepat Waktu vs Terlambat")

    delivery_counts = all_df["delivery_status"].value_counts()
    delivery_percent = delivery_counts / delivery_counts.sum() * 100

    fig, ax = plt.subplots(figsize=(2.5, 2.5))

    ax.pie(
        delivery_percent,
        labels=delivery_percent.index,
        autopct="%1.1f%%",
        startangle=90,
        textprops={"fontsize": 7}
    )
    ax.axis("equal")  # Supaya bulat

    st.pyplot(fig, use_container_width=False)

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
# 6️⃣ SEGMENTASI CUSTOMER (RFM)
# =========================
st.subheader("🔁 Analisis Segmentasi Customer (RFM)")

col1, col2, col3 = st.columns(3)

# =========================
# 1️⃣ JUMLAH CUSTOMER PER SEGMENT
# =========================
with col1:
    st.markdown("##### Jumlah Customer")

    segment_count = rfm["segment"].value_counts().reset_index()
    segment_count.columns = ["Segment", "Jumlah Customer"]

    fig, ax = plt.subplots(figsize=(4, 3))
    sns.barplot(
        data=segment_count,
        x="Segment",
        y="Jumlah Customer",
        ax=ax
    )

    ax.tick_params(axis="x", labelsize=6, rotation=30)
    ax.tick_params(axis="y", labelsize=6)
    ax.set_xlabel("")
    ax.set_ylabel("")

    st.pyplot(fig)


# =========================
# 2️⃣ TOTAL REVENUE PER SEGMENT
# =========================
with col2:
    st.markdown("##### Total Revenue")

    rfm_segment_revenue = (
        rfm
        .groupby("segment", as_index=False)["monetary"]
        .sum()
    )

    rfm_segment_revenue.columns = ["Segment", "Total Revenue"]

    fig, ax = plt.subplots(figsize=(4, 3))
    sns.barplot(
        data=rfm_segment_revenue,
        x="Segment",
        y="Total Revenue",
        ax=ax
    )

    ax.tick_params(axis="x", labelsize=6, rotation=30)
    ax.tick_params(axis="y", labelsize=6)
    ax.ticklabel_format(style="plain", axis="y")
    ax.set_xlabel("")
    ax.set_ylabel("")

    st.pyplot(fig)


# =========================
# 3️⃣ RATA-RATA JUMLAH ORDER
# =========================
with col3:
    st.markdown("##### Rata-rata Jumlah Order")

    order_count = (
        all_df
        .groupby("customer_unique_id")["order_id"]
        .nunique()
        .reset_index(name="total_orders")
    )

    rfm_repeat = rfm.merge(
        order_count,
        on="customer_unique_id",
        how="left"
    )

    repeat_by_segment = (
        rfm_repeat
        .groupby("segment")["total_orders"]
        .mean()
        .sort_values(ascending=False)
    )

    fig, ax = plt.subplots(figsize=(4, 3))
    repeat_by_segment.plot(kind="bar", ax=ax)

    ax.tick_params(axis="x", labelsize=6, rotation=30)
    ax.tick_params(axis="y", labelsize=6)
    ax.set_xlabel("")
    ax.set_ylabel("")

    st.pyplot(fig)