import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    silhouette_score
)

# ------------------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------------------
st.set_page_config(
    page_title="Supply Chain BI Dashboard",
    page_icon="📦",
    layout="wide"
)

# ------------------------------------------------------------
# CUSTOM CSS
# ------------------------------------------------------------
st.markdown("""
<style>
.main {
    background-color: #f8f6ff;
}
.block-container {
    padding-top: 1rem;
    padding-bottom: 2rem;
}
.card {
    background: white;
    padding: 18px;
    border-radius: 18px;
    box-shadow: 0 4px 14px rgba(0,0,0,0.08);
    margin-bottom: 12px;
}
.metric-box {
    background: linear-gradient(135deg, #ece3ff, #ffffff);
    padding: 20px;
    border-radius: 18px;
    box-shadow: 0 4px 14px rgba(90, 60, 130, 0.12);
    text-align: center;
}
.metric-title {
    font-size: 16px;
    color: #5a3c82;
    font-weight: 600;
}
.metric-value {
    font-size: 28px;
    color: #2e1f47;
    font-weight: 700;
}
.small-text {
    color: #666;
    font-size: 13px;
}
h1, h2, h3 {
    color: #4b2e83;
}
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------
# TITLE
# ------------------------------------------------------------
st.title("📦 Supply Chain Data Analytics Dashboard")
st.caption("Business Intelligence Dashboard using Streamlit | DataCo Supply Chain Dataset")

# ------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------
def find_col(df, names):
    lower_map = {c.lower().strip(): c for c in df.columns}
    for n in names:
        if n.lower().strip() in lower_map:
            return lower_map[n.lower().strip()]
    for n in names:
        for c in df.columns:
            if n.lower().strip() in c.lower().strip():
                return c
    return None


# Cache for column mappings to avoid repeated find_col calls
@st.cache_data
def get_column_mappings(df):
    """Cache column mappings to avoid repeated find_col calls"""
    return {
        'country': find_col(df, ["Order Country", "Customer Country"]),
        'segment': find_col(df, ["Customer Segment"]),
        'category': find_col(df, ["Category Name"]),
        'shipmode': find_col(df, ["Shipping Mode"]),
        'sales': find_col(df, ["Sales", "Order Item Total", "Sales per customer"]),
        'profit': find_col(df, ["Order Profit Per Order", "Benefit per order"]),
        'risk': find_col(df, ["Late_delivery_risk"]),
        'order_qty': find_col(df, ["Order Item Quantity"]),
        'order_date': find_col(df, ["order date (dateorders)", "order date"]),
        'product': find_col(df, ["Product Name"]),
        'delivery_status': find_col(df, ["Delivery Status"])
    }


@st.cache_data
def get_filter_options(df, cols):
    """Precompute filter options to avoid repeated astype(str) conversions"""
    options = {}
    for key, col_name in cols.items():
        if col_name and col_name in df.columns:
            # Convert to string once and cache unique values
            options[key] = ["All"] + sorted(df[col_name].astype(str).dropna().unique().tolist())
        else:
            options[key] = ["All"]
    return options


def clean_country_column(df):
    """
    Clean and standardize country column values.
    Detects country column, converts to string, strips whitespace,
    maps variations to standard English names, and applies title case.
    """
    # Detect country column - prioritize Order Country as it has more diverse values
    country_col = find_col(df, ["Order Country", "Customer Country"])
    if not country_col:
        return df
    
    # Country name mapping dictionary
    country_mapping = {
        "ee. uu.": "United States",
        "estados unidos": "United States",
        "usa": "United States",
        "u.s.a.": "United States",
        "us": "United States",
        "united states of america": "United States",
        "américa": "United States",
        "japón": "Japan",
        "japon": "Japan",
        "india": "India",
        "indonesia": "Indonesia",
        "puerto rico": "Puerto Rico",
        "china": "China",
        "australia": "Australia",
        "méxico": "Mexico",
        "mexico": "Mexico",
        "canadá": "Canada",
        "canada": "Canada",
        "alemania": "Germany",
        "germany": "Germany",
        "francia": "France",
        "france": "France",
        "italia": "Italy",
        "españa": "Spain",
        "spain": "Spain",
        "reino unido": "United Kingdom",
        "united kingdom": "United Kingdom",
        "uk": "United Kingdom",
        "brasil": "Brazil",
        "brazil": "Brazil",
        "argentina": "Argentina",
        "colombia": "Colombia",
        "perú": "Peru",
        "peru": "Peru",
        "chile": "Chile",
        "ecuador": "Ecuador",
        "panamá": "Panama",
        "panama": "Panama",
        "costa rica": "Costa Rica",
        "nicaragua": "Nicaragua",
        "honduras": "Honduras",
        "el salvador": "El Salvador",
        "guatemala": "Guatemala",
        "belice": "Belize",
        "república dominicana": "Dominican Republic",
        "dominican republic": "Dominican Republic",
        "haití": "Haiti",
        "haiti": "Haiti",
        "jamaica": "Jamaica",
        "trinidad y tobago": "Trinidad and Tobago",
        "barbados": "Barbados",
        "bahamas": "Bahamas",
        "cuba": "Cuba",
        "corea del sur": "South Korea",
        "south korea": "South Korea",
        "corea del norte": "North Korea",
        "north korea": "North Korea",
        "taiwán": "Taiwan",
        "taiwan": "Taiwan",
        "hong kong": "Hong Kong",
        "singapur": "Singapore",
        "singapore": "Singapore",
        "malasia": "Malaysia",
        "malaysia": "Malaysia",
        "tailandia": "Thailand",
        "thailand": "Thailand",
        "vietnam": "Vietnam",
        "filipinas": "Philippines",
        "philippines": "Philippines",
        "rusia": "Russia",
        "russia": "Russia",
        "ucrania": "Ukraine",
        "ukraine": "Ukraine",
        "polonia": "Poland",
        "poland": "Poland",
        "suecia": "Sweden",
        "sweden": "Sweden",
        "noruega": "Norway",
        "norway": "Norway",
        "dinamarca": "Denmark",
        "denmark": "Denmark",
        "finlandia": "Finland",
        "finland": "Finland",
        "holanda": "Netherlands",
        "netherlands": "Netherlands",
        "bélgica": "Belgium",
        "belgium": "Belgium",
        "suiza": "Switzerland",
        "switzerland": "Switzerland",
        "austria": "Austria",
        "portugal": "Portugal",
        "irlanda": "Ireland",
        "irland": "Ireland",
        "grecia": "Greece",
        "greece": "Greece",
        "turquía": "Turkey",
        "turkey": "Turkey",
        "israel": "Israel",
        "arabia saudita": "Saudi Arabia",
        "saudi arabia": "Saudi Arabia",
        "emiratos árabes unidos": "United Arab Emirates",
        "united arab emirates": "United Arab Emirates",
        "uae": "United Arab Emirates",
        "egipto": "Egypt",
        "egypt": "Egypt",
        "sudáfrica": "South Africa",
        "south africa": "South Africa",
        "nigeria": "Nigeria",
        "kenia": "Kenya",
        "kenya": "Kenya",
        "ghana": "Ghana",
        "marruecos": "Morocco",
        "morocco": "Morocco",
        "argelia": "Algeria",
        "algeria": "Algeria",
        "túnez": "Tunisia",
        "tunisia": "Tunisia",
        "nueva zelanda": "New Zealand",
        "new zealand": "New Zealand"
    }
    
    # Clean the country column
    df = df.copy()
    df[country_col] = df[country_col].astype(str).str.strip().str.lower()
    
    # Apply mapping
    df[country_col] = df[country_col].map(country_mapping).fillna(df[country_col])
    
    # Apply title case
    df[country_col] = df[country_col].str.title()
    
    return df


@st.cache_data
def load_data():
    df = pd.read_csv("DataCoSupplyChainDataset.csv", encoding="latin1")
    return df


@st.cache_data
def preprocess_data(df):
    # Avoid unnecessary copy at the beginning - only copy when needed
    # remove duplicates
    df = df.drop_duplicates()

    # Clean country column first
    df = clean_country_column(df)

    # convert likely date columns
    date_cols = [
        find_col(df, ["order date (dateorders)", "order date"]),
        find_col(df, ["shipping date (dateorders)", "shipping date"])
    ]
    for c in date_cols:
        if c:
            df[c] = pd.to_datetime(df[c], errors="coerce")

    # numeric conversion - only convert columns that exist and need conversion
    numeric_candidates = [
        "Days for shipping (real)",
        "Days for shipment (scheduled)",
        "Benefit per order",
        "Sales per customer",
        "Late_delivery_risk",
        "Category Id",
        "Customer Id",
        "Department Id",
        "Latitude",
        "Longitude",
        "Order Item Cardprod Id",
        "Order Item Discount",
        "Order Item Discount Rate",
        "Order Item Id",
        "Order Item Product Price",
        "Order Item Profit Ratio",
        "Order Item Quantity",
        "Order Item Total",
        "Order Profit Per Order",
        "Order Zipcode",
        "Product Card Id",
        "Product Category Id",
        "Product Description",
        "Product Image",
        "Product Price",
        "Product Status",
        "shipping date (DateOrders)"
    ]

    # Only process columns that exist in the dataframe
    existing_numeric_cols = [c for c in numeric_candidates if c in df.columns]
    for c in existing_numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # fill missing values - optimize by doing this once for all numeric columns
    numeric_cols = df.select_dtypes(include=np.number).columns
    for c in numeric_cols:
        if df[c].isnull().sum() > 0:
            df[c] = df[c].fillna(df[c].median())

    # fill missing values for object columns - optimize by batch processing
    object_cols = df.select_dtypes(include="object").columns
    for c in object_cols:
        if df[c].isnull().sum() > 0:
            mode_val = df[c].mode()
            df[c] = df[c].fillna(mode_val.iloc[0] if not mode_val.empty else "Unknown")

    return df


def metric_card(title, value):
    st.markdown(
        f"""
        <div class="metric-box">
            <div class="metric-title">{title}</div>
            <div class="metric-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def safe_group_sum(df, group_col, value_col, top_n=None):
    temp = df.groupby(group_col)[value_col].sum().reset_index()
    temp = temp.sort_values(value_col, ascending=False)
    if top_n:
        temp = temp.head(top_n)
    return temp


def safe_group_count(df, group_col):
    return df[group_col].value_counts().reset_index().rename(
        columns={"index": group_col, group_col: "Count"}
    )


@st.cache_data
def run_kmeans(df):
    sales_col = find_col(df, ["Sales", "Order Item Total", "Sales per customer"])
    profit_col = find_col(df, ["Order Profit Per Order", "Benefit per order"])

    if not sales_col or not profit_col:
        return None

    cluster_df = df[[sales_col, profit_col]].copy().dropna()
    
    # Add sampling to limit to max 10,000 rows for performance
    cluster_df = cluster_df.sample(min(10000, len(cluster_df)), random_state=42)
    
    cluster_df.columns = ["Sales", "Profit"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(cluster_df)

    model = KMeans(n_clusters=3, random_state=42, n_init=10)
    labels = model.fit_predict(X_scaled)
    cluster_df["Cluster"] = labels.astype(str)

    sil = silhouette_score(X_scaled, labels)

    centers = scaler.inverse_transform(model.cluster_centers_)
    centers_df = pd.DataFrame(centers, columns=["Sales", "Profit"])
    centers_df["Cluster"] = centers_df.index.astype(str)

    return {
        "data": cluster_df,
        "silhouette": sil,
        "centers": centers_df
    }


@st.cache_data
def run_logistic(df):
    target_col = find_col(df, ["Late_delivery_risk"])
    feature_cols = [
        find_col(df, ["Days for shipping (real)"]),
        find_col(df, ["Days for shipment (scheduled)"]),
        find_col(df, ["Order Item Quantity"]),
        find_col(df, ["Order Item Total", "Sales"]),
        find_col(df, ["Benefit per order", "Order Profit Per Order"]),
        find_col(df, ["Product Price", "Order Item Product Price"])
    ]
    feature_cols = [c for c in feature_cols if c is not None]

    if target_col is None or len(feature_cols) < 2:
        return None

    model_df = df[feature_cols + [target_col]].copy().dropna()

    for c in feature_cols + [target_col]:
        model_df[c] = pd.to_numeric(model_df[c], errors="coerce")

    model_df = model_df.dropna()

    X = model_df[feature_cols]
    y = model_df[target_col].astype(int)

    if y.nunique() < 2:
        return None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_sc, y_train)
    y_pred = model.predict(X_test_sc)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    coef_df = pd.DataFrame({
        "Feature": feature_cols,
        "Coefficient": model.coef_[0]
    }).sort_values("Coefficient", ascending=False)

    report_df = pd.DataFrame(classification_report(
        y_test, y_pred, output_dict=True, zero_division=0
    )).transpose()

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "cm": cm,
        "coef_df": coef_df,
        "report_df": report_df,
        "features": feature_cols
    }


def generate_bi_insights(df, total_sales, total_profit, late_risk_pct,
                         top_category_df, top_country_df, logistic_result, kmeans_result):
    insights = []

    if total_sales > 0:
        insights.append(f"Total sales generated from the filtered dataset are {total_sales:,.2f}, showing the overall business volume.")

    if total_profit > 0:
        insights.append(f"Total profit is {total_profit:,.2f}, which helps evaluate whether sales growth is producing healthy returns.")
    else:
        insights.append("Profit is low or negative in the filtered data, which indicates a need to review pricing, discounts, or operational efficiency.")

    if late_risk_pct >= 50:
        insights.append(f"Late delivery risk is {late_risk_pct:.2f}%, which is high and suggests that logistics performance needs improvement.")
    else:
        insights.append(f"Late delivery risk is {late_risk_pct:.2f}%, which is relatively manageable but should still be monitored.")

    if top_category_df is not None and not top_category_df.empty:
        top_cat = top_category_df.iloc[0, 0]
        top_cat_sales = top_category_df.iloc[0, 1]
        insights.append(f"The highest sales category is {top_cat} with sales of {top_cat_sales:,.2f}, so inventory and promotions can be focused there.")

    if top_country_df is not None and not top_country_df.empty:
        top_country = top_country_df.iloc[0, 0]
        top_country_sales = top_country_df.iloc[0, 1]
        insights.append(f"The top revenue-generating country is {top_country} with sales of {top_country_sales:,.2f}, making it an important market.")

    if logistic_result is not None:
        insights.append(
            f"The Logistic Regression model achieved accuracy of {logistic_result['accuracy']*100:.2f}%, "
            f"precision of {logistic_result['precision']*100:.2f}%, recall of {logistic_result['recall']*100:.2f}% "
            f"and F1-score of {logistic_result['f1']*100:.2f}% for late delivery prediction."
        )

    if kmeans_result is not None:
        insights.append(
            f"K-Means clustering formed 3 product/order performance groups with silhouette score "
            f"{kmeans_result['silhouette']:.4f}, showing meaningful separation in sales-profit behavior."
        )

    decisions = [
        "Focus more stock and marketing on high-performing categories and products.",
        "Investigate late deliveries in risky shipping patterns and improve logistics planning.",
        "Use delivery risk prediction to identify problematic orders early.",
        "Review low-profit clusters to reduce losses or redesign pricing and discount strategies.",
        "Concentrate business expansion efforts on the top-performing countries or markets."
    ]

    return insights, decisions


# ------------------------------------------------------------
# SIDEBAR
# ------------------------------------------------------------

raw_df = load_data()
df = preprocess_data(raw_df)

# Get column mappings once and cache them
cols = get_column_mappings(df)
# Precompute filter options to avoid repeated conversions
filter_options = get_filter_options(df, cols)

st.sidebar.header("🔎 Filters")

filtered_df = df.copy()

if cols['country']:
    selected_country = st.sidebar.selectbox("Country", filter_options['country'])
    if selected_country != "All":
        filtered_df = filtered_df[filtered_df[cols['country']].astype(str) == selected_country]

if cols['segment']:
    selected_segment = st.sidebar.selectbox("Customer Segment", filter_options['segment'])
    if selected_segment != "All":
        filtered_df = filtered_df[filtered_df[cols['segment']].astype(str) == selected_segment]

if cols['category']:
    selected_category = st.sidebar.selectbox("Category", filter_options['category'])
    if selected_category != "All":
        filtered_df = filtered_df[filtered_df[cols['category']].astype(str) == selected_category]

if cols['shipmode']:
    selected_shipmode = st.sidebar.selectbox("Shipping Mode", filter_options['shipmode'])
    if selected_shipmode != "All":
        filtered_df = filtered_df[filtered_df[cols['shipmode']].astype(str) == selected_shipmode]

st.sidebar.markdown("---")
st.sidebar.write(f"**Rows:** {filtered_df.shape[0]:,}")
st.sidebar.write(f"**Columns:** {filtered_df.shape[1]}")

# ------------------------------------------------------------
# MAIN KPI SECTION
# ------------------------------------------------------------
total_sales = filtered_df[cols['sales']].sum() if cols['sales'] else 0
total_profit = filtered_df[cols['profit']].sum() if cols['profit'] else 0
total_orders = len(filtered_df)
total_qty = filtered_df[cols['order_qty']].sum() if cols['order_qty'] else 0
late_risk_pct = filtered_df[cols['risk']].mean() * 100 if cols['risk'] else 0

k1, k2, k3, k4, k5 = st.columns(5)
with k1:
    metric_card("Total Orders", f"{total_orders:,}")
with k2:
    metric_card("Total Sales", f"{total_sales:,.2f}")
with k3:
    metric_card("Total Profit", f"{total_profit:,.2f}")
with k4:
    metric_card("Items Sold", f"{int(total_qty):,}")
with k5:
    metric_card("Late Delivery Risk %", f"{late_risk_pct:.2f}%")

# ------------------------------------------------------------
# CHART DATA
# ------------------------------------------------------------
monthly_sales = pd.DataFrame()
if cols['order_date'] and cols['sales']:
    temp = filtered_df[[cols['order_date'], cols['sales']]].dropna().copy()
    temp["Month"] = temp[cols['order_date']].dt.to_period("M").astype(str)
    monthly_sales = temp.groupby("Month")[cols['sales']].sum().reset_index()
    monthly_sales.columns = ["Month", "Sales"]

top_products_df = pd.DataFrame()
if cols['product'] and cols['sales']:
    top_products_df = safe_group_sum(filtered_df, cols['product'], cols['sales'], top_n=10)
    top_products_df.columns = ["Product", "Sales"]

top_category_df = pd.DataFrame()
if cols['category'] and cols['sales']:
    top_category_df = safe_group_sum(filtered_df, cols['category'], cols['sales'], top_n=10)
    top_category_df.columns = ["Category", "Sales"]

top_country_df = pd.DataFrame()
if cols['country'] and cols['sales']:
    top_country_df = safe_group_sum(filtered_df, cols['country'], cols['sales'], top_n=10)
    top_country_df.columns = ["Country", "Sales"]

delivery_status_df = pd.DataFrame()
if cols['delivery_status']:
    delivery_status_df = filtered_df[cols['delivery_status']].value_counts().reset_index()
    delivery_status_df.columns = ["Delivery Status", "Count"]

shipping_mode_df = pd.DataFrame()
if cols['shipmode']:
    shipping_mode_df = filtered_df[cols['shipmode']].value_counts().reset_index()
    shipping_mode_df.columns = ["Shipping Mode", "Count"]

# ------------------------------------------------------------
# MODELS
# ------------------------------------------------------------
kmeans_result = run_kmeans(filtered_df)
logistic_result = run_logistic(filtered_df)

# ------------------------------------------------------------
# DASHBOARD LAYOUT
# ------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Business Dashboard",
    "🤖 Machine Learning Results",
    "📌 BI Insights & Decisions",
    "🗂 Data Preview"
])

# ------------------------------------------------------------
# TAB 1
# ------------------------------------------------------------
with tab1:
    c1, c2 = st.columns(2)

    with c1:
        if not monthly_sales.empty:
            fig = px.line(
                monthly_sales,
                x="Month",
                y="Sales",
                markers=True,
                title="Monthly Sales Trend"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Monthly sales trend not available.")

    with c2:
        if not shipping_mode_df.empty:
            fig = px.pie(
                shipping_mode_df,
                names="Shipping Mode",
                values="Count",
                hole=0.5,
                title="Shipping Mode Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Shipping mode distribution not available.")

    c3, c4 = st.columns(2)

    with c3:
        if not top_category_df.empty:
            fig = px.bar(
                top_category_df,
                x="Sales",
                y="Category",
                orientation="h",
                title="Top Categories by Sales"
            )
            fig.update_layout(yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Category chart not available.")

    with c4:
        if not top_country_df.empty:
            fig = px.bar(
                top_country_df,
                x="Country",
                y="Sales",
                title="Top Countries by Sales"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Country chart not available.")

    c5, c6 = st.columns(2)

    with c5:
        if not top_products_df.empty:
            fig = px.bar(
                top_products_df,
                x="Sales",
                y="Product",
                orientation="h",
                title="Top 10 Products by Sales"
            )
            fig.update_layout(yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Top products chart not available.")

    with c6:
        if not delivery_status_df.empty:
            fig = px.bar(
                delivery_status_df,
                x="Delivery Status",
                y="Count",
                title="Delivery Status Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Delivery status chart not available.")

# ------------------------------------------------------------
# TAB 2
# ------------------------------------------------------------
with tab2:
    left, right = st.columns(2)

    with left:
        st.subheader("K-Means Clustering Result")
        if kmeans_result is not None:
            st.metric("Silhouette Score", f"{kmeans_result['silhouette']:.4f}")

            sample_cluster = kmeans_result["data"].sample(
                min(5000, len(kmeans_result["data"])),
                random_state=42
            )

            fig = px.scatter(
                sample_cluster,
                x="Sales",
                y="Profit",
                color="Cluster",
                title="Cluster Distribution: Sales vs Profit"
            )
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("**Cluster Centers**")
            st.dataframe(kmeans_result["centers"], use_container_width=True)
        else:
            st.warning("K-Means could not be run with the available columns.")

    with right:
        st.subheader("Logistic Regression Result")
        if logistic_result is not None:
            m1, m2 = st.columns(2)
            with m1:
                st.metric("Accuracy", f"{logistic_result['accuracy']*100:.2f}%")
                st.metric("Precision", f"{logistic_result['precision']*100:.2f}%")
            with m2:
                st.metric("Recall", f"{logistic_result['recall']*100:.2f}%")
                st.metric("F1 Score", f"{logistic_result['f1']*100:.2f}%")

            cm = logistic_result["cm"]
            cm_df = pd.DataFrame(
                cm,
                index=["Actual 0", "Actual 1"],
                columns=["Predicted 0", "Predicted 1"]
            )
            st.markdown("**Confusion Matrix**")
            st.dataframe(cm_df, use_container_width=True)

            fig = px.bar(
                logistic_result["coef_df"],
                x="Coefficient",
                y="Feature",
                orientation="h",
                title="Feature Influence on Late Delivery Risk"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Logistic Regression could not be run with the available columns.")

    if logistic_result is not None:
        st.subheader("Detailed Classification Report")
        st.dataframe(logistic_result["report_df"], use_container_width=True)

# ------------------------------------------------------------
# TAB 3
# ------------------------------------------------------------
with tab3:
    insights, decisions = generate_bi_insights(
        filtered_df,
        total_sales,
        total_profit,
        late_risk_pct,
        top_category_df,
        top_country_df,
        logistic_result,
        kmeans_result
    )

    st.subheader("Generated Business Insights")
    for i, insight in enumerate(insights, start=1):
        st.markdown(f"**{i}.** {insight}")

    st.subheader("BI Decisions Based on Results")
    for i, decision in enumerate(decisions, start=1):
        st.markdown(f"**{i}.** {decision}")

    st.subheader("Dashboard Summary")
    st.write("""
    This dashboard demonstrates the actual outcome of the BI mini project:
    - sales and profit analysis
    - shipping and delivery trend analysis
    - top categories, products, and countries
    - K-Means clustering for performance grouping
    - Logistic Regression for late delivery prediction
    - measurable model results such as accuracy, precision, recall, F1-score, and confusion matrix
    """)

# ------------------------------------------------------------
# TAB 4
# ------------------------------------------------------------
with tab4:
    st.subheader("Dataset Preview")
    st.dataframe(filtered_df.head(20), use_container_width=True)

    st.subheader("Dataset Information")
    info_df = pd.DataFrame({
        "Column": filtered_df.columns,
        "Dtype": [str(x) for x in filtered_df.dtypes],
        "Missing Values": filtered_df.isnull().sum().values
    })
    st.dataframe(info_df, use_container_width=True)

    st.subheader("Numerical Summary")
    st.dataframe(filtered_df.describe(include="all").transpose(), use_container_width=True)

# ------------------------------------------------------------
# FOOTER
# ------------------------------------------------------------
st.markdown("---")
st.caption("Built with Streamlit, Plotly, Pandas and Scikit-learn")