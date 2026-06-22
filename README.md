# Supply Chain Analytics using Business Intelligence & Machine Learning

## Overview

This project analyzes real-world supply chain data to uncover insights related to sales performance, profitability, product demand, and delivery operations. By combining Business Intelligence techniques with Machine Learning, the project transforms raw operational data into actionable insights that support strategic business decisions.

The analysis is performed on the DataCo Supply Chain Dataset containing approximately 180,000 records and 53 attributes related to customer orders, products, shipping, and sales transactions.

---

## Business Problem

Modern supply chains generate large amounts of data from orders, products, customers, and logistics operations. However, businesses often struggle to identify high-performing products, understand regional sales trends, predict delivery risks, and optimize inventory levels.

This project addresses these challenges by analyzing supply chain data and extracting insights that can improve operational efficiency, profitability, and customer satisfaction.

---

## Dataset

**Dataset:** DataCo Supply Chain Dataset

**Dataset Size:** ~180,000 records | 53 attributes

**Key Features:**

* Order ID
* Product Name
* Customer Country
* Sales
* Profit Per Order
* Order Quantity
* Shipping Mode
* Delivery Status

---

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-Learn

---

## Data Preprocessing

To ensure reliable analysis, the dataset was cleaned and prepared by:

* Handling missing values
* Removing duplicate records
* Converting data into appropriate formats
* Selecting relevant business features
* Preparing data for machine learning models

---

## Machine Learning Techniques

### K-Means Clustering

K-Means clustering was used to group products based on sales and profit performance. This helps identify high-performing, medium-performing, and low-performing products, enabling businesses to make better inventory and product management decisions.

### Logistic Regression

Logistic Regression was used to predict delivery-related risks based on order characteristics. This helps organizations identify potential shipment issues early and improve logistics planning.

---

## Key Insights

The analysis revealed several valuable business insights:

* Identified products generating the highest sales and profit.
* Segmented products into performance-based groups using clustering.
* Analyzed delivery patterns to understand factors contributing to shipment delays.
* Examined sales trends and customer demand patterns across different regions.
* Highlighted opportunities to improve inventory allocation and supply chain efficiency.

---

## Visualizations & Dashboard

Interactive charts and dashboards were created to make the analysis easier to understand and support business decision-making.

Visualizations include:

* Sales and revenue trend analysis
* Product performance comparison
* Profitability analysis
* Delivery performance monitoring
* Business intelligence dashboards for operational insights

These visualizations help stakeholders quickly identify trends, opportunities, and areas requiring attention.

---

## Business Impact

The insights generated from this project can help businesses:

* **Improve Inventory Planning:** Maintain optimal stock levels for high-demand products and reduce stock shortages.
* **Increase Profitability:** Focus on products and regions that contribute the most revenue and profit.
* **Reduce Delivery Delays:** Use predictive insights to proactively manage shipment risks.
* **Enhance Decision-Making:** Provide data-driven recommendations through clear visualizations and analytics.
* **Optimize Supply Chain Operations:** Improve overall efficiency by identifying performance bottlenecks and demand patterns.

---

##🚀 Live Demo

Experience the interactive Supply Chain Analytics dashboard here:

🔗 Live Application:
https://supplychain-analytics.streamlit.app/
