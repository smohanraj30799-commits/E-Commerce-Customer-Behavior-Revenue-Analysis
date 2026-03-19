import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt

# Load dataset
df = pd.read_csv('online_retail.csv', encoding='ISO-8859-1')

# ----------------------
# DATA CLEANING
# ----------------------
df = df.dropna()
df = df[df['Quantity'] > 0]
df = df[df['UnitPrice'] > 0]

# Create Total Price
df['TotalPrice'] = df['Quantity'] * df['UnitPrice']

# Convert date
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

# ----------------------
# REVENUE ANALYSIS
# ----------------------
df['Month'] = df['InvoiceDate'].dt.to_period('M')

monthly_revenue = df.groupby('Month')['TotalPrice'].sum()

plt.figure()
monthly_revenue.plot()
plt.title("Monthly Revenue Trend")
plt.xlabel("Month")
plt.ylabel("Revenue")
plt.show()

# ----------------------
# TOP PRODUCTS
# ----------------------
top_products = df.groupby('Description')['Quantity'].sum().sort_values(ascending=False).head(10)

plt.figure()
top_products.plot(kind='bar')
plt.title("Top 10 Selling Products")
plt.show()

# ----------------------
# COUNTRY ANALYSIS
# ----------------------
country_sales = df.groupby('Country')['TotalPrice'].sum().sort_values(ascending=False).head(10)

plt.figure()
country_sales.plot(kind='bar')
plt.title("Top Countries by Revenue")
plt.show()

# ----------------------
# RFM ANALYSIS
# ----------------------
snapshot_date = df['InvoiceDate'].max() + dt.timedelta(days=1)

rfm = df.groupby('CustomerID').agg({
    'InvoiceDate': lambda x: (snapshot_date - x.max()).days,
    'InvoiceNo': 'nunique',
    'TotalPrice': 'sum'
})

rfm.columns = ['Recency', 'Frequency', 'Monetary']

# RFM Scoring
rfm['R_score'] = pd.qcut(rfm['Recency'], 4, labels=[4,3,2,1])
rfm['F_score'] = pd.qcut(rfm['Frequency'], 4, labels=[1,2,3,4])
rfm['M_score'] = pd.qcut(rfm['Monetary'], 4, labels=[1,2,3,4])

rfm['RFM_Score'] = rfm[['R_score','F_score','M_score']].astype(int).sum(axis=1)

# ----------------------
# CUSTOMER SEGMENTS
# ----------------------
def segment(score):
    if score >= 10:
        return "High Value"
    elif score >= 6:
        return "Mid Value"
    else:
        return "Low Value"

rfm['Segment'] = rfm['RFM_Score'].apply(segment)

segment_counts = rfm['Segment'].value_counts()

plt.figure()
segment_counts.plot(kind='bar')
plt.title("Customer Segments")
plt.show()

# ----------------------
# SAVE OUTPUT
# ----------------------
rfm.to_csv("rfm_output.csv")

print("Analysis Completed Successfully!")