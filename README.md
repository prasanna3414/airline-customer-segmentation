# README  
## AeroSeg – Airline Customer Segmentation and Intelligence

The purpose of this project is to develop a machine learning–based system that can effectively segment airline customers based on their behavioral characteristics using **K-Means clustering**. The project aims to support data-driven decision-making for airline marketing, retention, and customer relationship management.

---

## Introduction

Airline companies manage large volumes of customer data with diverse travel behaviors. Identifying valuable customer segments such as loyal customers, potential loyalists, and at-risk customers is essential for improving profitability and customer retention.

The objective of this project is to build an end-to-end **customer segmentation system** that groups airline customers into meaningful segments based on loyalty, recency, frequency, monetary value, and discount usage. The final solution is delivered through an **interactive Streamlit application** that provides both analytical insights and business recommendations.

---

## Data

The dataset used in this project is inspired by publicly available airline customer datasets.  
It contains customer-level airline behavior data with the following attributes:

- Frequent Flyer enrollment date
- Last flight activity
- Number of flights taken
- Revenue generated across multiple years
- Average discount usage

To ensure robustness and avoid schema inconsistencies, the project uses a **standardized CSV template** that users must follow when uploading data.

The dataset does not contain categorical labels and is suitable for **unsupervised learning**.

---

## Methods

The following steps were used to develop the customer segmentation system:

### Data Preprocessing
- Converted date columns to datetime format
- Engineered behavioral features such as **Loyalty Days** and **Monetary Value**
- Handled missing values using **median imputation**
- Standardized numerical features using **StandardScaler**

### Feature Engineering
Customer behavior was represented using five key features:
- Loyalty (tenure with airline)
- Recency (days since last flight)
- Frequency (number of flights)
- Monetary value (total revenue)
- Discount usage

### Model Training
- **K-Means clustering** was applied to segment customers
- The number of clusters was fixed at **five**, based on business interpretability
- Customers were assigned to clusters and mapped to meaningful segment labels

### Model Interpretation
Cluster-wise averages were calculated to interpret each segment’s behavioral characteristics.  
A **cluster profile table** was generated to summarize customer behavior across segments.

---

## Results

The model successfully segmented airline customers into five meaningful groups:

1. Loyal High-Value Customers  
2. Regular Customers  
3. Potential Loyalists  
4. At-Risk Customers  
5. Inactive Customers  

Each segment exhibited distinct behavioral patterns in terms of loyalty, flight frequency, revenue contribution, and recency. The segmentation results were visualized through interactive charts and summarized using cluster profiles.

The final application allows users to:
- View customer distribution across segments
- Analyze dominant behavioral traits per segment
- Download segmentation results for further analysis

---

## Conclusion

This project demonstrates that **K-Means clustering** is an effective approach for airline customer segmentation when combined with meaningful behavioral features. The resulting segments provide actionable insights that can be used for targeted marketing, customer retention, and strategic decision-making.

By integrating machine learning with an interactive dashboard, the project bridges the gap between technical analysis and real-world business applications in the airline industry.

---

## About

This project was developed as an **end-to-end data science and machine learning application**, covering data preprocessing, feature engineering, clustering, interpretation, and deployment using Streamlit.

It is intended for **academic learning, portfolio demonstration, and practical understanding** of unsupervised machine learning in a business context.

---

## Languages Used

- Python  
- Jupyter Notebook  
- Streamlit  

---

© 2025 GitHub, Inc.
