Fit.ly Customer Churn Analysis

Project Overview

This project explores customer churn patterns using Python and data analysis techniques. It was developed as part of a DataCamp Data Analyst certification practical assessment.

The objective was to clean and analyse customer information, investigate differences in churn across subscription plans, and communicate findings through descriptive statistics and data visualisations.

Tools and Technologies

- Python
- Pandas and NumPy
- Matplotlib and Seaborn
- Data cleaning and validation
- Exploratory data analysis
- Data visualisation

Dataset

The analysis uses three datasets:

- Customer accounts: 400 customer records containing subscription information and churn indicators.
- Customer support: 918 support records.
- User activity: 445 activity records.

The original assessment datasets are not included in this repository.

Methodology

1. Imported and inspected the datasets for missing values, duplicates and inconsistent data types.
2. Cleaned customer information and standardised churn indicators.
3. Standardised customer identifiers to enable matching across datasets.
4. Aggregated customer support and activity records before merging.
5. Conducted exploratory analysis of subscription plans and customer churn.
6. Calculated descriptive KPIs and produced visualisations.

Key Findings

- Of 400 customer accounts, 114 were marked as churned, representing 28.5%.
- The Free subscription plan had the highest observed churn proportion (41.0%).
- The Pro subscription plan had an observed churn proportion of 22.4%.
- Customer records were successfully matched to support and activity datasets after standardising identifiers.

Important limitation: The analysis treats blank churn indicators as customers not marked as churned. Missing support or activity records do not necessarily indicate an absence of customer interactions. These results are descriptive and do not establish the causes of churn.

Skills Demonstrated

This project demonstrates practical experience in data cleaning, dataset validation, data integration, exploratory data analysis, KPI calculation and communicating analytical findings.

Project Status

Initial analysis completed. Documentation and visualisations are being refined for portfolio presentation.


## Data Visualisations

### 1. Customer Churn Distribution
![Customer Churn Distribution](figures/figures/01_churn_distribution.png)

### 2. Subscription Plan Distribution
![Subscription Plan Distribution](figures/figures/02_plan_distribution.png)

### 3. Churn Rate by Subscription Plan
![Churn Rate by Subscription Plan](figures/figures/03_churn_by_plan.png)

### 4. Activity Count by Churn Status
![Activity Count by Churn Status](figures/figures/04_activity_by_churn.png)

### 5. Average Support Tickets by Churn Status
![Average Support Tickets by Churn Status](figures/figures/05_support_by_churn.png)

