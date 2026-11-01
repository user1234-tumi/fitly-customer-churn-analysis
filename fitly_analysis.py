import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style='whitegrid')

# STEP 1 — LOAD DATA
account_info = pd.read_csv('C:/Users/mkhab/Downloads/Data analytics/certificate/da_fitly_account_info.csv')
account_info = account_info.rename(
    columns={"churn_status": "churn"}
)
customer_support = pd.read_csv('C:/Users/mkhab/Downloads/Data analytics/certificate/da_fitly_customer_support.csv')
user_activity = pd.read_csv('C:/Users/mkhab/Downloads/Data analytics/certificate/da_fitly_user_activity.csv')

expected = {
    'account_info': ['customer_id','email','state','plan','plan_list_price','churn'],
    'customer_support': ['ticket_time','user_id','channel','topic','resolution_time_hours','state'],
    'user_activity': ['event_time','user_id','event_type']
}
raw = {'account_info':account_info,'customer_support':customer_support,'user_activity':user_activity}
for name, df in raw.items():
    df.columns = df.columns.str.strip()
    missing_columns = set(expected[name]) - set(df.columns)
    if missing_columns:
        raise ValueError(f'{name}: missing required columns {missing_columns}')
    print(f'\n=== {name}: raw validation ===')
    print('Shape:',df.shape, 'Duplicates:',df.duplicated().sum())
    print('Missing by column:\n',df.isna().sum())
    print('Data types:\n',df.dtypes)
    print('Examples:\n',df.head())

# STEP 2 — CLEAN / VALIDATE
account_clean = account_info.copy()
support_clean = customer_support.copy()
activity_clean = user_activity.copy()
for df, cols in [(account_clean,['customer_id','email','state','plan']),
                 (support_clean,['user_id','channel','topic','state']),
                 (activity_clean,['user_id','event_type'])]:
    for col in cols:
        df[col] = df[col].astype('string').str.strip()
        df[col] = df[col].replace('',pd.NA)
account_clean['email'] = account_clean['email'].str.lower()
account_clean['plan_list_price'] = pd.to_numeric(account_clean['plan_list_price'],errors='coerce')
support_clean['resolution_time_hours'] = pd.to_numeric(support_clean['resolution_time_hours'],errors='coerce')
support_clean['ticket_time'] = pd.to_datetime(support_clean['ticket_time'],errors='coerce',utc=True)
activity_clean['event_time'] = pd.to_datetime(activity_clean['event_time'],errors='coerce',utc=True)

print('\nRaw churn labels:\n',account_clean['churn'].value_counts(dropna=False))
# Map standard churn labels; unknown labels remain missing and require review.
# Convert churn status:
# Y = churned (1)
# Blank = not recorded as churned (0), assumed for this analysis

churn_text = (
    account_clean["churn"]
    .astype("string")
    .str.strip()
    .str.upper()
)

print("Original churn values:")
print(churn_text.value_counts(dropna=False))

# Check for unexpected values
unexpected = churn_text[
    churn_text.notna() & ~churn_text.isin(["Y", ""])
]

if not unexpected.empty:
    raise ValueError(
        f"Unexpected churn values: {unexpected.unique().tolist()}"
    )

account_clean["churn"] = (
    churn_text.eq("Y")
    .fillna(False)
    .astype(int)
)

print("Churn after cleaning:")
print(account_clean["churn"].value_counts())
print('Unrecognised/missing churn rows:',account_clean['churn'].isna().sum())
print('Negative plan prices:',account_clean['plan_list_price'].lt(0).sum())
print('Negative resolution hours:',support_clean['resolution_time_hours'].lt(0).sum())
print('Missing/invalid ticket dates:',support_clean['ticket_time'].isna().sum())
print('Missing/invalid activity dates:',activity_clean['event_time'].isna().sum())
print('Duplicate account customer IDs:',account_clean['customer_id'].duplicated().sum())
if account_clean['customer_id'].isna().any() or account_clean['customer_id'].duplicated().any():
    raise ValueError('Resolve missing/duplicate customer_id values before merging.')

for name, df in [('account_clean',account_clean),('support_clean',support_clean),('activity_clean',activity_clean)]:
    print(f'\n=== {name}: post-conversion review ===')
    print(df.isna().sum())
    for col in ['state','plan','channel','topic','event_type']:
        if col in df:
            print(f'{col} values:\n',df[col].value_counts(dropna=False).head(20))

# STEP 3 — AGGREGATE BEFORE MERGING (avoid many-to-many duplication)
# STEP 3 — STANDARDISE IDs AND MERGE DATASETS

# Create matching IDs without changing original customer IDs
account_clean['match_id'] = (
    account_clean['customer_id']
    .astype('string')
    .str.strip()
    .str.replace(r'^C(?=\d+$)', '', regex=True)
)

support_clean['match_id'] = (
    support_clean['user_id']
    .astype('string')
    .str.strip()
)

activity_clean['match_id'] = (
    activity_clean['user_id']
    .astype('string')
    .str.strip()
)

# Check matching records
print("\nCUSTOMER ID MATCHES")
print("Customers:", account_clean['customer_id'].nunique())

print(
    "Customers with activity:",
    account_clean['match_id'].isin(
        activity_clean['match_id']
    ).sum()
)

print(
    "Customers with support:",
    account_clean['match_id'].isin(
        support_clean['match_id']
    ).sum()
)

# Summarise support tickets by customer
support_summary = (
    support_clean
    .dropna(subset=['match_id'])
    .groupby('match_id')
    .agg(
        total_tickets=('match_id', 'size'),
        avg_resolution_time=('resolution_time_hours', 'mean')
    )
    .reset_index()
)

# Summarise activity events by customer
activity_summary = (
    activity_clean
    .dropna(subset=['match_id'])
    .groupby('match_id')
    .agg(
        total_activities=('match_id', 'size'),
        last_activity=('event_time', 'max')
    )
    .reset_index()
)

# Merge with account data
merged_df = account_clean.merge(
    support_summary,
    on='match_id',
    how='left',
    validate='one_to_one'
)

merged_df = merged_df.merge(
    activity_summary,
    on='match_id',
    how='left',
    validate='one_to_one'
)

# Zero means no matching record in the supplied extracts
merged_df['total_tickets'] = (
    merged_df['total_tickets'].fillna(0).astype(int)
)

merged_df['total_activities'] = (
    merged_df['total_activities'].fillna(0).astype(int)
)

# Check that merging has not duplicated customers
assert len(merged_df) == len(account_clean)
assert merged_df['customer_id'].is_unique

print("\nMerged dataset shape:", merged_df.shape)

# STEP 4 — EXPLORATORY ANALYSIS
analysis_df = merged_df.dropna(subset=['churn']).copy()
analysis_df['churn'] = analysis_df['churn'].astype(int)
print('\nChurn counts:\n',merged_df['churn'].value_counts(dropna=False))
print('\nChurn by plan:\n',analysis_df.groupby('plan',dropna=False)['churn'].agg(['count','mean']))
print('\nEngagement by churn:\n',analysis_df.groupby('churn')['total_activities'].agg(['count','mean','median']))
print('\nSupport by churn:\n',analysis_df.groupby('churn').agg(avg_tickets=('total_tickets','mean'),avg_resolution_hours=('avg_resolution_time','mean')))
print('\nChurn by account state:\n',analysis_df.groupby('state',dropna=False)['churn'].agg(['count','mean']).sort_values('mean',ascending=False))

# STEP 5 — REQUIRED GRAPHS
plt.figure(figsize=(7,5))
sns.countplot(data=analysis_df,x='churn',order=[0,1])
plt.title('Customer Churn Distribution');plt.xlabel('Churn (0=Retained, 1=Churned)');plt.ylabel('Customers');plt.tight_layout();plt.show()

plan_counts = merged_df['plan'].fillna('Unknown').value_counts()
if not plan_counts.empty:
    plt.figure(figsize=(7,7))
    plt.pie(plan_counts.values,labels=plan_counts.index,autopct='%1.1f%%',startangle=90)
    plt.title('Subscription Plan Distribution');plt.tight_layout();plt.show()

plan_churn = analysis_df.groupby('plan',dropna=False)['churn'].agg(['size','mean']).reset_index()
plan_churn['churn_rate'] = plan_churn['mean']*100
plt.figure(figsize=(8,5))
sns.barplot(data=plan_churn,x='plan',y='churn_rate',errorbar=None)
plt.title('Churn Rate by Subscription Plan');plt.xlabel('Plan');plt.ylabel('Churn Rate (%)');plt.tight_layout();plt.show()

plt.figure(figsize=(7,5))
sns.boxplot(data=analysis_df,x='churn',y='total_activities',order=[0,1])
plt.title('Activity Count by Churn Status');plt.xlabel('Churn (0=Retained, 1=Churned)');plt.ylabel('Recorded Activities');plt.tight_layout();plt.show()

plt.figure(figsize=(7,5))
sns.barplot(data=analysis_df,x='churn',y='total_tickets',order=[0,1],errorbar=None)
plt.title('Average Support Tickets by Churn Status');plt.xlabel('Churn (0=Retained, 1=Churned)');plt.ylabel('Average Tickets');plt.tight_layout();plt.show()

# STEP 6 — BUSINESS KPIs
n = len(analysis_df)
churned = int(analysis_df['churn'].sum())
retained = n-churned
churn_rate = churned/n*100 if n else np.nan
retention_rate = retained/n*100 if n else np.nan
kpi_summary = pd.DataFrame({'Metric':['Customers with known churn status','Churned customers','Retained customers','Churn proportion (%)','Retention proportion (%)','Mean activity events/customer','Mean support tickets/customer','Mean resolution hours (customers with tickets)'],
                            'Value':[n,churned,retained,churn_rate,retention_rate,merged_df['total_activities'].mean(),merged_df['total_tickets'].mean(),merged_df['avg_resolution_time'].mean()]})
print('\n=== KPI SUMMARY ===\n',kpi_summary.to_string(index=False))
print('\n=== PLAN CHURN ===\n',plan_churn.to_string(index=False))
# These are proportions in the supplied snapshot, not time-specific monthly churn rates.
# Validate time windows, missing records, plan prices, categories, and churn mapping before reporting.
kpi_summary.to_csv('fitly_kpi_summary.csv',index=False)
plan_churn.to_csv('fitly_plan_churn.csv',index=False)
print('\nSaved KPI tables. Use actual results to complete the report and slides.')
# FINAL DATA CHECK

print("\n===== FINAL DATA CHECK =====")

print("Customers:", len(account_clean))

print("Customers matching activity:",
      account_clean["match_id"].isin(
          activity_clean["match_id"]
      ).sum())

print("Customers matching support:",
      account_clean["match_id"].isin(
          support_clean["match_id"]
      ).sum())

print("Marked as churned:",
      (account_clean["churn"] == 1).sum())

print("Not marked as churned:",
      (account_clean["churn"] == 0).sum())
print("\nCUSTOMER ID MATCHES")

print("Customers:", account_clean["customer_id"].nunique())

print(
    "Customers with activity:",
    account_clean["customer_id"].isin(
        activity_clean["user_id"]
    ).sum()
)

print(
    "Customers with support:",
    account_clean["customer_id"].isin(
        support_clean["user_id"]
    ).sum()
)

print("\nCHURN COUNTS")
print(account_clean["churn"].value_counts())
print("\nACCOUNT CUSTOMER IDs")
print(account_clean["customer_id"].head(5).tolist())

print("\nACTIVITY USER IDs")
print(activity_clean["user_id"].head(5).tolist())

print("\nSUPPORT USER IDs")
print(support_clean["user_id"].head(5).tolist())
