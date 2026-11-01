"""Fit.ly customer churn analysis — revised from a DataCamp certification project.

Place the three original CSVs in a local `data/` directory. Do not publish
those files without permission from the data provider.
"""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data'
OUTPUT = ROOT / 'outputs'
OUTPUT.mkdir(exist_ok=True)


def load(name, required):
    df = pd.read_csv(DATA / name)
    df.columns = df.columns.str.strip()
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f'{name}: missing columns {sorted(missing)}')
    print(f'{name}: {len(df)} rows, {df.duplicated().sum()} exact duplicates')
    return df


def normalized_id(series):
    """Map C10000 and 10000 to the same ID, without altering original IDs."""
    return series.astype('string').str.strip().str.replace(r'^C(?=\d+$)', '', regex=True)


def main():
    accounts = load('da_fitly_account_info.csv',
                    ['customer_id', 'plan', 'plan_list_price', 'churn_status'])
    support = load('da_fitly_customer_support.csv',
                   ['user_id', 'resolution_time_hours', 'ticket_time'])
    activity = load('da_fitly_user_activity.csv', ['user_id', 'event_time'])

    accounts['match_id'] = normalized_id(accounts['customer_id'])
    support['match_id'] = normalized_id(support['user_id'])
    activity['match_id'] = normalized_id(activity['user_id'])
    if accounts['match_id'].isna().any() or accounts['match_id'].duplicated().any():
        raise ValueError('Customer match IDs must be present and unique')

    # Dataset assumption: Y = churned; blank = not marked as churned.
    labels = accounts['churn_status'].astype('string').str.strip().str.upper()
    invalid = labels.notna() & ~labels.isin(['Y', ''])
    if invalid.any():
        raise ValueError(f'Unexpected churn labels: {labels[invalid].unique().tolist()}')
    accounts['churn'] = labels.eq('Y').fillna(False).astype(int)
    accounts['plan'] = accounts['plan'].astype('string').str.strip().fillna('Unknown')
    accounts['plan_list_price'] = pd.to_numeric(accounts['plan_list_price'], errors='coerce')
    support['resolution_time_hours'] = pd.to_numeric(support['resolution_time_hours'], errors='coerce')
    support['ticket_time'] = pd.to_datetime(support['ticket_time'], errors='coerce', utc=True)
    activity['event_time'] = pd.to_datetime(activity['event_time'], errors='coerce', utc=True)

    support_summary = support.dropna(subset=['match_id']).groupby('match_id').agg(
        total_tickets=('match_id', 'size'),
        avg_resolution_hours=('resolution_time_hours', 'mean')).reset_index()
    activity_summary = activity.dropna(subset=['match_id']).groupby('match_id').agg(
        total_events=('match_id', 'size'),
        last_activity=('event_time', 'max')).reset_index()

    merged = accounts.merge(support_summary, on='match_id', how='left', validate='one_to_one')
    merged = merged.merge(activity_summary, on='match_id', how='left', validate='one_to_one')
    assert len(merged) == len(accounts)

    merged['has_support_record'] = merged['total_tickets'].notna()
    merged['has_activity_record'] = merged['total_events'].notna()
    print('Accounts matched to support:', merged['has_support_record'].sum())
    print('Accounts matched to activity:', merged['has_activity_record'].sum())
    if not merged['has_support_record'].any() or not merged['has_activity_record'].any():
        raise ValueError('No cross-dataset matches: inspect customer identifiers')

    churned = int(merged['churn'].sum())
    n = len(merged)
    print(f'Churned: {churned}/{n} ({churned/n:.1%}); blank statuses assumed not churned')
    plan_summary = merged.groupby('plan', dropna=False)['churn'].agg(
        customers='size', churned='sum', churn_rate='mean').reset_index()
    plan_summary['churn_rate_pct'] = (plan_summary['churn_rate'] * 100).round(1)
    plan_summary.drop(columns='churn_rate').to_csv(OUTPUT / 'plan_churn_summary.csv', index=False)
    print(plan_summary.to_string(index=False))

    # A missing support/activity row is NOT treated as confirmed zero activity.
    # This table compares observed records only, without making causal claims.
    observed = merged.groupby('churn').agg(
        customers=('match_id', 'size'),
        accounts_with_activity=('has_activity_record', 'sum'),
        accounts_with_support=('has_support_record', 'sum'),
        mean_events_among_matched=('total_events', 'mean'),
        mean_tickets_among_matched=('total_tickets', 'mean')).reset_index()
    observed.to_csv(OUTPUT / 'engagement_support_summary.csv', index=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    ordered = plan_summary.sort_values('churn_rate_pct', ascending=False)
    ax.bar(ordered['plan'], ordered['churn_rate_pct'])
    ax.set(title='Observed churn proportion by subscription plan',
           xlabel='Subscription plan', ylabel='Accounts marked churned (%)')
    ax.set_ylim(0, max(50, ordered['churn_rate_pct'].max() * 1.15))
    fig.tight_layout()
    fig.savefig(OUTPUT / 'churn_by_plan.png', dpi=160)
    plt.close(fig)
    print('Outputs saved in:', OUTPUT)


if __name__ == '__main__':
    main()
