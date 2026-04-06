import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
from scipy import stats
import statsmodels.formula.api as smf
import statsmodels.api as sm
 
# 1. LOAD DATA
df = pd.read_csv('435data_clean_v3.csv')
 
# 2. DEFINE PARAMETERS
years = [2011, 2013, 2015, 2017, 2019, 2021, 2023]
state_map = {0: 'None', 1: 'Stocks Only', 2: 'Debt Only', 3: 'Both'}
 
loan_cols = {
    2011: 'Amount of student loans 2011', 2013: 'Amount student loans 2013',
    2015: 'Amount student loans 2015', 2017: 'Amount student loans 2017',
    2019: 'Amount student loans 2019', 2021: 'Amount student loans 2021',
    2023: 'Amount student loans 2023',
}
 
stocks_cols = {
    2011: 'Imp value stocks 2011', 2013: 'Imp value stocks 2013',
    2015: 'Imp value stocks 2015', 2017: 'Imp value stocks 2017',
    2019: 'Imp value stocks 2019', 2021: 'Imp value stocks 2021',
    2023: 'Imp value stocks 2023',
}
 
# 3. CREATE THE 'agg' DATAFRAME (rename columns for easier looping)
loan_rename = {col: f'loan_{yr}' for yr, col in loan_cols.items()}
stocks_rename = {col: f'stocks_{yr}' for yr, col in stocks_cols.items()}
 
agg = df[list(loan_cols.values()) + list(stocks_cols.values())].copy()
agg = agg.rename(columns={**loan_rename, **stocks_rename})
 
# Clean data: Treat negative or massive outlier values as NaN
for col in agg.columns:
    agg[col] = agg[col].where((agg[col] >= 0) & (agg[col] < 9999990))
 
# Assign financial states for every year
# 0: None, 1: Stocks Only, 2: Debt Only, 3: Both
for yr in years:
    has_debt = (agg[f'loan_{yr}'] > 0)
    has_stocks = (agg[f'stocks_{yr}'] > 0)
    
    agg[f'state_{yr}'] = np.nan
    agg.loc[~has_debt & ~has_stocks, f'state_{yr}'] = 0
    agg.loc[~has_debt &  has_stocks, f'state_{yr}'] = 1
    agg.loc[ has_debt & ~has_stocks, f'state_{yr}'] = 2
    agg.loc[ has_debt &  has_stocks, f'state_{yr}'] = 3
 
# 4. ANALYSIS FUNCTIONS
def get_markov_data(df_agg, year_pairs):
    """Returns raw counts and normalized percentages for given year pairs."""
    transitions = []
    for y_from, y_to in year_pairs:
        valid = df_agg[[f'state_{y_from}', f'state_{y_to}']].dropna()
        transitions.append(valid.values)
    
    t_all = np.vstack(transitions)
    df_trans = pd.DataFrame(t_all, columns=['From', 'To'])
    
    counts = pd.crosstab(df_trans['From'], df_trans['To'])
    probs = pd.crosstab(df_trans['From'], df_trans['To'], normalize='index')
    
    # Fill missing states to ensure 4x4 matrix
    counts = counts.reindex(index=[0,1,2,3], columns=[0,1,2,3], fill_value=0)
    probs = probs.reindex(index=[0,1,2,3], columns=[0,1,2,3], fill_value=0)
    
    return counts, probs
 
# 5. EXECUTION
# Define the periods
all_pairs = [(years[i], years[i+1]) for i in range(len(years)-1)]
early_pairs = [(2011, 2013), (2013, 2015)]
recent_pairs = [(2019, 2021), (2021, 2023)]
 
# Calculate matrices
c_all, p_all = get_markov_data(agg, all_pairs)
c_early, p_early = get_markov_data(agg, early_pairs)
c_recent, p_recent = get_markov_data(agg, recent_pairs)
 
# -------------------------------
# RESULTS DISPLAY
# -------------------------------
 
print("--- 1. OVERALL MARKOV CHAIN (2011-2023) ---")
print(p_all.rename(index=state_map, columns=state_map))
print("\n" + "="*60 + "\n")
 
print("--- 2. FIRST FEW YEARS (2011-2015) ---")
print(p_early.rename(index=state_map, columns=state_map))
print("\n" + "="*60 + "\n")
 
print("--- 3. RECENT YEARS (2019-2023) ---")
print(p_recent.rename(index=state_map, columns=state_map))
print("\n" + "="*60 + "\n")
 
# 6. EFFECT SIZE ANALYSIS (REPLACES CHI-SQUARE TEST)
print("--- 4. EFFECT SIZE: EARLY VS. RECENT ---")
 
def total_variation(p, q):
    """Compute Total Variation Distance between two distributions."""
    return 0.5 * np.sum(np.abs(p - q))
 
effect_results = []
 
for state_code in [0, 1, 2, 3]:
    row_e = p_early.loc[state_code].values
    row_r = p_recent.loc[state_code].values
    
    tv = total_variation(row_e, row_r)
    
    effect_results.append({
        'Starting State': state_map[state_code],
        'TV Distance': round(tv, 4),
        'Interpretation': (
            'Very Small' if tv < 0.02 else
            'Small' if tv < 0.05 else
            'Moderate' if tv < 0.1 else
            'Large'
        )
    })
 
print(pd.DataFrame(effect_results))
 
print("\n" + "="*60 + "\n")
 
# 7. OVERALL MATRIX DIFFERENCE
fro_norm = np.linalg.norm(p_early.values - p_recent.values)
 
print("--- 5. OVERALL TRANSITION MATRIX DIFFERENCE ---")
print(f"Frobenius Norm: {fro_norm:.4f}")
 
if fro_norm < 0.05:
    interpretation = "Very small overall change"
elif fro_norm < 0.1:
    interpretation = "Small overall change"
elif fro_norm < 0.2:
    interpretation = "Moderate overall change"
else:
    interpretation = "Large overall change"
 
print(f"Interpretation: {interpretation}")
 
sig_results = []
for state_code in [0, 1, 2, 3]:
    row_e = c_early.loc[state_code].values
    row_r = c_recent.loc[state_code].values
    table = np.vstack([row_e, row_r])
    
    if table.sum() > 0:
        table = table[:, table.sum(axis=0) > 0]
        if table.shape[1] > 1:
            chi2, p, _, _ = chi2_contingency(table)
            sig_results.append({
                'Starting State': state_map[state_code],
                'p-value': round(p, 4),
                'Significant?': 'YES' if p < 0.05 else 'No'
            })
 
print(pd.DataFrame(sig_results))
 
def calculate_cohens_h(p1, p2):
    prop1 = p1 / 100 if p1 > 1 else p1
    prop2 = p2 / 100 if p2 > 1 else p2
    prop1 = np.clip(prop1, 0, 1)
    prop2 = np.clip(prop2, 0, 1)
    if np.isnan(prop1) or np.isnan(prop2):
        return np.nan
    h = 2 * (np.arcsin(np.sqrt(prop1)) - np.arcsin(np.sqrt(prop2)))
    return abs(h)
 
def analyze_high_borrower_participation(year):
    q_labels = pd.qcut(agg[f'loan_{year}'].rank(method='first'), 4, labels=False)
    has_stocks = (agg[f'stocks_{year}'] > 0)
    q3_participation = has_stocks[q_labels == 3].mean() * 100
    rest_participation = has_stocks[q_labels < 3].mean() * 100
    contingency_table = pd.crosstab(q_labels == 3, has_stocks)
    chi2, p_val, _, _ = stats.chi2_contingency(contingency_table)
    print(f"--- {year} STOCK MARKET PARTICIPATION ---")
    print(f"High Borrowers (Q3) who invest: {q3_participation:.2f}%")
    print(f"Others (Q0-Q2) who invest:      {rest_participation:.2f}%")
    print(f"P-value: {p_val:.4f}")
    effect_size = calculate_cohens_h(q3_participation, rest_participation)
    print(f"Cohen's h: {effect_size:.4f}")
    if effect_size < 0.2:
        print("Verdict: Negligible difference (even if p-value is significant)")
    elif effect_size < 0.5:
        print("Verdict: Small but meaningful difference")
    else:
        print("Verdict: Large, practically significant difference")
 
analyze_high_borrower_participation(2023)
 
# 8. ENTRY VS EXIT FLOWS
flows = []
counts = c_all
 
for state in [0, 1, 2, 3]:
    row = counts.loc[state]
    col = counts[state]
    stayed = counts.loc[state, state]
    exit_flow = row.sum() - stayed
    entry_flow = col.sum() - stayed
    net_flow = entry_flow - exit_flow
    flows.append({
        'State': state_map[state],
        'Entry Flow': int(entry_flow),
        'Exit Flow': int(exit_flow),
        'Net Flow': int(net_flow),
        'Interpretation': (
            'Sink (net inflow)' if net_flow > 0 else
            'Source (net outflow)' if net_flow < 0 else
            'Balanced'
        )
    })
 
df_flows = pd.DataFrame(flows)
print("\n--- 6. ENTRY VS EXIT FLOWS (2011–2023) ---")
print(df_flows)
 
def compute_net_flows(counts):
    flows = {}
    for s in [0,1,2,3]:
        row = counts.loc[s]
        col = counts[s]
        stayed = counts.loc[s, s]
        exit_flow = row.sum() - stayed
        entry_flow = col.sum() - stayed
        flows[s] = entry_flow - exit_flow
    return flows
 
net_early = compute_net_flows(c_early)
net_recent = compute_net_flows(c_recent)
 
print("Early net flows:", net_early)
print("Recent net flows:", net_recent)
 
from scipy.stats import ttest_rel
early_vals = np.array(list(net_early.values()))
recent_vals = np.array(list(net_recent.values()))
t, p = ttest_rel(early_vals, recent_vals)
print("Net-flow paired t-test p-value:", p)
 
# ── COHORT ANALYSIS: transitions from each state → state 0 (None) ──
 
def cohort_exit_values(agg, years, loan_cols_renamed, stocks_cols_renamed):
    records = []
    for i in range(len(years) - 1):
        y_from, y_to = years[i], years[i + 1]
        s_from = agg[f'state_{y_from}']
        s_to   = agg[f'state_{y_to}']
        for origin_state in [1, 2, 3]:
            mask = (s_from == origin_state) & (s_to == 0)
            cohort = agg[mask]
            for idx, row in cohort.iterrows():
                records.append({
                    'origin_state': origin_state,
                    'year_from':    y_from,
                    'loan_val':     row[f'loan_{y_from}'],
                    'stocks_val':   row[f'stocks_{y_from}'],
                })
    return pd.DataFrame(records)
 
df_exits = cohort_exit_values(agg, years, loan_cols, stocks_cols)
 
state_labels = {1: 'Stocks Only → None', 2: 'Debt Only → None', 3: 'Both → None'}
 
print("=== MEDIAN VALUES AT EXIT (by cohort) ===")
for state, label in state_labels.items():
    g = df_exits[df_exits['origin_state'] == state]
    print(f"\n{label}  (n={len(g)})")
    print(f"  Stocks — median: ${g['stocks_val'].median():,.0f}  |  mean: ${g['stocks_val'].mean():,.0f}  |  p25: ${g['stocks_val'].quantile(0.25):,.0f}  |  p75: ${g['stocks_val'].quantile(0.75):,.0f}")
    print(f"  Loans  — median: ${g['loan_val'].median():,.0f}  |  mean: ${g['loan_val'].mean():,.0f}  |  p25: ${g['loan_val'].quantile(0.25):,.0f}  |  p75: ${g['loan_val'].quantile(0.75):,.0f}")
 
bins = [0, 5000, 15000, 30000, 60000, float('inf')]
bin_labels = ['$0–5k', '$5–15k', '$15–30k', '$30–60k', '$60k+']
 
print("\n=== STOCK VALUE DISTRIBUTION (% of cohort) ===")
for state in [1, 3]:
    g = df_exits[df_exits['origin_state'] == state]['stocks_val'].dropna()
    cuts = pd.cut(g, bins=bins, labels=bin_labels)
    result = (cuts.value_counts(normalize=True).reindex(bin_labels) * 100).round(1)
    print(f"\n{state_labels[state]}")
    for label, val in result.items():
        print(f"  {label}: {val}%")
 
print("\n=== DEBT VALUE DISTRIBUTION (% of cohort) ===")
for state in [2, 3]:
    g = df_exits[df_exits['origin_state'] == state]['loan_val'].dropna()
    cuts = pd.cut(g, bins=bins, labels=bin_labels)
    result = (cuts.value_counts(normalize=True).reindex(bin_labels) * 100).round(1)
    print(f"\n{state_labels[state]}")
    for label, val in result.items():
        print(f"  {label}: {val}%")
 
 
# ══════════════════════════════════════════════════════════════════
# 9. DOES DEBT DELAY ENTRY INTO STOCK MARKET? (LAGGED PANEL ANALYSIS)
# ══════════════════════════════════════════════════════════════════
 
print("\n" + "="*60)
print("--- 9. DOES DEBT DELAY ENTRY INTO STOCK MARKET? ---")
print("="*60 + "\n")
 
# ── 9a. BUILD LONG-FORMAT PANEL ──
# Each row = one person-period. Outcome = do they invest NEXT period?
records = []
for i in range(len(years) - 1):
    y_now  = years[i]
    y_next = years[i + 1]
    for idx, row in agg.iterrows():
        loan_now    = row[f'loan_{y_now}']
        stocks_now  = row[f'stocks_{y_now}']
        stocks_next = row[f'stocks_{y_next}']
        if pd.isna(loan_now) or pd.isna(stocks_next):
            continue
        records.append({
            'person_id':        idx,
            'year':             y_now,
            'loan':             loan_now,
            'has_stocks_now':   int(stocks_now > 0) if not pd.isna(stocks_now) else np.nan,
            'has_stocks_next':  int(stocks_next > 0),
            'loan_log':         np.log1p(loan_now),
            'has_debt':         int(loan_now > 0),
        })
 
panel = pd.DataFrame(records)
 
# Debt quartile (among borrowers only)
borrowers_mask = panel['loan'] > 0
panel.loc[borrowers_mask, 'debt_quartile'] = pd.qcut(
    panel.loc[borrowers_mask, 'loan'].rank(method='first'), 4, labels=[1, 2, 3, 4]
)
 
print(f"Panel observations: {len(panel)}")
print(f"Unique individuals: {panel['person_id'].nunique()}")
print(f"Overall rate of investing next period: {panel['has_stocks_next'].mean():.3f}\n")
 
 
# ── 9b. ENTRY RATE: DO NON-INVESTORS WITH DEBT ENTER MARKET LESS? ──
# Restrict to people who do NOT currently invest
non_investors = panel[panel['has_stocks_now'] == 0].copy()
 
print("--- Entry into investing (among current non-investors) ---")
print(f"n = {len(non_investors)}\n")
 
entry_by_debt = (
    non_investors.groupby('has_debt')['has_stocks_next']
    .agg(pct_entered='mean', n='count')
    .assign(pct_entered=lambda x: (x['pct_entered'] * 100).round(2))
)
entry_by_debt.index = ['No debt', 'Has debt']
print(entry_by_debt.to_string())
 
ct = pd.crosstab(non_investors['has_debt'], non_investors['has_stocks_next'])
chi2_val, p_val, _, _ = stats.chi2_contingency(ct)
print(f"\nChi-square p-value: {p_val:.4f}")
 
# Cohen's h for the two entry rates
h = calculate_cohens_h(
    entry_by_debt.loc['Has debt', 'pct_entered'],
    entry_by_debt.loc['No debt', 'pct_entered']
)
print(f"Cohen's h: {h:.4f}")
if h < 0.2:
    print("Verdict: Negligible difference")
elif h < 0.5:
    print("Verdict: Small but meaningful difference")
else:
    print("Verdict: Large, practically significant difference")
 
 
# ── 9c. ENTRY RATE BY DEBT QUARTILE ──
print("\n--- Entry rate by debt quartile (non-investors with debt) ---")
non_inv_borrowers = non_investors[non_investors['has_debt'] == 1].copy()
quartile_entry = (
    non_inv_borrowers.groupby('debt_quartile', observed=True)['has_stocks_next']
    .agg(pct_entered='mean', n='count')
    .assign(pct_entered=lambda x: (x['pct_entered'] * 100).round(2))
)
print(quartile_entry.to_string())
 
# Chi-square across quartiles
ct_q = pd.crosstab(non_inv_borrowers['debt_quartile'], non_inv_borrowers['has_stocks_next'])
chi2_q, p_q, _, _ = stats.chi2_contingency(ct_q)
print(f"\nChi-square across quartiles p-value: {p_q:.4f}")
 
 
# ── 9d. LOGISTIC REGRESSION: DEBT LOAD → P(INVEST NEXT PERIOD) ──
# Controls for current investor status and year fixed effects
print("\n--- Logistic regression: P(invest next period) ---")
print("Controls: current investor status, year fixed effects\n")
 
logit_data = panel.dropna(subset=['loan_log', 'has_stocks_next', 'has_stocks_now']).copy()
logit_data['year_cat'] = logit_data['year'].astype('category')
 
model = smf.logit(
    'has_stocks_next ~ loan_log + has_stocks_now + C(year)',
    data=logit_data
).fit(disp=0)
 
coef_table = model.summary2().tables[1][['Coef.', 'Std.Err.', 'z', 'P>|z|']]
print(coef_table.to_string())
 
# Average marginal effect of log(debt) on P(invest)
marginal = model.get_margeff()
ame_frame = marginal.summary_frame()
print("\nMarginal effect column names:", ame_frame.columns.tolist())  # debug line

# Grab the loan_log row however the columns are named
ame_row = ame_frame.loc['loan_log']
dy_dx   = ame_row.iloc[0]
se      = ame_row.iloc[1]
pval    = ame_row.iloc[3]  # usually index 3 is p-value

print(f"\nAverage marginal effect of log(debt) on P(invest next period):")
print(f"  dy/dx = {dy_dx:.4f}  |  SE = {se:.4f}  |  p = {pval:.4f}")
print("  Interpretation: a 1-unit increase in log(debt) changes P(invest) by this amount on average")
 
# ── 9e. YEAR-BY-YEAR ENTRY RATES ──
# Shows whether the debt penalty on entry changed over time
print("\n--- Year-by-year entry rate gap (non-investors) ---")
print(f"{'Year':<8} {'No debt %':>12} {'Has debt %':>12} {'Gap (pp)':>10} {'p-value':>10}")
print("-" * 55)
 
for yr in years[:-1]:
    sub = non_investors[non_investors['year'] == yr]
    if len(sub) < 20:
        continue
    no_debt_rate  = sub[sub['has_debt'] == 0]['has_stocks_next'].mean() * 100
    has_debt_rate = sub[sub['has_debt'] == 1]['has_stocks_next'].mean() * 100
    gap = has_debt_rate - no_debt_rate
    ct_yr = pd.crosstab(sub['has_debt'], sub['has_stocks_next'])
    if ct_yr.shape == (2, 2):
        _, p_yr, _, _ = stats.chi2_contingency(ct_yr)
        sig = '*' if p_yr < 0.05 else ''
        print(f"{yr:<8} {no_debt_rate:>11.1f}% {has_debt_rate:>11.1f}% {gap:>+9.1f}pp {p_yr:>9.4f}{sig}")
    else:
        print(f"{yr:<8} {no_debt_rate:>11.1f}% {has_debt_rate:>11.1f}% {gap:>+9.1f}pp {'—':>10}")
 
print("\n* = significant at p < 0.05")
print("\nPositive gap means debt holders are MORE likely to invest next period.")
print("Negative gap means debt holders are LESS likely (delayed entry).")
