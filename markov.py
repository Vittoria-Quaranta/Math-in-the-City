import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
from scipy import stats

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
    # Contingency table (counts of where people went from a specific starting state)
    row_e = c_early.loc[state_code].values
    row_r = c_recent.loc[state_code].values
    table = np.vstack([row_e, row_r])
    
    # Only test if there is data in the table
    if table.sum() > 0:
        # Clean table of columns that are 0 in both rows
        table = table[:, table.sum(axis=0) > 0]
        if table.shape[1] > 1:
            chi2, p, _, _ = chi2_contingency(table)
            sig_results.append({
                'Starting State': state_map[state_code],
                'p-value': round(p, 4),
                'Significant?': 'YES' if p < 0.05 else 'No'
            })

print(pd.DataFrame(sig_results))

# explain: markov chain, frobenius norm, psi square, p-values
def calculate_cohens_h(p1, p2):
    """
    p1, p2: Participation rates (can be 0-100 or 0-1)
    """
    # Ensure we are working with proportions (0 to 1)
    # If the user passed 15.0, convert to 0.15
    prop1 = p1 / 100 if p1 > 1 else p1
    prop2 = p2 / 100 if p2 > 1 else p2
    
    # Clip values to be strictly between 0 and 1 to avoid math errors
    prop1 = np.clip(prop1, 0, 1)
    prop2 = np.clip(prop2, 0, 1)
    
    # Check for NaN before calculating
    if np.isnan(prop1) or np.isnan(prop2):
        return np.nan

    # Arcsine transformation: 2 * (arcsin(sqrt(p1)) - arcsin(sqrt(p2)))
    h = 2 * (np.arcsin(np.sqrt(prop1)) - np.arcsin(np.sqrt(prop2)))
    
    return abs(h)

def analyze_high_borrower_participation(year):
    # 1. Create the Quartile Labels (Q3 = Top 25% of borrowers)
    # Using rank(method='first') to handle the "too many zeros" problem
    q_labels = pd.qcut(agg[f'loan_{year}'].rank(method='first'), 4, labels=False)
    
    # 2. Define who has stocks (Binary: True/False)
    has_stocks = (agg[f'stocks_{year}'] > 0)
    
    # 3. Calculate percentages
    # Mean of a boolean series gives the percentage of 'True' values
    q3_participation = has_stocks[q_labels == 3].mean() * 100
    rest_participation = has_stocks[q_labels < 3].mean() * 100
    
    # 4. Statistical Test (Chi-Squared)
    # We compare the count of [investors, non-investors] between the two groups
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

    # Run it
analyze_high_borrower_participation(2023)

# 8. ENTRY VS EXIT FLOWS
flows = []

counts = c_all  # your 4x4 count matrix

for state in [0, 1, 2, 3]:
    row = counts.loc[state]          # transitions FROM this state
    col = counts[state]              # transitions TO this state
    
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
