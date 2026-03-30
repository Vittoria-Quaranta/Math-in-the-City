import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('435data_clean_v3.csv')

# Student loan debt columns by year
loan_cols = {
    2011: 'Amount of student loans 2011',
    2013: 'Amount student loans 2013',
    2015: 'Amount student loans 2015',
    2017: 'Amount student loans 2017',
    2019: 'Amount student loans 2019',
    2021: 'Amount student loans 2021',
    2023: 'Amount student loans 2023',
}
# family income columns by year
income_cols = {
    2011: 'Total family income 2011',
    2013: 'Total family income 2013',
    2015: 'Total family income 2015',
    2017: 'Total family income 2017',
    2019: 'Total family income 2019',
    2021: 'Total family income 2021',
    2023: 'Total family income 2023',
}

# Savings/accounts columns by year
savings_cols = {
    1999: 'Amount all accounts 1999',
    2001: 'Amount all accounts 2001',
    2003: 'Amount all accounts 2003',
    2005: 'Amount all accounts 2005',
    2007: 'Amount all accounts 2007',
    2009: 'Amount all accounts 2009',
    2011: 'Amount all accounts 2011',
    2013: 'Amount all accounts 2013',
    2015: 'Amount all accounts 2015',
    2017: 'Amount all accounts 2017',
    2019: 'Amount ck/saving account 2019',
    2021: 'Amount ck/savings acct 2021',
    2023: 'Amount ck/savings account 2023',
}

# Stocks/investments columns by year
stocks_cols = {
    1984: 'Imp value stocks 1984',
    1989: 'Imp value stocks 1989',
    1994: 'Imp value stocks 1994',
    1999: 'Imp value stocks 1999',
    2001: 'Imp value stocks 2001',
    2003: 'Imp value stocks 2003',
    2005: 'Imp value stocks 2005',
    2007: 'Imp value stocks 2007',
    2009: 'Imp value stocks 2009',
    2011: 'Imp value stocks 2011',
    2013: 'Imp value stocks 2013',
    2015: 'Imp value stocks 2015',
    2017: 'Imp value stocks 2017',
    2019: 'Imp value stocks 2019',
    2021: 'Imp value stocks 2021',
    2023: 'Imp value stocks 2023',
}

# Rename to readable names
loan_rename    = {col: f'loan_{yr}'    for yr, col in loan_cols.items()}
savings_rename = {col: f'savings_{yr}' for yr, col in savings_cols.items()}
stocks_rename  = {col: f'stocks_{yr}'  for yr, col in stocks_cols.items()}
income_rename = {col: f'income_{yr}' for yr, col in income_cols.items()}

years_savings_stocks = [1999, 2001, 2003, 2005, 2007, 2009, 2011, 2013, 2015, 2017, 2019, 2021, 2023]
years_loan           = [2011, 2013, 2015, 2017, 2019, 2021, 2023]

# Build agg — one row per person, all variables aligned
all_cols = (
    list(loan_cols.values()) +
    list(savings_cols.values()) +
    list(stocks_cols.values()) +
    list(income_cols.values())
)
agg = df[all_cols].copy()
agg = agg.rename(columns={
    **loan_rename,
    **savings_rename,
    **stocks_rename,
    **income_rename
})

# Clean sentinel values
for col in agg.columns:
    agg[col] = agg[col].where((agg[col] >= 0) & (agg[col] < 9999990))

print(agg.head())
print('\nShape:', agg.shape)

# --- ALL CORRELATIONS FIRST ---

print('\n--- CORRELATION: Savings vs Stocks ---')
for yr in years_savings_stocks:
    pair = agg[[f'savings_{yr}', f'stocks_{yr}']].dropna()
    corr = pair[f'savings_{yr}'].corr(pair[f'stocks_{yr}'])
    print(f"{yr}: r = {corr:.4f}  (n={len(pair):,})")

print('\n--- CORRELATION: Savings vs Debt ---')
for yr in years_loan:
    pair = agg[[f'savings_{yr}', f'loan_{yr}']].dropna()
    corr = pair[f'savings_{yr}'].corr(pair[f'loan_{yr}'])
    print(f"{yr}: r = {corr:.4f}  (n={len(pair):,})")

print('\n--- CORRELATION: Stocks vs Debt ---')
for yr in years_loan:
    pair = agg[[f'stocks_{yr}', f'loan_{yr}']].dropna()
    corr = pair[f'stocks_{yr}'].corr(pair[f'loan_{yr}'])
    print(f"{yr}: r = {corr:.4f}  (n={len(pair):,})")

print('\n--- CORRELATION: Savings vs Debt (Borrowers Only) ---')
for yr in years_loan:
    pair = agg[[f'savings_{yr}', f'loan_{yr}']].dropna()
    pair = pair[pair[f'loan_{yr}'] > 0]
    corr = pair[f'savings_{yr}'].corr(pair[f'loan_{yr}'])
    print(f"{yr}: r = {corr:.4f}  (n={len(pair):,})")

print('\n--- CORRELATION: Stocks vs Debt (Borrowers Only) ---')
for yr in years_loan:
    pair = agg[[f'stocks_{yr}', f'loan_{yr}']].dropna()
    pair = pair[pair[f'loan_{yr}'] > 0]
    corr = pair[f'stocks_{yr}'].corr(pair[f'loan_{yr}'])
    print(f"{yr}: r = {corr:.4f}  (n={len(pair):,})")

print('\n--- CORRELATION: Savings vs Debt (Log-transformed, Borrowers Only) ---')
for yr in years_loan:
    pair = agg[[f'savings_{yr}', f'loan_{yr}']].dropna()
    pair = pair[(pair[f'loan_{yr}'] > 0) & (pair[f'savings_{yr}'] > 0)]
    corr = np.log(pair[f'savings_{yr}']).corr(np.log(pair[f'loan_{yr}']))
    print(f"{yr}: r = {corr:.4f}  (n={len(pair):,})")

print('\n--- CORRELATION: Stocks vs Debt (Log-transformed, Borrowers Only) ---')
for yr in years_loan:
    pair = agg[[f'stocks_{yr}', f'loan_{yr}']].dropna()
    pair = pair[(pair[f'loan_{yr}'] > 0) & (pair[f'stocks_{yr}'] > 0)]
    corr = np.log(pair[f'stocks_{yr}']).corr(np.log(pair[f'loan_{yr}']))
    print(f"{yr}: r = {corr:.4f}  (n={len(pair):,})")

print('\n--- SPEARMAN RANK CORRELATION: Savings vs Debt (Borrowers Only) ---')
for yr in years_loan:
    pair = agg[[f'savings_{yr}', f'loan_{yr}']].dropna()
    pair = pair[pair[f'loan_{yr}'] > 0]
    corr = pair[f'savings_{yr}'].corr(pair[f'loan_{yr}'], method='spearman')
    print(f"{yr}: r = {corr:.4f}  (n={len(pair):,})")

print('\n--- SPEARMAN RANK CORRELATION: Stocks vs Debt (Borrowers Only) ---')
for yr in years_loan:
    pair = agg[[f'stocks_{yr}', f'loan_{yr}']].dropna()
    pair = pair[pair[f'loan_{yr}'] > 0]
    corr = pair[f'stocks_{yr}'].corr(pair[f'loan_{yr}'], method='spearman')
    print(f"{yr}: r = {corr:.4f}  (n={len(pair):,})")

from scipy.stats import mannwhitneyu
borrowers = agg[agg['loan_2023'] > 0]['savings_2023'].dropna()
non_borrowers = agg[agg['loan_2023'] == 0]['savings_2023'].dropna()
stat, p = mannwhitneyu(borrowers, non_borrowers)
print(f"p = {p:.4f}")

agg['debt_change'] = agg['loan_2023'] - agg['loan_2011']
agg['savings_change'] = agg['savings_2023'] - agg['savings_2011']
pair = agg[['debt_change', 'savings_change']].dropna()
corr = pair['debt_change'].corr(pair['savings_change'], method='spearman')
print(f"spearman r = {corr:.4f}")

# stuff given by grok
import statsmodels.api as sm
from scipy.stats import ks_2samp
import matplotlib.pyplot as plt

# ------------------------------------------------------------------
# 1. MULTIPLE REGRESSION: Savings ~ Debt + Stocks (controls for wealth)
#    Do this for both full sample and borrowers-only
# ------------------------------------------------------------------
print("\n--- OLS REGRESSION: Savings ~ Debt + Stocks (Borrowers Only) ---")
for yr in years_loan:
    pair = agg[[f'savings_{yr}', f'loan_{yr}', f'stocks_{yr}']].dropna()
    pair = pair[pair[f'loan_{yr}'] > 0]                     # borrowers only
    if len(pair) > 30:                                      # need enough obs
        X = sm.add_constant(pair[[f'loan_{yr}', f'stocks_{yr}']])
        y = pair[f'savings_{yr}']
        model = sm.OLS(y, X).fit()
        debt_coef = model.params[f'loan_{yr}']
        debt_p = model.pvalues[f'loan_{yr}']
        print(f"{yr}: Debt coef = {debt_coef:.4f} (p={debt_p:.4f}), "
              f"Stocks coef = {model.params[f'stocks_{yr}']:.4f}, "
              f"R² = {model.rsquared:.3f} (n={len(pair):,})")

# Same regression but on the full sample (includes non-borrowers)
print("\n--- OLS REGRESSION: Savings ~ Debt + Stocks (Full Sample) ---")
for yr in years_loan:
    pair = agg[[f'savings_{yr}', f'loan_{yr}', f'stocks_{yr}']].dropna()
    if len(pair) > 30:
        X = sm.add_constant(pair[[f'loan_{yr}', f'stocks_{yr}']])
        y = pair[f'savings_{yr}']
        model = sm.OLS(y, X).fit()
        debt_coef = model.params[f'loan_{yr}']
        debt_p = model.pvalues[f'loan_{yr}']
        print(f"{yr}: Debt coef = {debt_coef:.4f} (p={debt_p:.4f}), "
              f"Stocks coef = {model.params[f'stocks_{yr}']:.4f}, "
              f"R² = {model.rsquared:.3f} (n={len(pair):,})")

# ------------------------------------------------------------------
# 2. LAGGED / CROSS-LAGGED CORRELATIONS
#    e.g., savings_2011 vs loan_2013, savings_2011 vs loan_2015, etc.
# ------------------------------------------------------------------
print("\n--- LAGGED CORRELATIONS: Savings_t vs Loan_{t+k} (Borrowers Only) ---")
lags = [(2011,2013), (2011,2015), (2011,2017), (2013,2015), (2013,2017),
        (2015,2017), (2015,2019), (2017,2019), (2017,2021), (2019,2021),
        (2019,2023), (2021,2023)]
for t1, t2 in lags:
    pair = agg[[f'savings_{t1}', f'loan_{t2}']].dropna()
    pair = pair[pair[f'loan_{t2}'] > 0]
    if len(pair) > 100:
        r = pair[f'savings_{t1}'].corr(pair[f'loan_{t2}'])
        print(f"Savings {t1} → Loan {t2}: r = {r:.4f} (n={len(pair):,})")

print("\n--- LAGGED CORRELATIONS: Stocks_t vs Loan_{t+k} (Borrowers Only) ---")
for t1, t2 in lags:
    pair = agg[[f'stocks_{t1}', f'loan_{t2}']].dropna()
    pair = pair[pair[f'loan_{t2}'] > 0]
    if len(pair) > 100:
        r = pair[f'stocks_{t1}'].corr(pair[f'loan_{t2}'])
        print(f"Stocks {t1} → Loan {t2}: r = {r:.4f} (n={len(pair):,})")

# ------------------------------------------------------------------
# 3. CHANGE-SCORE ANALYSIS (within-person) for multiple intervals
# ------------------------------------------------------------------
print("\n--- CHANGE CORRELATIONS (Spearman) ---")
change_pairs = [(2011,2013), (2013,2015), (2015,2017), (2017,2019),
                (2019,2021), (2021,2023)]
for t1, t2 in change_pairs:
    agg[f'debt_ch_{t1}_{t2}'] = agg[f'loan_{t2}'] - agg[f'loan_{t1}']
    agg[f'sav_ch_{t1}_{t2}']  = agg[f'savings_{t2}'] - agg[f'savings_{t1}']
    pair = agg[[f'debt_ch_{t1}_{t2}', f'sav_ch_{t1}_{t2}']].dropna()
    if len(pair) > 100:
        r = pair[f'debt_ch_{t1}_{t2}'].corr(pair[f'sav_ch_{t1}_{t2}'], method='spearman')
        print(f"ΔDebt {t1}-{t2} vs ΔSavings {t1}-{t2}: r = {r:.4f} (n={len(pair):,})")

# ------------------------------------------------------------------
# 4. DISTRIBUTIONAL TESTS (KS) + Mann-Whitney for every year
# ------------------------------------------------------------------
print("\n--- DISTRIBUTION TESTS: Borrowers vs Non-Borrowers (Savings) ---")
for yr in years_loan:
    borrowers = agg[agg[f'loan_{yr}'] > 0][f'savings_{yr}'].dropna()
    non = agg[agg[f'loan_{yr}'] == 0][f'savings_{yr}'].dropna()
    if len(borrowers) > 30 and len(non) > 30:
        stat, p_mwu = mannwhitneyu(borrowers, non, alternative='two-sided')
        ks_stat, p_ks = ks_2samp(borrowers, non)
        print(f"{yr} Savings: MWU p = {p_mwu:.4f}, KS p = {p_ks:.4f} "
              f"(n_borrow={len(borrowers):,}, n_non={len(non):,})")

# Same for stocks
print("\n--- DISTRIBUTION TESTS: Borrowers vs Non-Borrowers (Stocks) ---")
for yr in years_loan:
    borrowers = agg[agg[f'loan_{yr}'] > 0][f'stocks_{yr}'].dropna()
    non = agg[agg[f'loan_{yr}'] == 0][f'stocks_{yr}'].dropna()
    if len(borrowers) > 30 and len(non) > 30:
        stat, p_mwu = mannwhitneyu(borrowers, non, alternative='two-sided')
        ks_stat, p_ks = ks_2samp(borrowers, non)
        print(f"{yr} Stocks: MWU p = {p_mwu:.4f}, KS p = {p_ks:.4f} "
              f"(n_borrow={len(borrowers):,}, n_non={len(non):,})")

# ------------------------------------------------------------------
# 5. NET WEALTH = Savings + Stocks - Debt (years where all exist)
#    and its correlation with debt level
# ------------------------------------------------------------------
print("\n--- NET WEALTH ANALYSIS ---")
net_years = [2011, 2013, 2015, 2017, 2019, 2021, 2023]
for yr in net_years:
    agg[f'net_wealth_{yr}'] = (agg[f'savings_{yr}'] + agg[f'stocks_{yr}'] 
                               - agg[f'loan_{yr}'].fillna(0))
    pair = agg[[f'net_wealth_{yr}', f'loan_{yr}']].dropna()
    pair = pair[pair[f'loan_{yr}'] > 0]   # borrowers only
    if len(pair) > 30:
        r = pair[f'net_wealth_{yr}'].corr(pair[f'loan_{yr}'])
        print(f"{yr} Net Wealth vs Debt (borrowers): r = {r:.4f} (n={len(pair):,})")

# ------------------------------------------------------------------
# 6. QUICK VISUAL: Correlation trends over time (optional but useful)
# ------------------------------------------------------------------
# Collect the savings-vs-debt (borrowers) correlations you already printed
savings_debt_r = [0.0616, 0.0650, 0.0776, 0.0414, 0.0565, 0.0939, 0.0398]
years_plot = [2011, 2013, 2015, 2017, 2019, 2021, 2023]

plt.figure(figsize=(8,5))
plt.plot(years_plot, savings_debt_r, marker='o', label='Savings vs Debt (borrowers)')
plt.axhline(0, color='gray', linestyle='--')
plt.title('Savings–Debt Correlation Among Borrowers Over Time')
plt.xlabel('Year')
plt.ylabel('Pearson r')
plt.legend()
plt.grid(True)
plt.show()

## trying to get anything
agg['income_2019_q'] = pd.qcut(agg['income_2019'], 4, labels=False)
for q in range(4):
    subset = agg[agg['income_2019_q'] == q]
    pair = subset[['savings_2019', 'loan_2019']].dropna()
    print(f"Income quartile {q}: r =", pair.corr().iloc[0,1])