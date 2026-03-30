import pandas as pd
import numpy as np

df = pd.read_csv('435data_clean_v3.csv')

# -------------------------------
# COLUMN MAPS
# -------------------------------

loan_cols = {
    2011: 'Amount of student loans 2011',
    2013: 'Amount student loans 2013',
    2015: 'Amount student loans 2015',
    2017: 'Amount student loans 2017',
    2019: 'Amount student loans 2019',
    2021: 'Amount student loans 2021',
    2023: 'Amount student loans 2023',
}

savings_cols = {
    2011: 'Amount all accounts 2011',
    2013: 'Amount all accounts 2013',
    2015: 'Amount all accounts 2015',
    2017: 'Amount all accounts 2017',
    2019: 'Amount ck/saving account 2019',
    2021: 'Amount ck/savings acct 2021',
    2023: 'Amount ck/savings account 2023',
}

stocks_cols = {
    2011: 'Imp value stocks 2011',
    2013: 'Imp value stocks 2013',
    2015: 'Imp value stocks 2015',
    2017: 'Imp value stocks 2017',
    2019: 'Imp value stocks 2019',
    2021: 'Imp value stocks 2021',
    2023: 'Imp value stocks 2023',
}

years = [2011, 2013, 2015, 2017, 2019, 2021, 2023]

# -------------------------------
# RENAME
# -------------------------------

loan_rename    = {col: f'loan_{yr}'    for yr, col in loan_cols.items()}
savings_rename = {col: f'savings_{yr}' for yr, col in savings_cols.items()}
stocks_rename  = {col: f'stocks_{yr}'  for yr, col in stocks_cols.items()}

all_cols = (
    list(loan_cols.values()) +
    list(savings_cols.values()) +
    list(stocks_cols.values())
)

agg = df[all_cols].copy()
agg = agg.rename(columns={
    **loan_rename,
    **savings_rename,
    **stocks_rename
})

# -------------------------------
# CLEAN DATA
# -------------------------------

for col in agg.columns:
    agg[col] = agg[col].where((agg[col] >= 0) & (agg[col] < 9999990))

print("Shape:", agg.shape)

# =========================================================
# 🔥 1. DEBT QUARTILE ANALYSIS (FIXED)
# =========================================================

print("\n--- DEBT-SEGMENTED CORRELATIONS (LOG SCALE) ---")

for yr in years:
    print(f"\nYEAR {yr}")
    
    loan = f'loan_{yr}'
    sav = f'savings_{yr}'
    
    temp = agg[[loan, sav]].dropna().copy()
    
    # 🔥 FIX: only borrowers
    temp = temp[temp[loan] > 0]
    
    if len(temp) < 50:
        print("  Not enough borrowers")
        continue
    
    # 🔥 FIX: safe qcut
    try:
        temp['debt_q'] = pd.qcut(temp[loan], 4, labels=False, duplicates='drop')
    except:
        print("  Could not create quartiles")
        continue
    
    print("  Groups:", sorted(temp['debt_q'].dropna().unique()))
    
    for q in sorted(temp['debt_q'].dropna().unique()):
        subset = temp[temp['debt_q'] == q]
        
        subset = subset[
            (subset[sav] > 0) &
            (subset[loan] > 0)
        ]
        
        if len(subset) > 30:
            r = np.log(subset[sav]).corr(np.log(subset[loan]))
            print(f"    Debt Q{int(q)}: r = {r:.4f} (n={len(subset)})")
        else:
            print(f"    Debt Q{int(q)}: not enough data")

# =========================================================
# 🔥 2. STOCK PARTICIPATION
# =========================================================

print("\n--- STOCK vs NO STOCK ANALYSIS ---")

for yr in years:
    print(f"\nYEAR {yr}")
    
    loan = f'loan_{yr}'
    sav = f'savings_{yr}'
    stock = f'stocks_{yr}'
    
    temp = agg[[loan, sav, stock]].dropna().copy()
    
    # borrowers only + valid values
    temp = temp[
        (temp[loan] > 0) &
        (temp[sav] > 0)
    ]
    
    has_stock = temp[temp[stock] > 0]
    no_stock  = temp[temp[stock] == 0]
    
    if len(has_stock) > 30:
        r = np.log(has_stock[sav]).corr(np.log(has_stock[loan]))
        print(f"  Has stocks: r = {r:.4f} (n={len(has_stock)})")
    else:
        print("  Has stocks: not enough data")
    
    if len(no_stock) > 30:
        r = np.log(no_stock[sav]).corr(np.log(no_stock[loan]))
        print(f"  No stocks: r = {r:.4f} (n={len(no_stock)})")
    else:
        print("  No stocks: not enough data")

# =========================================================
# 🔥 3. BORROWERS ONLY BASELINE
# =========================================================

print("\n--- BORROWERS ONLY (LOG CORRELATION) ---")

for yr in years:
    loan = f'loan_{yr}'
    sav = f'savings_{yr}'
    
    pair = agg[[loan, sav]].dropna()
    
    pair = pair[
        (pair[loan] > 0) &
        (pair[sav] > 0)
    ]
    
    if len(pair) > 30:
        r = np.log(pair[sav]).corr(np.log(pair[loan]))
        print(f"{yr}: r = {r:.4f} (n={len(pair)})")
    else:
        print(f"{yr}: not enough data")