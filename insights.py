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

# Define states for each year
# 0: None, 1: Stocks Only, 2: Debt Only, 3: Both
for yr in [2011, 2013, 2015, 2017, 2019, 2021, 2023]:
    # Booleans for debt and stocks
    has_debt = (agg[f'loan_{yr}'] > 0)
    has_stocks = (agg[f'stocks_{yr}'] > 0)
    
    agg[f'state_{yr}'] = np.nan
    agg.loc[~has_debt & ~has_stocks, f'state_{yr}'] = 0
    agg.loc[~has_debt &  has_stocks, f'state_{yr}'] = 1
    agg.loc[ has_debt & ~has_stocks, f'state_{yr}'] = 2
    agg.loc[ has_debt &  has_stocks, f'state_{yr}'] = 3

# Collect all 2-year transitions
transitions = []
for i in range(len(years) - 1):
    y_from, y_to = years[i], years[i+1]
    valid = agg[[f'state_{y_from}', f'state_{y_to}']].dropna()
    transitions.append(valid.values)

# Aggregate into a single Markov transition matrix
t_all = np.vstack(transitions)
df_trans = pd.DataFrame(t_all, columns=['From', 'To'])
matrix = pd.crosstab(df_trans['From'], df_trans['To'], normalize='index')

# Label for readability
state_map = {0: 'None', 1: 'Stocks Only', 2: 'Debt Only', 3: 'Both'}
matrix.index = [state_map[i] for i in matrix.index]
matrix.columns = [state_map[i] for i in matrix.columns]

print(matrix)

#Key Insights: The "Investment Fragility" of Debt Holders
#The Debt-Stock Trade-off: The most revealing insight is the transition from Both to Debt Only. There is a 31.5% probability that a household holding both student debt and stocks will liquidate their stocks within two years while keeping their debt.
#Reduced Retention:
#If you have No Debt, your probability of staying invested (remaining in the "Stocks Only" state) is 65.5%.
#If you have Debt, your probability of staying invested (remaining in the "Both" state) is only 44.9%.

#Finding: Having student debt makes an investment portfolio nearly 20% less stable. Debt holders are far more likely to "exit" the market during financial pressure.

#Entry Barriers: Households with "Debt Only" have only a 4.6% chance of starting an investment portfolio within two years, compared to a 5.4% chance for those with no debt at all. While the difference seems small, it compounds over decades into a massive wealth gap.

#The "Exit" Priority: Those in the "Both" category are more likely to move to "Debt Only" (31.5%) than to move to "Stocks Only" (14.2%). This suggests that when these households experience a change, they are twice as likely to lose their investments than to pay off their debt.