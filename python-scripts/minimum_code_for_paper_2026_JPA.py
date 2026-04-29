"""minimum_code_for_paper_2026.py
"""
import pandas as pd
from tabulate import tabulate

from boilerplate_dea import dea_add_frontier_point_estimates
from boilerplate_filter import filter_oil_producers

# from Local file
df = pd.read_excel('data/pwt110.xlsx', sheet_name='Data')

index_cols = ['country', 'year']
df = df.set_index(index_cols, drop=False)
pwt110_df = df.rename(columns={'country': 'label', 'year': 'time'})

pwt110_df['working_hours'] =pwt110_df['emp']*pwt110_df['avh']
pwt110_df['rgdpo_per_hour_worked'] =pwt110_df['rgdpo']/pwt110_df['working_hours']
pwt110_df['rgdpo_per_worker'] =pwt110_df['rgdpo']/pwt110_df['emp']
pwt110_df['capital_output_ratio'] =pwt110_df['rnna']/pwt110_df['rgdpna']
pwt110_df['rn'] =pwt110_df['capital_output_ratio']*pwt110_df['rgdpo']
pwt110_df['rn_per_hour_worked'] =pwt110_df['rn']/pwt110_df['working_hours']
pwt110_df['rn_per_worker'] =pwt110_df['rn']/pwt110_df['emp']

# --- Data scrutiny ---
pwt110_df = filter_oil_producers(pwt110_df, mode='final', invert=True)

# --- the Malmquist decomposition ---
_, result = dea_add_frontier_point_estimates(pwt110_df, year_t=2000, year_t1=2010, inputs=['rn_per_hour_worked'], outputs=['rgdpo_per_hour_worked'], keep_columns=['countrycode'], total_growth=None)
result_tabulated = tabulate(result, headers='keys', tablefmt='pretty', showindex=True)
print("Table 1")
print(result_tabulated)

_, result = dea_add_frontier_point_estimates(pwt110_df, year_t=2000, year_t1=2010, inputs=['rn_per_hour_worked', 'hc'], outputs=['rgdpo_per_hour_worked'], keep_columns=['countrycode'], total_growth=None)
result_tabulated = tabulate(result, headers='keys', tablefmt='pretty', showindex=True)
print("Table 2")
print(result_tabulated)

