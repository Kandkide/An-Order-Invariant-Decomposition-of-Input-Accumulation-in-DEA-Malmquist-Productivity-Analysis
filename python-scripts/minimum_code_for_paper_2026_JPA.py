"""minimum_code_for_paper_2026.py
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from boilerplate_dea import dea_add_frontier_point_estimates, dea_expand_all_years
from boilerplate_tabulate import tabulate_simple
from boilerplate_filter import filter_multiindex, filter_oil_producers
from boilerplate_plot import plot_multiindex_scatter_by_column
from boilerplate_plot3d import plot_multiindex_scatter_3d, plot_multiindex_scatter_3d_overlay
from boilerplate_plot_kde import plot_multiindex_kde
from boilerplate_estimate import estimate_simple_ols

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
result_tabulated = tabulate_simple(result, showindex=True)
print("Table 1")
print(result_tabulated)

_, result = dea_add_frontier_point_estimates(pwt110_df, year_t=2000, year_t1=2010, inputs=['rn_per_hour_worked', 'hc'], outputs=['rgdpo_per_hour_worked'], keep_columns=['countrycode'], total_growth=None)
result_tabulated = tabulate_simple(result, showindex=True)
print("Table 2")
print(result_tabulated)

