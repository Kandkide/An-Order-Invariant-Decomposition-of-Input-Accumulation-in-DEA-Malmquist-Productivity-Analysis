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
extra=['Türkiye', 'Syrian Arab Republic', 'Qatar', 'Poland', 'New Zealand', 'Norway', 'Mexico', 'Luxembourg', 'Kuwait', 'Ireland', 'Gabon', 'Egypt', 'Bulgaria', 'Bosnia and Herzegovina', 'Myanmar','United Arab Emirates' , 'Azerbaijan', 'Iraq' , 'Romania' , 'Saudi Arabia', "Trinidad and Tobago",'Oman','Sudan','Angola' ,'Bahrain','Venezuela (Bolivarian Republic of)','Algeria']
pwt110_df = filter_oil_producers(pwt110_df, mode='extra_only', invert=True, extra=extra)


# --- Plot Kernel Distributions ---
df_1965_1990_2023 = filter_multiindex(pwt110_df, level=-1, items=[1965, 1990, 2023])
plot_multiindex_kde(df_1965_1990_2023, column='rgdpo_per_hour_worked', level='year', values=None, bw_method='scott', kernel='gaussian', grid_points=512, ax=None, fill=False, legend=True, show=True, block=True, close_after_show=True, palette=None, linestyles=[":","-","--"], linewidth=None, alpha=0.6); print('kde plot, limit years before application')


# --- the Malmquist decomposition ---
# to get 2D results, remove 'hc' 
_, result = dea_add_frontier_point_estimates(pwt110_df, year_t=1990, year_t1=2023, inputs=['rn_per_hour_worked', 'hc'], outputs=['rgdpo_per_hour_worked'], keep_columns=['countrycode'])
result_tabulated = tabulate_simple(result, showindex=True)
print(result_tabulated)
OLS_on_the_result = tabulate_simple(estimate_simple_ols(result))
print(OLS_on_the_result)  # table in chapter 6


# --- 2D frontier plot ---
# This takes time
pwt110_df = dea_expand_all_years(pwt110_df, inputs=['rn_per_hour_worked'], outputs=['rgdpo_per_hour_worked'], rts=['VRS','DRS'] )

# for overlay
df_1965_1990 = filter_multiindex(pwt110_df, level=-1, items=[1965, 1990])
plot_multiindex_scatter_by_column(df_1965_1990, level=-1, x_column='rn_per_hour_worked', y_columns=['proj_rgdpo_per_hour_worked_VRS', 'rgdpo_per_hour_worked'], show_regression=False, show_corr=False, label_column='countrycode', label_condition = [None, 'all'], legend=False, overlay=True, plot_kind=['line', 'scatter'], line_order=['x_asc', None],line_kwargs=[{'lw':2,'ls':'--','color': 'orange'},{'ls':'-'}], scatter_kwargs=[{},{'marker': 'o','color': 'blue'}], **dict())

# for all
plot_multiindex_scatter_by_column(pwt110_df, level=-1, x_column='rn_per_hour_worked', y_columns=['proj_rgdpo_per_hour_worked_VRS'], show_regression=False, show_corr=False, label_column='countrycode', label_condition = None, legend=False, plot_kind='line', line_order='x_asc', **dict())

#for intervals
pwt110_df_intervals = filter_multiindex(pwt110_df, level=-1, items=[1950, 1960, 1970, 1980, 1990, 2000, 2010, 2020, 2023])
plot_multiindex_scatter_by_column(pwt110_df_intervals, level=-1, x_column='rn_per_hour_worked', y_columns=['proj_rgdpo_per_hour_worked_VRS'], show_regression=False, show_corr=False, label_column='countrycode', label_condition = ['USA'], legend=True, plot_kind='line', line_order='x_asc', **dict())


# # to plot 3D frontier, un-comment the following
# # --- 3D frontier plot ---
# # This takes time
# pwt110_df = dea_expand_all_years(pwt110_df, inputs=['rn_per_hour_worked', 'hc'], outputs=['rgdpo_per_hour_worked'], rts=['VRS', 'DRS'])

# # for overlay
# df_2023 = filter_multiindex(pwt110_df, level=-1, items=[2023])
# plot_multiindex_scatter_3d_overlay(df_2023, level=-1, x_column=['rn_per_hour_worked', 'rn_per_hour_worked'], y_column=['hc', 'hc'], z_column=['proj_rgdpo_per_hour_worked_VRS', 'rgdpo_per_hour_worked'], label_column='countrycode', label_condition=[None,['USA','JPN','CHE','TWN','ISL','DEU']], label_kwargs=None, color=None, marker=None, size=None, envelope=[True,False], envelope_groupwise=True, envelope_facecolor=None, envelope_alpha=0.5, envelope_edgecolor='k', envelope_edgewidth=0, figsize=(8, 6), elev=0, azim=-90, show_legend=False, title_and_labels=['','','',''], show_ticks=True, **dict())

# #for intervals
# pwt110_df_intervals = filter_multiindex(pwt110_df, level=-1, items=[1950, 1960, 1970, 1980, 1990, 2000, 2010, 2020, 2023])
# plot_multiindex_scatter_3d(pwt110_df_intervals, level=-1, x_column='rn_per_hour_worked', y_column='hc', z_column='proj_rgdpo_per_hour_worked_VRS', label_column='countrycode', label_condition=['USA'], label_kwargs=None, color=None, marker=None, size=None, figsize=(8, 6), elev=0, azim=-90, show_legend=True, envelope=True, title_and_labels=['', 'rn', 'hc', 'rgdp'], **dict())

# # for table in chapter 4
# OLS_on_df_2023 = estimate_simple_ols(df_2023, 'proj_rgdpo_per_hour_worked_VRS', ['rn_per_hour_worked','hc'])
# print(OLS_on_df_2023)

# df_2023_left_half = df_2023[df_2023['rn_per_hour_worked'] <= 270.296134]
# OLS_on_df_2023_left_half = estimate_simple_ols(df_2023_left_half, 'proj_rgdpo_per_hour_worked_VRS', ['rn_per_hour_worked','hc'])
# print(OLS_on_df_2023_left_half)

# df_2023_right_half = df_2023[df_2023['rn_per_hour_worked'] >= 270]
# OLS_on_df_2023_right_half = estimate_simple_ols(df_2023_right_half, 'proj_rgdpo_per_hour_worked_VRS', ['rn_per_hour_worked','hc'])
# print(OLS_on_df_2023_right_half)

