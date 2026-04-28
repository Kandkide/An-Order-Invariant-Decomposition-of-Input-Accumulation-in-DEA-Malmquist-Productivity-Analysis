"""minimum_code_for_paper_2026_replicate_KR.py
"""
import pandas as pd

from boilerplate_dea import dea_add_frontier_point_estimates, dea_expand_all_years
from boilerplate_tabulate import tabulate_simple
from boilerplate_filter import filter_multiindex
from boilerplate_plot import plot_multiindex_scatter_by_column

# from Local file
df = pd.read_excel('data/pwt56_forweb.xls', sheet_name='PWT56')

index_cols = ['Country', 'Year']
df = df.set_index(index_cols, drop=False)
df = df.rename(columns={'Country': 'label', 'Year': 'time'})
df = filter_multiindex(df, level=0, items=['VENEZUELA', 'IRAN'], exclude=True)
df = filter_multiindex(df, level=-1, items=slice(1950, 1964), exclude=True)

# KR like Malmquist index calculation
_, result = dea_add_frontier_point_estimates(df, year_t=1965, year_t1=1990, inputs=['KAPW'], outputs=['RGDPW'], total_growth=True)
result_tabulated = tabulate_simple(result, showindex=True)
print(result_tabulated)

# This takes time
df = dea_expand_all_years(df, inputs=['KAPW'], outputs=['RGDPW'], rts=['VRS','DRS'] )

# for overlay
df_1965_1990 = filter_multiindex(df, level=-1, items=[1965, 1990])
plot_multiindex_scatter_by_column(df_1965_1990, level=-1, x_column='KAPW', y_columns=['proj_RGDPW_VRS', 'RGDPW'], show_regression=False, show_corr=False, label_column='label', label_condition = [None, 'all'], legend=False, overlay=True, plot_kind=['line', 'scatter'], line_order=['x_asc', None],line_kwargs=[{'lw':2,'ls':'--','color': 'orange'},{'ls':'-'}], scatter_kwargs=[{},{'marker': 'o','color': 'blue'}], **dict()); print('plot with frontier line, with kwargs')

# for all
plot_multiindex_scatter_by_column(df, level=-1, x_column='KAPW', y_columns=['proj_RGDPW_VRS'], show_regression=False, show_corr=False, label_column='label', label_condition = None, legend=False, plot_kind='line', line_order='x_asc', **dict()); print('plot frontiers by year intervals, pwt56')

#for intervals
df_intervals = filter_multiindex(df, level=-1, items=[1965, 1970, 1980, 1990])
plot_multiindex_scatter_by_column(df_intervals, level=-1, x_column='KAPW', y_columns=['proj_RGDPW_VRS'], show_regression=False, show_corr=False, label_column='label', label_condition = ['U.S.A.'], legend=True, plot_kind='line', line_order='x_asc', **dict()); print('plot frontiers by year intervals, pwt56')
