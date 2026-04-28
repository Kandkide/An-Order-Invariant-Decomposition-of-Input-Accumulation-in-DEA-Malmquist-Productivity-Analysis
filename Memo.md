# Del eventually
default_script.py
example.py

# 操作
## End virtual environment
deactivate 

## Exec selected text as python code
shift+enter

## as python file
VS CodeのPython拡張機能がインストールされていれば、以下のキーで即座に実行できます。

    F5: デバッグモードで実行（ブレークポイントなどで止めることができる）

    Ctrl + F5: デバッグなしで実行（通常のスクリプト実行）

これらは「ファイルを保存してから実行」する挙動になります。


## personal memo
pwt56用の履歴ショートカットのメモ(使用上の注意を含む)
 - IRAN, VENEZUELAはあらかじめ除外する

temporary only for pwt56
 - pwt56_df = pwt56_df.rename(columns={'country': 'label'})
 - plot_multiindex_scatter_by_column(pwt56_df, level=None, x_column='KAPW', y_columns=['RGDPW'], show_regression=False, show_corr=True, label_column='label', label_condition='all', label_kwargs=None, overlay=False, plot_kind='scatter', line_kwargs=None, scatter_kwargs=None, line_order=None, **dict())
 - plot_multiindex_scatter_by_column(pwt56_df, level=-1, x_column='KAPW', y_columns=['proj_RGDPW_VRS'], show_regression=False, show_corr=False, label_column='country', label_condition = ['USA'], legend=True, plot_kind='line', line_order='x_asc', line_kwargs = [{"ls": "-", "lw": 3.0}, {"ls": "--", "lw": 2.0}, {"ls": "-.", "lw": 2.0}, {"ls": ":", "lw": 2.0}, {"ls": (0, (1, 1)), "lw": 1.0}, {"ls": (0, (5, 10)), "lw": 1.5}, {"ls": (0, (3, 5, 1, 5, 1, 5)), "lw": 1.5}, {"ls": (0, (3, 1)), "lw": 2.5}, {"ls": (0, (10, 2)), "lw": 1.0}, {"ls": (0, (1, 5)), "lw": 3.0}], **dict()); print('plot frontiers by year intervals, with label and line_kwargs, pwt56')
 - pwt56_df = _pwt56_df = dea_expand_all_years(pwt56_df, inputs=['KAPW'], outputs=['RGDPW'], rts=['VRS','DRS'] )
 - plot_multiindex_scatter_by_column(pwt56_df, level=-1, x_column='KAPW', y_columns=['proj_RGDPW_VRS'], show_regression=False, show_corr=False, label_column='country', label_condition = ['USA'], legend=False, plot_kind='line', line_order='x_asc', **dict()); print('plot frontiers by year intervals, pwt56')
 - plot_multiindex_scatter_by_column(pwt56_df, level=-1, x_column='KAPW', y_columns=['proj_RGDPW_VRS', 'RGDPW'], show_regression=False, show_corr=False, label_column='country', label_condition = [None, 'all'], legend=False, overlay=True, plot_kind=['line', 'scatter'], line_order=['x_asc', None],line_kwargs=[{'lw':2,'ls':'--','color': 'orange'},{'ls':'-'}], scatter_kwargs=[{},{'marker': 'o','color': 'blue'}], **dict()); print('plot with frontier line, with kwargs, pwt56')
 - tabulate_simple(df_mi, showindex=True)
 - df_new, df_mi = dea_add_frontier_point_estimates(pwt56_df, year_t=1965, year_t1=1990, inputs=['KAPW'], outputs=['RGDPW'], total_growth=True); print('KR like Malmquist index calculation 2D')
 - plot_multiindex_kde(pwt56_df, column='RGDPW', level='year', values=None, bw_method=None, kernel='gaussian', grid_points=512, ax=None, fill=False, legend=True, show=True, block=True, close_after_show=True, palette=None, linestyles=None, linewidth=None, alpha=0.6)
 - plot_multiindex_kde(pwt56_df, column='RGDPW', level='year', values=None, bw_method='scott', kernel='gaussian', grid_points=512, ax=None, fill=False, legend=True, show=True, block=True, close_after_show=True, palette=None, linestyles=None, linewidth=None, alpha=0.6)
 - plot_multiindex_kde(pwt56_df, column='RGDPW', level='year', values=None, bw_method=4000, kernel='gaussian', grid_points=512, ax=None, fill=False, legend=True, show=True, block=True, close_after_show=True, palette=None, linestyles=None, linewidth=None, alpha=0.6)


pwt110用の履歴ショートカットのメモ(使用上の注意を含む)

前半の2つがデータ準備、精査用
後半の2つが分析用
 - (注意)あらかじめ、年でフィルターをかけることが必要なコマンドあり(とくにtemporary-in-VRS-case)
 - (注意)あらかじめ、year, countryの列の削除か名前変更が必要なコマンドあり(本来は関数側を修正するのが望ましい)

Retry-init: ☓不要、- 必須
 - pwt110_df['working_hours']=_pwt110_df['working_hours']=_pwt110_df['emp']*_pwt110_df['avh']
 - pwt110_df['rgdpo_per_hour_worked']=_pwt110_df['rgdpo_per_hour_worked']=_pwt110_df['rgdpo']/_pwt110_df['working_hours']
 - pwt110_df['rgdpo_per_worker']=_pwt110_df['rgdpo_per_worker']=_pwt110_df['rgdpo']/_pwt110_df['emp']
 - pwt110_df['capital_output_ratio']=_pwt110_df['capital_output_ratio']=_pwt110_df['rnna']/_pwt110_df['rgdpna']
 - pwt110_df['rn']=_pwt110_df['rn']=_pwt110_df['capital_output_ratio']*_pwt110_df['rgdpo']
 - pwt110_df['rn_per_hour_worked']=_pwt110_df['rn_per_hour_worked']=_pwt110_df['rn']/_pwt110_df['working_hours']
 - pwt110_df['rn_per_worker']=_pwt110_df['rn_per_worker']=_pwt110_df['rn']/_pwt110_df['emp']
 ☓ pwt110_df = _pwt110_df = modify_add_suffix_by_year(pwt110_df, year_condition=lambda y: int(y) == 2023, suffix='+', target_col='countrycode', new_col=None)

Retry-extract-step-by-step: ☓不要か？、△選択、- 必須
 - extra=['Türkiye', 'Syrian Arab Republic', 'Qatar', 'Poland', 'New Zealand', 'Norway', 'Mexico', 'Luxembourg', 'Kuwait', 'Ireland', 'Gabon', 'Egypt', 'Bulgaria', 'Bosnia and Herzegovina', 'Myanmar']
 ☓ extra=['United Arab Emirates' , 'Azerbaijan', 'Iraq' , 'Romania' , 'Saudi Arabia', "Trinidad and Tobago"]
 ☓ extra=['Oman','Sudan']
 △ tabulate_simple(pwt110_df[pwt110_df['is_efficient_VRS']==True])
 △ pwt110_df[pwt110_df['is_efficient_VRS']==True]['country'].unique()
 ☓ extra=['Angola' ,'Bahrain','Venezuela (Bolivarian Republic of)','Algeria']
 △ pwt110_df['country'].nunique()
 - extra=['Türkiye', 'Syrian Arab Republic', 'Qatar', 'Poland', 'New Zealand', 'Norway', 'Mexico', 'Luxembourg', 'Kuwait', 'Ireland', 'Gabon', 'Egypt', 'Bulgaria', 'Bosnia and Herzegovina', 'Myanmar','United Arab Emirates' , 'Azerbaijan', 'Iraq' , 'Romania' , 'Saudi Arabia', "Trinidad and Tobago",'Oman','Sudan','Angola' ,'Bahrain','Venezuela (Bolivarian Republic of)','Algeria']
 - pwt110_df = _pwt110_df = filter_oil_producers(pwt110_df, mode='extra_only', invert=True, extra=extra)
 △ pwt110_df.groupby(level='year')['efficiency_DRS'].count().to_string()

temporary-in-VRS-Malmquist-calculation: - 選択必修(全部)
 - df_new, df_mi = dea_add_frontier_point_estimates(pwt110_df, year_t=1990, year_t1=2023, inputs=['rn','working_hours', 'hc'], outputs=['rgdpo'], keep_columns=['countrycode']); print('Malmquist index calculation 4D with countrycode')
 - df_new, df_mi = dea_add_frontier_point_estimates(pwt110_df, year_t=1990, year_t1=2023, inputs=['rn','working_hours', 'hc'], outputs=['rgdpo'], keep_columns=['countrycode'],rts='CRS'); print('Malmquist index calculation 4D with countrycode (CRS)')
 - tabulate_simple(df_new, showindex=True)
 - df_new, df_mi = dea_add_frontier_point_estimates(pwt110_df, year_t=1990, year_t1=2023, inputs=['rn_per_hour_worked', 'hc'], outputs=['rgdpo_per_hour_worked']); print('Malmquist index calculation 3D')
 - tabulate_simple(df_mi, showindex=True)
 - df_new, df_mi = dea_add_frontier_point_estimates(pwt110_df, year_t=1990, year_t1=2023, inputs=['rn_per_hour_worked', 'hc'], outputs=['rgdpo_per_hour_worked'], keep_columns=['countrycode']); print('Malmquist index calculation 3D with countrycode')
 - tabulate_simple(df_mi, showindex=True)
 - plot_multiindex_scatter_by_column(df_mi, level=None, x_column='ACCUM1 (CAGR %)', y_columns=['EFF (CAGR %)'], show_regression=True, show_corr=True, label_column='countrycode (1990)', label_condition=['all'], label_kwargs=None, overlay=False, plot_kind='scatter', line_kwargs=None, scatter_kwargs=None, line_order=None, legend=False, **dict())
 - df_new, df_mi = dea_add_frontier_point_estimates(pwt110_df, year_t=1990, year_t1=2023, inputs=['rn_per_hour_worked'], outputs=['rgdpo_per_hour_worked']); print('Malmquist index calculation 2D')
 - tabulate_simple(df_mi.corr(numeric_only=True), showindex=True)
 - plot_multiindex_scatter_by_column(df_mi, level=None, x_column='rgdpo_per_hour_worked (1990)', y_columns=None, show_regression=True, show_corr=True, label_column='countrycode (1990)', label_condition=['all'], label_kwargs=None, overlay=False, plot_kind='scatter', line_kwargs=None, scatter_kwargs=None, line_order=None, legend=False, **dict())

temporary-in-VRS-case: ☓不要か？、- 選択必修
 ☓ pwt110_df = modify_add_suffix_by_year(pwt110_df, year_condition=lambda y: int(y) == 1990, suffix='+', target_col='countrycode', new_col=None)
 - plot_multiindex_scatter_3d_overlay(pwt110_df, level=-1, x_column=['rn_per_hour_worked','rn_per_hour_worked'], y_column=['hc','hc'], z_column=['proj_rgdpo_per_hour_worked_VRS','rgdpo_per_hour_worked'], label_column='countrycode', label_condition=[None,['USA','JPN','CHE','TWN','ISL','DEU']], label_kwargs=None, color=None, marker=None, size=None, envelope=[True,False], envelope_groupwise=True, envelope_facecolor=None, envelope_alpha=0.5, envelope_edgecolor='k', envelope_edgewidth=0, figsize=(8, 6), elev=0, azim=-90, show_legend=False, title_and_labels=['','','',''], show_ticks=True, **dict()); print('3D plot overlay')
 - plot_multiindex_scatter_3d(pwt110_df, level=-1, x_column='rn_per_hour_worked', y_column='hc', z_column='proj_rgdpo_per_hour_worked_VRS', label_column='countrycode', label_condition=['USA'], label_kwargs=None, color=None, marker=None, size=None, figsize=(8, 6), elev=0, azim=-90, show_legend=True, envelope=True, title_and_labels=['', 'rn', 'hc', 'rgdp'], **dict()); print('3D plot for envalops')
 - plot_multiindex_scatter_by_column(pwt110_df, level=-1, x_column='rn_per_hour_worked', y_columns=['proj_rgdpo_per_hour_worked_VRS', 'rgdpo_per_hour_worked'], show_regression=False, show_corr=False, label_column='countrycode_tagged', label_condition = [None, 'all'], legend=False, overlay=True, plot_kind=['line', 'scatter'], line_order=['x_asc', None], **dict()); print('plot with frontier line')
 - plot_multiindex_scatter_by_column(pwt110_df, level=-1, x_column='rn_per_hour_worked', y_columns=['proj_rgdpo_per_hour_worked_VRS', 'rgdpo_per_hour_worked'], show_regression=False, show_corr=False, label_column='countrycode', label_condition = [None, 'all'], legend=False, overlay=True, plot_kind=['line', 'scatter'], line_order=['x_asc', None],line_kwargs=[{'lw':2,'ls':'--','color': 'orange'},{'ls':'-'}], scatter_kwargs=[{},{'marker': 'o','color': 'blue'}], **dict()); print('plot with frontier line, with kwargs')
 - plot_multiindex_scatter_by_column(pwt110_df, level=-1, x_column='rn_per_hour_worked', y_columns=['proj_rgdpo_per_hour_worked_VRS'], show_regression=False, show_corr=False, label_column='countrycode', label_condition = ['USA'], legend=True, plot_kind='line', line_order='x_asc', **dict()); print('plot frontiers by year intervals, with label')
 - plot_multiindex_scatter_by_column(pwt110_df, level=-1, x_column='rn_per_hour_worked', y_columns=['proj_rgdpo_per_hour_worked_VRS'], show_regression=False, show_corr=False, label_column='countrycode', label_condition = ['USA'], legend=True, plot_kind='line', line_order='x_asc', line_kwargs = [{"ls": "-", "lw": 3.0}, {"ls": "--", "lw": 2.0}, {"ls": "-.", "lw": 2.0}, {"ls": ":", "lw": 2.0}, {"ls": (0, (1, 1)), "lw": 1.0}, {"ls": (0, (5, 10)), "lw": 1.5}, {"ls": (0, (3, 5, 1, 5, 1, 5)), "lw": 1.5}, {"ls": (0, (3, 1)), "lw": 2.5}, {"ls": (0, (10, 2)), "lw": 1.0}, {"ls": (0, (1, 5)), "lw": 3.0}], **dict()); print('plot frontiers by year intervals, with label, line_kwargs')
 - plot_multiindex_scatter_by_column(pwt110_df, level=-1, x_column='rn_per_hour_worked', y_columns=['proj_rgdpo_per_hour_worked_VRS'], show_regression=False, show_corr=False, label_column='countrycode', label_condition = ['USA'], legend=True, plot_kind='line', line_order='x_asc', line_kwargs = [{"ls": "-", "lw": 3.0}, {"ls": "--", "lw": 2.0}, {"ls": "-.", "lw": 2.0}, {"ls": ":", "lw": 2.0}, {"ls": (0, (1, 1)), "lw": 1.0}, {"ls": (0, (5, 10)), "lw": 1.5}, {"ls": (0, (3, 5, 1, 5, 1, 5)), "lw": 1.5}, {"ls": (0, (3, 1)), "lw": 2.5}, {"ls": (0, (10, 2)), "lw": 1.0}, {"ls": (0, (1, 5)), "lw": 3.0}], **dict()); print('plot frontiers by year intervals, with label and line_kwargs')
 - plot_multiindex_kde(pwt110_df, column='rgdpo_per_hour_worked', level='year', values=None, bw_method='scott', kernel='gaussian', grid_points=512, ax=None, fill=False, legend=True, show=True, block=True, close_after_show=True, palette=None, linestyles=[":","-","--"], linewidth=None, alpha=0.6); print('kde plot, limit years before application')

# TODO
国際経済研究会レジュメ
ORCIDの取得(情報登録は後でゆっくり)

