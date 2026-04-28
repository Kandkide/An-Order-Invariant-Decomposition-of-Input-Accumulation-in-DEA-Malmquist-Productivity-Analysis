import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from scipy.stats import pearsonr

import tkinter as tk
from tkinter import messagebox

from boilerplate_filter import get_non_na_groups_by_year
from boilerplate_base import _prepare_y_columns

# alternative
# from adjustText import adjust_text

# 初期設定
def set_initial_style():
    """
    Matplotlib の描画スタイル初期設定を行う関数。

    この関数は、グラフのタイトル・軸ラベル・目盛・凡例などの
    フォントサイズや太さ、フォントファミリーを一括で設定します。
    スクリプトを読み込んだ時点で rcParams を更新します。

    【使い方】
        set_initial_style() を呼び出すと、以降に描画する全てのグラフに
        この設定が適用されます。

    【変更方法】
        - フォントサイズを変えたい場合は、mpl.rcParams.update() 内の
          'axes.titlesize', 'axes.labelsize', 'xtick.labelsize', 'ytick.labelsize',
          'legend.fontsize', 'legend.title_fontsize', 'figure.titlesize', 'font.size'
          の値を変更してください。
        - 太字／標準を切り替える場合は、'axes.titleweight', 'axes.labelweight',
          'figure.titleweight' の値を 'bold' または 'normal' に変更してください。
        - フォントファミリーを変える場合は、'font.family' を変更してください。

    【注意】
        この関数は rcParams をグローバルに変更するため、
        他のスクリプトやノートブックにも影響します。
        一時的に変更したい場合は、後から mpl.rcParams.update() を再度呼び出してください。
    """
    import matplotlib as mpl
    mpl.rcParams.update({
        'axes.titlesize': 20,         # サブプロットタイトル
        'axes.titleweight': 'normal', # タイトル太さ
        'axes.labelsize': 20,         # 軸ラベル
        'axes.labelweight': 'normal', # 軸ラベル太さ
        'xtick.labelsize': 20,        # x軸目盛
        'ytick.labelsize': 20,        # y軸目盛
        'legend.fontsize': 11,        # 凡例
        'legend.title_fontsize': 11,  # 凡例タイトル
        'figure.titlesize': 22,       # 図全体タイトル
        'figure.titleweight': 'bold', # 図全体タイトル太さ
        'font.size': 14,              # その他テキストのデフォルト
        'font.family': 'Noto Sans CJK JP'    # 日本語フォントを指定(ubuntuの場合)
        # 'font.family': 'IPAGothic'    # 日本語フォントを指定
    })


set_initial_style()


def plot_simple(df, **kwargs):
    df.plot(**kwargs)
    plt.show()


def confirm_large_plot_count(n_plots, threshold=5):
    """
    プロット枚数が閾値を超える場合に、ユーザーに続行確認を促すダイアログを表示。

    Parameters:
    n_plots (int): 予定されているプロット枚数
    threshold (int): 警告を出すプロット枚数の閾値（デフォルト: 5）

    Returns:
    bool: ユーザーが続行を選択した場合 True、キャンセルした場合 False
    """
    if n_plots <= threshold:
        return True

    root = tk.Tk()
    root.withdraw()  # メインウィンドウを非表示

    msg = f"{n_plots} 枚のグラフを描画しようとしています。\n表示に時間がかかる可能性があります。\n続行しますか？"
    result = messagebox.askyesno("Plot count warning", msg)

    root.destroy()
    return result


def plot_multiindex_by_group(df, level=0, y_columns=None, x_index=-1, plot_kind='line', stacked=False, **kwargs):
    """
    マルチインデックスのデータフレームをグループごとに分割してプロットする関数。

    Parameters:
    df (pd.DataFrame): マルチインデックスのデータフレーム
    level (int or str): グループ化するインデックスのレベル
    y_columns (list or None): プロットする列（None の場合は数値列すべて）
    x_index (int or str): x軸に使用するインデックスのレベル（デフォルトは -1）
    plot_kind (str): グラフの種類 ('line', 'bar' など)
    stacked (bool): 積み重ね棒グラフにするかどうか（bar の場合のみ有効）
    **kwargs: plot メソッドに渡す追加パラメータ

    Returns:
    None
    """
    y_columns = _prepare_y_columns(df, y_columns)
    groups = list(df.groupby(level=level))

    # プロット枚数が多い場合の警告
    if not confirm_large_plot_count(len(groups), threshold=5):
        print("プロットを中止しました。")
        return

    # モノクロ判別用の線種とマーカーの組み合わせ
    line_styles = ['-', '--', '-.', ':']
    markers = ['o', 's', 'D', '^', 'v', '<', '>', 'x', '+', '*']
    style_cycle = [
        line_styles[i % len(line_styles)] + markers[i % len(markers)]
        for i in range(len(y_columns))
    ]

    for group, data in groups:
        fig, ax = plt.subplots(figsize=(6, 4))
        x_values = data.index.get_level_values(x_index)

        if pd.api.types.is_integer_dtype(x_values):
            ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

        plot_df = data[y_columns].copy()
        plot_df.index = x_values

        if plot_kind == 'line':
            # kwargsにstyleが未指定なら自動生成
            if 'style' not in kwargs:
                # 列数に応じてスタイルを割り当て
                kwargs['style'] = style_cycle[:len(y_columns)]

        plot_df.plot(kind=plot_kind, stacked=stacked, ax=ax, **kwargs)

        x_label = df.index.names[x_index] if isinstance(x_index, int) else str(x_index)
        ax.set_xlabel(x_label)
        ax.set_title(str(group))
        ax.grid(True)
        plt.tight_layout()
        plt.show()


def plot_multiindex_by_column(df, level=0, y_columns=None, **kwargs):
    """
    各列（y_columns）の値を個別プロットし、各プロット内でマルチインデックスの指定レベルによりグループ化された系列を表示。

    Parameters:
    df (pd.DataFrame): マルチインデックスのデータフレーム
    level (int or str): グループ化に使うインデックスのレベル
    y_columns (list or None): 対象とする列（None の場合、数値列を使用）

    Returns:
    None
    """
    y_columns = _prepare_y_columns(df, y_columns)

    # プロット枚数が多い場合の警告
    if not confirm_large_plot_count(len(y_columns), threshold=5):
        print("プロットを中止しました。")
        return

    grouped = df.groupby(level=level)

    for col in y_columns:
        fig, ax = plt.subplots(figsize=(6, 4))
        for group_name, group_data in grouped:
            x_values = group_data.index.get_level_values(-1)
            y_values = group_data[col]
            ax.plot(x_values, y_values, marker='o', label=str(group_name), **kwargs)

        ax.set_title(f"{col}")
        ax.set_xlabel(df.index.names[-1])
        ax.set_ylabel(col)
        ax.legend(title=df.index.names[level])
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

        ax.grid(True)
        plt.show()


# --- グループ単位で解決するヘルパー（なければ追加） ---
def _get_per_group_param(param, y_col, group, y_idx, n_y, group_idx=None):
    """
    group-first の解決ルール:
      - None -> None
      - scalar -> そのまま
      - list/tuple -> group_idx が与えられればそれを優先、なければ y_idx を使う（末尾繰り返し）
      - dict:
          1) param[group] を試す（group-keyed があれば優先）
             - もし param[group] が dict で y_col キーを持つならそれを返す（group->y のネスト対応）
          2) param[y_col] を試す（y-keyed）
          3) dict の値を順序リストとみなし group_idx で選ぶ（group_idx が与えられている場合）
          4) None
    """
    if param is None:
        return None

    # scalar
    if not isinstance(param, (dict, list, tuple)):
        return param

    # list/tuple: group_idx 優先（グループ化時）、なければ y_idx
    if isinstance(param, (list, tuple)):
        if group_idx is not None:
            idx = group_idx
        else:
            idx = y_idx if y_idx is not None else 0
        if idx < len(param):
            return param[idx]
        return param[-1] if len(param) > 0 else None

    # dict の場合: group-first
    if isinstance(param, dict):
        # 1) group キー優先
        if group in param:
            val = param[group]
            # group -> y のネスト（例: {'A': {'temp': {...}}}）
            if isinstance(val, dict) and y_col in val:
                return val[y_col]
            return val
        # 2) y_col キー
        if y_col in param:
            val = param[y_col]
            # y -> group のネスト（例: {'temp': {'A': {...}}}）
            if isinstance(val, dict) and group in val:
                return val[group]
            return val
        # 3) フォールバック: dict の値を順序リストとみなし group_idx で選ぶ
        if group_idx is not None:
            vals = list(param.values())
            if len(vals) > 0:
                idx = group_idx if group_idx < len(vals) else -1
                return vals[idx]
        return None


def plot_multiindex_scatter_by_column(
    df,
    level=0,
    x_column=None,
    y_columns=None,
    show_regression=False,
    show_corr=True,
    label_column=None,
    label_condition=None,
    label_kwargs=None,
    overlay=False,
    plot_kind='scatter',         # 'scatter' | 'line' | 'both'  (scalar/list/dict 対応)
    line_kwargs=None,            # 線描画用オプション (scalar/list/dict)
    scatter_kwargs=None,         # 点描画用オプション (scalar/list/dict)
    line_order=None,             # 'column'|'x_asc'|'x_desc'|'y_asc'|'y_desc' or callable or list/dict
    **kwargs
):
    """
    Seaborn を用いたマルチインデックス DataFrame の散布図／折れ線プロットユーティリティ

    概要
    ----
    指定したインデックスレベルでグループ化したデータを可視化します。各グループごとに散布点、線、回帰線、相関係数表示、条件付きラベル付けを行え、
    さらに y 列ごと／グループごとに描画モードや描画オプションを個別に指定できます。複数の y 列を同一図に重ねる overlay モードにも対応します。

    主な機能
    - **level** で指定したインデックスレベルでグループ化してプロットする。
    - **plot_kind** により各系列を `'scatter'`（点のみ）/ `'line'`（線のみ）/ `'both'`（点＋線）で描画可能。
    - **overlay=True** で複数の y 列を同一軸に重ねて描画する。
    - **show_regression=True** で回帰線（seaborn.regplot）を表示できる（回帰線は scatter/both 時に適用）。
    - **show_corr=True** で各系列の Pearson 相関係数とサンプル数を図内に表示する。
    - **label_column / label_condition / label_kwargs** をスカラー・リスト・辞書で渡すと、y 列ごとに個別の注釈設定が可能。
    - **line_kwargs / scatter_kwargs** は *グループ優先（group-first）* の解決ルールを採用し、柔軟に指定できる。
    - **line_order** により線を結ぶ順序を制御できる（列順、x の昇降順、y の昇降順、または callable によるカスタム順序）。

    パラメータ
    ----------
    df : pandas.DataFrame
        マルチインデックスを持つデータフレーム（または通常の DataFrame）。`reset_index()` 後の列を参照して注釈等を行う。
    level : int or str or None, default 0
        グループ化に使うインデックスレベル。None の場合はグループ化せず全体をプロットする。
    x_column : str or None
        x 軸に使う列名。None の場合は自動選択される（数値列の先頭など）。
    y_columns : list or None
        y 軸に使う列名のリスト。None の場合は自動選択される。
    show_regression : bool, default False
        True のとき回帰線を表示する（regplot を使用）。回帰線は scatter/both モードで有効。
    show_corr : bool, default True
        True のとき各系列の Pearson 相関係数とサンプル数を図内に表示する。
    label_column : str or list or dict or None, default None
        注釈に使う列名。スカラー、リスト、辞書で y 列ごとに指定可能。None の場合は reset_index によって追加されたインデックス列を自動選択する。
    label_condition : callable or list or dict or scalar or None, default None
        注釈対象を絞る条件。スカラー／リスト／辞書／callable を受け付ける。callable は `(labels, x, y) -> boolean Series` を返す。
    label_kwargs : dict or list or dict of dict or None, default None
        `ax.annotate` に渡すオプション。y 列ごとに個別指定可能。
    overlay : bool, default False
        True のとき y_columns を同一図に重ねて描画する。
    plot_kind : str or list or dict, default 'scatter'
        各系列の描画モード。'scatter' | 'line' | 'both' を指定。リストや辞書で y 列ごとに指定可能。
    line_kwargs : dict or list or dict of dict or None
        線描画（ax.plot）に渡すオプション。**グループ優先（group-first）** の解決ルールを採用。
    scatter_kwargs : dict or list or dict of dict or None
        点描画（scatter/regplot の scatter_kws）に渡すオプション。**グループ優先（group-first）** の解決ルールを採用。
    line_order : str or callable or list or dict or None, default None
        線を結ぶ順序を指定。'column'（列順、デフォルト） / 'x_asc' / 'x_desc' / 'y_asc' / 'y_desc'、
        または `(df_group) -> df_sorted` のような callable を渡してカスタム順序を指定可能。リスト/辞書で y 列ごとに指定可。
    **kwargs : dict
        seaborn / matplotlib に渡す追加オプション。`legend` を含めると凡例表示を制御できる（デフォルト True）。

    戻り値
    ------
    なし（プロットを表示）。必要に応じて fig, ax を返すよう改修可能。

    実装上の注意（重要）
    -------------------
    **line_kwargs / scatter_kwargs の解決ルール（group-first）**
    1. **group-keyed** を最優先で探す（例: `{'A': {...}, 'B': {...}}`）。  
    - `param[group]` が dict でさらに y 列キーを含む場合（`param[group][y_col]`）はそれを使う（group→y のネスト対応）。
    2. 次に **y-keyed** を探す（例: `{'temp': {...}, 'precip': {...}}`）。  
    - `param[y_col]` が dict でさらに group キーを含む場合（`param[y_col][group]`）はそれを使う（y→group のネスト対応）。
    3. group / y のどちらのキーも見つからない場合、**group_idx（グループループの順序）** が与えられていれば、辞書の値を順序リストとみなし `group_idx` による順次適用を行う（キーを明示しなくても順に割り当て可能）。  
    4. `param` が **list/tuple** の場合は、**グループ化されているときは group_idx を優先**して選び、グループ化がない（または group_idx が与えられない）場合は y 列インデックス（y_idx）を使う。リストが短い場合は末尾を繰り返す。  
    5. `param` が **scalar** の場合は全ての系列／グループに同じ設定を適用する。  
    6. **level が None（グループ化なし）** のときは、従来どおり y 列ベース（y_idx）で解決される。

    その他の注意
    - `show_regression=True` と `plot_kind='both'` の組合せでは seaborn.regplot が点と回帰線を描画するため、点・線の細かな見た目は regplot の `scatter_kws` / `line_kws` で制御してください。
    - `line_order` に callable を渡す場合、関数は受け取った DataFrame を並べ替えて返すこと（返された行順が描画順となる）。
    - 注釈の重なりが気になる場合は外部パッケージ `adjustText` の導入を検討してください。
    - 多数系列を overlay すると凡例や視認性が悪くなるため、色・マーカー・透明度や凡例の列数を調整することを推奨します。

    kwargs の指定例（`line_kwargs` / `scatter_kwargs` の渡し方）
    ------------------------------------------------------------
    # 1) スカラー（全グループ・全 y に同じ設定）
    line_kwargs = {'lw': 1.5}
    scatter_kwargs = {'s': 30, 'alpha': 0.8}

    # 2) リスト（グループ順に順次適用したい場合）
    #    グループが 3 個なら group0 に 1 番目、group1 に 2 番目、group2 に 3 番目が適用される
    line_kwargs = [{'lw':2}, {'lw':1}, {'lw':0.5}]
    scatter_kwargs = [{'s':50}, {'s':30}, {'s':10}]

    # 3) group-keyed 辞書（group-first の典型）
    #    直接グループ名で指定する（最優先でマッチ）
    line_kwargs = {
        'A': {'lw':2, 'ls':'-'},
        'B': {'lw':1, 'ls':'--'}
    }
    scatter_kwargs = {
        'A': {'s':60, 'alpha':0.9},
        'B': {'s':20, 'alpha':0.6}
    }

    # 4) y-keyed 辞書（y 列ごとにグループ別設定をしたい場合）
    #    y -> group のネストもサポート（例: temp 列の A グループだけ別設定）
    line_kwargs = {
        'temp': {'A': {'lw':2}, 'B': {'lw':1}},
        'precip': {'lw':1.2}
    }

    # 5) y-first あるいは group-first の混在（柔軟なネスト）
    #    group-first 実装のため、group-keyed があればそれが優先され、次に y-keyed を参照します
    line_kwargs = {
        'A': {'temp': {'lw':2}, 'precip': {'lw':1}},
        'temp': {'lw':1.5}
    }

    # 6) キーを明示しない辞書（順序を使って group に割り当てたい場合）
    #    {'g1': {...}, 'g2': {...}} のようにキー名を気にせず順序で割り当てるときは
    #    group が存在する場合は group_idx に従って順次適用される
    line_kwargs = {'x1': {'lw':2}, 'x2': {'lw':1}, 'x3': {'lw':0.5}}

    使用例
    ------
    # 列ごとに個別設定して overlay で重ねる（グループ優先の辞書やリストを混在して使える）
    plot_multiindex_scatter_by_column(
        df,
        level=0,
        x_column='year',
        y_columns=['temp','precip','humidity'],
        overlay=True,
        plot_kind=['line','scatter','both'],
        line_kwargs=[{'lw':2}, {}, {'lw':1,'ls':'--'}],          # y 列ベースのリスト
        scatter_kwargs={'A': {'temp': {'s':60}}, 'B': {'temp': {'s':30}}},  # group-keyed の例
        label_column=['country', None, 'region'],
        label_condition=[None, lambda labels,x,y: y>100, 'all'],
        line_order=['column','x_asc','y_desc'],
        legend=True
    )

    # グループごとに順に適用したい（キーを明示しない辞書やリストを使う）
    plot_multiindex_scatter_by_column(
        df,
        level='group',
        x_column='time',
        y_columns=['A','B'],
        plot_kind='line',
        line_kwargs=[{'lw':2}, {'lw':1}],   # group_idx に従って適用
        scatter_kwargs={'s':20},             # 全体に同じ設定
    )

    # モノクロ用の線種セット（１０種）
    line_kwargs = [{"ls": "-", "lw": 3.0}, {"ls": "--", "lw": 2.0}, {"ls": "-.", "lw": 2.0}, {"ls": ":", "lw": 2.0}, {"ls": (0, (1, 1)), "lw": 1.0}, {"ls": (0, (5, 10)), "lw": 1.5}, {"ls": (0, (3, 5, 1, 5, 1, 5)), "lw": 1.5}, {"ls": (0, (3, 1)), "lw": 2.5}, {"ls": (0, (10, 2)), "lw": 1.0}, {"ls": (0, (1, 5)), "lw": 3.0}]
    """
    legend = kwargs.pop("legend", True)
    label_kwargs = label_kwargs or {}
    line_kwargs = line_kwargs or {}
    scatter_kwargs = scatter_kwargs or {}

    # --- ヘルパー: y_columns に対して param を返す ---
    def _get_per_y_param(param, y_col, idx, n):
        if param is None:
            return None
        if isinstance(param, dict):
            return param.get(y_col, None)
        if isinstance(param, (list, tuple)):
            if idx < len(param):
                return param[idx]
            else:
                return param[-1] if len(param) > 0 else None
        return param

    def _maybe_sort_xy_for_line(x_series, y_series, data_df, order_spec):
        """
        x_series, y_series: pandas Series (index corresponds to data_df)
        data_df: 元の DataFrame（group 単位または全体）
        order_spec: None | 'column' | 'x_asc' | 'x_desc' | 'y_asc' | 'y_desc' | callable
        戻り値: (x_sorted, y_sorted) — Series（index preserved or reset）
        """
        if order_spec is None or order_spec == 'column':
            return x_series, y_series

        # callable: data_df を渡して並べ替えられた DataFrame を受け取る
        if callable(order_spec):
            try:
                df_sorted = order_spec(data_df.copy())
                # try to extract x and y by name; fall back to positional
                xname = x_series.name
                yname = y_series.name
                if xname in df_sorted.columns and yname in df_sorted.columns:
                    xs = df_sorted[xname]
                    ys = df_sorted[yname]
                else:
                    # fallback: assume first two numeric-like columns correspond
                    xs = df_sorted.iloc[:, 0]
                    ys = df_sorted.iloc[:, 1] if df_sorted.shape[1] > 1 else df_sorted.iloc[:, 0]
                # return with original index dropped to avoid index mismatches in plotting
                return xs.reset_index(drop=True), ys.reset_index(drop=True)
            except Exception:
                return x_series, y_series

        # キーワード指定によるソート
        if order_spec in ('x_asc', 'x_desc'):
            asc = order_spec == 'x_asc'
            idx_sorted = x_series.sort_values(ascending=asc).index
            return x_series.loc[idx_sorted], y_series.loc[idx_sorted]
        if order_spec in ('y_asc', 'y_desc'):
            asc = order_spec == 'y_asc'
            idx_sorted = y_series.sort_values(ascending=asc).index
            return x_series.loc[idx_sorted], y_series.loc[idx_sorted]

        # 未知の指定は元順序
        return x_series, y_series

    def _plot_group(
        ax, x, y, labels=None, label=None,
        show_regression=False,
        show_corr=True,
        corr_text_pos=(0.05, 0.95),
        scatter_kws=None,
        line_kws=None,
        kind='scatter',
        label_condition_local=None,
        label_kwargs_local=None,
        **kwargs
    ):
        scatter_kws = scatter_kws or {}
        line_kws = line_kws or {}
        label_kwargs_local = label_kwargs_local or {}

        valid = x.notna() & y.notna()
        x = x[valid]
        y = y[valid]
        if labels is not None:
            labels = labels[valid]

        # 描画ロジック: kind に応じて scatter / line / both を描く
        # regplot は show_regression=True かつ kind が 'both'/'scatter' の場合に優先的に使う（回帰線を引く）
        if show_regression and kind in ('scatter', 'both'):
            plot_label = label if legend else "_nolegend_"
            sns.regplot(x=x, y=y, ax=ax, label=plot_label, ci=None,
                        scatter_kws={**dict(s=40, alpha=0.8), **scatter_kws},
                        line_kws={**dict(lw=2), **line_kws},
                        **kwargs)
        else:
            # 点描画
            if kind in ('scatter', 'both'):
                sns.scatterplot(x=x, y=y, ax=ax,
                                label=(label if legend else "_nolegend_"),
                                legend=legend,
                                **{**dict(s=40, alpha=0.8), **scatter_kws})
            # 線描画（点は描かない or 両方のときは点と線両方）
            if kind in ('line', 'both'):
                # ax.plot を使う（pandas/Matplotlib の線描画）
                # ax.plot は Series の index を x 軸に使うため、x,y は Series（順序が重要）
                ax.plot(x.values, y.values, label=(label if legend else "_nolegend_"), **line_kws)

        # ラベル付け（条件が与えられていれば）
        if labels is not None and label_condition_local is not None:
            if label_condition_local == 'all':
                mask = pd.Series(True, index=labels.index)
            elif callable(label_condition_local):
                mask = label_condition_local(labels, x, y)
            else:
                if isinstance(label_condition_local, (list, set, tuple)):
                    mask = labels.isin(label_condition_local)
                else:
                    mask = labels == label_condition_local

            for lx, ly, lab in zip(x[mask], y[mask], labels[mask]):
                ax.annotate(
                    str(lab),
                    xy=(lx, ly),
                    xytext=(3, 3),
                    textcoords='offset points',
                    fontsize=8,
                    **label_kwargs_local
                )
        # 表示が、気に入らないので、当面はなし。 # alternative
        # if labels is not None and label_condition_local is not None:
        #     if label_condition_local == 'all':
        #         mask = pd.Series(True, index=labels.index)
        #     elif callable(label_condition_local):
        #         mask = label_condition_local(labels, x, y)
        #     else:
        #         if isinstance(label_condition_local, (list, set, tuple)):
        #             mask = labels.isin(label_condition_local)
        #         else:
        #             mask = labels == label_condition_local


        #     texts = []
        #     pts_x = x[mask]
        #     pts_y = y[mask]
        #     labs = labels[mask]

        #     # まずはテキストを配置（微小オフセットで重なり判定を安定化）
        #     for lx, ly, lab in zip(pts_x, pts_y, labs):
        #         t = ax.text(lx, ly, str(lab), fontsize=8, ha='left', va='bottom', **label_kwargs_local)
        #         texts.append(t)

        #     arrowprops = {
        #         'arrowstyle': '-',
        #         'lw': 0.35,
        #         'alpha': 0.6,
        #         'connectionstyle': 'arc3,rad=0.0',
        #         'mutation_scale': 6
        #     }

        #     adjust_text(
        #         texts,
        #         x=pts_x.tolist(),
        #         y=pts_y.tolist(),
        #         ax=ax,
        #         expand_text=(1.01, 1.02),    # テキスト間は少し余裕を持たせる
        #         expand_points=(1.00, 1.00),  # 点との余白は最小（点に近づける）
        #         force_text=0.10,             # テキスト同士の反発は小さめ
        #         force_points=0.0,            # 点との反発は無効化
        #         only_move={'text':'y'},      # テキストだけ動かす（点は固定）
        #         autoalign=False,             # 自動整列を切ると点に近い配置になりやすい
        #         arrowprops=arrowprops              # 矢印を出さない（必要なら控えめにする）
        #     )

        if show_corr and len(x) > 1 and len(y) > 1:
            try:
                corr, _ = pearsonr(x, y)
            except Exception:
                corr = float('nan')
            n = len(x)
            text = (
                f"{label + ': ' if label else ''}"
                f"r = {corr:.3f}, n = {n}"
            )
            ax.text(
                corr_text_pos[0], corr_text_pos[1], text,
                transform=ax.transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.5)
            )

    # --- y_columns の準備（既存の _prepare_y_columns を想定） ---
    y_columns = _prepare_y_columns(df, y_columns)
    if x_column is None:
        x_column = y_columns[0]

    exclusions = [x_column]
    if isinstance(level, str):
        exclusions.append(level)
    y_columns = [col for col in y_columns if col not in exclusions]

    if not confirm_large_plot_count(len(y_columns), threshold=5):
        print("プロットを中止しました。")
        return

    level_name = (
        df.index.names[level]
        if isinstance(level, int) and level is not None
        else str(level) if level is not None else None
    )

    df_reset = df.reset_index()

    if label_column is None and level is not None:
        if level_name is not None and level_name in df_reset.columns:
            label_column = level_name
        else:
            try:
                if isinstance(level, int):
                    label_column = df_reset.columns[level]
                else:
                    label_column = df_reset.columns[0]
            except Exception:
                label_column = df_reset.columns[0]

    # --- 各 y_col ごとにパラメータを決定してプロット ---
    n_y = len(y_columns)

    # overlay の外で定義しておく（常に利用可能にする）
    palette = sns.color_palette(n_colors=max(6, n_y))
    markers = ['o', 's', 'D', '^', 'v', 'P', 'X', '*']

    # overlay モード用に fig/ax を先に作る（必要なら）
    if overlay:
        fig, ax = plt.subplots(figsize=(8, 6))

    for j, y_col in enumerate(y_columns):
        # 各 y_col に対する個別パラメータを解決
        lc = _get_per_y_param(label_column, y_col, j, n_y)
        lcond = _get_per_y_param(label_condition, y_col, j, n_y)
        lkwargs = _get_per_y_param(label_kwargs, y_col, j, n_y) or {}
        kind = _get_per_y_param(plot_kind, y_col, j, n_y) or 'scatter'
        lkws = _get_per_y_param(line_kwargs, y_col, j, n_y) or {}
        skws = _get_per_y_param(scatter_kwargs, y_col, j, n_y) or {}
        order_spec = _get_per_y_param(line_order, y_col, j, n_y)

        # デフォルト色/マーカーを割り当て（overlay 時に見分けやすくする）
        color = palette[j % len(palette)] if overlay else None
        if color is not None:
            if 'color' not in lkws:
                lkws = {**lkws, 'color': color}
            if 'color' not in skws:
                skws = {**skws, 'color': color}
        if 'marker' not in skws:
            skws = {**skws, 'marker': markers[j % len(markers)]}


        # --- overlay / 個別図の差し替えブロック ---
        if overlay:
            # 同一 ax に重ねる
            if level_name is not None:
                for i, (group, data) in enumerate(df_reset.groupby(level_name)):
                    labels_series = data[lc] if lc is not None and lc in data.columns else None
                    label = f"{group} - {y_col}"

                    x_s = data[x_column]
                    y_s = data[y_col]

                    # 線描画の順序指定がある場合はソートしてから渡す
                    if kind in ('line', 'both'):
                        x_plot, y_plot = _maybe_sort_xy_for_line(x_s, y_s, data, order_spec)
                    else:
                        x_plot, y_plot = x_s, y_s

                    # --- グループごとに line/scatter オプションを解決 ---
                    lkws_group = _get_per_group_param(line_kwargs, y_col, group, j, n_y, group_idx=i) or {}
                    skws_group = _get_per_group_param(scatter_kwargs, y_col, group, j, n_y, group_idx=i) or {}

                    # デフォルト色/マーカーの割当（y 列ベースの色を優先しつつ、グループ指定で上書き可能）
                    color = palette[j % len(palette)] if overlay else None
                    if color is not None:
                        if 'color' not in lkws_group:
                            lkws_group = {**lkws_group, 'color': color}
                        if 'color' not in skws_group:
                            skws_group = {**skws_group, 'color': color}
                    if 'marker' not in skws_group:
                        skws_group = {**skws_group, 'marker': markers[j % len(markers)]}

                    _plot_group(
                        ax,
                        x_plot,
                        y_plot,
                        labels=labels_series,
                        label=label,
                        show_regression=show_regression,
                        show_corr=show_corr,
                        corr_text_pos=(0.05, 0.95 - 0.04 * (i + j*0.2)),
                        scatter_kws=skws_group,
                        line_kws=lkws_group,
                        kind=kind,
                        label_condition_local=lcond,
                        label_kwargs_local=lkwargs,
                        **kwargs
                    )
            else:
                # level_name がない場合は全体に対して y 列ベースの設定を使う（従来どおり）
                labels_series = df_reset[lc] if lc is not None and lc in df_reset.columns else None
                x_s = df_reset[x_column]
                y_s = df_reset[y_col]
                if kind in ('line', 'both'):
                    x_plot, y_plot = _maybe_sort_xy_for_line(x_s, y_s, df_reset, order_spec)
                else:
                    x_plot, y_plot = x_s, y_s

                # y 列ベースの既存解決を利用
                lkws_y = lkws or {}
                skws_y = skws or {}
                # ただし、line_kwargs/scatter_kwargs が dict/list の場合は _get_per_y_param を使う
                lkws_y = _get_per_y_param(line_kwargs, y_col, j, n_y) or {}
                skws_y = _get_per_y_param(scatter_kwargs, y_col, j, n_y) or {}

                # 色/マーカー割当
                color = palette[j % len(palette)]
                if 'color' not in lkws_y:
                    lkws_y = {**lkws_y, 'color': color}
                if 'color' not in skws_y:
                    skws_y = {**skws_y, 'color': color}
                if 'marker' not in skws_y:
                    skws_y = {**skws_y, 'marker': markers[j % len(markers)]}

                _plot_group(
                    ax,
                    x_plot,
                    y_plot,
                    labels=labels_series,
                    label=y_col,
                    show_regression=show_regression,
                    show_corr=show_corr,
                    corr_text_pos=(0.05, 0.95 - 0.04 * j),
                    scatter_kws=skws_y,
                    line_kws=lkws_y,
                    kind=kind,
                    label_condition_local=lcond,
                    label_kwargs_local=lkwargs,
                    **kwargs
                )

            # overlay の最後のループで描画設定
            if j == n_y - 1:
                if legend:
                    ax.legend()
                ax.set_xlabel(x_column)
                ax.set_ylabel(", ".join(y_columns))
                ax.set_title(f"{x_column} vs {', '.join(y_columns)} (overlay)")
                ax.grid(True)
                plt.tight_layout()
                plt.show()

        else:
            # 個別図モード: y_col ごとに図を作る
            fig, ax = plt.subplots(figsize=(7, 5))
            if level_name is not None:
                for i, (group, data) in enumerate(df_reset.groupby(level_name)):
                    labels_series = data[lc] if lc is not None and lc in data.columns else None

                    x_s = data[x_column]
                    y_s = data[y_col]
                    if kind in ('line', 'both'):
                        x_plot, y_plot = _maybe_sort_xy_for_line(x_s, y_s, data, order_spec)
                    else:
                        x_plot, y_plot = x_s, y_s

                    # --- グループごとに line/scatter オプションを解決 ---
                    lkws_group = _get_per_group_param(line_kwargs, y_col, group, j, n_y, group_idx=i) or {}
                    skws_group = _get_per_group_param(scatter_kwargs, y_col, group, j, n_y, group_idx=i) or {}

                    # デフォルト色/マーカーの割当
                    color = None  # 個別図では色を自動割当しないか、必要なら palette を使う
                    if color is not None:
                        if 'color' not in lkws_group:
                            lkws_group = {**lkws_group, 'color': color}
                        if 'color' not in skws_group:
                            skws_group = {**skws_group, 'color': color}
                    if 'marker' not in skws_group:
                        skws_group = {**skws_group, 'marker': markers[j % len(markers)]}

                    _plot_group(
                        ax, x_plot, y_plot,
                        labels=labels_series,
                        label=str(group),
                        show_regression=show_regression,
                        show_corr=show_corr,
                        corr_text_pos=(0.05, 0.95 - 0.05 * i),
                        scatter_kws=skws_group,
                        line_kws=lkws_group,
                        kind=kind,
                        label_condition_local=lcond,
                        label_kwargs_local=lkwargs,
                        **kwargs
                    )
                if legend:
                    ax.legend(title=level_name)
            else:
                labels_series = df_reset[lc] if lc is not None and lc in df_reset.columns else None
                x_s = df_reset[x_column]
                y_s = df_reset[y_col]
                if kind in ('line', 'both'):
                    x_plot, y_plot = _maybe_sort_xy_for_line(x_s, y_s, df_reset, order_spec)
                else:
                    x_plot, y_plot = x_s, y_s

                # y 列ベースの既存解決を利用
                lkws_y = _get_per_y_param(line_kwargs, y_col, j, n_y) or {}
                skws_y = _get_per_y_param(scatter_kwargs, y_col, j, n_y) or {}

                # 色/マーカー割当
                color = None
                if color is not None:
                    if 'color' not in lkws_y:
                        lkws_y = {**lkws_y, 'color': color}
                    if 'color' not in skws_y:
                        skws_y = {**skws_y, 'color': color}
                if 'marker' not in skws_y:
                    skws_y = {**skws_y, 'marker': markers[j % len(markers)]}

                _plot_group(
                    ax, x_plot, y_plot,
                    labels=labels_series,
                    show_regression=show_regression,
                    show_corr=show_corr,
                    corr_text_pos=(0.05, 0.95),
                    scatter_kws=skws_y,
                    line_kws=lkws_y,
                    kind=kind,
                    label_condition_local=lcond,
                    label_kwargs_local=lkwargs,
                    **kwargs
                )

            ax.set_xlabel(x_column)
            ax.set_ylabel(y_col)
            ax.set_title(f"{x_column} vs {y_col}")
            ax.grid(True)
            plt.tight_layout()
            plt.show()


# 2026-02-19 16:52:37　以前のもの
# def plot_multiindex_scatter_by_column(
#     df,
#     level=0,
#     x_column=None,
#     y_columns=None,
#     show_regression=False,
#     show_corr=True,
#     label_column=None,
#     label_condition=None,
#     label_kwargs=None,
#     **kwargs
# ):
#     """
#     Seaborn によるマルチインデックス散布図（回帰ライン・相関係数・条件付きラベル付け対応）

#     概要
#     ----
#     マルチインデックスの DataFrame をグループごとに散布図化します。各グループごとに回帰線表示、相関係数表示、
#     および第3カラム（例: 国名）を条件に応じて点に注釈（ラベル）する機能を持ちます。

#     主な特徴
#     - **level** で指定したインデックスレベルでグループ化してプロットする。
#     - **show_regression=True** で回帰線（seaborn.regplot）を表示。
#     - **show_corr=True** で各グループの Pearson 相関係数とサンプル数を図内に表示。
#     - **label_column** を指定すると、その列の値を注釈に使う。
#       - **label_column=None** の場合、`reset_index()` によって追加されたインデックス列（level で指定したレベルに対応する列）を自動的にラベル対象にする。
#     - **label_condition** により、注釈する点を条件で絞れる（callable または集合や単一値を受け付ける）。
#     - **label_kwargs** で `ax.annotate` に渡す描画オプションを指定できる（フォントサイズ、bbox など）。
#     - **kwargs** 経由でプロット関数に追加オプションを渡せる。`legend` を kwargs に含めると凡例表示の有無を制御する（デフォルトは True）。

#     Parameters
#     ----------
#     df : pd.DataFrame
#         マルチインデックスを持つデータフレーム。
#     level : int or str or None, default 0
#         グループ化に使うインデックスのレベル（整数インデックスまたはレベル名）。None の場合はグループ化しない。
#     x_column : str or None
#         x 軸に使う列名。None の場合は y_columns の先頭を x に使う。
#     y_columns : list or None
#         y 軸に使う列名のリスト。None の場合は自動選択される。
#     show_regression : bool, default False
#         True のとき各グループに回帰線を描画する。
#     show_corr : bool, default True
#         True のとき各グループの Pearson 相関係数とサンプル数を図内に表示する。
#     label_column : str or None, default None
#         注釈に使う列名。None の場合は `reset_index()` によって追加されたインデックス列（level に対応）を自動で使う。
#     label_condition : callable or list/set/tuple or scalar or None, default None
#         注釈対象を絞る条件。
#         - **callable**: `(labels, x, y) -> boolean Series` を返す関数を渡す（各点ごとに True のものを注釈）。
#         - **list/set/tuple**: labels.isin(...) として扱う（指定した値のみ注釈）。
#         - **scalar**: labels == scalar として扱う。
#         - **None**: 全て注釈しない（注釈を行わない）。
#     label_kwargs : dict or None, default None
#         `ax.annotate` に渡す追加オプション（例: `fontsize`, `color`, `bbox`）。None の場合はデフォルト設定を使用。
#     **kwargs
#         seaborn のプロット関数や matplotlib に渡す追加オプション。`legend` を含めると凡例表示を制御できる（`legend=False` で凡例を表示しない）。

#     動作上の注意
#     --------------
#     - `label_column=None` のときは、`reset_index()` によって DataFrame に追加されたインデックス列を自動で選びます。
#       インデックス名が `None` の場合や多階層インデックスでは、`reset_index()` によって付与された列名（例: 'level_0'）や先頭の追加列を使います。
#     - `legend` は kwargs から取り出して関数内で制御します。`show_regression=True` の場合は seaborn.regplot が `legend` を扱わないため、
#       凡例を出したくないときは内部で `label='_nolegend_'` を使って凡例候補から除外します。
#     - 既存の列名と reset_index によって追加される列名が同じでも、**本関数は既存列を上書きしない**（label_column は既存列を参照するだけ）。
#     - 注釈の重なりが気になる場合は外部パッケージ `adjustText` を使うと改善できる（本関数はデフォルトでは使用しない）。

#     例
#     ---
#     # インデックスの第0レベルをラベルに使う（label_column を明示しない）
#     plot_multiindex_scatter_by_column(df, level=0, x_column='gdp', y_columns=['life_expectancy'])

#     # country 列をラベルに使い、y が 80 を超える点だけ注釈する
#     plot_multiindex_scatter_by_column(
#         df,
#         level=0,
#         x_column='gdp',
#         y_columns=['life_expectancy'],
#         label_column='country',
#         label_condition=lambda labels, x, y: y > 80,
#         label_kwargs=dict(fontsize=9, bbox=dict(facecolor='white', alpha=0.6))
#     )

#     """

#     # kwargs から legend を取り出す（デフォルト True）
#     legend = kwargs.pop("legend", True)
#     label_kwargs = label_kwargs or {}

#     def _plot_group(
#         ax, x, y, labels=None, label=None,
#         show_regression=False,
#         show_corr=True,
#         corr_text_pos=(0.05, 0.95),
#         **kwargs
#     ):
#         valid = x.notna() & y.notna()
#         x = x[valid]
#         y = y[valid]
#         if labels is not None:
#             labels = labels[valid]

#         if show_regression:
#             plot_label = label if legend else "_nolegend_"
#             sns.regplot(x=x, y=y, ax=ax, label=plot_label, **kwargs)
#         else:
#             sns.scatterplot(x=x, y=y, ax=ax, label=(label if legend else "_nolegend_"), legend=legend, **kwargs)

#         # ラベル付け（条件が与えられていれば）
#         if labels is not None and label_condition is not None:
#             # 'all' トークンを全件表示と扱う
#             if label_condition == 'all':
#                 mask = pd.Series(True, index=labels.index)
#             elif callable(label_condition):
#                 mask = label_condition(labels, x, y)
#             else:
#                 if isinstance(label_condition, (list, set, tuple)):
#                     # 空リストは「全件」扱いにしない（設計次第で変える）
#                     mask = labels.isin(label_condition)
#                 else:
#                     mask = labels == label_condition

#             # mask が True の点だけ注釈
#             for lx, ly, lab in zip(x[mask], y[mask], labels[mask]):
#                 ax.annotate(
#                     str(lab),
#                     xy=(lx, ly),
#                     xytext=(3, 3),
#                     textcoords='offset points',
#                     fontsize=8,
#                     **label_kwargs
#                 )

#         if show_corr and len(x) > 1 and len(y) > 1:
#             corr, _ = pearsonr(x, y)
#             n = len(x)
#             text = (
#                 f"{label + ': ' if label else ''}"
#                 f"r = {corr:.3f}, "
#                 f"n = {n}"
#             )
#             ax.text(
#                 corr_text_pos[0], corr_text_pos[1], text,
#                 transform=ax.transAxes,
#                 fontsize=10, verticalalignment='top',
#                 bbox=dict(boxstyle="round", facecolor="white", alpha=0.5)
#             )

#     # --- 以下は既存処理（略） ---
#     y_columns = _prepare_y_columns(df, y_columns)
#     if x_column is None:
#         x_column = y_columns[0]

#     exclusions = [x_column]
#     if isinstance(level, str):
#         exclusions.append(level)
#     y_columns = [col for col in y_columns if col not in exclusions]

#     if not confirm_large_plot_count(len(y_columns), threshold=5):
#         print("プロットを中止しました。")
#         return

#     # --- reset_index 前に level_name を決める ---
#     level_name = (
#         df.index.names[level]
#         if isinstance(level, int) and level is not None
#         else str(level) if level is not None else None
#     )

#     # reset_index 実行（既存の df をそのまま使う）
#     df_reset = df.reset_index()

#     # label_column が None のときは、reset_index によって追加された列を自動選択する
#     if label_column is None and level is not None:
#         # 可能な限り安全に該当列を見つけるロジック
#         if level_name is not None and level_name in df_reset.columns:
#             label_column = level_name
#         else:
#             # level が int の場合は、reset_index で挿入される列の位置は index レベルの順序に対応する
#             try:
#                 if isinstance(level, int):
#                     # index の先頭から数えて level 番目が挿入されているはず
#                     label_column = df_reset.columns[level]
#                 else:
#                     # level が文字列だが level_name が見つからない場合は、先頭のインデックス列を使う
#                     # （多階層インデックスで順序が分からないケースに備える）
#                     # reset_index によって追加されたインデックス列は DataFrame の先頭に来るため先頭列を選ぶ
#                     label_column = df_reset.columns[0]
#             except Exception:
#                 # 最終フォールバック：先頭列
#                 label_column = df_reset.columns[0]

#     for y_col in y_columns:
#         fig, ax = plt.subplots(figsize=(7, 5))

#         if level_name is not None:
#             for i, (group, data) in enumerate(df_reset.groupby(level_name)):
#                 labels_series = data[label_column] if label_column is not None else None
#                 _plot_group(
#                     ax, data[x_column], data[y_col],
#                     labels=labels_series,
#                     label=str(group),
#                     show_regression=show_regression,
#                     show_corr=show_corr,
#                     corr_text_pos=(0.05, 0.95 - 0.05 * i),
#                     **kwargs
#                 )
#             if legend:
#                 ax.legend(title=level_name)
#         else:
#             labels_series = df_reset[label_column] if label_column is not None else None
#             _plot_group(
#                 ax, df_reset[x_column], df_reset[y_col],
#                 labels=labels_series,
#                 show_regression=show_regression,
#                 show_corr=show_corr,
#                 corr_text_pos=(0.05, 0.95),
#                 **kwargs
#             )

#         ax.set_xlabel(x_column)
#         ax.set_ylabel(y_col)
#         ax.set_title(f"{x_column} vs {y_col}")
#         ax.grid(True)
#         plt.tight_layout()
#         plt.show()


def plot_non_na_counts_by_year(
    df, level=-1, y_columns=None, years_filter=None, **kwargs
):
    """
    各列（y_columns）について、指定したインデックスのレベル（通常は年）ごとに
    NaN でないデータの数をカウントし、折れ線グラフで表示する。
    さらに、years_filter に一致する年に対して、非NAデータを持つグループ名を辞書にまとめて返す。

    Parameters:
    df (pd.DataFrame): マルチインデックスを持つデータフレーム
    level (int or str): 年を含むインデックスのレベル（通常は年）
    y_columns (list or None): 対象とする列（None の場合は数値列を対象）
    years_filter (list, slice, or None): 年の選択条件（例: [1990, 2000] や slice(None, 1990)）
    **kwargs: plt.plot に渡す追加引数

    Returns:
    dict: {年: [非NAデータを持つグループ名]} の辞書
    """

    # if y_columns is None:
    #     y_columns = df.select_dtypes(include='number').columns.tolist()
    y_columns = _prepare_y_columns(df, y_columns)

    # 年ごとの非NAカウントを計算
    counts = df[y_columns].groupby(level=level).count()

    # グラフ描画
    ax = counts.plot(marker='o', **kwargs)
    ax.set_title("Non-NA Counts by Year")
    ax.set_xlabel(df.index.names[level])
    ax.set_ylabel("Non-NA Count")
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.grid(True)
    plt.show()

    # 年ごとの非NAグループを収集
    if years_filter is not None:
        return get_non_na_groups_by_year(df, level, y_columns, years_filter)
    else:
        return {}


# def remove_outliers(series, method='iqr', threshold=1.5):
#     """
#     指定した方法で外れ値を除外する。

#     Parameters:
#     series (pd.Series): 対象のデータ列
#     method (str): 外れ値除外手法（現在は 'iqr' のみ対応）
#     threshold (float): IQR に対する倍率（通常は 1.5）

#     Returns:
#     pd.Series: 外れ値を除外したデータ
#     """
#     s = series.dropna()
#     if method == 'iqr':
#         q1, q3 = s.quantile([0.25, 0.75])
#         iqr = q3 - q1
#         lower = q1 - threshold * iqr
#         upper = q3 + threshold * iqr
#         return s[(s >= lower) & (s <= upper)]
#     else:
#         raise ValueError(f"Unsupported method: {method}")


def _plot_stat_by_year(df, level, y_columns, stat_func, title_prefix, ylabel, plot_type='box', **kwargs):
    """
    共通の年次統計可視化ロジック。

    Parameters:
    df (pd.DataFrame): マルチインデックスを持つデータフレーム
    level (int or str): グループ化に用いるインデックスレベル
    y_columns (list): 対象の列名リスト
    stat_func (callable or None): 統計処理関数（例: pd.Series.std）または None（箱ひげ図の場合）
    title_prefix (str): グラフタイトルのプレフィックス
    ylabel (str): Y軸のラベル
    plot_type (str): 'box' または 'line'
    **kwargs: matplotlib に渡す追加引数
    """
    # def remove_outliers_and_stat(s):
    #     return stat_func(remove_outliers(s))

    # if y_columns is None:
    #     y_columns = df.select_dtypes(include='number').columns.tolist()
    y_columns = _prepare_y_columns(df, y_columns)
    grouped = df.groupby(level=level)

    for col in y_columns:
        fig, ax = plt.subplots(figsize=(6, 4))

        if plot_type == 'box':
            data_by_group = [group[col].dropna().values for _, group in grouped]
            # data_by_group = [
            #     remove_outliers(group[col]).values
            #     for _, group in grouped
            # ]

            labels = [str(name) for name, _ in grouped]
            ax.boxplot(data_by_group, labels=labels, **kwargs)
        elif plot_type == 'line':
            stats = grouped[col].apply(stat_func)
            # stats = grouped[col].apply(remove_outliers_and_stat)
            stats.plot(ax=ax, marker='o', linestyle='-', **kwargs)
        else:
            raise ValueError(f"Unsupported plot_type: {plot_type}")

        ax.set_title(f"{title_prefix} of {col}")
        ax.set_xlabel(df.index.names[level])
        ax.set_ylabel(ylabel)
        ax.grid(True)
        plt.tight_layout()
        plt.show()


def plot_box(df, level=-1, y_columns=None, **kwargs):
    """
    年ごとに各列の箱ひげ図を描画する。

    Parameters:
    df (pd.DataFrame): マルチインデックスを持つデータフレーム
    level (int or str): グループ化に用いるインデックスレベル
    y_columns (list or None): 対象の列名（None の場合は全数値列）
    **kwargs: matplotlib の boxplot に渡す追加引数
    """
    _plot_stat_by_year(df, level, y_columns, stat_func=None,
                       title_prefix="Yearly Box Plot", ylabel="Value", plot_type='box', **kwargs)


def plot_std(df, level=-1, y_columns=None, **kwargs):
    """
    年ごとに各列の標準偏差を折れ線グラフで描画する。

    Parameters:
    df (pd.DataFrame): マルチインデックスを持つデータフレーム
    level (int or str): グループ化に用いるインデックスレベル
    y_columns (list or None): 対象の列名（None の場合は全数値列）
    **kwargs: matplotlib の plot に渡す追加引数
    """
    _plot_stat_by_year(df, level, y_columns, stat_func=pd.Series.std,
                       title_prefix="Yearly Std Dev", ylabel="Standard Deviation", plot_type='line', **kwargs)


def plot_groupwise_missing_heatmap(df, group_level=0, label_level=-1, max_groups=10):
    """
    マルチインデックスDataFrameに対して、指定したインデックスレベルで
    グループごとの欠損値ヒートマップを表示する関数。
    縦軸にはマルチインデックスの指定レベルをラベルとして表示。

    Parameters:
    df (pd.DataFrame): 欠損値を可視化したいマルチインデックスDataFrame
    group_level (int or str): グループ化に使うインデックスレベル（例：0, 'company_id'）
    label_level (int or str or None): 縦軸ラベルに使うインデックスレベル（Noneなら最後のレベル）
    max_groups (int): 表示するグループの最大数

    Returns:
    None
    """
    # # ラベルに使うインデックスレベルを決定
    # if label_level is None:
    #     label_level = df.index.nlevels - 1  # 最後のレベル

    # 欠損のあるグループIDを取得
    group_ids = df.groupby(level=group_level).filter(lambda g: g.isnull().any().any()).index.get_level_values(group_level).unique()
    group_ids = group_ids[:max_groups]

    for gid in group_ids:
        group_df = df.xs(gid, level=group_level)

        # ラベル生成（指定インデックスレベルの値を文字列化）
        y_labels = group_df.index.get_level_values(label_level).astype(str)

        plt.figure(figsize=(12, 4))
        sns.heatmap(group_df.isnull(), cbar=False, cmap="mako", yticklabels=y_labels)
        plt.title(f"Missing Values Heatmap for Group: {gid}")
        plt.xlabel("Columns")
        plt.ylabel(f"Index Level: {label_level}")
        plt.tight_layout()
        plt.show()

