# import seaborn as sns
# import matplotlib.pyplot as plt
from tabulate import tabulate
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.tsa.api as tsa
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import grangercausalitytests
from linearmodels.panel import PanelOLS, RandomEffects, PooledOLS

from io import StringIO
import warnings
from collections import defaultdict

from itertools import product


def estimate_simple_ols(df, y_col=None, x_cols=None, add_constant=True, output_format='table', limit=100):
    """
    OLS（最小二乗法）推計を実行し、結果を pd.DataFrame 形式で一貫して返します。
    引数の組み合わせにより、単一の重回帰分析または変数間の総当たり単回帰を自動で切り替えます。

    Args:
        df (pd.DataFrame): 解析対象のデータフレーム。
        y_col (str, list, or None): 
            被説明変数（従属変数）。
            - str: 単一の変数を指定。
            - list: 指定された各変数について個別にループ推計。
            - None: df内の全数値列を対象（デフォルト）。
        x_cols (str, list, or None): 
            説明変数（独立変数）。
            - str: 単一の変数を指定。
            - list: **重回帰分析**として、これら全ての変数をモデルに同時に投入。
            - None: df内の全数値列を候補とし、単回帰の**総当たり**を実行（デフォルト）。
        add_constant (bool): 定数項を含めるかどうか。デフォルトは True。
        output_format (str): 
            - 'table': 統計量をまとめた pd.DataFrame を返します（デフォルト）。
            - 'results': statsmodels の結果オブジェクトを返します（単一推計時のみ）。
            - 'summary': statsmodels の標準サマリー（テキスト）を返します。
        limit (int): 総当たり時の最大モデル数。これを超えると続行の確認プロンプトが表示されます。

    Returns:
        pd.DataFrame or statsmodels object: 
            'table'指定時は、Dependent, Independent, Coef, P-value, Sig, Adj_R2, Obs の列を持つ表。
            定数項が含まれる場合は 'Independent' 列に 'const' として表示されます。

    Examples:
        >>> # 1. 【重回帰】 yに対してx1とx2を同時に投入
        >>> # 結果は const, x1, x2 の3行を含む DataFrame になります
        >>> df_mr = estimate_simple_ols(df, y_col='GDP', x_cols=['Capital', 'Labor'])
        >>> print(df_mr)

        >>> # 2. 【総当たり】 全数値変数の組み合わせで単回帰を網羅実行
        >>> # 決定係数(Adj_R2)が高い順に並び替えて、相関の強いペアを特定
        >>> results = estimate_simple_ols(df, limit=50)
        >>> top_5 = results.sort_values('Adj_R2', ascending=False).head(10) # 1モデル2行(const含)のため10行取得

        >>> # 3. 【特定yへの総当たり】 GDPに対して、他の全変数を個別にぶつける
        >>> res_gdp = estimate_simple_ols(df, y_col='GDP', x_cols=None)
        
        >>> # 4. 【応用】 有意な変数(Sig)だけを抽出してCSV保存
        >>> results[results['Sig'].str.contains('\*')].to_csv("significant_results.csv")

    Notes:
        - 内部で `df.select_dtypes(include=['number'])` を使用し、非数値列を自動で除外します。
        - 被説明変数と説明変数が重複する組み合わせは自動的にスキップされます。
        - 欠損値(NaN)はモデルごとに `dropna()` 処理されます。
    """
    # 1. 候補となる数値列のリストアップ
    num_cols = df.select_dtypes(include=['number']).columns.tolist()
    
    # 2. ターゲットの整理
    y_targets = [y_col] if isinstance(y_col, str) else (y_col if y_col is not None else num_cols)
    
    # 3. 総当たりモードの判定
    # y_colが複数指定されている、または x_colsがNone（全探索）の場合は「総当たり」とみなす
    # y_colが1つで x_colsがリストの場合は、単一の「重回帰」とみなす
    is_brute_force = (y_col is None) or (isinstance(y_col, list) and len(y_targets) > 1) or (x_cols is None)

    def get_stars(p):
        if p < 0.01: return '***'
        if p < 0.05: return '**'
        if p < 0.1:  return '*'
        return ''

    # --- A. 総当たりモード ---
    if is_brute_force:
        combinations = []
        if x_cols is None:
            # xが未指定なら、yごとに各xと「単回帰」の組み合わせを作る
            for y in y_targets:
                for x in num_cols:
                    if y == x: continue
                    combinations.append((y, x))
        else:
            # xが指定済み（リスト含む）なら、yごとにその「xセット」で回帰
            for y in y_targets:
                # yが説明変数セットに含まれていないか確認
                if (isinstance(x_cols, list) and y in x_cols) or (y == x_cols):
                    continue
                combinations.append((y, x_cols))

        if len(combinations) > limit:
            print(f"Warning: {len(combinations)} models. (Limit: {limit})")
            if input("Proceed? (y/n): ").strip().lower() != 'y': return pd.DataFrame()

        all_results = []
        for y, x in combinations:
            try:
                # 単一推計として再帰呼び出し（内部では is_brute_force=False になる）
                res_df = estimate_simple_ols(df, y, x, add_constant, output_format='table', limit=float('inf'))
                all_results.append(res_df)
            except: continue
        
        return pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()

    # --- B. 単一モデルモード (重回帰含む) ---
    actual_x = [x_cols] if isinstance(x_cols, str) else x_cols
    y_name = y_targets[0]
    
    # データの準備
    data = df[[y_name] + actual_x].dropna()
    if data.empty: raise ValueError(f"No valid data for {y_name} ~ {actual_x}")

    X = sm.add_constant(data[actual_x]) if add_constant else data[actual_x]
    results = sm.OLS(data[y_name], X).fit()

    if output_format == 'results': return results
    if output_format == 'summary': return results.summary()
    
    # DataFrame形式での結果作成
    rows = []
    for v in results.params.index:
        rows.append({
            'Dependent': y_name,
            'Independent': v,
            'Coef': round(results.params[v], 4),
            'P-value': round(results.pvalues[v], 4),
            'Sig': get_stars(results.pvalues[v]),
            'Adj_R2': round(results.rsquared_adj, 4),
            'Obs': int(results.nobs)
        })
    return pd.DataFrame(rows)
