import pandas as pd
import numpy as np

import pyperclip
import sys
import subprocess


def _prepare_y_columns(df, y_columns):
    """
    y_columns の式を評価し、必要に応じて df に列を追加して返す。
    
    Parameters:
    df (pd.DataFrame): 元のデータフレーム
    y_columns (list or None): プロット対象の列または演算式

    Returns:
    list: 処理後の y_columns リスト
    """
    if y_columns is None:
        y_columns = df.select_dtypes(include='number').columns.tolist()

    # 利用可能な関数群（必要に応じて追加可能）
    safe_funcs = {
        'log': np.log,
        'log1p': np.log1p,
        'exp': np.exp,
        'expm1': np.expm1,
        'sqrt': np.sqrt,
        'abs': np.abs,
        'clip': np.clip,
        'floor': np.floor,
        'ceil': np.ceil,
        'round': np.round,
        'maximum': np.maximum,
        'minimum': np.minimum,
    }

    for col_expr in y_columns:
        if isinstance(col_expr, str) and col_expr not in df.columns:
            try:
                df[col_expr] = df.eval(col_expr, engine='python', local_dict=safe_funcs)
                print(f"式 `{col_expr}` を評価して追加しました")
            except Exception as e:
                print(f"式 `{col_expr}` の評価に失敗しました: {e}")

    return y_columns


def copy_to_clipboard(text):
    try:
        import pyperclip
        pyperclip.copy(text)
    except Exception:
        if sys.platform.startswith("win"):
            subprocess.run("clip", input=text.strip().encode('utf-16'), shell=True)
        elif sys.platform.startswith("darwin"):
            subprocess.run("pbcopy", input=text.encode(), shell=True)
        elif sys.platform.startswith("linux"):
            subprocess.run("xclip -selection clipboard", input=text.encode(), shell=True)
        else:
            print("⚠️ クリップボードコピーはこの環境ではサポートされていません。")
            

def _parse_period_strings(period_strings, available_times):
    """
    "1870:1913" のような文字列リストを、available_times に基づいて分割する。
    無効な期間（範囲外・空）は警告を出してスキップする。

    Parameters:
        period_strings : list of str
            例: ['1870:1913', '1950:1970']
        available_times : list of int or str
            prepare_panel_data から得られる times

    Returns:
        list of list : 各期間に対応する時系列値のリスト（有効なもののみ）
    """
    parsed_periods = []
    for s in period_strings:
        try:
            start, end = s.split(":")
            start, end = type(available_times[0])(start), type(available_times[0])(end)
        except Exception as e:
            print(f"⚠️ 無効な期間指定 '{s}'（形式エラー）: {e}")
            continue

        # 範囲チェック
        if start > end:
            print(f"⚠️ 無効な期間指定 '{s}'（開始 > 終了）")
            continue

        period = [t for t in available_times if start <= t <= end]
        if not period:
            print(f"⚠️ 有効なデータが存在しない期間 '{s}'（available_times に該当なし）")
            continue

        parsed_periods.append(period)
    return parsed_periods

