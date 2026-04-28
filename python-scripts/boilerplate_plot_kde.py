import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KernelDensity
from typing import Any

# -------------------------
# 基本ユーティリティ
# -------------------------
def _compute_bandwidth_1d(x, method):
    x = np.asarray(x)
    x = x[~np.isnan(x)]
    n = x.size
    if n <= 1:
        return 1.0
    std = x.std(ddof=1)
    if std == 0 or np.isnan(std):
        return 1e-3

    if isinstance(method, (int, float)):
        bw = float(method)
    elif method is None:
        bw = 1.0
    elif callable(method):
        try:
            bw = float(method(x))
        except Exception:
            bw = 1.0
    else:
        m = str(method).lower()
        if m == "scott":
            bw = std * (n ** (-1.0 / 5.0))
        elif m == "silverman":
            iqr = np.subtract(*np.percentile(x, [75, 25]))
            sigma = min(std, iqr / 1.34 if iqr > 0 else std)
            bw = 0.9 * sigma * (n ** (-1.0 / 5.0))
            if bw <= 0:
                bw = std * (n ** (-1.0 / 5.0))
        else:
            try:
                bw = float(method)
            except Exception:
                bw = 1.0
    if bw <= 0 or np.isnan(bw):
        bw = 1e-3
    return float(bw)


def _kde_on_grid(values, bandwidth, kernel, grid):
    vals = np.asarray(values).reshape(-1, 1)
    kde = KernelDensity(bandwidth=bandwidth, kernel=kernel)
    kde.fit(vals)
    log_d = kde.score_samples(grid.reshape(-1, 1))
    return np.exp(log_d)


def _ensure_multiindex_like(df, level):
    if isinstance(df.index, pd.MultiIndex):
        if level is None:
            # groupなし扱い
            return df, None, None
        if isinstance(level, str):
            if level not in df.index.names:
                raise KeyError(f"指定したレベル名 '{level}' が MultiIndex に存在しません。")
            level_num = df.index.names.index(level)
            level_name = level
        elif isinstance(level, int):
            nlevels = df.index.nlevels
            if level < 0:
                level_num = nlevels + level
            else:
                level_num = level
            if not (0 <= level_num < nlevels):
                raise IndexError(f"指定したレベル番号 {level} が範囲外です。")
            level_name = df.index.names[level_num]
        else:
            raise TypeError("level は int または str か None で指定してください。")
        return df, level_num, level_name

    # 非 MultiIndex の場合
    df2 = df.reset_index()
    if level is None:
        return df2, None, None
    if isinstance(level, int):
        ncols = df2.shape[1]
        if level < 0:
            idx = ncols + level
        else:
            idx = level
        if not (0 <= idx < ncols):
            raise IndexError(f"指定したレベル番号 {level} が列範囲外です。")
        level_col = df2.columns[idx]
    elif isinstance(level, str):
        if level in df2.columns:
            level_col = level
        else:
            if df.index.name == level and df.index.name is not None:
                level_col = df.index.name
            else:
                raise KeyError(f"指定したレベル名 '{level}' が列にも index 名にも存在しません。")
    else:
        raise TypeError("level は int または str か None で指定してください。")
    df_prepared = df2.set_index(level_col)
    return df_prepared, 0, level_col


# -------------------------
# 汎用パラメータ解決ロジック（group-first）
# - param: scalar | list/tuple | dict | callable
# - label: グループ名（str）または y 列名など
# - idx: y 列インデックス（0..）
# - group_idx: グループループの順序（0..）
# -------------------------
def _resolve_param_group_first(param, label, idx=0, group_idx=None, group_keys=None):
    if param is None:
        return None
    # callable: label/idx/group_idx を渡す場合は呼ぶ（優先度高め）
    if callable(param):
        try:
            return param(label, idx, group_idx)
        except TypeError:
            try:
                return param(label)
            except Exception:
                try:
                    return param()
                except Exception:
                    return None
    # dict: group-keyed or label-keyed
    if isinstance(param, dict):
        # 直接キー一致（group-first を想定して label を優先）
        if label in param:
            return param[label]
        # 次に文字列化した label を探す
        key_str = str(label)
        if key_str in param:
            return param[key_str]
        # y 列名ベースの指定があるか（label が group でなく y の場合も対応）
        # 探索: param のキーが label と一致するものがなければ、順序割当（キーを順序リストとして扱う）
        # 順序割当: group_idx が与えられればそれを使う
        keys = list(param.keys())
        if group_idx is not None and len(keys) > 0:
            # key が明示的なラベルでない場合（順序割当を期待）
            # 例: {'x1': {...}, 'x2': {...}} を group_idx に従って割当
            k = keys[group_idx % len(keys)]
            return param[k]
        # 最後にラベルが y 列名として存在するかを試す
        if label in param:
            return param[label]
        return None
    # list/tuple: index による割当（優先は group_idx があればそれ）
    if isinstance(param, (list, tuple)):
        if group_idx is not None:
            i = group_idx
        else:
            i = idx
        if len(param) == 0:
            return None
        if i < len(param):
            return param[i]
        return param[-1]
    # scalar
    return param


def _prepare_arrays_for_nogroup(df):
    if isinstance(df, pd.Series):
        arr = df.dropna().to_numpy()
        if arr.size == 0:
            raise ValueError("描画可能なデータがありません。")
        return {"series": arr}
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) == 0:
        raise ValueError("数値列が見つかりません。描画できません。")
    arrays = {}
    for col in numeric_cols:
        arr = df[col].dropna().to_numpy()
        if arr.size == 0:
            warnings.warn(f"列 '{col}' のデータが空です。スキップします。")
            continue
        arrays[col] = arr
    if len(arrays) == 0:
        raise ValueError("描画可能なデータがありません。")
    return arrays


def _compute_common_grid(arrays, grid_points):
    all_vals = np.concatenate(list(arrays.values()))
    vmin, vmax = np.nanmin(all_vals), np.nanmax(all_vals)
    if vmin == vmax:
        vmin = vmin - 0.5
        vmax = vmax + 0.5
    padding = (vmax - vmin) * 0.05
    return np.linspace(vmin - padding, vmax + padding, grid_points)


def _normalize_palette(palette, group_keys):
    # palette: None | sequence | mapping
    if palette is None:
        return sns.color_palette(n_colors=max(6, len(group_keys)))
    if isinstance(palette, dict):
        # produce list aligned with group_keys
        return [palette.get(k, palette.get(str(k), sns.color_palette(n_colors=1)[0])) for k in group_keys]
    if isinstance(palette, (list, tuple)):
        if len(palette) >= len(group_keys):
            return list(palette[: len(group_keys)])
        # 足りない場合は繰り返す
        out = []
        for i, _ in enumerate(group_keys):
            out.append(palette[i % len(palette)])
        return out
    # 単一色
    return [palette for _ in group_keys]


# -------------------------
# メイン関数
# -------------------------
def plot_multiindex_kde(
    df,
    column=None,
    level=None,
    values=None,
    bw_method="scott",            # scalar | dict(label->bw) | callable(arr)->bw
    kernel="gaussian",
    grid_points=512,
    ax=None,
    fill=False,
    legend=True,
    show=True,
    block=True,
    close_after_show=True,
    palette=None,                 # None | sequence | mapping(label->color)
    linestyles=None,              # scalar | list | dict
    linewidth=None,               # scalar | list | dict
    alpha=0.6,                    # scalar or dict
):
    """
    概要
    ----
    MultiIndex（または通常の）DataFrame から 1 次元データのカーネル密度推定（KDE）を描画するユーティリティ。
    - 指定したインデックスレベルでグループ化して各グループの KDE を同一グリッド上に重ね描画する。
    - グループ化が不要な場合は `column=None` かつ `level=None` として DataFrame の数値列すべて（または Series）を描画できる。
    - 色・線種・線幅・透明度・bandwidth は柔軟に指定可能（scalar / list / dict / callable をサポート、group-first 解決ルール）。

    引数
    ----
    df
        pandas.DataFrame または pandas.Series。MultiIndex を想定するが、通常の DataFrame も受け付ける。
    column
        描画対象の列名（str）。`None` の場合は数値列すべて（grouping も None のとき）。
    level
        グループ化に使うインデックスレベル（int / str / None）。`None` のときはグループ化しない。
    values
        指定したレベルの値のサブセット（list）。None なら全グループ。
    bw_method
        帯域幅指定。'scott' / 'silverman' / float / None のほか、以下を受け付ける：
        - dict: {group_label: spec, ...}（グループごとに指定）
        - list/tuple: グループ順に割当（短ければ末尾を繰り返す）
        - callable: arr -> float（グループ配列を受け取り bandwidth を返す）
        詳細は下の「bw_method による描画の特徴」を参照。
    kernel
        sklearn.neighbors.KernelDensity に渡すカーネル名（例: 'gaussian'）。
    grid_points
        評価点数（x 軸分解能）。
    ax
        matplotlib Axes を渡すとその上に描画。None なら新規作成。
    fill
        True なら曲線下を塗りつぶす（alpha による透過あり）。
    legend
        凡例表示の有無（True/False）。
    show
        True のとき関数内で plt.show(block=block) を呼ぶ。
    block
        plt.show の block 引数（環境によって無視されることがある）。
    close_after_show
        show=True のとき表示後に図を閉じるか（True/False）。外部で fig を使う場合は False 推奨。
    palette
        色指定。None / sequence / mapping を受け付ける。
        - None: seaborn のパレットを自動割当
        - sequence: グループ順に割当（足りなければ繰り返す）
        - mapping: {label: color, ...}（ラベルで色を指定）
    linestyles
        線種指定。scalar / list / dict / callable を受け付ける（group-first 解決）。
    linewidth
        線幅指定。scalar / list / dict / callable を受け付ける（group-first 解決）。
    alpha
        透明度。scalar / dict / callable を受け付ける（group-first 解決）。塗りつぶし時は自動で調整される。

    戻り値
    ----
    matplotlib.axes.Axes
        描画に使った Axes を返す（show の有無に関わらず返す）。

    挙動の要点
    ----
    - `level` が指定されている場合はそのレベルで groupby して各グループを描画する。
    - `level` が None の場合はグループ化せず、`column` が指定されていればその列全体を描画、`column` も None の場合は DataFrame の数値列すべてを描画する。
    - `bw_method` の優先順位は: group-keyed dict > callable > list/tuple（順次割当）> scalar。
    - `palette` / `linestyles` / `linewidth` / `alpha` は group-first の解決ルールを採用（dict で group 指定、list で順次割当、scalar で全体適用）。
    - 欠損値は無視し、サンプル数が少ないグループには警告を出す。

    bw_method による描画の特徴
    ----
    - **概念**  
    帯域幅（bandwidth）は KDE の平滑化量を決める主要パラメータ。小さいほど局所的なピークを強調し、多峰性やノイズを拾いやすい。大きいほど滑らかになり細部が潰れて全体傾向が強調される。

    - **主要指定と直感的な効果**
    - `"scott"`  
        データの標準偏差とサンプル数に基づく自動推定。多くの実務で妥当な中庸の平滑化を与える。まず試すべき設定。
    - `"silverman"`  
        Scott よりやや滑らかになる傾向。ノイズ抑制を優先したいときに有用。
    - `float`（例: `0.1`, `0.5`）  
        明示的な固定帯域幅。小さい値は鋭いピーク、大きい値は滑らかな曲線を生成する。データの単位に依存するため注意。
    - `None`  
        本実装では固定の `bandwidth=1.0` として扱う（データのスケールに依存する）。データが標準化されている場合は妥当だが、未標準化データでは過度な平滑化または過剰適合を招く可能性がある。
    - `dict`（グループ別指定）  
        グループごとに異なる平滑化を適用できる。サンプル数や分散が大きく異なる群を見やすく比較する際に有効だが、比較の公平性に注意。
    - `list/tuple`（順次割当）  
        グループ順に帯域幅を割り当てる。少数グループで手早く試すときに便利。
    - `callable`（例: `lambda arr: max(0.05, arr.std() * len(arr)**(-1/5))`）  
        各グループのデータを見て動的に bandwidth を決定する。自動化ルールを組み込みたい場合に有効。

    - **視覚的な変化の例**
    - 非常に小さい帯域幅（例 `0.01`〜`0.1`）: 多くの鋭いピークが現れ、ノイズや離散サンプルの影響が強く出る（過剰適合の危険）。
    - 中程度（`"scott"` や `0.2`〜`0.5`）: 局所構造を残しつつノイズを抑えるバランスの良い描画。
    - 大きい帯域幅（例 `1.0` 以上）: 平滑化が強くなり複数の山が統合される。全体傾向は掴めるが細部は失われる。

    - **実務的な注意点**
    - データのスケールが異なるグループを同一の固定 `bandwidth`（特に `None`→1.0）で比較すると誤解を招く。標準化するか、グループ別に `bw_method` を指定することを検討する。
    - 比較目的ではまず共通の `bw_method`（例 `"scott"`）で描画し、必要に応じて固定値や callable で微調整する。
    - 手動で異なる bandwidth を使う場合は、図の注記や解析報告で使用した `bw_method` を明記する。

    使用例
    ----
    # 1) 基本（country レベルで value 列を描画、内部で表示して閉じる）
    plot_multiindex_kde(df, column="value", level="country")

    # 2) palette を辞書で指定、線種と線幅をグループごとに辞書で指定
    plot_multiindex_kde(
        df,
        column="value",
        level="country",
        palette={"Japan":"#1f77b4","USA":"#ff7f0e","Germany":"#2ca02c"},
        linestyles={"Japan":"-","USA":"--","Germany":":"},
        linewidth={"Japan":2.0,"USA":1.0,"Germany":1.5},
        show=True
    )

    # 3) bw_method を callable（グループ配列を受け取り bandwidth を返す）
    def bw_by_std(arr):
        n = len(arr)
        return max(0.05, arr.std(ddof=1) * (n ** (-1/5)))
    plot_multiindex_kde(df, column="value", level="country", bw_method=bw_by_std)

    # 4) bw_method の比較（複数を試す例）
    for bw in ["scott", "silverman", 0.1, 0.3, 1.0]:
        plot_multiindex_kde(df, column="value", level="country", bw_method=bw, show=False)
    plt.legend(title="bw_method")
    plt.show()

    注意
    ----
    - デフォルトで show=True にするとスクリプト実行環境では表示がブロッキングされる場合がある。バッチ処理や後でまとめて表示したい場合は show=False を推奨。
    - 異なる bandwidth をグループごとに使うと見た目の差が生じるため、比較目的では同一の bw_method を使うことを検討する。
    """
    if column is None and level is None:
        arrays = _prepare_arrays_for_nogroup(df)
        group_keys = list(arrays.keys())
        level_name = None
    elif column is not None and level is None:
        if column not in df.columns:
            raise KeyError(f"指定した列 '{column}' が DataFrame に存在しません。")
        arr = df[column].dropna().to_numpy()
        if arr.size == 0:
            raise ValueError("描画可能なデータがありません。")
        arrays = {column: arr}
        group_keys = [column]
        level_name = None
    else:
        if column is None:
            raise ValueError("level を指定する場合は column も指定してください。")
        if column not in df.columns:
            raise KeyError(f"指定した列 '{column}' が DataFrame に存在しません。")
        df_prepared, level_num, level_name = _ensure_multiindex_like(df, level)
        grouped = df_prepared.groupby(level=level_num) if level_num is not None else [(None, df_prepared)]
        group_keys = [k for k, _ in grouped] if level_num is not None else [None]
        # filter values if provided
        if values is not None:
            values_set = set(values)
            group_keys = [k for k in group_keys if k in values_set]
            if len(group_keys) == 0:
                raise ValueError("指定した values に該当するグループがありません。")
        arrays = {}
        # build arrays in same order as group_keys
        if level_name is not None:
            for i, key in enumerate(group_keys):
                grp = grouped.get_group(key)
                arr = grp[column].to_numpy()
                arr = arr[~pd.isna(arr)]
                if arr.size == 0:
                    warnings.warn(f"グループ '{key}' のデータが空です。スキップします。")
                    continue
                arrays[key] = arr
        else:
            # level が None after ensure -> treat whole df_prepared as single group
            arr = df_prepared[column].dropna().to_numpy()
            arrays[column] = arr
            group_keys = [column]

    if len(arrays) == 0:
        raise ValueError("描画可能なデータがありません。")

    # 共通グリッド
    grid = _compute_common_grid(arrays, grid_points)

    # Axes 準備
    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
        created_fig = True
    else:
        fig = ax.figure

    # パレット解決（group_keys の順に色を返す）
    colors = _normalize_palette(palette, list(arrays.keys()))

    # 描画ループ
    for i, (key, arr) in enumerate(arrays.items()):
        # bw_method の解決（group-keyed dict / callable / scalar / list）
        if isinstance(bw_method, dict):
            bw_spec = bw_method.get(key, bw_method.get(str(key), None))
            bw = _compute_bandwidth_1d(arr, bw_spec)
        elif isinstance(bw_method, (list, tuple)):
            bw_spec = bw_method[i % len(bw_method)]
            bw = _compute_bandwidth_1d(arr, bw_spec)
        elif callable(bw_method):
            try:
                bw = _compute_bandwidth_1d(arr, bw_method(arr))
            except Exception:
                bw = _compute_bandwidth_1d(arr, bw_method)
        else:
            bw = _compute_bandwidth_1d(arr, bw_method)

        try:
            density = _kde_on_grid(arr, bandwidth=bw, kernel=kernel, grid=grid)
        except Exception as e:
            warnings.warn(f"グループ '{key}' の KDE 計算でエラーが発生しました: {e}. スキップします。")
            continue

        # 色・線種・太さ・alpha を group-first ルールで解決
        color = _resolve_param_group_first(palette, key, idx=i, group_idx=i, group_keys=list(arrays.keys()))
        if color is None:
            color = colors[i % len(colors)]
        ls = _resolve_param_group_first(linestyles, key, idx=i, group_idx=i, group_keys=list(arrays.keys()))
        lw = _resolve_param_group_first(linewidth, key, idx=i, group_idx=i, group_keys=list(arrays.keys()))
        a = _resolve_param_group_first(alpha, key, idx=i, group_idx=i, group_keys=list(arrays.keys()))
        if a is None:
            a = 0.6

        plot_kwargs = {}
        if color is not None:
            plot_kwargs["color"] = color
        if ls is not None:
            # matplotlib accepts 'ls' or 'linestyle'
            plot_kwargs["linestyle"] = ls
        if lw is not None:
            plot_kwargs["linewidth"] = lw

        label = str(key) if level_name is not None else (column if column is not None else str(key))

        ax.plot(grid, density, label=(label if legend else "_nolegend_"), alpha=a, **plot_kwargs)
        if fill:
            ax.fill_between(grid, density, alpha=a * 0.5, color=plot_kwargs.get("color", None))

    # 軸ラベル・タイトル・凡例
    if level_name is None:
        xlabel = column if column is not None else "value"
        ax.set_title(f"KDE of '{xlabel}' (no grouping)")
        ax.set_xlabel(xlabel)
    else:
        ax.set_title(f"KDE of '{column}' grouped by '{level_name}'")
        ax.set_xlabel(column)
    ax.set_ylabel("Density")

    if legend:
        ax.legend(title=str(level_name) if level_name is not None else None, bbox_to_anchor=(1.02, 1), loc="upper left")
        plt.tight_layout()

    # show 制御
    if show:
        plt.show(block=block)
        if close_after_show and created_fig:
            plt.close(fig)

    return ax


# -------------------------
# 簡単な使用例（コメント）
# -------------------------
# 1) グループごとに色と線種を辞書で指定
# ax = plot_multiindex_kde(df, column="value", level="country",
#                          palette={"Japan":"#1f77b4","USA":"#ff7f0e"},
#                          linestyles={"Japan":"-","USA":"--"},
#                          linewidth={"Japan":2.0,"USA":1.0},
#                          alpha={"Japan":0.7,"USA":0.5},
#                          bw_method={"Japan":0.2,"USA":"scott"},
#                          show=True)
#
# 2) palette をリストで渡し、linestyles をリストで順次割当
# ax = plot_multiindex_kde(df, column="value", level="country",
#                          palette=["#1f77b4","#ff7f0e","#2ca02c"],
#                          linestyles=["-","--",":"],
#                          show=True)


# 使い方例（簡潔）
if __name__ == "__main__":
    np.random.seed(0)
    countries = ["Japan", "USA", "Germany"]
    years = [2018, 2019, 2020]
    rows = []
    for c in countries:
        for y in years:
            mu = {"Japan": 0, "USA": 2, "Germany": -1}[c] + 0.1 * (y - 2018)
            samples = np.random.normal(loc=mu, scale=1.0, size=100)
            for s in samples:
                rows.append((c, y, s))
    df_example = pd.DataFrame(rows, columns=["country", "year", "value"])
    df_example = df_example.set_index(["country", "year"])

    # デフォルト: show=True（関数内で表示して閉じる）
    ax = plot_multiindex_kde(df_example, column="value", level="country")

    # show を抑制して後で表示したい場合
    ax2 = plot_multiindex_kde(df_example, column="value", level="country", show=False)
    plt.show()
