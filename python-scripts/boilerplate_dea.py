import numpy as np
import pandas as pd
import cvxpy as cp

import itertools


def _prepare_df_for_year(df, year, inputs, outputs, label_level=None):
    """
    年ごとの共通前処理を行い、DEA に使うデータ行列とラベルを返す。

    処理内容
    - DataFrame のインデックスが MultiIndex かつ階層数 >= 2 の場合は末尾レベルを年とみなし抽出する。
    - そうでない場合は 'year' 列を参照して抽出する。
    - 指定した inputs と outputs 列のみを残し、欠損値を含む行を削除する。
    - 入力行列 X と出力行列 Y を numpy 配列（shape=(n_vars, n_dmus)）で返す。

    Parameters
    - **df** : pandas.DataFrame
        元データ。MultiIndex の末尾が年であることを想定するか、'year' 列を持つこと。
    - **year** : scalar
        抽出する年（MultiIndex の末尾レベル値または 'year' 列の値）。
    - **inputs** : sequence of str
        DEA の入力変数名リスト。
    - **outputs** : sequence of str
        DEA の出力変数名リスト。
    - **label_level** : int or None, optional
        ラベルに使うインデックスレベル。`None` の場合は `df_xy.index` 全体（タプルを含む）をラベルとする。

    Returns
    - **df_xy** : pandas.DataFrame
        指定年・指定変数で欠損を除去したデータ（元の Index / MultiIndex を保持）。
    - **dmu_labels** : list
        各 DMU のラベル（`label_level` 指定時はそのレベルの値、None 時は Index の要素）。
    - **X** : numpy.ndarray, shape=(len(inputs), n_dmus)
        入力行列（float）。
    - **Y** : numpy.ndarray, shape=(len(outputs), n_dmus)
        出力行列（float）。

    Raises
    - **ValueError**
        - 指定年に完全データを持つ DMU が存在しない場合。
        - MultiIndex でなく 'year' 列も存在しない場合。
    """
    df_sel = df.copy()
    if isinstance(df_sel.index, pd.MultiIndex) and df_sel.index.nlevels >= 2:
        mask = df_sel.index.get_level_values(-1) == year
        df_year = df_sel[mask]
    else:
        if 'year' in df_sel.columns:
            df_year = df_sel[df_sel['year'] == year]
        else:
            raise ValueError("DataFrame must have MultiIndex (country,...,year) or a 'year' column.")

    cols = list(inputs) + list(outputs)
    df_xy = df_year[cols].dropna(how='any').copy()
    if df_xy.shape[0] == 0:
        raise ValueError("No DMUs with complete data for the selected year and variables.")

    if label_level is None:
        dmu_labels = list(df_xy.index)  # タプルを含む可能性あり
    else:
        dmu_labels = list(df_xy.index.get_level_values(label_level))

    X = df_xy[inputs].to_numpy().T.astype(float)
    Y = df_xy[outputs].to_numpy().T.astype(float)
    return df_xy, dmu_labels, X, Y


def dea_from_dataframe(df, year, inputs, outputs, rts='VRS', orientation='output',
                       solver=None, label_level=None):
    """
    指定年の各 DMU について DEA を解き、効率値と各 DMU に対応する lambda ベクトルを返す。

    概要
    - 内部で `_prepare_df_for_year` を呼び、X, Y 行列を作成する。
    - 各 DMU ごとに線形計画問題を解き、効率値（input 方向なら theta、output 方向なら phi）を算出する。
    - 各 DMU に対する全 lambda ベクトル（他 DMU の重み）を辞書で返す。
    - 戻り値の `scores.index` は `_prepare_df_for_year` が返す `df_xy.index` に揃える。

    Parameters
    - **df** : pandas.DataFrame
        元データ（MultiIndex または 'year' 列を想定）。
    - **year** : scalar
        実行対象の年。
    - **inputs** : sequence of str
        入力変数名リスト。
    - **outputs** : sequence of str
        出力変数名リスト。
    - **rts** : {'CRS', 'VRS', 'DRS'}, default 'VRS'
        規模収益の仮定。'VRS' は可変規模、'CRS' は一定規模、'DRS' は収益逓減。
    - **orientation** : {'input', 'output'}, default 'output'
        DEA の方向。'input' は入力指向（theta を最小化）、'output' は出力指向（phi を最大化）。
    - **solver** : cvxpy ソルバー名または None
        使用するソルバー。None の場合はデフォルトソルバーを使用し、失敗時は SCS にフォールバックする。
    - **label_level** : int or None, optional
        ラベルに使うインデックスレベル（`_prepare_df_for_year` に渡す）。

    Returns
    - **scores** : pandas.Series
        各 DMU の効率値（index は `df_xy.index` に揃う）。
    - **lambdas_store** : dict
        {dmu_label: numpy.ndarray or None}。最適解が得られない場合は None。

    Notes
    - 最適解が得られない DMU は `scores` に `NaN` を格納し、`lambdas_store` では値を `None` にする。
    - `orientation` によって返される効率値の意味が異なる（input: theta ≤ 1、output: phi ≥ 1 を期待）。
    """
    df_xy, dmu_labels, X, Y = _prepare_df_for_year(df, year, inputs, outputs, label_level=label_level)
    n = X.shape[1]

    # 一旦 dmu_labels を使って Series を作るが、最後に index を df_xy.index に揃える
    scores = pd.Series(index=range(n), dtype=float)
    lambdas_store = {}

    for i in range(n):
        x0 = X[:, i]
        y0 = Y[:, i]

        lam = cp.Variable(n, nonneg=True)

        if orientation == 'input':
            # existing input-oriented formulation (minimize theta)
            theta = cp.Variable()
            constraints = [X @ lam <= theta * x0, Y @ lam >= y0]
            # RTS
            if rts == 'VRS':
                constraints.append(cp.sum(lam) == 1)
            elif rts == 'DRS':
                constraints.append(cp.sum(lam) <= 1)
            # nonnegativity / optional upper bound
            constraints.append(theta >= 0)
            # optional: constraints.append(theta <= 1.0)
            prob = cp.Problem(cp.Minimize(theta), constraints)
        elif orientation == 'output':
            # output-oriented formulation (maximize phi)
            phi = cp.Variable()
            constraints = [X @ lam <= x0, Y @ lam >= phi * y0]
            if rts == 'VRS':
                constraints.append(cp.sum(lam) == 1)
            elif rts == 'DRS':
                constraints.append(cp.sum(lam) <= 1)
            # phi typically >= 1 for output orientation
            constraints.append(phi >= 0)   # allow solver freedom; we will interpret phi>=1 as efficient
            # optional: constraints.append(phi >= 1.0)
            prob = cp.Problem(cp.Maximize(phi), constraints)
        else:
            raise ValueError("orientation must be 'input' or 'output'")

        try:
            if solver is not None:
                prob.solve(solver=solver, verbose=False)
            else:
                prob.solve(verbose=False)
        except Exception:
            prob.solve(solver=cp.SCS, verbose=False)

        label = df_xy.index[i]
        if prob.status in ["optimal", "optimal_inaccurate"]:
            if orientation == 'input':
                scores.iloc[i] = float(theta.value)
            else:
                scores.iloc[i] = float(phi.value)
            lambdas_store[label] = np.array(lam.value, dtype=float)
        else:
            scores.iloc[i] = np.nan
            lambdas_store[label] = None

    # ここで scores.index を df_xy.index に揃える（MultiIndex そのまま）
    scores.index = df_xy.index
    return scores, lambdas_store


def dea_with_integration(df, year, inputs, outputs, rts='VRS', orientation='output',
                         solver=None, eps=1e-8, label_level=None, return_lambdas=False,
                         tol=1e-6):
    """
    DEA を実行して、元データに効率値・ピア情報・投影値を統合した DataFrame を返す。

    概要
    - `dea_from_dataframe` を呼び出して効率値と lambda を取得する。
    - 各 DMU について非ゼロの lambda をピアとして抽出し、ピア重みを保存する。
    - 投影値（proj_<input>, proj_<output>）を計算して追加する。
    - `is_efficient` は効率値が 1 に近いかどうかを `tol` で判定する。

    Parameters
    - **df** : pandas.DataFrame
        元データ。
    - **year** : scalar
        実行対象の年。
    - **inputs** : sequence of str
        入力変数名リスト。
    - **outputs** : sequence of str
        出力変数名リスト。
    - **rts** : {'CRS','VRS','DRS'}, default 'VRS'
        規模収益の仮定。
    - **orientation** : {'input','output'}, default 'output'
        DEA の方向。
    - **solver** : cvxpy ソルバー名または None
        使用ソルバー。
    - **eps** : float, default 1e-8
        lambda を非ゼロとみなす閾値。
    - **label_level** : int or None, optional
        ラベルに使うインデックスレベル。
    - **return_lambdas** : bool, default False
        True の場合は (out_df, lambdas) を返す。lambdas は {label: array}。
    - **tol** : float, default 1e-6
        `is_efficient` 判定の許容誤差（効率値と 1 の差の絶対値が tol 以下で効率とみなす）。

    Returns
    - **out** : pandas.DataFrame
        `df_xy` を基に以下の列を追加した DataFrame:
        - **efficiency** : float
        - **is_efficient** : bool
        - **peers** : list of labels
        - **peer_weights** : list of floats
        - **proj_<input>** : float（各入力の投影値）
        - **proj_<output>** : float（各出力の投影値）
    - **lambdas** : dict, optional
        `return_lambdas=True` の場合に返る {label: numpy.ndarray}。

    Notes
    - 投影の定義は `orientation` に依存する:
      - input 指向: \(x_{proj} = \theta x_0\), \(y_{proj} = Y \lambda\)
      - output 指向: \(y_{proj} = \phi y_0\), \(x_{proj} = X \lambda\)
    - 最適化が失敗した DMU は peers と投影値が空または NaN になる。
    """
    df_xy, dmu_labels, X, Y = _prepare_df_for_year(df, year, inputs, outputs, label_level=label_level)
    scores, lambdas = dea_from_dataframe(df, year, inputs, outputs, rts=rts,
                                         orientation=orientation, solver=solver, label_level=label_level)

    dmu_order = list(scores.index)
    n = X.shape[1]

    proj_inputs = np.full((n, X.shape[0]), np.nan, dtype=float)
    proj_outputs = np.full((n, Y.shape[0]), np.nan, dtype=float)
    peers_list = []
    peer_weights_list = []
    is_eff_list = []

    for i, dmu in enumerate(dmu_order):
        score = scores.loc[dmu]          # score is theta (input) or phi (output)
        lam = lambdas.get(dmu)

        if lam is None or np.any(np.isnan(lam)) or np.isnan(score):
            peers_list.append([])
            peer_weights_list.append([])
            is_eff_list.append(False)
            continue

        lam = np.array(lam, dtype=float)
        nz_idx = np.where(lam > eps)[0]
        peers = [dmu_order[j] for j in nz_idx.tolist()]
        weights = lam[nz_idx].tolist()
        peers_list.append(peers)
        peer_weights_list.append(weights)

        # Projection: formulas depend on orientation and on meaning of `score`
        if orientation == 'input':
            # score == theta (<=1). projection: x_proj = theta * x0, y_proj = Y @ lam
            x0 = X[:, i]
            proj_x = float(score) * x0
            proj_y = (Y @ lam).flatten()
            proj_inputs[i, :] = proj_x
            proj_outputs[i, :] = proj_y
            is_eff_list.append(abs(score - 1.0) <= tol)
        else:
            # output-oriented: score == phi (>=1). projection: y_proj = phi * y0, x_proj = X @ lam
            y0 = Y[:, i]
            proj_y = float(score) * y0
            proj_x = (X @ lam).flatten()
            proj_inputs[i, :] = proj_x
            proj_outputs[i, :] = proj_y
            is_eff_list.append(abs(score - 1.0) <= tol)

    out = df_xy.copy()
    out['efficiency'] = scores.reindex(out.index).values
    out['is_efficient'] = is_eff_list
    out['peers'] = peers_list
    out['peer_weights'] = peer_weights_list

    for j, col in enumerate(inputs):
        out[f'proj_{col}'] = proj_inputs[:, j]
    for j, col in enumerate(outputs):
        out[f'proj_{col}'] = proj_outputs[:, j]

    if return_lambdas:
        return out, lambdas
    return out


def dea_expand_all_years(df, inputs, outputs, rts='all', orientation='output',
                         solver=None, eps=1e-8, label_level=None,
                         return_lambdas=False):
    """
    全年にわたり DEA を実行し、元 DataFrame に RTS ごとの結果列を追加する。

    概要
    - `rts` に 'all' を指定すると ['CRS','VRS','DRS'] を順に実行する。
    - 各 RTS ごとに列名の末尾にサフィックスを付けて結果を追加する:
      例: efficiency_VRS, is_efficient_CRS, peers_DRS, proj_input1_VRS など。
    - `return_lambdas=True` の場合は RTS をキーとする辞書で全 lambda を返す。

    Parameters
    - **df** : pandas.DataFrame
        元データ（MultiIndex の末尾が年であることを想定）。
    - **inputs** : sequence of str
        入力変数名リスト。
    - **outputs** : sequence of str
        出力変数名リスト。
    - **rts** : {'all', 'CRS', 'VRS', 'DRS'} or list, default 'all'
        実行する RTS。'all' は ['CRS','VRS','DRS'] に展開される。リストで複数指定可。
    - **orientation** : {'input','output'}, default 'output'
        DEA の方向。
    - **solver** : cvxpy ソルバー名または None
        使用ソルバー。
    - **eps** : float, default 1e-8
        lambda を非ゼロとみなす閾値（`dea_with_integration` に渡す）。
    - **label_level** : int or None, optional
        ラベルに使うインデックスレベル。
    - **return_lambdas** : bool, default False
        True の場合は (df_out, lambdas_all) を返す。`lambdas_all` は
        {rts: {label: numpy.ndarray}} の構造。

    Returns
    - **df_out** : pandas.DataFrame
        元の DataFrame に RTS ごとの結果列を追加した DataFrame。
    - **lambdas_all** : dict, optional
        `return_lambdas=True` の場合に返る辞書。構造は {rts: {label: lambda_array}}。

    Notes
    - 計算量は年数 × RTS 数に比例して増えるため、処理時間に注意すること。
    - 列名のサフィックスはアンダースコアと RTS 名（例: `_VRS`）を用いる。
    - 必要なら並列化や結果キャッシュの導入を検討する。
    """
    df_out = df.copy()

    # rts 引数の正規化
    if rts == 'all' or rts is None:
        rts_list = ['CRS', 'VRS', 'DRS']
    elif isinstance(rts, (list, tuple)):
        rts_list = list(rts)
    else:
        rts_list = [rts]

    # 年の一覧（MultiIndex の末尾が年であることを想定）
    years = sorted(df.index.get_level_values(-1).unique())

    lambdas_all = {} if return_lambdas else None

    for y in years:
        # 年ごとにデータを抽出しておく（_prepare_df_for_year を再利用）
        df_xy, dmu_labels, X, Y = _prepare_df_for_year(df, y, inputs, outputs, label_level=label_level)

        for r in rts_list:
            out = dea_with_integration(
                df, y,
                inputs=inputs,
                outputs=outputs,
                rts=r,
                orientation=orientation,
                solver=solver,
                eps=eps,
                label_level=label_level,
                return_lambdas=return_lambdas
            )

            if return_lambdas:
                out_df, lambdas = out
                # lambdas_all[r] を年ごと・DMUごとにまとめる（キー構造は自由に変更可）
                if lambdas_all is None:
                    lambdas_all = {}
                if r not in lambdas_all:
                    lambdas_all[r] = {}
                # lambdas は {label: array} なのでそのままマージ
                lambdas_all[r].update(lambdas)
            else:
                out_df = out

            # サフィックスを付けた列名で df_out に格納
            suffix = f'_{r}'
            # efficiency, is_efficient, peers, peer_weights
            df_out.loc[out_df.index, f'efficiency{suffix}'] = out_df['efficiency']
            df_out.loc[out_df.index, f'is_efficient{suffix}'] = out_df['is_efficient']
            df_out.loc[out_df.index, f'peers{suffix}'] = out_df['peers']
            df_out.loc[out_df.index, f'peer_weights{suffix}'] = out_df['peer_weights']

            # projected inputs/outputs
            for col in inputs:
                df_out.loc[out_df.index, f'proj_{col}{suffix}'] = out_df[f'proj_{col}']
            for col in outputs:
                df_out.loc[out_df.index, f'proj_{col}{suffix}'] = out_df[f'proj_{col}']

    if return_lambdas:
        return df_out, lambdas_all
    return df_out

# ----------------------------------- #


def _prepare_df_for_year_v2(df, year, inputs, outputs, label_level=None, keep_columns=None):
    """
    年ごとの共通前処理を行い、DEA に使うデータ行列とラベルを返す。
    変更点:
    - keep_columns を受け取り、存在する列だけを df_keep として返す。
    - df_year の index は DMU のみ（year レベルを drop）にする。
    戻り値: df_xy, dmu_labels, X, Y, df_keep
    """
    df_sel = df.copy()

    # --- MultiIndex の末尾レベルを year とみなし抽出 ---
    if isinstance(df_sel.index, pd.MultiIndex) and df_sel.index.nlevels >= 2:
        mask = df_sel.index.get_level_values(-1) == year
        df_year = df_sel[mask].copy()
        # year レベルを drop して DMU のみの index にする
        df_year.index = df_year.index.droplevel(-1)
    else:
        if 'year' in df_sel.columns:
            df_year = df_sel[df_sel['year'] == year].copy()
        else:
            raise ValueError("DataFrame must have MultiIndex (DMU,...,year) or a 'year' column.")

    # 必要列だけ残す（inputs/outputs は必須）
    cols = list(inputs) + list(outputs)
    # 入力・出力に欠損がある行は除外する
    df_xy = df_year[cols].dropna(how='any').copy()
    if df_xy.shape[0] == 0:
        raise ValueError("No DMUs with complete data for the selected year and variables.")

    # DMU ラベル
    if label_level is None:
        dmu_labels = list(df_xy.index)
    else:
        dmu_labels = list(df_xy.index.get_level_values(label_level))

    # 行列化（DEA 標準形）
    X = df_xy[inputs].to_numpy().T.astype(float)
    Y = df_xy[outputs].to_numpy().T.astype(float)

    # keep_columns の処理
    df_keep = None
    if keep_columns is not None:
        if isinstance(keep_columns, str):
            keep_columns = [keep_columns]
        existing = [c for c in keep_columns if c in df_year.columns]
        missing = [c for c in keep_columns if c not in df_year.columns]
        if missing:
            print(f"[WARN] For year {year}, these keep_columns not found and will be ignored: {missing}")
        if len(existing) > 0:
            # df_year は DMU を index にしているのでそのまま existing 列を抽出
            df_keep = df_year[existing].copy()

    return df_xy, dmu_labels, X, Y, df_keep


# --- 既存の _prepare_df_for_year_v2, dea_from_dataframe, dea_with_integration, dea_expand_all_years は省略せずそのまま残す ---
# （ここに元の関数群を置いてください。省略しないでください。）

# --- 追加ヘルパー: 参照フロンティア（ref_year）の X_ref, Y_ref に対して任意の評価点群を評価する ---
def _evaluate_against_reference(df, ref_year, eval_df, inputs, outputs, rts='VRS', orientation='output', solver=None):
    """
    ref_year のフロンティア (X_ref, Y_ref) に対して、eval_df の各行 (inputs, outputs) を評価する。
    - eval_df: pandas.DataFrame（index は評価対象 DMU のラベル、columns に inputs+outputs を含む）
    - 戻り値: pandas.Series of scores (index = eval_df.index)
    """
    # 参照フロンティアを準備
    df_ref, _, X_ref, Y_ref, _ = _prepare_df_for_year_v2(df, ref_year, inputs, outputs, label_level=None)
    n_ref = X_ref.shape[1]
    m_in = X_ref.shape[0]
    m_out = Y_ref.shape[0]

    scores = pd.Series(index=eval_df.index, dtype=float)

    # 各評価点について LP を解く（出力指向のみを想定しているが、orientation を引き継ぐ）
    for idx, row in eval_df.iterrows():
        x0 = row[inputs].to_numpy().astype(float)
        y0 = row[outputs].to_numpy().astype(float)

        lam = cp.Variable(n_ref, nonneg=True)

        if orientation == 'input':
            theta = cp.Variable()
            constraints = [X_ref @ lam <= theta * x0, Y_ref @ lam >= y0]
            if rts == 'VRS':
                constraints.append(cp.sum(lam) == 1)
            elif rts == 'DRS':
                constraints.append(cp.sum(lam) <= 1)
            constraints.append(theta >= 0)
            prob = cp.Problem(cp.Minimize(theta), constraints)
        else:
            phi = cp.Variable()
            constraints = [X_ref @ lam <= x0, Y_ref @ lam >= phi * y0]
            if rts == 'VRS':
                constraints.append(cp.sum(lam) == 1)
            elif rts == 'DRS':
                constraints.append(cp.sum(lam) <= 1)
            constraints.append(phi >= 0)
            prob = cp.Problem(cp.Maximize(phi), constraints)

        try:
            if solver is not None:
                prob.solve(solver=solver, verbose=False)
            else:
                prob.solve(verbose=False)
        except Exception:
            prob.solve(solver=cp.SCS, verbose=False)

        if prob.status in ["optimal", "optimal_inaccurate"]:
            if orientation == 'input':
                scores.loc[idx] = float(theta.value)
            else:
                scores.loc[idx] = float(phi.value)
        else:
            scores.loc[idx] = np.nan

    return scores

def _make_bit_labels(m):
    """長さ m の全ビット列を 'p' + bits で返す（例: p00, p01, ...）。順序は lexicographic (0..1)。"""
    labels = []
    for bits in itertools.product([0,1], repeat=m):
        labels.append('p' + ''.join(str(b) for b in bits))
    return labels

def _build_input_combo(x_t_row, x_t1_row, bits):
    """
    bits: tuple/list of 0/1 of length m
    x_t_row, x_t1_row: pandas Series or 1D array of length m (inputs order)
    戻り値: numpy array length m
    """
    x_t = np.asarray(x_t_row, dtype=float)
    x_t1 = np.asarray(x_t1_row, dtype=float)
    combo = np.where(np.array(bits)==1, x_t1, x_t)
    return combo

def _cagr(num, year_t, year_t1, label=None, total_growth=False):
    # 付加情報がある場合のプレフィックスを作成
    info = f" [{label}]" if label is not None else ""

    # 1. NaN チェック
    if pd.isna(num):
        print(f"[WARN]{info} CAGR undefined (nan): num={num}")
        return np.nan
    
    # 2. 0 のチェック
    if num == 0:
        print(f"[WARN]{info} CAGR undefined (zero): num={num}")
        return np.nan

    # 3. 負の値のチェック（比率がマイナスだと複素数になるため計算不可）
    if num < 0:
        print(f"[WARN]{info} CAGR negative input (undefined): num={num}")
        return np.nan

    # 4. 期間のチェック
    delta_t = year_t1 - year_t
    if delta_t <= 0:
        print(f"[WARN]{info} Invalid period: delta_t={delta_t}")
        return np.nan

    if (total_growth):
        # 累積成長率を返す
        return (num - 1) * 100 # KR like
    else:
        # 標準的なCAGRの計算式
        return (num ** (1 / delta_t) - 1) * 100

def _safe_div(a, b):
    try:
        if pd.isna(a) or pd.isna(b):
            return np.nan
        if b == 0:
            return np.nan
        return a / b
    except Exception:
        return np.nan


def dea_add_frontier_point_estimates(df, year_t, year_t1, inputs, outputs,
                                                  rts='VRS', orientation='output', solver=None,
                                                  label_level=None, include_proj_outputs=True, debug=False, keep_columns=None, total_growth=False, **kwargs):
    """
    2 期間 t・t+1 のデータについて、m 個の入力に対する 2^m 通りの
    「入力ミックス（ビットパターン）」をすべて生成し、それぞれの
    φ（効率値）を両期間のフロンティアで評価し、さらに
    Malmquist 型の生産性指標（EFF, TECH, ACCUM, ACCUM_k）を構成する。

    Parameters
    ----------
    df : pandas.DataFrame
        パネル形式のデータ。行は DMU、列は入力・出力・年など。
    year_t : int or str
        基準となる期間 t。
    year_t1 : int or str
        次期 t+1。
    inputs : list of str
        入力変数の列名。
    outputs : list of str
        出力変数の列名。
    rts : {'VRS', 'CRS'}, default 'VRS'
        規模に関する仮定（DEA ソルバーに渡す）。
    orientation : {'output', 'input'}, default 'output'
        DEA の指向性。Malmquist 分解では output 指向を想定。
    solver : optional
        `_evaluate_against_reference` に渡す DEA ソルバー。
    label_level : optional
        MultiIndex のどの階層が DMU 名かを指定する場合に使用。
    include_proj_outputs : bool, default True
        True の場合、各 φ 列に対して φ·y の投影出力列も作成。
    debug : bool, default False
        True の場合、ACCUM_calc や EC_calc など内部検算用の列を
        df_mi に追加する。
    keep_columns : list of str, optional
        df_new に引き継ぎたい元のデータフレームの列名リスト。
    total_growth : df_miの結果列をCAGRではなく累積成長率(%)で表示。デフォルトはFalse

    Returns
    -------
    df_new : pandas.DataFrame
        MultiIndex (DMU, year) を持つデータフレーム。
        内容は以下を含む：
        - 各 DMU の元の入力・出力（t, t+1）
        - 2^m 通りのビットパターン pXYZ に対する φ（onF0, onF1）
        - 必要に応じて φ·y の投影出力
        列名は次の形式：
            pXYZ_phi_onF0, pXYZ_phi_onF1
            pXYZ_phi_onF0_<output>, pXYZ_phi_onF1_<output>

    df_mi : pandas.DataFrame
        DMU ごとの Malmquist 型分解結果。
        含まれる指標：
        - Total (%) : y1/y0 の CAGR
        - TFP (%)   : EFF + TECH（総合的生産性変化）
        - EFF (%)   : 効率変化
        - TECH (%)  : 技術変化
        - ACCUM (%) : 入力量の蓄積効果
        - ACCUMk (%): 各入力 k の寄与（m > 1 の場合）
        debug=True の場合、ACCUM_calc などの検算列も追加。

    Notes
    -----
    • m 個の入力に対して 2^m 通りのビットパターンを生成する。
      ビット 0 は year_t の入力、1 は year_t1 の入力を選択する。
      これにより「仮想的な入力ミックス点」を作り、両期間の
      フロンティア F_t, F_{t+1} で φ を評価する。

    • φ を用いて次のように指標を構成する：
        EFF  = φ_{p0,onF0}(t) / φ_{p1,onF1}(t+1)
        TECH = sqrt( (F1_p0/F0_p0) * (F1_p1/F0_p1) )
        ACCUM = sqrt( (F0_p1/F0_p0) * (F1_p1/F1_p0) )
      ACCUM_k は、入力 k のビットだけを変化させたパターン間の比を
      幾何平均して求める。

    • すべての MI 指標は、year_t → year_t1 の CAGR (%) として返す。

    • 両期間に共通する DMU が存在しない場合、空の DataFrame を返す。

    """

    # --- 元の index レベル名を取得 ---
    if isinstance(df.index, pd.MultiIndex):
        orig_index_names = list(df.index.names)
        orig_dmu_name = orig_index_names[0] if orig_index_names[0] is not None else 'country'
        orig_year_name = orig_index_names[-1] if orig_index_names[-1] is not None else 'year'
    else:
        orig_dmu_name = df.index.name if df.index.name is not None else 'country'
        orig_year_name = 'year'

    # --- 年ごとの前処理（keep_columns を渡す） ---
    df_t, _, X_t, Y_t, df_t_keep = _prepare_df_for_year_v2(df, year_t, inputs, outputs,
                                                           label_level=label_level, keep_columns=keep_columns)
    df_t1, _, X_t1, Y_t1, df_t1_keep = _prepare_df_for_year_v2(df, year_t1, inputs, outputs,
                                                               label_level=label_level, keep_columns=keep_columns)

    # 共通 DMU の抽出
    idx_common = df_t.index.intersection(df_t1.index)
    if len(idx_common) == 0:
        return pd.DataFrame(), pd.DataFrame()

    # 新しい MultiIndex を元のレベル名で作成
    index_new = pd.MultiIndex.from_product([idx_common, [year_t, year_t1]], names=[orig_dmu_name, orig_year_name])
    df_new = pd.DataFrame(index=index_new)

    # 元の inputs / outputs をコピー
    for dmu in idx_common:
        df_new.loc[(dmu, year_t), inputs] = df_t.loc[dmu, inputs].values
        df_new.loc[(dmu, year_t), outputs] = df_t.loc[dmu, outputs].values
        df_new.loc[(dmu, year_t1), inputs] = df_t1.loc[dmu, inputs].values
        df_new.loc[(dmu, year_t1), outputs] = df_t1.loc[dmu, outputs].values

    # φ 計算の準備
    m = len(inputs)
    bit_tuples = list(itertools.product([0, 1], repeat=m))
    bit_labels = ['p' + ''.join(str(b) for b in bits) for bits in bit_tuples]

    for bits, label in zip(bit_tuples, bit_labels):
        eval_rows_t = []
        eval_rows_t1 = []
        for dmu in idx_common:
            x_t = df_t.loc[dmu, inputs]
            x_t1 = df_t1.loc[dmu, inputs]
            x_combo_t = _build_input_combo(x_t, x_t1, bits)
            x_combo_t1 = _build_input_combo(x_t, x_t1, bits)

            eval_rows_t.append(pd.Series(np.concatenate([x_combo_t, df_t.loc[dmu, outputs].to_numpy()]),
                                        index=list(inputs) + list(outputs), name=dmu))
            eval_rows_t1.append(pd.Series(np.concatenate([x_combo_t1, df_t1.loc[dmu, outputs].to_numpy()]),
                                         index=list(inputs) + list(outputs), name=dmu))

        eval_df_t = pd.DataFrame(eval_rows_t)
        eval_df_t1 = pd.DataFrame(eval_rows_t1)

        # φ を 4 通り計算
        phi_t_on_t = _evaluate_against_reference(df, year_t, eval_df_t, inputs, outputs,
                                                 rts=rts, orientation=orientation, solver=solver)
        phi_t1_on_t = _evaluate_against_reference(df, year_t, eval_df_t1, inputs, outputs,
                                                  rts=rts, orientation=orientation, solver=solver)
        phi_t_on_t1 = _evaluate_against_reference(df, year_t1, eval_df_t, inputs, outputs,
                                                  rts=rts, orientation=orientation, solver=solver)
        phi_t1_on_t1 = _evaluate_against_reference(df, year_t1, eval_df_t1, inputs, outputs,
                                                   rts=rts, orientation=orientation, solver=solver)

        for f, (phi_eval_t, phi_eval_t1) in enumerate(((phi_t_on_t, phi_t1_on_t), (phi_t_on_t1, phi_t1_on_t1))):
            phi_col = f"{label}_phi_onF{f}"
            if phi_col not in df_new.columns:
                df_new[phi_col] = np.nan

            for dmu in idx_common:
                df_new.loc[(dmu, year_t), phi_col] = phi_eval_t.loc[dmu]
                df_new.loc[(dmu, year_t1), phi_col] = phi_eval_t1.loc[dmu]

            if include_proj_outputs:
                for out in outputs:
                    proj_col = f"{phi_col}_{out}"
                    if proj_col not in df_new.columns:
                        df_new[proj_col] = np.nan

                    for dmu in idx_common:
                        y0 = df_t.loc[dmu, out]
                        phi_at_t = df_new.loc[(dmu, year_t), phi_col]
                        df_new.loc[(dmu, year_t), proj_col] = (phi_at_t * y0) if not pd.isna(phi_at_t) else np.nan

                        y1 = df_t1.loc[dmu, out]
                        phi_at_t1 = df_new.loc[(dmu, year_t1), phi_col]
                        df_new.loc[(dmu, year_t1), proj_col] = (phi_at_t1 * y1) if not pd.isna(phi_at_t1) else np.nan


    # --- keep_columns の列名リストを決定（存在する列のみ、順序は df_t_keep 優先） ---
    keep_cols = []
    if keep_columns is not None:
        if df_t_keep is not None:
            keep_cols = list(df_t_keep.columns)
        elif df_t1_keep is not None:
            keep_cols = list(df_t1_keep.columns)
        # 重複や None を排除
        keep_cols = [c for c in keep_cols if c is not None]



    # MI/TC/MI_k/EC の計算（従来ロジック）
    rows = []
    for dmu in idx_common:
        out = outputs[0]
        p0 = 'p' + '0' * m
        p1 = 'p' + '1' * m

        F0_p0 = df_new.loc[(dmu, year_t), f"{p0}_phi_onF0_{out}"]
        F1_p0 = df_new.loc[(dmu, year_t1), f"{p0}_phi_onF1_{out}"]
        F0_p1 = df_new.loc[(dmu, year_t), f"{p1}_phi_onF0_{out}"]
        F1_p1 = df_new.loc[(dmu, year_t1), f"{p1}_phi_onF1_{out}"]

        EFF = df_new.loc[(dmu, year_t), f"{p0}_phi_onF0"] / df_new.loc[(dmu, year_t1), f"{p1}_phi_onF1"]

        t_p1, t_p2 = F1_p0 / F0_p0, F1_p1 / F0_p1
        a_p1, a_p2 = F0_p1 / F0_p0, F1_p1 / F1_p0

        TECH = np.sqrt(t_p1 * t_p2)
        ACCUM = np.sqrt(a_p1 * a_p2)

        if np.isnan(TECH) or np.isnan(ACCUM):
            print(f"\n[DEBUG] NaN detected for DMU: {dmu} ({year_t} -> {year_t1})")
            if np.isnan(TECH):
                if not np.isnan(t_p1) and np.isnan(t_p2):
                    TECH = t_p1
                    print(f"  > TECH: Part2 is NaN. Substituted with Part1: {t_p1:.4f}")
                elif np.isnan(t_p1) and not np.isnan(t_p2):
                    TECH = t_p2
                    print(f"  > TECH: Part1 is NaN. Substituted with Part2: {t_p2:.4f}")
            if np.isnan(ACCUM):
                if not np.isnan(a_p1) and np.isnan(a_p2):
                    ACCUM = a_p1
                    print(f"  > ACCUM: Part2 is NaN. Substituted with Part1: {a_p1:.4f}")
                elif np.isnan(a_p1) and not np.isnan(a_p2):
                    ACCUM = a_p2
                    print(f"  > ACCUM: Part1 is NaN. Substituted with Part2: {a_p2:.4f}")
            if np.isnan(TECH) or np.isnan(ACCUM):
                print(f"  > Critical: Recalculation failed. Inputs: F0_p0={F0_p0}, F1_p0={F1_p0}, F0_p1={F0_p1}, F1_p1={F1_p1}")

        ACCUM_k_list = []
        ACCUM_calc = 1.0

        if m == 1:
            ACCUM_calc = ACCUM
        else:
            for k in range(m):
                ratios = []
                nan_detected = False
                for other_bits in itertools.product([0, 1], repeat=m - 1):
                    bits0 = list(other_bits); bits0.insert(k, 0); label0 = 'p' + ''.join(str(b) for b in bits0)
                    bits1 = list(other_bits); bits1.insert(k, 1); label1 = 'p' + ''.join(str(b) for b in bits1)

                    F0_0 = df_new.loc[(dmu, year_t), f"{label0}_phi_onF0_{out}"]
                    F0_1 = df_new.loc[(dmu, year_t), f"{label1}_phi_onF0_{out}"]
                    F1_0 = df_new.loc[(dmu, year_t1), f"{label0}_phi_onF1_{out}"]
                    F1_1 = df_new.loc[(dmu, year_t1), f"{label1}_phi_onF1_{out}"]

                    r1 = _safe_div(F0_1, F0_0)
                    r2 = _safe_div(F1_1, F1_0)

                    if pd.isna(r1) or pd.isna(r2):
                        nan_detected = True
                    if not pd.isna(r1):
                        ratios.append(r1)
                    if not pd.isna(r2):
                        ratios.append(r2)

                if nan_detected or len(ratios) == 0:
                    print(f"[WARN][{dmu}] ACCUM_k: NaN detected for input {k}. ratios={ratios}")
                    ACCUM_k = np.nan
                else:
                    ACCUM_k = np.prod(ratios) ** (1.0 / len(ratios))

                if debug:
                    ACCUM_calc = ACCUM_calc * (ACCUM_k if not pd.isna(ACCUM_k) else 1.0)

                ACCUM_k_cagr = _cagr(ACCUM_k, year_t, year_t1, dmu, total_growth) if not pd.isna(ACCUM_k) else np.nan
                ACCUM_k_list.append(ACCUM_k_cagr)

        y0 = df_new.loc[(dmu, year_t), out]
        y1 = df_new.loc[(dmu, year_t1), out]
        if debug:
            EC_calc = y1 / y0 / TECH / ACCUM

        if debug:
            # --- まず CAGR を計算 ---
            total_cagr = _cagr(y1 / y0, year_t, year_t1, dmu, total_growth)
            eff_cagr   = _cagr(EFF, year_t, year_t1, dmu, total_growth)
            tech_cagr  = _cagr(TECH, year_t, year_t1, dmu, total_growth)
            accum_cagr = _cagr(ACCUM, year_t, year_t1, dmu, total_growth)
            accum_calc_cagr = _cagr(ACCUM_calc, year_t, year_t1, dmu, total_growth)
            ec_calc_cagr = _cagr(EC_calc, year_t, year_t1, dmu, total_growth)

            # --- 新設: TFP = EFF_cagr + TECH_cagr ---
            tfp_cagr = eff_cagr + tech_cagr

            row = [
                dmu,
                total_cagr,      # Total
                tfp_cagr,        # ← 新設 TFP
                eff_cagr,        # EFF
                ec_calc_cagr,    # EFF_calc
                tech_cagr,       # TECH
                accum_cagr,      # ACCUM
                accum_calc_cagr  # ACCUM_calc
            ] + ACCUM_k_list

        else:
            # --- まず CAGR を計算 ---
            total_cagr = _cagr(y1 / y0, year_t, year_t1, dmu, total_growth)
            eff_cagr   = _cagr(EFF, year_t, year_t1, dmu, total_growth)
            tech_cagr  = _cagr(TECH, year_t, year_t1, dmu, total_growth)
            accum_cagr = _cagr(ACCUM, year_t, year_t1, dmu, total_growth)

            # --- 新設: TFP = EFF_cagr + TECH_cagr ---
            tfp_cagr = eff_cagr + tech_cagr

            row = [
                dmu,
                total_cagr,   # Total
                tfp_cagr,     # ← 新設 TFP
                eff_cagr,     # EFF
                tech_cagr,    # TECH
                accum_cagr    # ACCUM
            ] + ACCUM_k_list

        # --- ここで year_t 時点の出力値を追加 ---
        # outputs は関数引数の list[str]
        y0_list = []
        for out_col in outputs:
            try:
                y0_val = df_new.loc[(dmu, year_t), out_col]
            except KeyError:
                # 万一 df_new に出力列が無ければ NaN を入れる
                y0_val = np.nan
            y0_list.append(y0_val)

        row = row + y0_list

        # --- 追加: keep_columns の year_t 値を row に追加 ---
        if keep_cols:
            keep_vals = []
            for kc in keep_cols:
                # df_t_keep は year_t の時点の keep 列を持つ（index は DMU）
                if df_t_keep is not None and kc in df_t_keep.columns and dmu in df_t_keep.index:
                    keep_vals.append(df_t_keep.loc[dmu, kc])
                else:
                    # 存在しない場合は NaN
                    keep_vals.append(np.nan)
            row = row + keep_vals

        rows.append(row)

    _unit = '%'
    if debug:
        colnames = [f'{orig_dmu_name}', f'Total ({_unit})', f'TFP ({_unit})', f'EFF ({_unit})', f'EFF_calc ({_unit})',
                    f'TECH ({_unit})', f'ACCUM ({_unit})', f'ACCUM_calc ({_unit})']
    else:
        colnames = [f'{orig_dmu_name}', f'Total ({_unit})', f'TFP ({_unit})', f'EFF ({_unit})', f'TECH ({_unit})', f'ACCUM ({_unit})']

    if m > 1:
        colnames += [f'ACCUM{k} ({_unit})' for k in range(m)]

    # --- ここで year_t 時点の出力列名を追加 ---
    # 表示は "output_name (year_t)" とする（year_t を文字列化）
    year_label = str(year_t)
    colnames += [f"{out} ({year_label})" for out in outputs]

    # --- 追加: keep_columns の列名を追加（表示は "<col> (year_t)"） ---
    if keep_cols:
        colnames += [f"{kc} ({year_label})" for kc in keep_cols]

    # df_mi の index 名を元の DMU 名にする
    df_mi = pd.DataFrame(rows, columns=colnames).set_index(orig_dmu_name)

    # --- keep_columns マージ処理（元の index レベル名を完全に使う） ---
    if keep_columns is not None:
        # orig_dmu_name, orig_year_name は関数冒頭で決定済みとする
        # フォールバック
        if orig_dmu_name is None:
            orig_dmu_name = 'country'
        if orig_year_name is None:
            orig_year_name = 'year'

        # df_t_keep / df_t1_keep を parts に入れる際に、必ず元の index 名で列ができるよう正規化する
        parts = []
        if df_t_keep is not None:
            tmp = df_t_keep.copy().reset_index()
            # reset_index() によって元の DMU レベル名が列になっているはず
            if orig_dmu_name not in tmp.columns:
                first_col = tmp.columns[0]
                if first_col != orig_year_name:
                    tmp = tmp.rename(columns={first_col: orig_dmu_name})
            # 年列は元のレベル名で付与する（文字列化）
            tmp[orig_year_name] = str(year_t)
            parts.append(tmp)

        if df_t1_keep is not None:
            tmp = df_t1_keep.copy().reset_index()
            if orig_dmu_name not in tmp.columns:
                first_col = tmp.columns[0]
                if first_col != orig_year_name:
                    tmp = tmp.rename(columns={first_col: orig_dmu_name})
            tmp[orig_year_name] = str(year_t1)
            parts.append(tmp)

        if parts:
            # 各パートを正規化して concat
            norm_parts = []
            for p in parts:
                q = p.copy()
                # 年列を統一（もし orig_year_name が別名で存在していれば 'orig_year_name' を使う）
                if orig_year_name in q.columns:
                    q[orig_year_name] = q[orig_year_name].astype(str)
                else:
                    # ここは通常起きないはずだが念のため
                    q[orig_year_name] = q.get('year', '').astype(str)
                # DMU 列が orig_dmu_name でない場合は最初の列を DMU にする
                if orig_dmu_name not in q.columns:
                    first_col = q.columns[0]
                    if first_col != orig_year_name:
                        q = q.rename(columns={first_col: orig_dmu_name})
                norm_parts.append(q)

            df_keep_all = pd.concat(norm_parts, ignore_index=True)

            # df_new 側を正規化してマージ
            df_new_reset = df_new.reset_index()
            # df_new_reset に orig_dmu_name 列がなければ最初の列をそれにする
            if orig_dmu_name not in df_new_reset.columns:
                first_col = df_new_reset.columns[0]
                if first_col != orig_year_name:
                    df_new_reset = df_new_reset.rename(columns={first_col: orig_dmu_name})
            # df_new_reset の年列を orig_year_name に揃える（reset_index() で既にその名前ならそのまま）
            if orig_year_name not in df_new_reset.columns and 'year' in df_new_reset.columns:
                df_new_reset = df_new_reset.rename(columns={'year': orig_year_name})
            # 文字列化して比較を安定化
            df_new_reset[orig_year_name] = df_new_reset[orig_year_name].astype(str)
            df_keep_all[orig_year_name] = df_keep_all[orig_year_name].astype(str)

            # デバッグ用（必要なら有効化）
            # print("df_keep_all.columns:", df_keep_all.columns.tolist())
            # print("df_new_reset.columns:", df_new_reset.columns.tolist())

            # マージ（キーは元の DMU レベル名 と orig_year_name）
            df_new_merged = pd.merge(
                df_new_reset,
                df_keep_all,
                left_on=[orig_dmu_name, orig_year_name],
                right_on=[orig_dmu_name, orig_year_name],
                how='left'
            )

            # インデックスを元のレベル名で戻す
            df_new = df_new_merged.set_index([orig_dmu_name, orig_year_name])

    return df_new, df_mi





#     # 呼び出し（keep_columns を渡す）
#     df_t, _, X_t, Y_t, df_t_keep = _prepare_df_for_year_v2(df, year_t, inputs, outputs, label_level=label_level, keep_columns=keep_columns)
#     df_t1, _, X_t1, Y_t1, df_t1_keep = _prepare_df_for_year_v2(df, year_t1, inputs, outputs, label_level=label_level, keep_columns=keep_columns)

#     idx_common = df_t.index.intersection(df_t1.index)
#     if len(idx_common) == 0:
#         return pd.DataFrame(), pd.DataFrame()

#     index_new = pd.MultiIndex.from_product([idx_common, [year_t, year_t1]], names=['DMU', 'year'])
#     df_new = pd.DataFrame(index=index_new)

#     # 元の inputs / outputs をコピー
#     for dmu in idx_common:
#         df_new.loc[(dmu, year_t), inputs] = df_t.loc[dmu, inputs].values
#         df_new.loc[(dmu, year_t), outputs] = df_t.loc[dmu, outputs].values
#         df_new.loc[(dmu, year_t1), inputs] = df_t1.loc[dmu, inputs].values
#         df_new.loc[(dmu, year_t1), outputs] = df_t1.loc[dmu, outputs].values

#     m = len(inputs)
#     bit_tuples = list(itertools.product([0,1], repeat=m))
#     bit_labels = ['p' + ''.join(str(b) for b in bits) for bits in bit_tuples]

#     for bits, label in zip(bit_tuples, bit_labels):
#         # eval データ作成
#         eval_rows_t = []
#         eval_rows_t1 = []
#         for dmu in idx_common:
#             x_t = df_t.loc[dmu, inputs]
#             x_t1 = df_t1.loc[dmu, inputs]
#             x_combo_t  = _build_input_combo(x_t,  x_t1, bits)
#             x_combo_t1 = _build_input_combo(x_t,  x_t1, bits)

#             eval_rows_t.append(pd.Series(np.concatenate([x_combo_t,  df_t.loc[dmu, outputs].to_numpy()]),
#                                         index=list(inputs)+list(outputs), name=dmu))
#             eval_rows_t1.append(pd.Series(np.concatenate([x_combo_t1, df_t1.loc[dmu, outputs].to_numpy()]),
#                                          index=list(inputs)+list(outputs), name=dmu))

#         eval_df_t  = pd.DataFrame(eval_rows_t)
#         eval_df_t1 = pd.DataFrame(eval_rows_t1)

#         # φ を 4 通り計算
#         phi_t_on_t   = _evaluate_against_reference(df, year_t,  eval_df_t,  inputs, outputs,
#                                                    rts=rts, orientation=orientation, solver=solver)
#         phi_t1_on_t  = _evaluate_against_reference(df, year_t,  eval_df_t1, inputs, outputs,
#                                                    rts=rts, orientation=orientation, solver=solver)
#         phi_t_on_t1  = _evaluate_against_reference(df, year_t1, eval_df_t,  inputs, outputs,
#                                                    rts=rts, orientation=orientation, solver=solver)
#         phi_t1_on_t1 = _evaluate_against_reference(df, year_t1, eval_df_t1, inputs, outputs,
#                                                    rts=rts, orientation=orientation, solver=solver)

#         # φ 列名と投影列名を決定（phi_col と proj_col）
#         for f, (phi_eval_t, phi_eval_t1) in enumerate(((phi_t_on_t, phi_t1_on_t), (phi_t_on_t1, phi_t1_on_t1))):
#             phi_col = f"{label}_phi_onF{f}"
#             if phi_col not in df_new.columns:
#                 df_new[phi_col] = np.nan

#             # 各行に対応する φ を格納（eval 年に対応する φ をその行に入れる）
#             for dmu in idx_common:
#                 df_new.loc[(dmu, year_t),  phi_col] = phi_eval_t.loc[dmu]
#                 df_new.loc[(dmu, year_t1), phi_col] = phi_eval_t1.loc[dmu]

#             # 投影列名は "<phi_col>_<output>"
#             if include_proj_outputs:
#                 for out in outputs:
#                     proj_col = f"{phi_col}_{out}"
#                     if proj_col not in df_new.columns:
#                         df_new[proj_col] = np.nan

#                     # 各行について、その行の phi とその行の y を掛ける
#                     for dmu in idx_common:
#                         # year_t row
#                         y0 = df_t.loc[dmu, out]
#                         phi_at_t = df_new.loc[(dmu, year_t), phi_col]
#                         df_new.loc[(dmu, year_t), proj_col] = (phi_at_t * y0) if not pd.isna(phi_at_t) else np.nan

#                         # year_t1 row
#                         y1 = df_t1.loc[dmu, out]
#                         phi_at_t1 = df_new.loc[(dmu, year_t1), phi_col]
#                         df_new.loc[(dmu, year_t1), proj_col] = (phi_at_t1 * y1) if not pd.isna(phi_at_t1) else np.nan

#     # MI/TC/MI_k/EC の計算（従来ロジックを phiベースの列名に合わせて参照）
#     rows = []
#     for dmu in idx_common:
#         out = outputs[0]
#         p0 = 'p' + '0'*m
#         p1 = 'p' + '1'*m

#         F0_p0 = df_new.loc[(dmu, year_t),  f"{p0}_phi_onF0_{out}"]
#         F1_p0 = df_new.loc[(dmu, year_t1), f"{p0}_phi_onF1_{out}"]
#         F0_p1 = df_new.loc[(dmu, year_t),  f"{p1}_phi_onF0_{out}"]
#         F1_p1 = df_new.loc[(dmu, year_t1), f"{p1}_phi_onF1_{out}"]

#         EFF = df_new.loc[(dmu, year_t),  f"{p0}_phi_onF0"]/df_new.loc[(dmu, year_t1),  f"{p1}_phi_onF1"]
#         # TECH = np.sqrt((F1_p0 / F0_p0) * (F1_p1 / F0_p1))
#         # ACCUM = np.sqrt((F0_p1 / F0_p0) * (F1_p1 / F1_p0))

#         # --- 通常の計算（メインフロー） ---
#         t_p1, t_p2 = F1_p0 / F0_p0, F1_p1 / F0_p1
#         a_p1, a_p2 = F0_p1 / F0_p0, F1_p1 / F1_p0

#         TECH = np.sqrt(t_p1 * t_p2)
#         ACCUM = np.sqrt(a_p1 * a_p2)

#         # --- 異常検知・再計算・ログ出力 ---
#         if np.isnan(TECH) or np.isnan(ACCUM):
#             print(f"\n[DEBUG] NaN detected for DMU: {dmu} ({year_t} -> {year_t1})")
            
#             # TECHの救済処理
#             if np.isnan(TECH):
#                 if not np.isnan(t_p1) and np.isnan(t_p2):
#                     TECH = t_p1  # (t_p1 * t_p1)のルートと同じ意味
#                     print(f"  > TECH: Part2 is NaN. Substituted with Part1 (case of first inter-frontier move): {t_p1:.4f}")
#                 elif np.isnan(t_p1) and not np.isnan(t_p2):
#                     TECH = t_p2
#                     print(f"  > TECH: Part1 is NaN. Substituted with Part2 (case of last inter-frontier move): {t_p2:.4f}")

#             # ACCUMの救済処理
#             if np.isnan(ACCUM):
#                 if not np.isnan(a_p1) and np.isnan(a_p2):
#                     ACCUM = a_p1
#                     print(f"  > ACCUM: Part2 is NaN. Substituted with Part1 (case of last inter-frontier move): {a_p1:.4f}")
#                 elif np.isnan(a_p1) and not np.isnan(a_p2):
#                     ACCUM = a_p2
#                     print(f"  > ACCUM: Part1 is NaN. Substituted with Part2 (case of first inter-frontier move): {a_p2:.4f}")

#             # それでもNaNの場合や、根本的な原因の表示
#             if np.isnan(TECH) or np.isnan(ACCUM):
#                 print(f"  > Critical: Recalculation failed. Inputs: F0_p0={F0_p0}, F1_p0={F1_p0}, F0_p1={F0_p1}, F1_p1={F1_p1}")

#         ACCUM_k_list = []
#         ACCUM_calc = 1.0

#         if m == 1:
#             ACCUM_calc = ACCUM
#         # elif m == 2:
#         #     # p01, p10 の φ を取得
#         #     F0_p01 = df_new.loc[(dmu, year_t),  f'p01_phi_onF0_{out}']
#         #     F1_p01 = df_new.loc[(dmu, year_t1), f'p01_phi_onF1_{out}']
#         #     F0_p10 = df_new.loc[(dmu, year_t),  f'p10_phi_onF0_{out}']
#         #     F1_p10 = df_new.loc[(dmu, year_t1), f'p10_phi_onF1_{out}']

#         #     F0_p00 = df_new.loc[(dmu, year_t),  f'p00_phi_onF0_{out}']
#         #     F1_p00 = df_new.loc[(dmu, year_t1), f'p00_phi_onF1_{out}']
#         #     F0_p11 = df_new.loc[(dmu, year_t),  f'p11_phi_onF0_{out}']
#         #     F1_p11 = df_new.loc[(dmu, year_t1), f'p11_phi_onF1_{out}']

#         #     # --- ACCUM0（input 0 の寄与） ---
#         #     ACCUM0 = (
#         #         (F0_p10 / F0_p00) * (F0_p11 / F0_p01) *
#         #         (F1_p10 / F1_p00) * (F1_p11 / F1_p01)
#         #     ) ** 0.25

#         #     # --- ACCUM1（input 1 の寄与） ---
#         #     ACCUM1 = (
#         #         (F0_p01 / F0_p00) * (F0_p11 / F0_p10) *
#         #         (F1_p01 / F1_p00) * (F1_p11 / F1_p10)
#         #     ) ** 0.25

#         #     ACCUM_k_list = [ACCUM0, ACCUM1]
#         #     ACCUM_calc = ACCUM0 * ACCUM1
#         else:
#             for k in range(m):
#                 ratios = []
#                 nan_detected = False  # ← NaN が出たかどうか

#                 for other_bits in itertools.product([0,1], repeat=m-1):
#                     bits0 = list(other_bits); bits0.insert(k, 0); label0 = 'p' + ''.join(str(b) for b in bits0)
#                     bits1 = list(other_bits); bits1.insert(k, 1); label1 = 'p' + ''.join(str(b) for b in bits1)

#                     F0_0 = df_new.loc[(dmu, year_t),  f"{label0}_phi_onF0_{out}"]
#                     F0_1 = df_new.loc[(dmu, year_t),  f"{label1}_phi_onF0_{out}"]
#                     F1_0 = df_new.loc[(dmu, year_t1), f"{label0}_phi_onF1_{out}"]
#                     F1_1 = df_new.loc[(dmu, year_t1), f"{label1}_phi_onF1_{out}"]

#                     r1 = _safe_div(F0_1, F0_0)
#                     r2 = _safe_div(F1_1, F1_0)

#                     # NaN を検知したらフラグを立てる
#                     if pd.isna(r1) or pd.isna(r2):
#                         nan_detected = True

#                     # NaN でなければ ratios に追加
#                     if not pd.isna(r1):
#                         ratios.append(r1)
#                     if not pd.isna(r2):
#                         ratios.append(r2)

#                 # ひとつでも NaN があれば ACCUM_k は NaN
#                 if nan_detected or len(ratios) == 0:
#                     print(f"[WARN][{dmu}] ACCUM_k: NaN detected for input {k}. ratios={ratios}")
#                     ACCUM_k = np.nan
#                 else:
#                     ACCUM_k = np.prod(ratios) ** (1.0 / len(ratios))

#                 if debug:
#                     ACCUM_calc = ACCUM_calc * (ACCUM_k if not pd.isna(ACCUM_k) else 1.0)

#                 ACCUM_k_cagr = _cagr(ACCUM_k, year_t, year_t1, dmu) if not pd.isna(ACCUM_k) else np.nan
#                 ACCUM_k_list.append(ACCUM_k_cagr)


#         y0 = df_new.loc[(dmu, year_t),  out]
#         y1 = df_new.loc[(dmu, year_t1), out]
#         if debug:
#             EC_calc = y1 / y0 / TECH / ACCUM

#         if debug:
#             row = [dmu, _cagr(y1/y0, year_t, year_t1, dmu), _cagr(EFF, year_t, year_t1, dmu), _cagr(EC_calc, year_t, year_t1, dmu), _cagr(TECH, year_t, year_t1, dmu), _cagr(ACCUM, year_t, year_t1, dmu), _cagr(ACCUM_calc, year_t, year_t1, dmu)] + ACCUM_k_list
#         else:
#             row = [dmu, _cagr(y1/y0, year_t, year_t1, dmu), _cagr(EFF, year_t, year_t1, dmu), _cagr(TECH, year_t, year_t1, dmu), _cagr(ACCUM, year_t, year_t1, dmu)] + ACCUM_k_list

#         rows.append(row)

#     _unit = '%'
#     if debug:
#         colnames = [f'DMU', f'Total ({_unit})', f'EFF ({_unit})', f'EFF_calc ({_unit})', f'TECH ({_unit})', f'ACCUM ({_unit})', f'ACCUM_calc ({_unit})']
#     else:
#         colnames = [f'DMU', f'Total ({_unit})', f'EFF ({_unit})', f'TECH ({_unit})', f'ACCUM ({_unit})']

#     # m > 1 の場合のみ ACCUM0, ACCUM1... を追加
#     if m > 1:
#         colnames += [f'ACCUM{k} ({_unit})' for k in range(m)]
#     df_mi = pd.DataFrame(rows, columns=colnames).set_index('DMU')


# # ... 既存処理で df_new を作成 ...

#     # マージ処理（より単純で安全）
#     if keep_columns is not None:
#         # df_t_keep / df_t1_keep は DMU を index に持つ。年列を作ってから concat してマージする
#         parts = []
#         if df_t_keep is not None:
#             tmp = df_t_keep.copy()
#             tmp = tmp.reset_index().rename(columns={'index': 'DMU'}) if tmp.index.name is None else tmp.reset_index()
#             tmp['year'] = str(year_t)
#             parts.append(tmp)
#         if df_t1_keep is not None:
#             tmp = df_t1_keep.copy()
#             tmp = tmp.reset_index().rename(columns={'index': 'DMU'}) if tmp.index.name is None else tmp.reset_index()
#             tmp['year'] = str(year_t1)
#             parts.append(tmp)

#         if parts:
#             df_keep_all = pd.concat(parts, ignore_index=True)
#             # df_new の year を文字列化しておく
#             df_new_reset = df_new.reset_index()
#             df_new_reset['year'] = df_new_reset['year'].astype(str)
#             df_new_merged = pd.merge(df_new_reset, df_keep_all, on=['DMU', 'year'], how='left')
#             df_new = df_new_merged.set_index(['DMU', 'year'])


#     return df_new, df_mi



# m = 2まではOK
# def dea_add_frontier_point_estimates(df, year_t, year_t1, inputs, outputs,
#                                                   rts='VRS', orientation='output', solver=None,
#                                                   label_level=None, include_proj_outputs=True):
#     df_t, _, X_t, Y_t = _prepare_df_for_year_v2(df, year_t, inputs, outputs, label_level=label_level)
#     df_t1, _, X_t1, Y_t1 = _prepare_df_for_year_v2(df, year_t1, inputs, outputs, label_level=label_level)

#     idx_common = df_t.index.intersection(df_t1.index)
#     if len(idx_common) == 0:
#         return pd.DataFrame(), pd.DataFrame()

#     index_new = pd.MultiIndex.from_product([idx_common, [year_t, year_t1]], names=['DMU', 'year'])
#     df_new = pd.DataFrame(index=index_new)

#     # 元の inputs / outputs をコピー
#     for dmu in idx_common:
#         df_new.loc[(dmu, year_t), inputs] = df_t.loc[dmu, inputs].values
#         df_new.loc[(dmu, year_t), outputs] = df_t.loc[dmu, outputs].values
#         df_new.loc[(dmu, year_t1), inputs] = df_t1.loc[dmu, inputs].values
#         df_new.loc[(dmu, year_t1), outputs] = df_t1.loc[dmu, outputs].values

#     m = len(inputs)
#     bit_tuples = list(itertools.product([0,1], repeat=m))
#     bit_labels = ['p' + ''.join(str(b) for b in bits) for bits in bit_tuples]

#     for bits, label in zip(bit_tuples, bit_labels):
#         # eval データ作成
#         eval_rows_t = []
#         eval_rows_t1 = []
#         for dmu in idx_common:
#             x_t = df_t.loc[dmu, inputs]
#             x_t1 = df_t1.loc[dmu, inputs]
#             x_combo_t  = _build_input_combo(x_t,  x_t1, bits)
#             x_combo_t1 = _build_input_combo(x_t,  x_t1, bits)

#             eval_rows_t.append(pd.Series(np.concatenate([x_combo_t,  df_t.loc[dmu, outputs].to_numpy()]),
#                                         index=list(inputs)+list(outputs), name=dmu))
#             eval_rows_t1.append(pd.Series(np.concatenate([x_combo_t1, df_t1.loc[dmu, outputs].to_numpy()]),
#                                          index=list(inputs)+list(outputs), name=dmu))

#         eval_df_t  = pd.DataFrame(eval_rows_t)
#         eval_df_t1 = pd.DataFrame(eval_rows_t1)

#         # φ を 4 通り計算
#         phi_t_on_t   = _evaluate_against_reference(df, year_t,  eval_df_t,  inputs, outputs,
#                                                    rts=rts, orientation=orientation, solver=solver)
#         phi_t1_on_t  = _evaluate_against_reference(df, year_t,  eval_df_t1, inputs, outputs,
#                                                    rts=rts, orientation=orientation, solver=solver)
#         phi_t_on_t1  = _evaluate_against_reference(df, year_t1, eval_df_t,  inputs, outputs,
#                                                    rts=rts, orientation=orientation, solver=solver)
#         phi_t1_on_t1 = _evaluate_against_reference(df, year_t1, eval_df_t1, inputs, outputs,
#                                                    rts=rts, orientation=orientation, solver=solver)

#         # φ 列名と投影列名を決定（phi_col と proj_col）
#         for f, (phi_eval_t, phi_eval_t1) in enumerate(((phi_t_on_t, phi_t1_on_t), (phi_t_on_t1, phi_t1_on_t1))):
#             phi_col = f"{label}_phi_onF{f}"
#             if phi_col not in df_new.columns:
#                 df_new[phi_col] = np.nan

#             # 各行に対応する φ を格納（eval 年に対応する φ をその行に入れる）
#             for dmu in idx_common:
#                 df_new.loc[(dmu, year_t),  phi_col] = phi_eval_t.loc[dmu]
#                 df_new.loc[(dmu, year_t1), phi_col] = phi_eval_t1.loc[dmu]

#             # 投影列名は "<phi_col>_<output>"
#             if include_proj_outputs:
#                 for out in outputs:
#                     proj_col = f"{phi_col}_{out}"
#                     if proj_col not in df_new.columns:
#                         df_new[proj_col] = np.nan

#                     # 各行について、その行の phi とその行の y を掛ける
#                     for dmu in idx_common:
#                         # year_t row
#                         y0 = df_t.loc[dmu, out]
#                         phi_at_t = df_new.loc[(dmu, year_t), phi_col]
#                         df_new.loc[(dmu, year_t), proj_col] = (phi_at_t * y0) if not pd.isna(phi_at_t) else np.nan

#                         # year_t1 row
#                         y1 = df_t1.loc[dmu, out]
#                         phi_at_t1 = df_new.loc[(dmu, year_t1), phi_col]
#                         df_new.loc[(dmu, year_t1), proj_col] = (phi_at_t1 * y1) if not pd.isna(phi_at_t1) else np.nan

#     # MI/TC/MI_k/EC の計算（従来ロジックを phiベースの列名に合わせて参照）
#     rows = []
#     for dmu in idx_common:
#         out = outputs[0]
#         p0 = 'p' + '0'*m
#         p1 = 'p' + '1'*m

#         F0_p0 = df_new.loc[(dmu, year_t),  f"{p0}_phi_onF0_{out}"]
#         F1_p0 = df_new.loc[(dmu, year_t1), f"{p0}_phi_onF1_{out}"]
#         F0_p1 = df_new.loc[(dmu, year_t),  f"{p1}_phi_onF0_{out}"]
#         F1_p1 = df_new.loc[(dmu, year_t1), f"{p1}_phi_onF1_{out}"]

#         EC = df_new.loc[(dmu, year_t),  f"{p0}_phi_onF0"]/df_new.loc[(dmu, year_t1),  f"{p1}_phi_onF1"]
#         TC = np.sqrt((F1_p0 / F0_p0) * (F1_p1 / F0_p1))
#         MI = np.sqrt((F0_p1 / F0_p0) * (F1_p1 / F1_p0))

#         MI_k_list = []
#         MI_calc = 1.0

#         if m == 2:
#             # p01, p10 の φ を取得
#             F0_p01 = df_new.loc[(dmu, year_t),  f'p01_phi_onF0_{out}']
#             F1_p01 = df_new.loc[(dmu, year_t1), f'p01_phi_onF1_{out}']
#             F0_p10 = df_new.loc[(dmu, year_t),  f'p10_phi_onF0_{out}']
#             F1_p10 = df_new.loc[(dmu, year_t1), f'p10_phi_onF1_{out}']

#             F0_p00 = df_new.loc[(dmu, year_t),  f'p00_phi_onF0_{out}']
#             F1_p00 = df_new.loc[(dmu, year_t1), f'p00_phi_onF1_{out}']
#             F0_p11 = df_new.loc[(dmu, year_t),  f'p11_phi_onF0_{out}']
#             F1_p11 = df_new.loc[(dmu, year_t1), f'p11_phi_onF1_{out}']

#             # --- MI0（input 0 の寄与） ---
#             MI0 = (
#                 (F0_p10 / F0_p00) * (F0_p11 / F0_p01) *
#                 (F1_p10 / F1_p00) * (F1_p11 / F1_p01)
#             ) ** 0.25

#             # --- MI1（input 1 の寄与） ---
#             MI1 = (
#                 (F0_p01 / F0_p00) * (F0_p11 / F0_p10) *
#                 (F1_p01 / F1_p00) * (F1_p11 / F1_p10)
#             ) ** 0.25

#             MI_k_list = [MI0, MI1]
#             MI_calc = MI0 * MI1
#         else:
#             for k in range(m):
#                 ratios = []
#                 for other_bits in itertools.product([0,1], repeat=m-1):
#                     bits0 = list(other_bits); bits0.insert(k, 0); label0 = 'p' + ''.join(str(b) for b in bits0)
#                     bits1 = list(other_bits); bits1.insert(k, 1); label1 = 'p' + ''.join(str(b) for b in bits1)

#                     F0_0 = df_new.loc[(dmu, year_t),  f"{label0}_phi_onF0_{out}"]
#                     F0_1 = df_new.loc[(dmu, year_t),  f"{label1}_phi_onF1_{out}"]
#                     F1_0 = df_new.loc[(dmu, year_t1), f"{label0}_phi_onF0_{out}"]
#                     F1_1 = df_new.loc[(dmu, year_t1), f"{label1}_phi_onF1_{out}"]

#                     ratios.append(F0_1 / F0_0)
#                     ratios.append(F1_1 / F1_0)

#                 MI_k = np.prod(ratios) ** (1.0 / len(ratios))
#                 MI_calc = MI_calc * MI_k
#                 MI_k = _cagr(MI_k, year_t, year_t1) 
#                 MI_k_list.append(MI_k)

#         y0 = df_new.loc[(dmu, year_t),  out]
#         y1 = df_new.loc[(dmu, year_t1), out]
#         EC_calc = y1 / y0 / TC / MI

#         row = [dmu, _cagr(EC, year_t, year_t1), _cagr(EC_calc, year_t, year_t1), _cagr(TC, year_t, year_t1), _cagr(MI, year_t, year_t1), _cagr(MI_calc, year_t, year_t1)] + MI_k_list
#         rows.append(row)

#     colnames = ['DMU', 'EC (%)', 'EC_calc (%)', 'TC (%)', 'MI (%)', 'MI_calc (%)'] + [f'MI{k} (%)' for k in range(m)]
#     df_mi = pd.DataFrame(rows, columns=colnames).set_index('DMU')

#     return df_new, df_mi


# _data0などの区別あり
# def dea_add_frontier_point_estimates(df, year_t, year_t1, inputs, outputs,
#                                      rts='VRS', orientation='output', solver=None,
#                                      label_level=None, include_proj_outputs=True, max_combos_warn=64):

#     df_t, _, X_t, Y_t = _prepare_df_for_year_v2(df, year_t, inputs, outputs, label_level=label_level)
#     df_t1, _, X_t1, Y_t1 = _prepare_df_for_year_v2(df, year_t1, inputs, outputs, label_level=label_level)

#     idx_common = df_t.index.intersection(df_t1.index)
#     if len(idx_common) == 0:
#         return pd.DataFrame(), pd.DataFrame()

#     index_new = pd.MultiIndex.from_product([idx_common, [year_t, year_t1]], names=['DMU', 'year'])
#     df_new = pd.DataFrame(index=index_new)

#     # 元の inputs / outputs をコピー
#     for dmu in idx_common:
#         df_new.loc[(dmu, year_t), inputs] = df_t.loc[dmu, inputs].values
#         df_new.loc[(dmu, year_t), outputs] = df_t.loc[dmu, outputs].values
#         df_new.loc[(dmu, year_t1), inputs] = df_t1.loc[dmu, inputs].values
#         df_new.loc[(dmu, year_t1), outputs] = df_t1.loc[dmu, outputs].values

#     m = len(inputs)
#     bit_tuples = list(itertools.product([0,1], repeat=m))
#     bit_labels = ['p' + ''.join(str(b) for b in bits) for bits in bit_tuples]

#     for bits, label in zip(bit_tuples, bit_labels):

#         eval_rows_t = []
#         eval_rows_t1 = []

#         for dmu in idx_common:
#             x_t = df_t.loc[dmu, inputs]
#             x_t1 = df_t1.loc[dmu, inputs]
#             x_combo = _build_input_combo(x_t, x_t1, bits)

#             eval_rows_t.append(pd.Series(
#                 np.concatenate([x_combo, df_t.loc[dmu, outputs].to_numpy()]),
#                 index=list(inputs)+list(outputs), name=dmu
#             ))
#             eval_rows_t1.append(pd.Series(
#                 np.concatenate([x_combo, df_t1.loc[dmu, outputs].to_numpy()]),
#                 index=list(inputs)+list(outputs), name=dmu
#             ))

#         eval_df_t = pd.DataFrame(eval_rows_t)
#         eval_df_t1 = pd.DataFrame(eval_rows_t1)

#         # φ 計算（4通り）
#         phi_t_on_t  = _evaluate_against_reference(df, year_t,  eval_df_t,  inputs, outputs,
#                                                   rts=rts, orientation=orientation, solver=solver)
#         phi_t1_on_t = _evaluate_against_reference(df, year_t,  eval_df_t1, inputs, outputs,
#                                                   rts=rts, orientation=orientation, solver=solver)

#         phi_t_on_t1  = _evaluate_against_reference(df, year_t1, eval_df_t,  inputs, outputs,
#                                                    rts=rts, orientation=orientation, solver=solver)
#         phi_t1_on_t1 = _evaluate_against_reference(df, year_t1, eval_df_t1, inputs, outputs,
#                                                    rts=rts, orientation=orientation, solver=solver)

#         # φ を保存（対応する eval の行にのみ格納）
#         for f, phi_pair in enumerate(((phi_t_on_t, phi_t1_on_t), (phi_t_on_t1, phi_t1_on_t1))):
#             col_phi = f"{label}_phi_onF{f}"
#             if col_phi not in df_new.columns:
#                 df_new[col_phi] = np.nan
#             phi_eval_t, phi_eval_t1 = phi_pair
#             for dmu in idx_common:
#                 df_new.loc[(dmu, year_t),  col_phi] = phi_eval_t.loc[dmu]
#                 df_new.loc[(dmu, year_t1), col_phi] = phi_eval_t1.loc[dmu]

#         # 投影出力を保存（対応するデータ年の行にのみ格納）
#         if include_proj_outputs:
#             for out in outputs:
#                 col_F0_label_F0 = f"F_data0_{label}_onF0_{out}"  # data=year_t on frontier0
#                 col_F1_label_F0 = f"F_data1_{label}_onF0_{out}"  # data=year_t1 on frontier0
#                 col_F0_label_F1 = f"F_data0_{label}_onF1_{out}"  # data=year_t on frontier1
#                 col_F1_label_F1 = f"F_data1_{label}_onF1_{out}"  # data=year_t1 on frontier1

#                 for col in (col_F0_label_F0, col_F1_label_F0, col_F0_label_F1, col_F1_label_F1):
#                     if col not in df_new.columns:
#                         df_new[col] = np.nan

#                 for dmu in idx_common:
#                     y0 = df_t.loc[dmu, out]
#                     y1 = df_t1.loc[dmu, out]

#                     F_data0_onF0 = phi_t_on_t.loc[dmu]   * y0
#                     F_data1_onF0 = phi_t1_on_t.loc[dmu]  * y1
#                     F_data0_onF1 = phi_t_on_t1.loc[dmu]  * y0
#                     F_data1_onF1 = phi_t1_on_t1.loc[dmu] * y1

#                     # store only on the corresponding data-year row
#                     df_new.loc[(dmu, year_t),  col_F0_label_F0] = F_data0_onF0
#                     df_new.loc[(dmu, year_t1), col_F1_label_F0] = F_data1_onF0

#                     df_new.loc[(dmu, year_t),  col_F0_label_F1] = F_data0_onF1
#                     df_new.loc[(dmu, year_t1), col_F1_label_F1] = F_data1_onF1

#     # ============================================================
#     # MI, TC, MI_k の計算（出力列名に合わせて取得）
#     # ============================================================
#     rows = []
#     for dmu in idx_common:
#         out = outputs[0]
#         p0 = 'p' + '0'*m
#         p1 = 'p' + '1'*m

#         F0_p0 = df_new.loc[(dmu, year_t),  f"F_data0_{p0}_onF0_{out}"]
#         F1_p0 = df_new.loc[(dmu, year_t1), f"F_data1_{p0}_onF0_{out}"]
#         F0_p1 = df_new.loc[(dmu, year_t),  f"F_data0_{p0}_onF1_{out}"]
#         F1_p1 = df_new.loc[(dmu, year_t1), f"F_data1_{p0}_onF1_{out}"]

#         TC = np.sqrt((F1_p0 / F0_p0) * (F1_p1 / F0_p1))
#         MI = np.sqrt((F0_p1 / F0_p0) * (F1_p1 / F1_p0))

#         MI_k_list = []
#         for k in range(m):
#             ratios = []
#             for other_bits in itertools.product([0,1], repeat=m-1):
#                 bits0 = list(other_bits); bits0.insert(k, 0); label0 = 'p' + ''.join(str(b) for b in bits0)
#                 bits1 = list(other_bits); bits1.insert(k, 1); label1 = 'p' + ''.join(str(b) for b in bits1)

#                 F0_0 = df_new.loc[(dmu, year_t),  f"F_data0_{label0}_onF0_{out}"]
#                 F0_1 = df_new.loc[(dmu, year_t),  f"F_data0_{label1}_onF0_{out}"]
#                 F1_0 = df_new.loc[(dmu, year_t1), f"F_data1_{label0}_onF0_{out}"]
#                 F1_1 = df_new.loc[(dmu, year_t1), f"F_data1_{label1}_onF0_{out}"]

#                 ratios.append(F0_1 / F0_0)
#                 ratios.append(F1_1 / F1_0)

#             MI_k = np.prod(ratios) ** (1.0 / len(ratios))
#             MI_k_list.append(np.log(MI_k) * 100)

#         y0 = df_new.loc[(dmu, year_t),  out]
#         y1 = df_new.loc[(dmu, year_t1), out]
#         EC = y1 / y0 / TC / MI

#         row = [dmu, np.log(EC)*100, np.log(TC)*100, np.log(MI)*100] + MI_k_list
#         rows.append(row)

#     colnames = ['DMU', 'EC (%)', 'TC (%)', 'MI (%)'] + [f'MI{k} (%)' for k in range(m)]
#     df_mi = pd.DataFrame(rows, columns=colnames).set_index('DMU')

#     return df_new, df_mi







# # EC とEFFの定義の違いによる混乱（復活、％へ、EC逆算）
# def dea_add_frontier_point_estimates(df, year_t, year_t1, inputs, outputs,
#                                      rts='VRS', orientation='output', solver=None,
#                                      label_level=None, include_proj_outputs=True, max_combos_warn=64):

#     # --- 年ごとの抽出 ---
#     df_t, _, X_t, Y_t = _prepare_df_for_year_v2(df, year_t, inputs, outputs, label_level=label_level)
#     df_t1, _, X_t1, Y_t1 = _prepare_df_for_year_v2(df, year_t1, inputs, outputs, label_level=label_level)

#     idx_common = df_t.index.intersection(df_t1.index)
#     if len(idx_common) == 0:
#         return pd.DataFrame(), pd.DataFrame()

#     # --- 新しい DataFrame の骨格 ---
#     index_new = pd.MultiIndex.from_product([idx_common, [year_t, year_t1]], names=['DMU', 'year'])
#     df_new = pd.DataFrame(index=index_new)

#     # 元の inputs / outputs をコピー
#     for dmu in idx_common:
#         df_new.loc[(dmu, year_t), inputs] = df_t.loc[dmu, inputs].values
#         df_new.loc[(dmu, year_t), outputs] = df_t.loc[dmu, outputs].values
#         df_new.loc[(dmu, year_t1), inputs] = df_t1.loc[dmu, inputs].values
#         df_new.loc[(dmu, year_t1), outputs] = df_t1.loc[dmu, outputs].values

#     # --- 全ビット組合せ ---
#     m = len(inputs)
#     bit_tuples = list(itertools.product([0,1], repeat=m))
#     bit_labels = ['p' + ''.join(str(b) for b in bits) for bits in bit_tuples]

#     # --- 各ビット組合せごとに計算 ---
#     for bits, label in zip(bit_tuples, bit_labels):

#         eval_rows_t = []
#         eval_rows_t1 = []

#         for dmu in idx_common:
#             x_t = df_t.loc[dmu, inputs]
#             x_t1 = df_t1.loc[dmu, inputs]
#             x_combo = _build_input_combo(x_t, x_t1, bits)

#             eval_rows_t.append(pd.Series(
#                 np.concatenate([x_combo, df_t.loc[dmu, outputs].to_numpy()]),
#                 index=list(inputs)+list(outputs), name=dmu
#             ))
#             eval_rows_t1.append(pd.Series(
#                 np.concatenate([x_combo, df_t1.loc[dmu, outputs].to_numpy()]),
#                 index=list(inputs)+list(outputs), name=dmu
#             ))

#         eval_df_t = pd.DataFrame(eval_rows_t)
#         eval_df_t1 = pd.DataFrame(eval_rows_t1)

#         # --- phi 計算 ---
#         phi_t_on_t = _evaluate_against_reference(df, year_t, eval_df_t, inputs, outputs,
#                                                  rts=rts, orientation=orientation, solver=solver)
#         phi_t_on_t1 = _evaluate_against_reference(df, year_t1, eval_df_t, inputs, outputs,
#                                                   rts=rts, orientation=orientation, solver=solver)

#         # --- φ を単一列に統合 ---
#         col_phi = f'{label}_phi'
#         df_new[col_phi] = np.nan

#         for dmu in idx_common:
#             df_new.loc[(dmu, year_t), col_phi] = phi_t_on_t.loc[dmu]
#             df_new.loc[(dmu, year_t1), col_phi] = phi_t_on_t1.loc[dmu]

#         # --- 投影出力 ---
#         if include_proj_outputs:
#             for out in outputs:
#                 col_proj = f'{label}_proj_{out}'
#                 df_new[col_proj] = np.nan

#                 for dmu in idx_common:
#                     y_orig = df_t.loc[dmu, out]
#                     df_new.loc[(dmu, year_t), col_proj] = phi_t_on_t.loc[dmu] * y_orig
#                     df_new.loc[(dmu, year_t1), col_proj] = phi_t_on_t1.loc[dmu] * y_orig

#     # ============================================================
#     # ここから第2戻り値（EC, TC, MI, MI_k）を計算
#     # ============================================================

#     rows = []

#     for dmu in idx_common:

#         # --- 出力名（1つを想定） ---
#         out = outputs[0]

#         # --- p000...0, p111...1 の投影アウトプットを取得 ---
#         p0 = 'p' + '0'*m
#         p1 = 'p' + '1'*m

#         F0_p0 = df_new.loc[(dmu, year_t),  f"{p0}_proj_{out}"]
#         F1_p0 = df_new.loc[(dmu, year_t1), f"{p0}_proj_{out}"]
#         F0_p1 = df_new.loc[(dmu, year_t),  f"{p1}_proj_{out}"]
#         F1_p1 = df_new.loc[(dmu, year_t1), f"{p1}_proj_{out}"]

#         # # --- EC（これは φ でOK） ---
#         # EC = df_new.loc[(dmu, year_t),  f"{p0}_phi"] / df_new.loc[(dmu, year_t1), f"{p1}_phi"]

#         # --- TC（投影アウトプットで計算） ---
#         TC = np.sqrt((F1_p0 / F0_p0) * (F1_p1 / F0_p1))

#         # --- MI（投影アウトプットで計算） ---
#         MI = np.sqrt((F0_p1 / F0_p0) * (F1_p1 / F1_p0))

#         # --- MI_k（投影アウトプットで計算） ---
#         MI_k_list = []

#         for k in range(m):
#             ratios = []
#             for other_bits in itertools.product([0,1], repeat=m-1):

#                 bits0 = list(other_bits)
#                 bits0.insert(k, 0)
#                 label0 = 'p' + ''.join(str(b) for b in bits0)

#                 bits1 = list(other_bits)
#                 bits1.insert(k, 1)
#                 label1 = 'p' + ''.join(str(b) for b in bits1)

#                 F0_0 = df_new.loc[(dmu, year_t),  f"{label0}_proj_{out}"]
#                 F0_1 = df_new.loc[(dmu, year_t),  f"{label1}_proj_{out}"]
#                 F1_0 = df_new.loc[(dmu, year_t1), f"{label0}_proj_{out}"]
#                 F1_1 = df_new.loc[(dmu, year_t1), f"{label1}_proj_{out}"]

#                 ratios.append(F0_1 / F0_0)
#                 ratios.append(F1_1 / F1_0)

#             n = len(ratios)
#             MI_k = np.prod(ratios) ** (1.0 / n)
#             MI_k_list.append(np.log(MI_k)*100)


#         # --- y1/y0 = EC * TC * MI の検算 ---
#         y0 = df_new.loc[(dmu, year_t), outputs[0]]
#         y1 = df_new.loc[(dmu, year_t1), outputs[0]]

#         EC = y1/ y0 / TC / MI


#         # lhs = y1 / y0
#         # rhs = EC * TC * MI
#         # diff = lhs - rhs

#         # print(f"[CHECK] {dmu}: y1/y0={lhs:.6f}, EC*TC*MI={rhs:.6f}, diff={diff:.6e}")


#         row = [dmu, np.log(EC)*100, np.log(TC)*100, np.log(MI)*100] + MI_k_list
#         rows.append(row)

#     colnames = ['DMU', 'EC (%)', 'TC (%)', 'MI (%)'] + [f'MI{k} (%)' for k in range(m)]
#     df_mi = pd.DataFrame(rows, columns=colnames).set_index('DMU')

#     return df_new, df_mi

# 自分でやってみるが、なにかおかしい
# def dea_add_frontier_point_estimates(df, year_t, year_t1, inputs, outputs,
#                                      rts='VRS', orientation='output', solver=None,
#                                      label_level=None, include_proj_outputs=True, max_combos_warn=64):

#     # --- 年ごとの抽出 ---
#     df_t, _, X_t, Y_t = _prepare_df_for_year_v2(df, year_t, inputs, outputs, label_level=label_level)
#     df_t1, _, X_t1, Y_t1 = _prepare_df_for_year_v2(df, year_t1, inputs, outputs, label_level=label_level)

#     idx_common = df_t.index.intersection(df_t1.index)
#     if len(idx_common) == 0:
#         return pd.DataFrame(), pd.DataFrame()

#     # --- 新しい DataFrame の骨格 ---
#     index_new = pd.MultiIndex.from_product([idx_common, [year_t, year_t1]], names=['DMU', 'year'])
#     df_new = pd.DataFrame(index=index_new)

#     # 元の inputs / outputs をコピー
#     for dmu in idx_common:
#         df_new.loc[(dmu, year_t), inputs] = df_t.loc[dmu, inputs].values
#         df_new.loc[(dmu, year_t), outputs] = df_t.loc[dmu, outputs].values
#         df_new.loc[(dmu, year_t1), inputs] = df_t1.loc[dmu, inputs].values
#         df_new.loc[(dmu, year_t1), outputs] = df_t1.loc[dmu, outputs].values

#     # --- 全ビット組合せ ---
#     m = len(inputs)
#     bit_tuples = list(itertools.product([0,1], repeat=m))
#     bit_labels = ['p' + ''.join(str(b) for b in bits) for bits in bit_tuples]

#     # --- 各ビット組合せごとに計算 ---
#     for bits, label in zip(bit_tuples, bit_labels):

#         eval_rows_t = []
#         eval_rows_t1 = []

#         for dmu in idx_common:
#             x_t = df_t.loc[dmu, inputs]
#             x_t1 = df_t1.loc[dmu, inputs]
#             x_combo = _build_input_combo(x_t, x_t1, bits)

#             eval_rows_t.append(pd.Series(
#                 np.concatenate([x_combo, df_t.loc[dmu, outputs].to_numpy()]),
#                 index=list(inputs)+list(outputs), name=dmu
#             ))
#             eval_rows_t1.append(pd.Series(
#                 np.concatenate([x_combo, df_t1.loc[dmu, outputs].to_numpy()]),
#                 index=list(inputs)+list(outputs), name=dmu
#             ))

#         eval_df_t = pd.DataFrame(eval_rows_t)
#         eval_df_t1 = pd.DataFrame(eval_rows_t1)

#         # --- phi 計算 ---
#         phi_t_on_t = _evaluate_against_reference(df, year_t, eval_df_t, inputs, outputs,
#                                                  rts=rts, orientation=orientation, solver=solver)
#         phi_t_on_t1 = _evaluate_against_reference(df, year_t1, eval_df_t1, inputs, outputs,
#                                                   rts=rts, orientation=orientation, solver=solver)

#         # --- φ を単一列に統合 ---
#         col_phi = f'{label}_phi'
#         df_new[col_phi] = np.nan

#         for dmu in idx_common:
#             df_new.loc[(dmu, year_t), col_phi] = phi_t_on_t.loc[dmu]
#             df_new.loc[(dmu, year_t1), col_phi] = phi_t_on_t1.loc[dmu]

#         # --- 投影出力 ---
#         if include_proj_outputs:
#             for out in outputs:
#                 col_proj = f'{label}_proj_{out}'
#                 df_new[col_proj] = np.nan

#                 for dmu in idx_common:
#                     y_orig = df_t.loc[dmu, out]
#                     df_new.loc[(dmu, year_t), col_proj] = phi_t_on_t.loc[dmu] * y_orig
#                     df_new.loc[(dmu, year_t1), col_proj] = phi_t_on_t1.loc[dmu] * y_orig

#     # ============================================================
#     # ここから第2戻り値（EC, TC, MI, MI_k）を計算
#     # ============================================================

#     rows = []

#     for dmu in idx_common:

#         out = outputs[0]

#         # 実アウトプット
#         y0 = df_new.loc[(dmu, year_t),  out]
#         y1 = df_new.loc[(dmu, year_t1), out]

#         # p000...0, p111...1
#         p0 = 'p' + '0'*m
#         p1 = 'p' + '1'*m

#         # --- EC（φベースのままでOK） ---
#         EC = df_new.loc[(dmu, year_t),  f"{p0}_phi"] / df_new.loc[(dmu, year_t1), f"{p1}_phi"]

#         # F0(p) = φ0(p) * y0
#         F0_p0 = df_new.loc[(dmu, year_t),  f"{p0}_phi"] * y0
#         F0_p1 = df_new.loc[(dmu, year_t),  f"{p1}_phi"] * y1

#         # F1(p) = φ1(p) * y1
#         F1_p0 = df_new.loc[(dmu, year_t1), f"{p0}_phi"] * y0
#         F1_p1 = df_new.loc[(dmu, year_t1), f"{p1}_phi"] * y1

#         TC = np.sqrt((F1_p0 / F0_p0) * (F1_p1 / F0_p1))

#         MI = np.sqrt((F0_p1 / F0_p0) * (F1_p1 / F1_p0))

#         MI_k_list = []

#         if m == 2:
#             # p01, p10 の φ を取得
#             F0_p01 = df_new.loc[(dmu, year_t),  f'p01_proj_{out}']
#             F1_p01 = df_new.loc[(dmu, year_t1), f'p01_proj_{out}']
#             F0_p10 = df_new.loc[(dmu, year_t),  f'p10_proj_{out}']
#             F1_p10 = df_new.loc[(dmu, year_t1), f'p10_proj_{out}']

#             F0_p00 = df_new.loc[(dmu, year_t),  f'p00_proj_{out}']
#             F1_p00 = df_new.loc[(dmu, year_t1), f'p00_proj_{out}']
#             F0_p11 = df_new.loc[(dmu, year_t),  f'p11_proj_{out}']
#             F1_p11 = df_new.loc[(dmu, year_t1), f'p11_proj_{out}']

#             # --- MI0（input 0 の寄与） ---
#             MI0 = (
#                 (F0_p10 / F0_p00) * (F0_p11 / F0_p01) *
#                 (F1_p10 / F1_p00) * (F1_p11 / F1_p01)
#             ) ** 0.25

#             # --- MI1（input 1 の寄与） ---
#             MI1 = (
#                 (F0_p01 / F0_p00) * (F0_p11 / F0_p10) *
#                 (F1_p01 / F1_p00) * (F1_p11 / F1_p10)
#             ) ** 0.25

#             MI_k_list = [MI0, MI1]
#             # MI = MI0 * MI1

#         # --- 検算 ---
#         lhs = y1 / y0
#         rhs = EC * TC * MI
#         diff = lhs - rhs
#         print(f"[CHECK] {dmu}: y1/y0={lhs:.6f}, EC*TC*MI={rhs:.6f}, diff={diff:.6e}")

#         row = [dmu, EC, TC, MI] + MI_k_list
#         rows.append(row)

#     colnames = ['DMU', 'EC', 'TC', 'MI'] + [f'MI{k}' for k in range(m)]
#     df_mi = pd.DataFrame(rows, columns=colnames).set_index('DMU')

#     return df_new, df_mi



