import numpy as np
import pandas as pd
import cvxpy as cp

import itertools
import math

try:
    from tabulate import tabulate
except ImportError:
    tabulate = None


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


# --- 追加ヘルパー: 参照フロンティア（ref_year）の X_ref, Y_ref に対して任意の評価点群を評価する ---
def _evaluate_against_reference_xy(X_ref, Y_ref, eval_df, inputs, outputs,
                                  rts='VRS', orientation='output', solver=None):
    """
    参照フロンティアを (X_ref, Y_ref) として直接受け取り、
    eval_df の各行 (inputs, outputs) を評価して φ (or θ) を返す。
    戻り値: pandas.Series (index = eval_df.index)
    """
    n_ref = X_ref.shape[1]

    scores = pd.Series(index=eval_df.index, dtype=float)

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

def _cagr(num, year_t=0, year_t1=1, label=None, total_growth=False):
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
    
    # 4. 期間のチェック (None の場合も期間が不正なら落とす仕様を維持)
    delta_t = year_t1 - year_t
    if delta_t <= 0:
        print(f"[WARN]{info} Invalid period: delta_t={delta_t}")
        return np.nan
    
    # --- 戻り値の分岐処理 ---
    if total_growth is None:
        # 【追加】バリデーション通過後、計算せずにそのまま返す
        return num
    elif total_growth:
        # 累積成長率を返す (True の場合)
        return (num - 1) * 100
    else:
        # 標準的なCAGRの計算式 (False の場合)
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


def _get_F_value(df_new, dmu, year_t, out, label, frontier_id):
    """
    F(p, t) を返すヘルパー。
    - frontier_id: 0 -> period t frontier, 1 -> period t+1 frontier
    - まず proj 列（phi_col_out）があればそれを使う。
    - なければ phi 列 × y(year_t) で F を復元する。
      （あなたの実装では phi は y(year_t) を使って評価しているため一致する）
    """
    phi_col = f"{label}_phi_onF{frontier_id}"
    proj_col = f"{phi_col}_{out}"

    if proj_col in df_new.columns:
        return df_new.loc[(dmu, year_t), proj_col]

    # proj 列が無い場合は φ×y で復元
    y_base = df_new.loc[(dmu, year_t), out]
    phi = df_new.loc[(dmu, year_t), phi_col]
    if pd.isna(phi) or pd.isna(y_base):
        return np.nan
    return float(phi) * float(y_base)


def dea_add_frontier_point_estimates(df, year_t, year_t1, inputs, outputs,
                                                  rts='VRS', orientation='output', solver=None,
                                                  label_level=None, debug=False, keep_columns=None, total_growth=False, **kwargs):
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
    debug : bool, default False
        True の場合、ACCUM_calc や EC_calc など内部検算用の列を
        df_mi に追加する。
    keep_columns : list of str, optional
        df_new に引き継ぎたい元のデータフレームの列名リスト。
    total_growth : bool or None, default False
        df_mi の結果列（成長率・指数）の計算方法と単位を指定する。
        - False (デフォルト): 年平均成長率 (CAGR %) として算出。
        - True: 期間全体の累積成長率 (%) として算出。
        - None: 演算（-1 や指数計算）を行わず、元の倍率 (raw ratio) のまま算出。

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
        表示単位は `total_growth` の値により次のように動的に変化する：
        - total_growth=None  : 'raw ratio' (例: 1.05)
        - total_growth=True  : 'Total %'    (例: 5.0)
        - total_growth=False : 'CAGR %'     (例: 2.47)

        含まれる主な指標：
        - Total   : y1/y0 の変化
        - TFP     : EFF * TECH（総合的生産性変化）
        - EFF     : 効率変化 (Efficiency Change)
        - TECH    : 技術変化 (Technical Change)
        - ACCUM   : 入力量の蓄積効果 (Accumulation Effect)
        - ACCUM_k : 各入力 k の寄与（m > 1 の場合）
        ※ debug=True の場合、内部検算用の列も追加される。

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
        ACCUM_k は、入力 k のビットだけを変化させたパターン間の比を幾何平均して求める。
      ※ total_growth が None 以外の場合は、これらに対して期間 (year_t1 - year_t) 
         に基づく CAGR または累積成長率の変換が行われる。

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

        # φ を 4 通り計算（参照フロンティアは X_t/Y_t, X_t1/Y_t1 を使い回す）
        phi_t_on_t   = _evaluate_against_reference_xy(X_t,  Y_t,  eval_df_t,  inputs, outputs,
                                                    rts=rts, orientation=orientation, solver=solver)
        phi_t1_on_t  = _evaluate_against_reference_xy(X_t,  Y_t,  eval_df_t1, inputs, outputs,
                                                    rts=rts, orientation=orientation, solver=solver)
        phi_t_on_t1  = _evaluate_against_reference_xy(X_t1, Y_t1, eval_df_t,  inputs, outputs,
                                                    rts=rts, orientation=orientation, solver=solver)
        phi_t1_on_t1 = _evaluate_against_reference_xy(X_t1, Y_t1, eval_df_t1, inputs, outputs,
                                                    rts=rts, orientation=orientation, solver=solver)

        for f, (phi_eval_t, phi_eval_t1) in enumerate(((phi_t_on_t, phi_t1_on_t), (phi_t_on_t1, phi_t1_on_t1))):
            phi_col = f"{label}_phi_onF{f}"
            if phi_col not in df_new.columns:
                df_new[phi_col] = np.nan

            for dmu in idx_common:
                df_new.loc[(dmu, year_t), phi_col] = phi_eval_t.loc[dmu]
                df_new.loc[(dmu, year_t1), phi_col] = phi_eval_t1.loc[dmu]

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

    # --- via p10 / via p01 と symm(G1,G2) の比較表 ---
    path_compare_rows = []

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

        ACCUM_pre = ACCUM  # NaN補正前のACCUM（比較用）

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

        # --- 新設: TFP = EFF * TECH --- 2026-03-21 00:46:10に変更
        TFP = EFF * TECH

        # --- ここから ACCUM_k 計算の置き換え本体 ---
        ACCUM_k_list = []
        ACCUM_calc = 1.0

        # --- ∏G_k を作るために raw の G_k（=ACCUM_k）も保持 ---
        ACCUM_k_raw_list = []

        if m == 1:
            ACCUM_calc = ACCUM
        else:
            m_fact = math.factorial(m)  # m!
            for k in range(m):

                # 論文の定義：
                # G_k = Π_{σ∈S_m} ( r^{(t)}_{σ,posσ(k)} r^{(t+1)}_{σ,posσ(k)} )^{1/(2m!)}
                # を「直前集合 S の重み |S|!(m-|S|-1)!」で集計して実装する [1](https://kwanseio365-my.sharepoint.com/personal/fjn61288_nuc_kwansei_ac_jp/Documents/Microsoft%20Copilot%20%E3%83%81%E3%83%A3%E3%83%83%E3%83%88%20%E3%83%95%E3%82%A1%E3%82%A4%E3%83%AB/ForJPA_v3.pdf)

                log_sum = 0.0
                valid = True

                for other_bits in itertools.product([0, 1], repeat=m - 1):
                    s = int(sum(other_bits))  # |S|
                    weight = math.factorial(s) * math.factorial(m - s - 1)  # |S|!(m-|S|-1)!

                    # bits0: kが0、他は other_bits
                    bits0 = list(other_bits)
                    bits0.insert(k, 0)
                    label0 = 'p' + ''.join(str(b) for b in bits0)

                    # bits1: kが1、他は other_bits
                    bits1 = list(other_bits)
                    bits1.insert(k, 1)
                    label1 = 'p' + ''.join(str(b) for b in bits1)

                    # F_t(·) と F_{t+1}(·) を取得（年は year_t 行を統一的に使用）
                    F0_0 = _get_F_value(df_new, dmu, year_t, out, label0, frontier_id=0)  # F(label0, t)
                    F0_1 = _get_F_value(df_new, dmu, year_t, out, label1, frontier_id=0)  # F(label1, t)

                    F1_0 = _get_F_value(df_new, dmu, year_t, out, label0, frontier_id=1)  # F(label0, t+1)
                    F1_1 = _get_F_value(df_new, dmu, year_t, out, label1, frontier_id=1)  # F(label1, t+1)

                    r_t  = _safe_div(F0_1, F0_0)
                    r_t1 = _safe_div(F1_1, F1_0)

                    # どれかが欠損/非正なら G_k は定義不能（log が取れない/分母ゼロ等）
                    if pd.isna(r_t) or pd.isna(r_t1) or (r_t <= 0) or (r_t1 <= 0):
                        valid = False
                        if debug:
                            print(f"[WARN][{dmu}] ACCUM_k invalid for k={k}, other_bits={other_bits}, "
                                f"r_t={r_t}, r_t1={r_t1}")
                        break

                    # log 空間で重み付き加算： (1/(2m!)) Σ weight*(log r_t + log r_t1)
                    log_sum += weight * (np.log(r_t) + np.log(r_t1))

                if not valid:
                    ACCUM_k = np.nan
                else:
                    ACCUM_k = float(np.exp(log_sum / (2.0 * m_fact)))  # 2m! で割る（両期間分）[1](https://kwanseio365-my.sharepoint.com/personal/fjn61288_nuc_kwansei_ac_jp/Documents/Microsoft%20Copilot%20%E3%83%81%E3%83%A3%E3%83%83%E3%83%88%20%E3%83%95%E3%82%A1%E3%82%A4%E3%83%AB/ForJPA_v3.pdf)

                if debug:
                    ACCUM_calc = ACCUM_calc * (ACCUM_k if not pd.isna(ACCUM_k) else 1.0)

                ACCUM_k_cagr = _cagr(ACCUM_k, year_t, year_t1, dmu, total_growth) if not pd.isna(ACCUM_k) else np.nan
                ACCUM_k_list.append(ACCUM_k_cagr)
                ACCUM_k_raw_list.append(ACCUM_k)
        # --- 置き換えここまで ---

        # ===== via p10 / via p01 の分解値と symm(G1,G2) を比較（m==2のみ） =====
        if m == 2:
            out0 = outputs[0]  # 既に out = outputs[0] を使っている前提なら out でOK

            # F_t(·) と F_{t+1}(·) を取得（あなたの _get_F_value を利用）
            F0_00 = _get_F_value(df_new, dmu, year_t, out0, "p00", frontier_id=0)
            F0_10 = _get_F_value(df_new, dmu, year_t, out0, "p10", frontier_id=0)
            F0_01 = _get_F_value(df_new, dmu, year_t, out0, "p01", frontier_id=0)
            F0_11 = _get_F_value(df_new, dmu, year_t, out0, "p11", frontier_id=0)

            F1_00 = _get_F_value(df_new, dmu, year_t, out0, "p00", frontier_id=1)
            F1_10 = _get_F_value(df_new, dmu, year_t, out0, "p10", frontier_id=1)
            F1_01 = _get_F_value(df_new, dmu, year_t, out0, "p01", frontier_id=1)
            F1_11 = _get_F_value(df_new, dmu, year_t, out0, "p11", frontier_id=1)

            # 安全割り算（あなたの _safe_div を利用）
            r0_10_00 = _safe_div(F0_10, F0_00)  # Ft: p10/p00
            r1_10_00 = _safe_div(F1_10, F1_00)  # Ft1: p10/p00
            r0_11_10 = _safe_div(F0_11, F0_10)  # Ft: p11/p10
            r1_11_10 = _safe_div(F1_11, F1_10)  # Ft1: p11/p10

            r0_01_00 = _safe_div(F0_01, F0_00)  # Ft: p01/p00
            r1_01_00 = _safe_div(F1_01, F1_00)  # Ft1: p01/p00
            r0_11_01 = _safe_div(F0_11, F0_01)  # Ft: p11/p01
            r1_11_01 = _safe_div(F1_11, F1_01)  # Ft1: p11/p01

            def _geom2(a, b):
                if pd.isna(a) or pd.isna(b) or a <= 0 or b <= 0:
                    return np.nan
                return float(np.sqrt(a * b))

            # 経路A（via p10）：入力1→入力2
            G1_p10 = _geom2(r0_10_00, r1_10_00)
            G2_p10 = _geom2(r0_11_10, r1_11_10)

            # 経路B（via p01）：入力2→入力1
            G2_p01 = _geom2(r0_01_00, r1_01_00)
            G1_p01 = _geom2(r0_11_01, r1_11_01)

            # 対称化（あなたの実装で得た G1,G2）
            G1_symm = ACCUM_k_raw_list[0] if len(ACCUM_k_raw_list) >= 1 else np.nan
            G2_symm = ACCUM_k_raw_list[1] if len(ACCUM_k_raw_list) >= 2 else np.nan

            # 差分を先に作る（NaNならNaN）
            dG1_p10 = (G1_symm - G1_p10) if (not pd.isna(G1_symm) and not pd.isna(G1_p10)) else np.nan
            dG2_p10 = (G2_symm - G2_p10) if (not pd.isna(G2_symm) and not pd.isna(G2_p10)) else np.nan
            dG1_p01 = (G1_symm - G1_p01) if (not pd.isna(G1_symm) and not pd.isna(G1_p01)) else np.nan
            dG2_p01 = (G2_symm - G2_p01) if (not pd.isna(G2_symm) and not pd.isna(G2_p01)) else np.nan

            # gap = max abs（4つの差分の絶対値の最大）
            _abs_list = []
            for v in (dG1_p10, dG2_p10, dG1_p01, dG2_p01):
                if not pd.isna(v):
                    _abs_list.append(abs(v))
            gap = max(_abs_list) if len(_abs_list) > 0 else np.nan

            path_compare_rows.append({
                orig_dmu_name: dmu,
                "year_t": year_t,
                "year_t1": year_t1,
                "ACCUM_post": float(ACCUM) if not pd.isna(ACCUM) else np.nan,

                "G1_symm": G1_symm,
                "G2_symm": G2_symm,

                "G1_via_p10": G1_p10,
                "G2_via_p10": G2_p10,

                "G1_via_p01": G1_p01,
                "G2_via_p01": G2_p01,

                "dG1(symm-p10)": dG1_p10,
                "dG2(symm-p10)": dG2_p10,
                "dG1(symm-p01)": dG1_p01,
                "dG2(symm-p01)": dG2_p01,

                "gap(max|dG|)": gap
            })
        # ===== ここまで =====

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
            tfp_cagr = _cagr(TFP, year_t, year_t1, dmu, total_growth) # 2026-03-21 00:46:10に変更
            accum_calc_cagr = _cagr(ACCUM_calc, year_t, year_t1, dmu, total_growth)
            ec_calc_cagr = _cagr(EC_calc, year_t, year_t1, dmu, total_growth)

            # # --- 新設: TFP = EFF_cagr + TECH_cagr ---
            # tfp_cagr = eff_cagr + tech_cagr # 2026-03-21 00:46:10に変更(コメント化)

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
            tfp_cagr = _cagr(TFP, year_t, year_t1, dmu, total_growth) # 2026-03-21 00:46:10に変更

            # # --- 新設: TFP = EFF_cagr + TECH_cagr ---
            # tfp_cagr = eff_cagr + tech_cagr # 2026-03-21 00:46:10に変更(コメント化)

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

    _unit = 'raw ratio' if total_growth is None else ('Total %' if total_growth else 'CAGR %')
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

    # --- via p10 / via p01 と symm(G1,G2) の比較表を出力（m==2のみ） ---
    df_path_compare = pd.DataFrame(path_compare_rows)
    if not df_path_compare.empty and not kwargs.get("return_path_compare", False):
        df_path_compare = df_path_compare.set_index(orig_dmu_name)

        print("\n[CHECK] Path-dependent decomposition vs Symmetrized (G1,G2)")
        if tabulate is not None:
            print(tabulate(df_path_compare, headers="keys", tablefmt="github",
                        showindex=True, floatfmt=".6f"))
        else:
            print(df_path_compare.to_string(float_format=lambda x: f"{x:.6f}"))


    # df_gap_only = df_path_compare[["gap(max|dG|)"]].copy()
    # print("\n[CHECK] gap only (for manual sorting later)")
    # if tabulate is not None:
    #     print(tabulate(df_gap_only, headers="keys", tablefmt="github",
    #                 showindex=True, floatfmt=".6f"))
    # else:
    #     print(df_gap_only.to_string(float_format=lambda x: f"{x:.6f}"))

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

    if kwargs.get("return_path_compare", False):
        # 既に return_accum_check を付けているなら、返し方は要調整（どちらを優先するか）
        return df_new, df_mi, df_path_compare

    return df_new, df_mi


