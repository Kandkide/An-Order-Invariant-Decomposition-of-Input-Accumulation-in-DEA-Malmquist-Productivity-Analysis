import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import pandas as pd
import numpy as np

try:
    from scipy.spatial import ConvexHull
except Exception:
    ConvexHull = None


def _get_per_param(param, key, idx, n):
    if param is None:
        return None
    if isinstance(param, dict):
        return param.get(key, None)
    if isinstance(param, (list, tuple)):
        return param[idx] if idx < len(param) else param[-1]
    return param


def _broadcast_to_list(x, n):
    if isinstance(x, (list, tuple)):
        if len(x) >= n:
            return list(x[:n])
        else:
            return list(x) + [x[-1]] * (n - len(x))
    else:
        return [x] * n


def _plot_envelope_convex(ax, pts, face_color='lightgrey', alpha=0.4, edge_color='k', edge_width=0):
    if ConvexHull is None:
        raise RuntimeError("scipy is required for envelope plotting (ConvexHull).")
    if pts.shape[0] < 4:
        return
    hull = ConvexHull(pts)
    faces = hull.simplices
    tri_verts = [pts[s] for s in faces]
    ec = 'none' if edge_color is None or edge_width == 0 else edge_color
    lw = edge_width if edge_width is not None else 0
    poly = Poly3DCollection(tri_verts, facecolor=face_color, edgecolor=ec, linewidths=lw, alpha=alpha)
    if ec == 'none' or lw == 0:
        try:
            poly.set_edgecolor('none')
            poly.set_linewidth(0)
        except Exception:
            pass
    ax.add_collection3d(poly)
    ax.auto_scale_xyz(pts[:,0], pts[:,1], pts[:,2])


def _resolve_title_and_labels(title_and_labels, x_column, y_column, z_column):
    title = xlabel = ylabel = zlabel = None
    if title_and_labels is not None:
        if isinstance(title_and_labels, dict):
            title = title_and_labels.get('title', None)
            xlabel = title_and_labels.get('xlabel', None)
            ylabel = title_and_labels.get('ylabel', None)
            zlabel = title_and_labels.get('zlabel', None)
        elif isinstance(title_and_labels, (list, tuple)):
            vals = list(title_and_labels) + [None] * (4 - len(title_and_labels))
            title, xlabel, ylabel, zlabel = vals[:4]
        if title is None:
            title = f"{x_column} vs {y_column} vs {z_column} (3D scatter)"
        if xlabel is None:
            xlabel = x_column
        if ylabel is None:
            ylabel = y_column
        if zlabel is None:
            zlabel = z_column
    else:
        title = f"{x_column} vs {y_column} vs {z_column} (3D scatter)"
        xlabel = x_column
        ylabel = y_column
        zlabel = z_column
    return title, xlabel, ylabel, zlabel


def plot_multiindex_scatter_3d_overlay(
    df,
    level=0,
    x_column=None,
    y_column=None,
    z_column=None,
    label_column=None,
    label_condition=None,
    label_kwargs=None,
    color=None,
    marker=None,
    size=None,
    envelope=False,
    envelope_groupwise=True,
    envelope_facecolor=None,
    envelope_alpha=0.25,
    envelope_edgecolor='k',
    envelope_edgewidth=0,
    figsize=(8,6),
    elev=0,
    azim=-90,
    show_legend=True,
    title_and_labels=None,
    show_ticks=True,
    **kwargs
):
    """
    3次元散布図を複数レイヤーで重ねて描画するユーティリティ。

    概要
    ----
    - DataFrame の指定列を x/y/z に割り当て、複数のプロット仕様を同一 Axes に順次重ねて描画します。
    - 各引数はスカラー（単一プロット）またはリスト（複数プロット）で与えられます。
    - envelope 関連の引数（envelope, envelope_groupwise, envelope_facecolor, envelope_alpha, envelope_edgecolor, envelope_edgewidth）は
      各プロットごとに個別指定可能です（リストで与えるとプロット毎に適用、短いリストは末尾要素でブロードキャスト）。

    主な引数
    -------------
    - df: pandas.DataFrame（マルチインデックスでも可。内部で reset_index して扱います）
    - level: グルーピングに使うインデックスレベル（デフォルト 0）
    - x_column, y_column, z_column: 列名または列名のリスト（リスト長 = 重ねるプロット数）
    - label_column: ラベル列名またはリスト
    - label_condition: ラベル表示条件（'all' / callable / 値リスト / 単一値）またはリスト
    - label_kwargs: ax.text に渡す辞書（フォントサイズ等）またはリスト
    - color, marker, size: 描画スタイル（スカラーまたはリスト）
    - envelope: bool または bool のリスト。True のプロットは包絡面を描画（個別指定可）
    - envelope_groupwise: bool またはリスト。True ならグループ毎に包絡、False なら全体包絡に含める
    - title_and_labels: None / list / dict。None はデフォルトラベル、'' は非表示、None 要素は列名で埋める
    - show_ticks: 目盛り表示の有無

    挙動
    ----
    - リスト引数が混在する場合、関数は最長のリスト長を基準にブロードキャストして複数プロットを作成します。
    - envelope_groupwise=False のプロットは全体包絡用の点集合に追加され、最後にまとめて一つの包絡面を描きます。
    - label_condition が callable の場合は (labels, x, y, z) を受け取り、同じインデックスを持つ boolean Series を返すことを期待します。

    使用例
    ------
    # 単一プロット（従来どおり）
    plot_multiindex_scatter_3d_overlay(
        df,
        x_column='proj_rn_per_hour_worked_DRS',
        y_column='proj_hc_DRS',
        z_column='proj_rgdpo_per_hour_worked_DRS',
        envelope=True,
        envelope_facecolor='C0',
        envelope_alpha=0.2
    )

    # 2つのプロットを重ねる（各プロットで色・マーカー・包絡の有無を指定）
    plot_multiindex_scatter_3d_overlay(
        df,
        x_column=['proj_rn_per_hour_worked_DRS','proj_rn_per_hour_worked_DRS'],
        y_column=['proj_hc_DRS','proj_hc_DRS'],
        z_column=['proj_rgdpo_per_hour_worked_DRS','proj_rgdpo_per_hour_worked_DRS'],
        color=['C0','C1'],
        marker=['o','^'],
        size=[30, 50],
        envelope=[True, False],
        envelope_groupwise=[True, False],
        envelope_facecolor=['C0', None],
        envelope_alpha=[0.25, 0.1],
        title_and_labels=['', 'RN per hour', 'HC', 'RGDPO']
    )
    """
    if x_column is None or y_column is None or z_column is None:
        raise ValueError("x_column, y_column, z_column を指定してください。")

    label_kwargs = label_kwargs or {}
    df_reset = df.reset_index()

    if label_column is None and level is not None:
        if isinstance(level, int) and level < len(df.index.names):
            name = df.index.names[level]
            if name is not None and name in df_reset.columns:
                label_column = name
        if label_column is None:
            label_column = df_reset.columns[0]

    level_name = None
    if level is not None:
        level_name = df.index.names[level] if isinstance(level, int) else str(level)
        if level_name not in df_reset.columns:
            level_name = None

    # --- determine number of overlaid plots by max length among list params ---
    candidate_params = [x_column, y_column, z_column, label_column, label_condition, label_kwargs, color, marker, size, envelope, envelope_groupwise, envelope_facecolor, envelope_alpha, envelope_edgecolor, envelope_edgewidth]
    lengths = [len(p) for p in candidate_params if isinstance(p, (list, tuple))]
    n_plots = max(lengths) if lengths else 1

    # broadcast all params to length n_plots
    x_list = _broadcast_to_list(x_column, n_plots)
    y_list = _broadcast_to_list(y_column, n_plots)
    z_list = _broadcast_to_list(z_column, n_plots)
    label_col_list = _broadcast_to_list(label_column, n_plots)
    label_cond_list = _broadcast_to_list(label_condition, n_plots)
    label_kwargs_list = _broadcast_to_list(label_kwargs, n_plots)
    color_list = _broadcast_to_list(color, n_plots)
    marker_list = _broadcast_to_list(marker, n_plots)
    size_list = _broadcast_to_list(size, n_plots)
    # envelope-related broadcast
    envelope_list = _broadcast_to_list(envelope, n_plots)
    envelope_groupwise_list = _broadcast_to_list(envelope_groupwise, n_plots)
    envelope_facecolor_list = _broadcast_to_list(envelope_facecolor, n_plots)
    envelope_alpha_list = _broadcast_to_list(envelope_alpha, n_plots)
    envelope_edgecolor_list = _broadcast_to_list(envelope_edgecolor, n_plots)
    envelope_edgewidth_list = _broadcast_to_list(envelope_edgewidth, n_plots)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=elev, azim=azim)

    palette = plt.rcParams.get('axes.prop_cycle').by_key().get('color', ['C0','C1','C2','C3'])
    default_markers = ['o','s','^','D','v','P','X','*']

    all_pts_for_global_envelope = []

    for k in range(n_plots):
        x_col = x_list[k]
        y_col = y_list[k]
        z_col = z_list[k]
        lc = label_col_list[k]
        lcond = label_cond_list[k]
        lkwargs = label_kwargs_list[k] or {}
        col = color_list[k] or palette[k % len(palette)]
        m = marker_list[k] or default_markers[k % len(default_markers)]
        s = size_list[k] or 30

        env_flag = bool(envelope_list[k])
        env_groupwise = bool(envelope_groupwise_list[k])
        env_fc = envelope_facecolor_list[k]
        env_alpha = envelope_alpha_list[k]
        env_ec = envelope_edgecolor_list[k]
        env_ew = envelope_edgewidth_list[k]

        if level_name is not None:
            groups = list(df_reset.groupby(level_name))
        else:
            groups = [(None, df_reset)]

        for i, (group, data) in enumerate(groups):
            x_all = pd.to_numeric(data[x_col], errors='coerce')
            y_all = pd.to_numeric(data[y_col], errors='coerce')
            z_all = pd.to_numeric(data[z_col], errors='coerce')

            valid = x_all.notna() & y_all.notna() & z_all.notna()
            x = x_all[valid]; y = y_all[valid]; z = z_all[valid]
            labels = data.loc[valid, lc] if (lc is not None and lc in data.columns) else None

            if len(x) > 0:
                ax.scatter(x.values, y.values, z.values, c=[col], marker=m, s=s,
                           label=(f"{group} - plot{k}" if level_name is not None else f"plot{k}"),
                           depthshade=True, **kwargs)

            if labels is not None and lcond is not None:
                if lcond == 'all':
                    mask_for_labels = pd.Series(True, index=labels.index)
                elif callable(lcond):
                    try:
                        mask_for_labels = lcond(labels, x, y, z)
                        if not isinstance(mask_for_labels, pd.Series):
                            mask_for_labels = pd.Series(mask_for_labels, index=labels.index)
                        elif not mask_for_labels.index.equals(labels.index):
                            mask_for_labels = pd.Series(mask_for_labels.values, index=labels.index)
                    except Exception:
                        mask_for_labels = pd.Series(False, index=labels.index)
                else:
                    if isinstance(lcond, (list, set, tuple)):
                        mask_for_labels = labels.isin(lcond)
                    else:
                        mask_for_labels = labels == lcond
                mask_for_labels = mask_for_labels.astype(bool)
                for xi, yi, zi, lab in zip(x[mask_for_labels].values, y[mask_for_labels].values, z[mask_for_labels].values, labels[mask_for_labels].values):
                    ax.text(xi, yi, zi, str(lab), **lkwargs)

            pts = np.column_stack([x.values, y.values, z.values])
            if env_flag:
                if env_groupwise:
                    if pts.shape[0] >= 4:
                        fc = env_fc or col
                        _plot_envelope_convex(ax, pts, face_color=fc, alpha=env_alpha, edge_color=env_ec, edge_width=env_ew)
                else:
                    if pts.shape[0] > 0:
                        all_pts_for_global_envelope.append(pts)

    if any(envelope_list) and (not any(envelope_groupwise_list)) and len(all_pts_for_global_envelope) > 0:
        pts_all = np.vstack(all_pts_for_global_envelope)
        if pts_all.shape[0] >= 4:
            # use first non-None facecolor/alpha/ec/ew among plots, fallback to defaults
            fc = next((c for c in envelope_facecolor_list if c is not None), None) or 'lightgrey'
            alpha_val = next((a for a in envelope_alpha_list if a is not None), envelope_alpha)
            ec = next((e for e in envelope_edgecolor_list if e is not None), envelope_edgecolor)
            ew = next((w for w in envelope_edgewidth_list if w is not None), envelope_edgewidth)
            _plot_envelope_convex(ax, pts_all, face_color=fc, alpha=alpha_val, edge_color=ec, edge_width=ew)

    if show_legend:
        try:
            ax.legend()
        except Exception:
            pass

    title, xlabel, ylabel, zlabel = _resolve_title_and_labels(title_and_labels, x_list[0], y_list[0], z_list[0])
    ax.set_title('' if title == '' else title)
    ax.set_xlabel('' if xlabel == '' else xlabel)
    ax.set_ylabel('' if ylabel == '' else ylabel)
    ax.set_zlabel('' if zlabel == '' else zlabel)

    if not show_ticks:
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])

    try:
        plt.tight_layout()
    except Exception:
        pass

    plt.show()


def plot_multiindex_scatter_3d(
    df,
    level=0,
    x_column=None,
    y_column=None,
    z_column=None,
    label_column=None,
    label_condition=None,
    label_kwargs=None,
    color=None,
    marker=None,
    size=None,
    envelope=False,
    envelope_groupwise=True,
    envelope_facecolor=None,
    envelope_alpha=0.25,
    envelope_edgecolor='k',
    envelope_edgewidth=0,
    figsize=(8,6),
    elev=0,
    azim=-90,
    show_legend=True,
    title_and_labels=None,        # list of 4 or dict {'title','xlabel','ylabel','zlabel'}
    show_ticks=True,
    **kwargs
):
    """
    単一仕様の 3次元散布図を描画するユーティリティ。

    概要
    ----
    - 1つのプロット仕様（x/y/z/スタイル）に基づいて 3D 散布図を描画します。
    - overlay（重ね描画）を使わないシンプルな用途向けに設計されています。
    - title_and_labels によるラベル制御をサポート。要素が None の場合は列名を使い、'' は非表示になります。

    主な引数
    -------------
    - df: pandas.DataFrame
    - level: グルーピングに使うインデックスレベル
    - x_column, y_column, z_column: 列名（必須）
    - label_column, label_condition, label_kwargs: 注釈ラベル関連
    - color, marker, size: 描画スタイル
    - envelope, envelope_groupwise, envelope_facecolor, envelope_alpha, envelope_edgecolor, envelope_edgewidth:
      包絡面の描画（スカラー指定。複雑な個別指定は overlay 版を使用）

    使用例
    ------
    # 基本的な単一プロット
    plot_multiindex_scatter_3d(
        df,
        x_column='proj_rn_per_hour_worked_DRS',
        y_column='proj_hc_DRS',
        z_column='proj_rgdpo_per_hour_worked_DRS',
        color='C0',
        marker='o',
        size=40,
        envelope=True,
        envelope_facecolor='C0',
        envelope_alpha=0.2,
        title_and_labels=[None, 'RN per hour', 'HC', 'RGDPO']  # None はデフォルト（列名）表示
    )

    # ラベルを全て消して余白を詰めたい場合
    plot_multiindex_scatter_3d(
        df,
        x_column='proj_rn_per_hour_worked_DRS',
        y_column='proj_hc_DRS',
        z_column='proj_rgdpo_per_hour_worked_DRS',
        title_and_labels=['', '', '', '']  # '' は非表示
    )
    """
    if x_column is None or y_column is None or z_column is None:
        raise ValueError("x_column, y_column, z_column を指定してください。")

    label_kwargs = label_kwargs or {}
    df_reset = df.reset_index()

    # label_column 自動選択（level が指定されている場合）
    if label_column is None and level is not None:
        if isinstance(level, int) and level < len(df.index.names):
            name = df.index.names[level]
            if name is not None and name in df_reset.columns:
                label_column = name
        if label_column is None:
            label_column = df_reset.columns[0]

    level_name = None
    if level is not None:
        level_name = df.index.names[level] if isinstance(level, int) else str(level)
        if level_name not in df_reset.columns:
            level_name = None

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=elev, azim=azim)

    palette = plt.rcParams.get('axes.prop_cycle').by_key().get('color', ['C0','C1','C2','C3'])
    markers = ['o','s','^','D','v','P','X','*']

    if level_name is not None:
        groups = list(df_reset.groupby(level_name))
    else:
        groups = [(None, df_reset)]

    all_pts = []

    for i, (group, data) in enumerate(groups):
        x_all = pd.to_numeric(data[x_column], errors='coerce')
        y_all = pd.to_numeric(data[y_column], errors='coerce')
        z_all = pd.to_numeric(data[z_column], errors='coerce')

        valid = x_all.notna() & y_all.notna() & z_all.notna()

        x = x_all[valid]
        y = y_all[valid]
        z = z_all[valid]

        labels = None
        if label_column in data.columns:
            labels = data.loc[valid, label_column]

        col = _get_per_param(color, group if group is not None else 'all', i, len(groups)) or palette[i % len(palette)]
        m = _get_per_param(marker, group if group is not None else 'all', i, len(groups)) or markers[i % len(markers)]
        s = _get_per_param(size, group if group is not None else 'all', i, len(groups)) or 30

        if len(x) > 0:
            ax.scatter(x.values, y.values, z.values, c=[col], marker=m, s=s, label=str(group) if group is not None else z_column, depthshade=True, **kwargs)

        # ラベル付け
        if labels is not None and label_condition is not None:
            if label_condition == 'all':
                mask_for_labels = pd.Series(True, index=labels.index)
            elif callable(label_condition):
                try:
                    mask_for_labels = label_condition(labels, x, y, z)
                    if not isinstance(mask_for_labels, pd.Series):
                        mask_for_labels = pd.Series(mask_for_labels, index=labels.index)
                    elif not mask_for_labels.index.equals(labels.index):
                        mask_for_labels = pd.Series(mask_for_labels.values, index=labels.index)
                except Exception:
                    mask_for_labels = pd.Series(False, index=labels.index)
            else:
                if isinstance(label_condition, (list, set, tuple)):
                    mask_for_labels = labels.isin(label_condition)
                else:
                    mask_for_labels = labels == label_condition

            mask_for_labels = mask_for_labels.astype(bool)

            for xi, yi, zi, lab in zip(x[mask_for_labels].values, y[mask_for_labels].values, z[mask_for_labels].values, labels[mask_for_labels].values):
                ax.text(xi, yi, zi, str(lab), **label_kwargs)

        pts = np.column_stack([x.values, y.values, z.values])
        if envelope:
            if envelope_groupwise:
                if pts.shape[0] >= 4:
                    fc = envelope_facecolor or col
                    _plot_envelope_convex(ax, pts, face_color=fc, alpha=envelope_alpha, edge_color=envelope_edgecolor, edge_width=envelope_edgewidth)
            else:
                if pts.shape[0] > 0:
                    all_pts.append(pts)

    if envelope and not envelope_groupwise and len(all_pts) > 0:
        pts_all = np.vstack(all_pts)
        if pts_all.shape[0] >= 4:
            fc = envelope_facecolor or 'lightgrey'
            _plot_envelope_convex(ax, pts_all, face_color=fc, alpha=envelope_alpha, edge_color=envelope_edgecolor, edge_width=envelope_edgewidth)

    # 凡例（元のシンプル挙動）
    if show_legend and level_name is not None:
        ax.legend(title=level_name)

    # ラベル解決（None -> default, '' -> hide）
    title, xlabel, ylabel, zlabel = _resolve_title_and_labels(title_and_labels, x_column, y_column, z_column)

    # 空文字 '' は非表示、それ以外は設定
    ax.set_title('' if title == '' else title)
    ax.set_xlabel('' if xlabel == '' else xlabel)
    ax.set_ylabel('' if ylabel == '' else ylabel)
    ax.set_zlabel('' if zlabel == '' else zlabel)

    # 目盛り表示制御
    if not show_ticks:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])

    # 元のように tight_layout で自動調整（余白ロジックは元に戻す）
    try:
        plt.tight_layout()
    except Exception:
        pass

    plt.show()

