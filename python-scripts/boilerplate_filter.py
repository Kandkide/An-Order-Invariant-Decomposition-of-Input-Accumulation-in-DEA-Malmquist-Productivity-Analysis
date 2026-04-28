import pandas as pd


def filter_multiindex(df, level, items, exclude=False):
    """
    マルチインデックスの特定の階層（レベル）を指定してデータフレームを抽出または除外する。

    Args:
        df (pd.DataFrame): フィルタリング対象のデータフレーム。
        level (int or str): インデックスの階層番号（例: 0, -1）または名前（例: 'year'）。
        items (scalar, list, or slice): 抽出・除外したい要素。
        exclude (bool): Trueにすると、itemsで指定した要素「以外」を抽出する。デフォルトはFalse。

    Returns:
        pd.DataFrame: フィルタリングされたデータフレーム。

    Examples:
        >>> # 2020年「以外」のデータをすべて残す
        >>> df_not_2020 = filter_multiindex(df, level='year', items=2020, exclude=True)

        >>> # 1965年から1990年の範囲「外」のデータを抽出
        >>> df_outside = filter_multiindex(df, level=-1, items=slice(1965, 1990), exclude=True)
        
        >>> # リストに含まれる特定の国「以外」を抽出
        >>> df_other_countries = filter_multiindex(df, level=0, items=['Japan', 'USA'], exclude=True)
    """
    # 階層（level）の全値を取得
    level_values = df.index.get_level_values(level)

    # フィルタ条件に一致するマスク（True/False）を作成
    if isinstance(items, slice):
        # スライスの場合（start <= x <= stop）
        # ※ラベルベースのスライスなので両端を含む
        mask = (level_values >= items.start) & (level_values <= items.stop)
    elif isinstance(items, (list, tuple, pd.Index, pd.Series)):
        # リスト等の集合に含まれるか
        mask = level_values.isin(items)
    else:
        # 単一要素（スカラー）
        mask = level_values == items

    # 除外設定ならビット反転
    if exclude:
        mask = ~mask

    return df[mask]


def filter_dataframe_by_regex(df, column='INDICATOR', pattern=r'(?i)^[a-m]', na=False):
    """
    指定した列の値が正規表現 `pattern` にマッチする行のみを抽出する関数。

    Args:
        df (pd.DataFrame): フィルタリング対象のデータフレーム。
        column (str): フィルタリングする列の名前。
        pattern (str, optional): マッチング条件を指定する正規表現。デフォルトは '^[a-m]'。
        na (bool, optional): 欠損値(NaN)の扱い。デフォルトは False。

    Returns:
        pd.DataFrame: 条件に合う行のみを含むデータフレーム。
    """
    return df[df[column].str.match(pattern, na=na)]


def get_non_na_groups_by_year(df, level, y_columns, years_filter, method='all'):
    """
    指定された年ごとに、非NAデータを持つグループ名を収集する。

    Parameters:
    df (pd.DataFrame): マルチインデックスを持つデータフレーム
    level (int or str): 年を含むインデックスのレベル
    y_columns (list): 対象とする列
    years_filter (list, slice, or None): 年の選択条件
    method (str): 'all'（全て非NA）または 'any'（どれか1つ非NA）。デフォルトは 'all'。

    Returns:
    dict or list: {年: [非NAグループ]} または [非NAグループ]
    """
    if method not in ('all', 'any'):
        raise ValueError("method must be 'all' or 'any'")

    result = {}
    index_values = df.index.get_level_values(level)
    target_years = index_values.unique().sort_values()

    if isinstance(years_filter, slice):
        target_years = target_years[years_filter]
    else:
        target_years = [y for y in target_years if y in years_filter]

    for year in target_years:
        sub_df = df.xs(year, level=level, drop_level=False)
        na_mask = sub_df[y_columns].notna()
        match_mask = na_mask.all(axis=1) if method == 'all' else na_mask.any(axis=1)
        non_na_index = sub_df[match_mask].index.droplevel(level)
        result[year] = non_na_index.unique().tolist()

    return next(iter(result.values())) if len(result) == 1 else result


def filter_df_by_non_na_groups(df, year, level=-1, y_columns=None, method='all'):
    """
    指定年において条件を満たすグループのみに基づき、
    全期間の df から該当行だけを抽出する。

    Parameters:
    df (pd.DataFrame): マルチインデックスのデータフレーム
    year (int or str): 判定対象となる年
    level (int or str): 年に対応するインデックスレベル（デフォルト -1）
    y_columns (list or None): 対象列（None の場合は数値列）
    method (str): 'all' または 'any'（デフォルト 'all'）

    Returns:
    pd.DataFrame: 指定年に条件を満たしたグループの全期間データ
    """
    if y_columns is None:
        y_columns = df.select_dtypes(include='number').columns.tolist()

    # 対象グループのリストを取得（年を除いたインデックス構造）
    valid_groups = get_non_na_groups_by_year(df, level, y_columns, [year], method=method)

    # 年のレベルを除いたインデックスと突き合わせ
    reduced_index = df.index.droplevel(level)
    match_mask = reduced_index.isin(valid_groups)
    return df[match_mask]

# def filter_df_by_non_na_groups(df, year, level=-1, y_columns=None, method='all', target_level=0):
#     """
#     指定年において条件を満たすグループのみを、全ての年を含めて抽出する。

#     Parameters:
#     df (pd.DataFrame): マルチインデックスを持つデータフレーム
#     year (int or str): フィルタ判定に使う単一の年
#     level (int or str): 年が格納されたインデックスレベル（デフォルト -1）
#     y_columns (list or None): 対象列（None の場合は数値列）
#     method (str): 'all' または 'any'（デフォルト 'all'）
#     target_level (int or str): グループ抽出対象のインデックスレベル（デフォルト 0）

#     Returns:
#     pd.DataFrame: 条件を満たしたグループに絞った全期間のデータ
#     """
#     if y_columns is None:
#         y_columns = df.select_dtypes(include='number').columns.tolist()

#     groups = get_non_na_groups_by_year(df, level, y_columns, [year], method=method)
#     # 全体から対象のグループのみ抽出（指定された target_level に絞る）
#     filtered = df[df.index.get_level_values(target_level).isin(groups)]
#     return filtered


FINAL_COMPACT = {
    "Saudi Arabia", "United Arab Emirates", "Kuwait", "Qatar",
    "Iraq", "Iran (Islamic Republic of)", "Oman", "Yemen",
    "Nigeria", "Angola", "Gabon", "Equatorial Guinea",
    "Chad", "South Sudan", "Sudan", "Congo",
    "Venezuela", "Ecuador", "Trinidad and Tobago",
    "Venezuela (Bolivarian Republic of)", "Iran", "Myanmar"
}

SECOND_COMPACT = FINAL_COMPACT | {
    "Azerbaijan", "Kazakhstan", "Russian Federation"
}


def mask_oil_producers(df, mode="final", extra=None):
    """
    Return a boolean mask selecting rows belonging to oil-dependent countries.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame. Country names must be either:
        - MultiIndex: level 0 contains country names
        - Single index: a column named 'country' contains country names
    mode : {"final", "second", "extra_only"}, default "final"
        "final"      -> use FINAL_COMPACT
        "second"     -> use SECOND_COMPACT
        "extra_only" -> use only the countries provided in `extra`
    extra : list or set of str, optional
        Additional country names to include (for "final"/"second") or
        the sole target set when mode == "extra_only".

    Returns
    -------
    pandas.Series (bool)
        Boolean mask aligned with df.index.
    """

    # validate mode
    if mode not in {"final", "second", "extra_only"}:
        raise ValueError("mode must be 'final', 'second', or 'extra_only'")

    # build target set
    if mode == "final":
        target = set(FINAL_COMPACT)
        if extra:
            target |= set(extra)
    elif mode == "second":
        target = set(SECOND_COMPACT)
        if extra:
            target |= set(extra)
    else:  # extra_only
        if not extra:
            raise ValueError("When mode='extra_only', `extra` must be a non-empty list or set of country names.")
        target = set(extra)

    # MultiIndex case
    if isinstance(df.index, pd.MultiIndex):
        countries = df.index.get_level_values(0)
        return countries.isin(target)

    # Single index → expect a 'country' column
    if "country" not in df.columns:
        raise KeyError("DataFrame must have a 'country' column when index is not MultiIndex.")

    return df["country"].isin(target)


def filter_oil_producers(df, mode="final", invert=False, extra=None):
    """
    Filter the DataFrame by oil-dependent countries.

    Parameters
    ----------
    df : pandas.DataFrame
    mode : {"final", "second", "extra_only"}, default "final"
    invert : bool, default False
        False -> return only oil-dependent countries
        True  -> return only NON-oil-dependent countries
    extra : list or set of str, optional
        Additional country names or sole target list when mode == "extra_only".

    Returns
    -------
    pandas.DataFrame
    """

    mask = mask_oil_producers(df, mode=mode, extra=extra)
    if invert:
        mask = ~mask
    return df.loc[mask]


