import pandas as pd


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


