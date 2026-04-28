from tabulate import tabulate


def tabulate_simple(df, showindex=True):
    return tabulate(df, headers='keys', tablefmt='pretty', showindex=showindex)
