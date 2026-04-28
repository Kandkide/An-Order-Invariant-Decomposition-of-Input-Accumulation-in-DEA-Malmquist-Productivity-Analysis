"""minimum_code_for_paper.py
"""
import matplotlib.pyplot as plt
import numpy as np

### in case of independent script ####
import pandas as pd
import os

# # code 1
# if os.path.exists('data.csv'):
#     df = pd.read_csv('data.csv', index_col=False)
# else:
#     df = pd.read_excel(
#         r"https://dataverse.nl/api/access/datafile/354095", sheet_name='Data')
#     df.to_csv('data.csv', index=False)
# # df = pd.read_excel(r"C:\Users\kandk\Downloads\data_file_downloaded\pwt1001.xlsx", sheet_name='Data')
# # df = pd.read_csv(r"C:\Users\kandk\Downloads\data_file_downloaded\pwt1001\csv_from_sheet\Data.csv")

# #### in case of python_from_cpp ####
# from init_pwt import pwt1001_df
# df = pwt1001_df

# from Local file
df = pd.read_excel('data/pwt1001.xlsx', sheet_name='Data')

df = df.set_index(['countrycode', 'year'])
cgdpopl = df['cgdpo']/(df['emp']*df['avh'])
df['yius'] = cgdpopl/cgdpopl.loc['USA']
df['qius'] = df['yius']/df['ctfp']
df[['yius', 'ctfp', 'qius']].dropna().describe().to_clipboard()

# # code 2
# df[['yius', 'qius']].dropna().groupby(level=1).apply(
#     lambda g: np.log(g['qius']).var()/np.log(g['yius']).var()).plot(title='measure of success')
# plt.show()

# code 3
_rgdpnapl = df['rgdpna']/(df['emp']*df['avh'])
df['yt2017'] = _rgdpnapl.groupby(level=0, group_keys=False).apply(
    lambda g: g/g.loc[(slice(None), 2017)])
df['qt2017'] = df['yt2017']/df['rtfpna']
print(df[['yt2017', 'rtfpna', 'qt2017']].dropna().describe())

# # code 4 # do not use in the paper
# gb = df[['yt2017', 'rtfpna', 'qt2017']].loc[['USA', 'JPN', 'DEU']].dropna()
# print(gb.groupby(level=0).apply(lambda g: (np.log(g.iloc[-1]) - np.log(g.iloc[0]))*100))

# code 5: (code 4 in the paper)
labsh_mean = df['labsh'].groupby(level=1).transform(lambda g: g.mean())
lab_contrib = df['hc']**labsh_mean
c_cap_contrib = (df['cn']/(df['emp']*df['avh']))**(1-labsh_mean)
r_cap_contrib = (df['rnna']/(df['emp']*df['avh']))**(1-labsh_mean)
df['qius_calc'] = \
(c_cap_contrib/c_cap_contrib.loc['USA'])*(lab_contrib/lab_contrib.loc['USA'])
df['ctfp_calc'] = df['yius']/df['qius_calc']
df['kt2017'] = r_cap_contrib.groupby(level=0, group_keys=False)\
.apply(lambda g: g/g.loc[(slice(None), 2017)])
df['hct2017'] = lab_contrib.groupby(level=0, group_keys=False)\
.apply(lambda g: g/g.loc[(slice(None), 2017)])
df['qt2017_calc'] = df['kt2017'] * df['hct2017']
df['rtfpna_calc'] = df['yt2017']/df['qt2017_calc']

# code 6
countries = ['BGD', 'KHM', 'PAK', 'MMR', 'VNM']
df.groupby(level=1).apply(
    lambda df: np.log(df['qius_calc']).var()/np.log(df['yius']).var()
    ).plot(title='measure of success: all countries')
plt.show()
df.loc[list(countries)].groupby(level=1).apply(
    lambda df: np.log(df['qius_calc']).var()/np.log(df['yius']).var()
    ).plot(title='measure of success: five countries')
plt.show()

# code 7
variables = ['rtfpna_calc', 'yt2017', 'qt2017_calc']
gb = df.loc[countries, variables].dropna().groupby(level=0, group_keys=False)
gb.apply(
    lambda df: ((np.log(df.iloc[-1]) - np.log(df.iloc[0]))*100)
    ).dropna().to_clipboard()
