from scipy.stats import normaltest
import numpy as np

def find_outliers(df):
    num_col = df.select_dtypes(include=np.number).columns
    gaussian_col = []
    non_gaussian_col  = []
    for col in num_col:
        stat,p = normaltest(df[col])
        alpha = 0.05
        if p>alpha:
            gaussian_col.append(col)
        else:
            non_gaussian_col.append(col)
    col_with_out =[] 
    capped_data = df.copy()
    for i in gaussian_col:
        ds = df[i].describe()
        Highest_allowed = (ds.iloc[1] + 3*ds.iloc[2])
        Lowest_allowed = (ds.iloc[1] - 3*ds.iloc[2])
        ds = df[(df[i] >Highest_allowed) | (df[i] < Lowest_allowed)]
        if ds.shape[0] > 0:
            col_with_out.append(i)
            capped_data.loc[capped_data[i]> Highest_allowed,i] = Highest_allowed
            capped_data.loc[capped_data[i]<Lowest_allowed,col]=Lowest_allowed
    return col_with_out,capped_data