from 

def del_dupli_rows(dataFrame):

    dupli_rows = dataFrame[dataFrame.duplicated()]

    dataFrame = dataFrame.drop_duplicated(keep='first')
    return dataFrame 

def drop_nd_fill_na(dataFrame):

    drop_col = dataFrame.coloumns[dataFrame.isnull().mean > 0.5]

    dataFrame = dataFrame.drop(drop_col,axis=1)

    dataFrame = dataFrame.fillna(dataframe.mean())

    return dataframe


def split_data(dataFrame,)

