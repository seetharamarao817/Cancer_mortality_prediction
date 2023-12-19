from sklearn.preprocessing import OneHotEncoder


def bin_to_num(data):
    binnedinc = []

    for i in data["binnedInc"]:

        # remove parentheses or brackets 
        i = i.strip("()[]")
        #seperate on comma
        i = i.split(",")
        #convert to tuple
        i = tuple(i)
        # convert to float
        i = tuple(map(float,i))
        # convert to list
        i = list(i)
        # append to the list 
        binnedinc.append(i)
    data["binnedInc"] = binnedinc
    
    # Create new seperate coloumns for data in the list 
    data['lower_bound'] = [i[0] for i in data["binnedInc"]]
    data['upper_bound'] = [i[1] for i in data["binnedInc"]]
    data['mean'] = (data["lower_bound"]+data['upper_bound'])/2

    data = data.drop("binnedInc",axis=1)

    return data

def cal_col(data):

    data['county'] = [i.split(',')[0] for i in data["Geography"]]
    data['state'] = [i.split(',')[1] for i in data["Geography"]]

    data = data.drop("Geography",axis=1)
    return data 

def one_hot_enc(data):
    cat_cols = data.select_dtypes(include =['object']).columns
    
    one_hot = OneHotEncoder(sparse =False,handle_unknown ='ignore')

    one_hot_enc = one_hot.fit_transform(data[cal_cols])

    one_hot_enc = pd.DataFrame(
        one_hot_enc,coloumns=one_hot.get_feature_names_out(cal_cols)

    )
    data = data.drop(cat_cols,axis=1)
    data = pd.concat([data,one_hot_enc],axis=1)

    return data 


