from sklearn.naive_bayes import MultinomialNB
from scipy.stats import multinomial
import pandas as pd, numpy as np, scpiy as sp

def prob_gen(df):
    vals, probs = [], []
    for i in df.columns:
        vals += [df[i].unique()]  
        probs += [df[i].value_counts(normalize=True)]
    return vals, probs

def rv_gen(values, dist, length):
    temp = [dist[i] for i in values]
    return multinomial.rvs(n=1, p=temp, size=length)@np.array(values)

def table_gen(df, start, end, freq):
    train = df.drop(['time','Status'], axis=1)
    
    ################
    #generates dates
    ################
    
    date_rng = pd.date_range(start=start, end=end, freq=freq)
    steps = len(date_rng)
    df1 = pd.DataFrame(date_rng, columns=['date'])
    
    ##########################################
    #generates covariates with multinomial rv gen
    #########################################
    
    vals, probs = prob_gen(train) 
    data = np.matrix([rv_gen(vals[x], probs[x], steps) for x in range(len(vals))]).T
    df2 = pd.DataFrame(data, columns = [i for i in train.columns])
    
    #########################################
    #generates failures with naive bayes model
    #########################################
    
    target = df['Status']
    mnb = MultinomialNB()
    y_pred = mnb.fit(train, target)
    fails = y_pred.predict(df2)
    df3 = pd.DataFrame(fails, columns = ['Status'])
    
    res = pd.concat([df1, df2, df3], axis=1)
    
    return res

