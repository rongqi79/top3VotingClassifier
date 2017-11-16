import pandas as pd;
from sklearn import preprocessing;
import scipy.stats as sp;
from sklearn import decomposition;
from DataFrameImputer import DataFrameImputer

def dataPreprocessing(df):

    #imp = preprocessing.Imputer(missing_values='NaN', strategy='most_frequent', axis=0, verbose=1)
    #imp.fit(df)

    imputerResult = DataFrameImputer().fit_transform(df)

    imputerResult = imputerResult.apply(preprocessing.LabelEncoder().fit_transform)
    #df=df.apply(preprocessing.StandardScaler().fit_transform)
    #df=df.apply(preprocessing.MinMaxScaler().fit_transform)
    return imputerResult

def featureSelection(df):
    # feature selection based on chi square
    listPVal = []
    for i in range(len(df.columns) - 1):
        freqtab = pd.crosstab(df.iloc[:, i], df.iloc[:, -1])
        chi2, pval, dof, expected = sp.chi2_contingency(freqtab)
        listPVal.append(pval)
    for i in range(len(df.columns) - 1):
        if (listPVal[i] > 0.005):
            df.drop(df.columns[i], axis=1, inplace=True)
    print("now the column number is ", df.shape[1])
    k=int(input("how many feature numbers you want to keep?"))
    data = df.iloc[:,:len(df.columns)-1]
    Y=df.iloc[:,len(df.columns)-1]
    #Xdata=pd.DataFrame(data)
    pca = decomposition.PCA(n_components=k)
    pca.fit(data)
    Xdata = pd.DataFrame(pca.transform(data))
    Xdata[k]=Y
    return Xdata

def most_common(lst):
    return max(set(lst), key=lst.count)

def getX(df):
    k = df.shape[1]-1
    return df.drop(k, 1).values

def getY(df):
    k = df.shape[1]-1
    print(k)
    return df[k].values.astype(int)
