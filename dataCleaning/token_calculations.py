import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

def tokenization_ftm(X,X_test):

    #For train data we fit and transform
    v = TfidfVectorizer(stop_words='english', ngram_range=(2,2), analyzer='word')
    x_train = v.fit_transform(X['final_text'])
    x_train = pd.DataFrame(x_train.toarray(), columns=v.get_feature_names())
    X = X.drop('final_text', axis=1)
    res_train = pd.concat([X.reset_index(drop=True), x_train], axis=1)


    #For test data we transform
    x_test = v.transform(X_test['final_text'])
    x_test = pd.DataFrame(x_test.toarray(), columns=v.get_feature_names())
    X_test = X_test.drop('final_text',axis=1)
    res_test = pd.concat([X_test.reset_index(drop=True), x_test], axis=1)


    return res_train, res_test






