import os
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import TruncatedSVD


def import_tr_te(train_file_name, test_file_name, columns, data_folder):
    train = pd.read_csv(os.getcwd() + '/' + data_folder + '/' + train_file_name, header=None, names=columns, sep='\t')
    test = pd.read_csv(os.getcwd() + '/' + data_folder + '/' + test_file_name, header=None, names=columns[1:], sep='\t')
    return train, test


def draw_ct(join, response, factor):
    df = pd.crosstab(join[response], join[factor])
    # Normalize the cross tab to sum to 1:
    ndf = df.div(df.sum(1).astype(float), axis=0)

    ndf = ndf.sort(columns=[1, 0])
    return ndf, ndf.plot(kind='bar', stacked=True, title=('Clicks by' + factor))


def count_vect(df_join, col):
    # could sum tfidf vect along columns
    fs = df_join[col].value_counts()
    fs = fs[fs > 1]
    fs = pd.cut(fs, bins=10, labels=False)
    df = pd.DataFrame({col: fs.index, 'freq_' + col: fs.values})
    return df


def get_dummies(join, col):
    df = pd.get_dummies(join[col])
    df['id'] = join['id']
    return df


def tfidf(join, col='user_tags'):
    # use to convert user_tags to tfidf features
    tfidfvect = TfidfVectorizer()
    tsvd = TruncatedSVD(n_components=10)
    tfidf_fs = tfidfvect.fit_transform(join[col])
    tfidf_fs = tsvd.fit_transform(tfidf_fs)
    df = pd.DataFrame(tfidf_fs.toarray(), columns=tfidfvect.vocabulary_)
    df['id'] = join['id']
    return df


def split_user_agent(join, col='user_agent'):
    # use to split user_agent info and then use it with tfidf transform
    split = join[col].map(lambda x: x.split("_"))
    df = pd.DataFrame()
    df['sys'] = split.map(lambda x: x[0])
    df['browser'] = split.map(lambda x: x[1])
    df['id'] = join['id']
    return df


def split_timestamp(join, col='timestamp'):
    df = pd.DataFrame()
    df['ty'] = join[col].map(lambda x: str(x)[0:4])
    df['tm'] = join[col].map(lambda x: str(x)[4:6])
    df['td'] = join[col].map(lambda x: str(x)[6:8])
    df['id'] = join['id']
    return df


def part_of_day(join, col='hour'):
    df = pd.DataFrame()
    df['h_binned'] = pd.cut(join[col], bins=6, labels=False)
    df['id'] = join['id']
    return df


def multiply(join, col1='height', col2='width'):
    df = pd.DataFrame()
    df['area'] = join[col1] * join[col2]
    df['id'] = join['id']
    return df
