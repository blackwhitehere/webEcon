import os
import pandas as pd


def import_tr_te(train_file_name, test_file_name, columns, data_folder):
    train = pd.read_csv(os.getcwd()+'/'+data_folder+'/'+train_file_name, header=None, names=columns, sep='\t')
    test = pd.read_csv(os.getcwd()+'/'+data_folder+'/'+test_file_name, header=None, names=columns[1:], sep='\t')
    return train, test


def draw_ct(join, colName1, colName2):
    df = pd.crosstab(join[colName1], join[colName2])
    ndf = df.div(df.sum(1).astype(float), axis=0)

    # Normalize the cross tab to sum to 1:
    ndf = ndf.sort(columns=[2.0, 1.0, 0.0])
    return ndf, ndf.plot(kind='bar', stacked=True, title=('Fault Severity by ' + colName))