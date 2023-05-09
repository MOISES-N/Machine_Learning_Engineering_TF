import pandas as pd
import numpy as np
import os

def read_file_csv( filename ):
    """
    Reads a csv file an outputs a pandas dataframe

    :param filename (str) name of the csv file
    :returns DataFrame pandas
    
    >>> read_file_csv( 'Dataset_train.csv' )
    """

    path = os.path.join('../data/raw/', filename)

    df = pd.read_csv( path )

    print(filename, ' cargado correctamente')

    return df

def data_preparation(df):
    
    """
    Prepares data for trainning

    :param df (pandas DataFrame) trainning, validation or test dataset
    :returns DataFrame pandas
    
    >>> data_preparation( df )
    """

    df[ 'Delta Temperature[K]' ] = df[ 'Process temperature [K]' ] - df[ 'Air temperature [K]' ]

    print('Transformación de datos completa')

    return df

def data_exporting( df , features , filename ) :

    """
    Exports prepared data as csv

    :param df (pandas DataFrame) Processed trainning, validation or test dataset
    :param features (list) list of features to be preserved in the dataset
    :param filename (str) filename of the processed dataset
    
    >>> data_exporting( df , ['Air temperature [K]' , 'Process temperature [K]' , 'Rotational speed [rpm]' ] , 'Processed_dataset.csv' )
    """
    
    dfp = df[ features ]
    
    path = os.path.join('../data/processed/', filename)
    
    dfp.to_csv( path )
    
    print(filename, 'exportado correctamente en la carpeta processed')

def main():

    # Matriz de Entrenamiento
    df1 = read_file_csv('Dataset_train.csv')
    tdf1 = data_preparation(df1)
    data_exporting(tdf1, [ 'Delta Temperature[K]','Rotational speed [rpm]','Torque [Nm]','Tool wear [min]' , 'label' ],'Dataset_train_processed.csv')
    
    # Matriz de Validación
    df1 = read_file_csv('Dataset_validation.csv')
    tdf1 = data_preparation(df1)
    data_exporting(tdf1, [ 'Delta Temperature[K]','Rotational speed [rpm]','Torque [Nm]','Tool wear [min]' , 'label' ],'Dataset_validation_processed.csv')
    
    # Matriz de Scoring
    df1 = read_file_csv('Dataset_test.csv')
    tdf1 = data_preparation(df1)
    data_exporting(tdf1, [ 'Delta Temperature[K]','Rotational speed [rpm]','Torque [Nm]','Tool wear [min]' ],'Dataset_test_processed.csv')

if __name__ == '__main__' :

    main()