from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
import os
import pandas as pd
import pickle

# Cargar la tabla transformada
def read_file_csv( filename ):

    """
    Reads a csv file an outputs a pandas dataframe

    :param filename (str) name of the csv file
    :returns DataFrame pandas trainning and labels
    
    >>> read_file_csv( 'Dataset_train.csv' )
    """
    path = os.path.join('../data/processed', filename )

    df = pd.read_csv( path )

    X_train = df.drop(['label'],axis=1)

    y_train = df[['label']]

    print(filename, ' cargado correctamente')

    return X_train, y_train


def Transformaciones_train( X_train ):

    """
    Prepares data for trainning

    :param df (pandas DataFrame) Processed trainning, validation or test dataset
    :returns DataFrame pandas
    
    >>> Transformaciones_train( X_train )
    """

    Scaler = MinMaxScaler()

    variables_objetivo = [ 'Delta Temperature[K]','Rotational speed [rpm]','Torque [Nm]','Tool wear [min]']

    ct = ColumnTransformer([
            ('Scaler', Scaler, variables_objetivo)
        ], remainder='passthrough')
    
    Pipeline_transformacion = Pipeline( [ ('Transformaciones' , ct ) ] )

    Pipeline_transformacion.fit( X_train )

    pickle.dump( Pipeline_transformacion , open( '../models/pipeline.pkl', 'wb'))

    print('Pipeline exportado correctamente en la carpeta models')

    return Pipeline_transformacion


def Entrenamiento( X_train, y_train , Pipeline_transformacion ):

    """
    Performs modeling in the trainning dataset

    :param X_train (pandas DataFrame) feature dataset for trainning
    :param y_train (pandas DataFrame) label dataset
    :param Pipeline_transformacion fitted sklearn pipeline
    
    >>> Entrenamiento( X_train, y_train , Pipeline )
    """

    Trainning_df = Pipeline_transformacion.transform( X_train )

    logreg = LogisticRegression(solver='liblinear', random_state=0)

    logreg.fit( Trainning_df , y_train )

    print('Modelo entrenado')

    # Guardamos el modelo entrenado para usarlo en produccion
    package = '../models/best_model.pkl'

    pickle.dump( logreg , open(package, 'wb'))

    print('Modelo exportado correctamente en la carpeta models')


# Entrenamiento completo
def main():

    X_train, y_train = read_file_csv('Dataset_train_processed.csv')

    Pipeline_transformacion = Transformaciones_train( X_train )

    Entrenamiento( X_train, y_train , Pipeline_transformacion )

    print('Finalizó el entrenamiento del Modelo')


if __name__ == "__main__":

    main()