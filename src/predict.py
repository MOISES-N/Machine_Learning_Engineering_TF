import pandas as pd
import pickle
import os


# Cargar la tabla transformada
def score_model(filename, failure_prediction ):
    df = pd.read_csv(os.path.join('../data/processed', filename))
    print(filename, ' cargado correctamente')

    # Leemos pipeline
    package = '../models/pipeline.pkl'
    pipeline_transformacion = pickle.load(open(package, 'rb'))
    print('pipeline importado correctamente')

    df = pipeline_transformacion.transform( df )

    # Leemos el modelo entrenado para usarlo
    package = '../models/best_model.pkl'
    model = pickle.load(open(package, 'rb'))
    print('Modelo importado correctamente')

    # Predecimos sobre el set de datos de Scoring    
    res = model.predict(df).reshape(-1,1)
    pred = pd.DataFrame(res, columns=['PREDICT'])
    pred.to_csv(os.path.join('../data/scores/', failure_prediction ))
    print( failure_prediction , 'exportado correctamente en la carpeta scores')


# Scoring desde el inicio
def main():
    df = score_model( 'Dataset_test_processed.csv','failure_prediction.csv')
    print('Finalizó el Scoring del Modelo')


if __name__ == "__main__":
    main()