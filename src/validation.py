import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import *
import os


# Cargar la tabla transformada
def eval_model(filename):
    df = pd.read_csv(os.path.join('../data/processed', filename))
    print(filename, ' cargado correctamente')

    # Leemos pipeline
    package = '../models/pipeline.pkl'
    pipeline_transformacion = pickle.load(open(package, 'rb'))
    print('pipeline importado correctamente')

    # Leemos el modelo entrenado para usarlo
    package = '../models/best_model.pkl'
    model = pickle.load(open(package, 'rb'))
    print('Modelo importado correctamente')

    # Predecimos sobre el set de datos de validación 
    X_test = df.drop(['label'],axis=1)
    X_test = pipeline_transformacion.transform( X_test )
    y_test = df[['label']]
    y_pred_test=model.predict(X_test)
    
    # Generamos métricas de diagnóstico
    cm_test = confusion_matrix(y_test,y_pred_test)
    print("Matriz de confusion: ")
    print(cm_test)
    accuracy_test=accuracy_score(y_test,y_pred_test)
    print("Accuracy: ", accuracy_test)
    precision_test=precision_score(y_test,y_pred_test)
    print("Precision: ", precision_test)
    recall_test=recall_score(y_test,y_pred_test)
    print("Recall: ", recall_test)


# Validación desde el inicio
def main():
    df = eval_model('Dataset_validation_processed.csv')
    print('Finalizó la validación del Modelo')


if __name__ == "__main__":
    main()