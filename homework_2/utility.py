from sklearn.metrics import classification_report
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score, train_test_split, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load_ds():
    return pd.read_csv("Draft_Dataset_5anomalie_3orelog.csv")

def clear_df(df):
    # Sostituisci con 'anomaly' se la label è diversa da 'normal'
    df['label'] = df['label'].apply(lambda x: 1 if x != 'normal' else 0)

    # Rimozione colonne con valori costanti
    constant_columns = [col for col in df.columns if df[col].nunique() == 1]
    df.drop(columns=constant_columns, inplace=True)

    # Rimuovi le righe con valori mancanti
    df.dropna(inplace=True)
    
    return df

def get_train_and_test_set(df):
    return train_test_split(df.drop(columns=['label']), df['label'], test_size=0.20, random_state=42)

def knn(df):
    # Inizializzazione KNN
    knn = KNeighborsClassifier(n_neighbors=5)
    
    # Divisione del dataframe in training e test set
    X_train, _, y_train, _ = get_train_and_test_set(df)

    # Fit
    knn.fit(X_train, y_train)

    # Visualizza il modello addestrato
    print(knn)

    return knn, df.drop(columns=['label']), df['label']

def cross_validation(knn, X, y):
    kf = KFold(n_splits=5, shuffle=False, random_state=None)  # Senza distribuzione delle classi
    kfold = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)  # Con distribuzione delle classi

    # Applica la cross-validation per MSE
    cv_kf_mse = cross_val_score(knn, X, y, cv=kf, scoring='neg_mean_squared_error')
    cv_kfold_mse = cross_val_score(knn, X, y, cv=kfold, scoring='neg_mean_squared_error')

    # Applica la cross-validation per accurcy
    cv_kf_accuracy = cross_val_score(knn, X, y, cv=kf, scoring='accuracy')
    cv_kfold_accuracy = cross_val_score(knn, X, y, cv=kfold, scoring='accuracy')

    # Plot dei punteggi di MSE e accuracy di ciascuna fold
    _create_plot(cv_kf_mse, cv_kfold_mse, 'MSE')

    print("Cross-validation accuracy senza distribuzione delle classi:", np.mean(cv_kf_accuracy))
    print("Cross-validation accuracy con classi distribuite:", np.mean(cv_kfold_accuracy))
    
    # Effettua la predizione con la cross-validation
    y_pred_kf = cross_val_predict(knn, X, y, cv=kf)
    y_pred_kfold = cross_val_predict(knn, X, y, cv=kfold)

    # Stampare il report di classificazione
    print("Report di classificazione 1:\n", classification_report(y, y_pred_kf))
    print("Report di classificazione 2:\n", classification_report(y, y_pred_kfold))


def _create_plot(cv_kf, cv_kfold, property):
    values = [abs(np.mean(cv_kf)), abs(np.mean(cv_kfold))] 

    # Etichette per le barre dell'istogramma
    labels = ['Classi non distribuite', 'Classi distribuite']

    # Posizioni delle barre
    x = range(len(values))

    # Creazione dell'istogramma
    plt.bar(x, values, tick_label=labels, color=['blue', 'red'])

    # Aggiunta di titoli e etichette
    plt.xlabel(property)
    plt.ylabel('Valore')
    plt.title(f'Confronto {property}')

    # Mostra l'istogramma
    plt.show()