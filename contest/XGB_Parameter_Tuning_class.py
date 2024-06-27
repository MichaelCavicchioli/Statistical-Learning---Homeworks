from sklearn.metrics import mean_squared_error, confusion_matrix, accuracy_score, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import xgboost as xgb
import seaborn as sns
import pandas as pd

class XGB_Parameter_Tuning:

    # Inizializza i parametri di default e carica il dataset
    def __init__(self) -> None:
        self.seed = 42
        self.params = {
            'eta': 0.1,
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'random_state': self.seed
        }
        self.nrounds = 1000
        self.early_stopping_rounds = 20
        self.df = pd.read_csv("dataset/Airline_Passenger_Satisfaction.csv")

    # Preprocessing dei dati: rimozione dei valori nulli ed encoding delle variabili categoriche
    def preprocess_data(self) -> None:
        self.df.dropna(inplace=True)
        label_encoders = {}
        for column in ['Gender', 'Customer.Type', 'Type.of.Travel', 'Class']:
            le = LabelEncoder()
            self.df[column] = le.fit_transform(self.df[column])
            label_encoders[column] = le        
        
        # Modifica LabelEncoder per 'satisfaction' per assegnare 0 e 1
        self.df['satisfaction'] = LabelEncoder().fit_transform(self.df['satisfaction'])
        self.X = self.df.iloc[:, :-1].values
        self.y = self.df.iloc[:, -1].values

    # Divisione del dataset in training set (67%) e test set (33%)
    def split_data(self) -> None:
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.33, random_state=self.seed
        )

    # Esegue la ricerca del miglior valore per diversi parametri
    def run_tuning(self) -> None:
        max_depth_results = self._tuning_parameter('max_depth', [1, 2, 3, 6])
        best_max_depth = max(max_depth_results, key=lambda x: x[3])[0]
        self.params['max_depth'] = best_max_depth

        min_child_weight_results = self._tuning_parameter('min_child_weight', [20, 30, 40, 50])
        best_min_child_weight = max(min_child_weight_results, key=lambda x: x[3])[0]
        self.params['min_child_weight'] = best_min_child_weight

        lambda_results = self._tuning_parameter('lambda', [1, 3, 5, 10])
        best_lambda = max(lambda_results, key=lambda x: x[3])[0]
        self.params['lambda'] = best_lambda

        alpha_results = self._tuning_parameter('alpha', [1, 3, 5, 10, 20, 50])
        best_alpha = max(alpha_results, key=lambda x: x[3])[0]
        self.params['alpha'] = best_alpha

        print("Best parameters chosen by Cross-Validation:")
        print(f"-Max Depth: {best_max_depth}")
        print(f"-Min. child weight: {best_min_child_weight}")
        print(f"-Lambda: {best_lambda}")
        print(f"-Alpha: {best_alpha}")

    # Allena il modello XGBoost utilizzando i parametri ottimizzati
    def train_best_model(self) -> None:
        dtrain = xgb.DMatrix(self.X_train, label=self.y_train)

        self.model = xgb.train(
            params = self.params,
            dtrain = dtrain,
            num_boost_round = self.nrounds,
            early_stopping_rounds = self.early_stopping_rounds,
            evals=[(dtrain, 'train')],
            verbose_eval = False
        )

    # Funzione per valutare le prestazioni del modello sul test set
    def evaluate_model(self) -> None:
        dtest = xgb.DMatrix(self.X_test, label=self.y_test)
        y_pred_prob = self.model.predict(dtest)
        y_pred = (y_pred_prob > 0.5).astype(int)

        self._plot(y_pred_prob, y_pred)
    
    # Grid search per cercare i migliori parametri del modello
    def grid_search(self) -> None:
        param_grid = {
            'max_depth': [2, 3, 6],
            'min_child_weight': [20, 30, 50],
            'lambda': [3, 5],
            'alpha': [3, 5, 10, 20],
            'eta': [0.1]
        }

        model = xgb.XGBClassifier(objective='binary:logistic', eval_metric='auc', use_label_encoder=False)

        grid_search = GridSearchCV(
            estimator = model,
            param_grid = param_grid,
            scoring = 'roc_auc',
            n_jobs = -1,
            cv = 5,
            verbose = False
        )

        grid_search.fit(self.X_train, self.y_train)
        print("Best parameters chosen by Grid search:")
        for k, v in grid_search.best_params_.items():
            print(f"-{str(k).title()}: {v}")

        # Predire i valori sul test set
        y_pred_prob = grid_search.predict(self.X_test)
        y_pred = (y_pred_prob > 0.5).astype(int)

        self._plot(y_pred_prob, y_pred)
       
    # Funzione per eseguire la cross-validation su diversi valori di un parametro
    def _tuning_parameter(self, param_name, param_values):
        results = []
        for value in param_values:
            params = self.params.copy()
            params[param_name] = value
            cv_results = xgb.cv(
                params = params,
                dtrain = xgb.DMatrix(self.X_train, label=self.y_train),
                num_boost_round = self.nrounds,
                nfold = 5,
                early_stopping_rounds = self.early_stopping_rounds,
                metrics = ["error", "auc"],
                as_pandas = True,
                seed = self.seed,
                verbose_eval = False
            )
            accuracy_mean = 1 - cv_results['test-error-mean'].iloc[-1]
            best_iteration = cv_results['test-error-mean'].idxmin()
            best_auc = cv_results['test-auc-mean'][best_iteration]
            results.append((value, best_iteration, accuracy_mean, best_auc))
            print(f"Value of {param_name}={value}: Accuracy={accuracy_mean}, AUC={best_auc}, nrounds={best_iteration}")
        
        return results

    # Crea i vari plot
    def _plot(self, y_pred_prob, y_pred) -> None:
        # MSE
        mse = mean_squared_error(self.y_test, y_pred_prob)
        print(f"MSE: {mse}")

        # Confusion Matrix
        conf_matrix = confusion_matrix(self.y_test, y_pred)
        accuracy = accuracy_score(self.y_test, y_pred)
        print(f"Accuracy: {accuracy}")

        sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
        plt.title('Confusion matrix')
        plt.xlabel('Predicted values')
        plt.ylabel('Real values')
        plt.show()

        # ROC-AUC
        roc_auc = roc_auc_score(self.y_test, y_pred_prob)
        print(f"ROC-AUC: {roc_auc}")

        # Curva ROC
        fpr, tpr, _ = roc_curve(self.y_test, y_pred_prob)
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.fill_between(fpr, tpr, alpha=0.2, color='darkorange')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC)')
        plt.legend(loc="lower right")
        plt.show()
