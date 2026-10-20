import os
import joblib
import pandas as pd
from imblearn.over_sampling import SMOTE
from lightgbm import LGBMClassifier
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import RobustScaler


class IndustrialModelTrainer:
    """Orquestra a divisão temporal, o balanceamento de classes e

    o treino dos modelos supervisionados e não supervisionados.
    """

    def __init__(
        self,
        caminho_dados: str = "data/processed/features_eng.parquet",
    ):
        self.caminho_dados = caminho_dados
        self.df: pd.DataFrame = pd.DataFrame()
        self.X_train, self.X_test = None, None
        self.y_train, self.y_test = None, None
        self.scaler = RobustScaler()
        self.feature_names = []

    def preparar_dados(
        self,
        target_col: str = "tipo_falha",
        proporcao_treino: float = 0.7,
    ):
        """Lê os dados, ordena cronologicamente e aplica a divisão temporal estrita."""
        self.df = pd.read_parquet(self.caminho_dados)
        self.df.sort_values(by="timestamp", inplace=True)
        self.df.reset_index(drop=True, inplace=True)

        colunas_ignorar = [
            "timestamp",
            "id_equipamento",
            "modo_operacao",
            target_col,
        ]
        self.feature_names = [
            col
            for col in self.df.columns
            if col not in colunas_ignorar
        ]

        X = self.df[self.feature_names]
        y = self.df[target_col].astype(int)

        # Divisão temporal (sem shuffle)
        ponto_corte = int(len(self.df) * proporcao_treino)
        X_train_raw, X_test_raw = (
            X.iloc[:ponto_corte],
            X.iloc[ponto_corte:],
        )
        self.y_train, self.y_test = (
            y.iloc[:ponto_corte],
            y.iloc[ponto_corte:],
        )

        # Normalização com RobustScaler ajustado exclusivamente no conjunto de treino
        self.X_train = pd.DataFrame(
            self.scaler.fit_transform(X_train_raw),
            columns=self.feature_names,
        )
        self.X_test = pd.DataFrame(
            self.scaler.transform(X_test_raw),
            columns=self.feature_names,
        )

        return self.X_train, self.X_test, self.y_train, self.y_test

    def balancear_treino_smote(self):
        """Aplica SMOTE exclusivamente aos dados de treino."""
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(
            self.X_train, self.y_train
        )
        return X_resampled, y_resampled

    def treinar_supervisionados(self, X_train, y_train):
        """Treina e otimiza hiperparâmetros de Random Forest e LightGBM."""
        modelos = {
            "RandomForest": {
                "estimator": RandomForestClassifier(
                    random_state=42, class_weight="balanced"
                ),
                "params": {
                    "n_estimators": [100, 200],
                    "max_depth": [10, 20, None],
                    "min_samples_split": [2, 5],
                },
            },
            "LightGBM": {
                "estimator": LGBMClassifier(
                    random_state=42, class_weight="balanced", verbose=-1
                ),
                "params": {
                    "n_estimators": [100, 200],
                    "learning_rate": [0.01, 0.05, 0.1],
                    "num_leaves": [31, 63],
                },
            },
        }

        melhores_modelos = {}
        for nome, config in modelos.items():
            search = RandomizedSearchCV(
                config["estimator"],
                config["params"],
                n_iter=4,
                scoring="recall_macro",
                cv=3,
                random_state=42,
                n_jobs=-1,
            )
            search.fit(X_train, y_train)
            melhores_modelos[nome] = search.best_estimator_

        return melhores_modelos

    def treinar_isolation_forest(self):
        """Treina o Isolation Forest estritamente sobre dados de operação normal (tipo_falha == 0)."""
        idx_normais = self.y_train[self.y_train == 0].index
        X_normais = self.X_train.loc[idx_normais]

        iso = IsolationForest(
            n_estimators=150,
            contamination=0.03,
            random_state=42,
            n_jobs=-1,
        )
        iso.fit(X_normais)
        return iso

    def salvar_artefactos(
        self,
        melhor_classificador,
        modelo_iso,
        pasta_destino: str = "models",
    ):
        os.makedirs(pasta_destino, exist_ok=True)
        joblib.dump(
            melhor_classificador,
            os.path.join(pasta_destino, "classifier_model.joblib"),
        )
        joblib.dump(
            modelo_iso,
            os.path.join(pasta_destino, "isolation_forest.joblib"),
        )
        joblib.dump(
            self.scaler, os.path.join(pasta_destino, "scaler.joblib")
        )
        joblib.dump(
            self.feature_names,
            os.path.join(pasta_destino, "feature_names.joblib"),
        )
        print(f"Artefactos exportados com sucesso em '{pasta_destino}/'.")