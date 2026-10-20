import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix


class IndustrialModelEvaluator:
    """Gera matrizes de confusão 4x4, métricas comparativas e

    gráficos de interpretabilidade física das decisões do modelo.
    """

    CLASSES = [
        "Normal",
        "Desgaste_Rolamento",
        "Desbalanceamento_Rotor",
        "Cavitacao_Superaquecimento",
    ]

    @staticmethod
    def avaliar_classificador(
        modelo,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        nome_modelo: str,
    ):
        y_pred = modelo.predict(X_test)
        report = classification_report(
            y_test,
            y_pred,
            target_names=IndustrialModelEvaluator.CLASSES,
            output_dict=True,
        )

        print(
            f"\n================ Desempenho: {nome_modelo} ================"
        )
        print(
            classification_report(
                y_test,
                y_pred,
                target_names=IndustrialModelEvaluator.CLASSES,
            )
        )

        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(7, 5))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=IndustrialModelEvaluator.CLASSES,
            yticklabels=IndustrialModelEvaluator.CLASSES,
        )
        plt.title(f"Matriz de Confusão 4x4 - {nome_modelo}")
        plt.xlabel("Classe Predita")
        plt.ylabel("Classe Real")
        plt.xticks(rotation=35, ha="right")
        plt.tight_layout()
        plt.show()

        return report

    @staticmethod
    def comparar_desempenho(relatorios: dict) -> pd.DataFrame:
        """Monta tabela comparativa de Acurácia, Recall Macro e F1 Macro."""
        linhas = []
        for nome, rep in relatorios.items():
            linhas.append(
                {
                    "Modelo": nome,
                    "Acurácia": round(rep["accuracy"], 4),
                    "Recall Macro": round(
                        rep["macro avg"]["recall"], 4
                    ),
                    "F1-Score Macro": round(
                        rep["macro avg"]["f1-score"], 4
                    ),
                    "Recall Desgaste": round(
                        rep["Desgaste_Rolamento"]["recall"], 4
                    ),
                    "Recall Desbalanceamento": round(
                        rep["Desbalanceamento_Rotor"]["recall"], 4
                    ),
                    "Recall Cavitação": round(
                        rep["Cavitacao_Superaquecimento"]["recall"], 4
                    ),
                }
            )
        tabela = pd.DataFrame(linhas)
        return tabela

    @staticmethod
    def plotar_feature_importance(
        modelo, feature_names: list, top_n: int = 10
    ):
        if hasattr(modelo, "feature_importances_"):
            importancias = modelo.feature_importances_
            indices = np.argsort(importancias)[-top_n:]

            plt.figure(figsize=(9, 5))
            plt.title(
                "Top Variáveis Mais Relevantes (Assinatura Física)"
            )
            plt.barh(
                range(len(indices)),
                importancias[indices],
                align="center",
                color="#008080",
            )
            plt.yticks(
                range(len(indices)), [feature_names[i] for i in indices]
            )
            plt.xlabel("Importância Relativa")
            plt.tight_layout()
            plt.show()