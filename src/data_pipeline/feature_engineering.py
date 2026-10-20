import pandas as pd

class IndustrialFeatureEngineer:
    """Implementa a extração de variáveis temporais e indicadores

    físicos multivariados a partir dos sinais estabilizados.
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def adicionar_estatisticas_moveis(self) -> pd.DataFrame:
        """Gera médias móveis e desvios-padrão móveis em janelas de 1h (6 passos)

        e 6h (36 passos) calculados por equipamento.
        """
        # Passos de 10 minutos: 1h = 6 períodos; 6h = 36 períodos
        janelas = {"1h": 6, "6h": 36}
        variaveis_alvo = ["vibracao_rms_mms", "temperatura_mancal_c"]

        self.df.sort_values(
            by=["id_equipamento", "timestamp"], inplace=True
        )

        for nome_janela, passos in janelas.items():
            for var in variaveis_alvo:
                # Rolling Mean
                col_media = f"{var}_rolling_mean_{nome_janela}"
                self.df[col_media] = (
                    self.df.groupby("id_equipamento")[var]
                    .transform(
                        lambda x: x.rolling(
                            window=passos, min_periods=1
                        ).mean()
                    )
                )

                # Rolling Std
                col_desvio = f"{var}_rolling_std_{nome_janela}"
                self.df[col_desvio] = (
                    self.df.groupby("id_equipamento")[var]
                    .transform(
                        lambda x: x.rolling(
                            window=passos, min_periods=1
                        ).std()
                    )
                    .fillna(0.0)
                )

        return self.df

    def calcular_severidade_dinamica(self) -> pd.DataFrame:
        """Calcula a razão entre Vibração RMS e Rotação (RPM)."""
        # Evita divisão por zero com valor mínimo residual
        rotacao_segura = self.df["rotacao_rpm"].replace(0, 1e-6)
        self.df["severidade_dinamica"] = (
            self.df["vibracao_rms_mms"] / rotacao_segura
        )
        return self.df

    def executar_pipeline(self) -> pd.DataFrame:
        """Executa a sequência completa de engenharia de atributos."""
        self.adicionar_estatisticas_moveis()
        self.calcular_severidade_dinamica()
        return self.df