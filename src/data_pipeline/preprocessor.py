import numpy as np
import pandas as pd

class IndustrialDataSanitizer:
    """Responsável pela limpeza, tratamento de outliers e interpolação

    temporal dos dados das bancadas de testes da AeroMotion Dynamics.
    """

    LIMITES_FISICOS = {
        "temperatura_mancal_c": {"min": 0.0, "max": 150.0},  # Faixa normal: 45°C a 85°C
        "vibracao_rms_mms": {"min": 0.0, "max": 25.0},  # Faixa normal: 1.0 a 4.5 mm/s
        "pressao_oleo_bar": {"min": 0.5, "max": 10.0},  # Faixa normal: 3.5 a 5.2 Bar
        "rotacao_rpm": {"min": 500.0, "max": 5000.0},  # Faixa normal: 2300 a 3600 RPM
        "emissao_acustica_db": {"min": 20.0, "max": 100.0},  # Faixa normal: 40 a 68 dB
        "corrente_eletrica_amp": {"min": 5.0, "max": 60.0},  # Faixa normal: 18 a 42 A
    }

    def __init__(self, caminho_ficheiro: str):
        self.caminho_ficheiro = caminho_ficheiro
        self.df: pd.DataFrame = pd.DataFrame()

    def carregar_dados(self) -> pd.DataFrame:
        """Carrega o ficheiro CSV e formata os carimbos temporais."""
        self.df = pd.read_csv(self.caminho_ficheiro)
        self.df["timestamp"] = pd.to_datetime(self.df["timestamp"])
        self.df.sort_values(
            by=["id_equipamento", "timestamp"], inplace=True
        )
        self.df.reset_index(drop=True, inplace=True)
        return self.df

    def inspecionar_estrutura(self) -> dict:
        """Avalia consistência de amostragem (10 min) e dados em falta por bancada."""
        metricas = {
            "total_registos": len(self.df),
            "equipamentos": self.df["id_equipamento"].unique().tolist(),
            "nulos_por_coluna": self.df.isnull().sum().to_dict(),
            "nulos_por_equipamento": self.df.groupby("id_equipamento")
            .apply(lambda x: x.isnull().sum())
            .to_dict(),
            "intervalos_temporais": {},
        }

        # Verificação do intervalo de amostragem de 10 minutos
        for equip, grupo in self.df.groupby("id_equipamento"):
            deltas = grupo["timestamp"].diff().dropna()
            fora_de_frequencia = (
                deltas != pd.Timedelta(minutes=10)
            ).sum()
            metricas["intervalos_temporais"][equip] = {
                "desvios_intervalo_10min": int(fora_de_frequencia)
            }

        return metricas

    def filtrar_ruidos_eletromagneticos(self) -> pd.DataFrame:
        """Converte em NaN leituras anómalas de interferência eletromagnética

        e valores que violem os limites físicos de operação.
        """
        # Remoção explícita de códigos de erro de sensores conhecidos
        self.df.loc[
            self.df["temperatura_mancal_c"] >= 900.0,
            "temperatura_mancal_c",
        ] = np.nan
        self.df.loc[
            self.df["vibracao_rms_mms"] < 0.0, "vibracao_rms_mms"
        ] = np.nan

        # Aplicação de limites físicos operacionais
        for coluna, limites in self.LIMITES_FISICOS.items():
            if coluna in self.df.columns:
                mascara_invalida = (self.df[coluna] < limites["min"]) | (
                    self.df[coluna] > limites["max"]
                )
                self.df.loc[mascara_invalida, coluna] = np.nan

        return self.df

    def interpolar_series_temporais(
        self, metodo: str = "time"
    ) -> pd.DataFrame:
        """Aplica interpolação temporal por equipamento para preencher

        falhas contínuas respeitando a física do processo.
        """
        colunas_numericas = [
            "temperatura_mancal_c",
            "vibracao_rms_mms",
            "pressao_oleo_bar",
            "rotacao_rpm",
            "emissao_acustica_db",
            "corrente_eletrica_amp",
            "horas_operacao_acumuladas",
        ]

        # Garante a ordenação cronológica antes da interpolação
        self.df.sort_values(
            by=["id_equipamento", "timestamp"], inplace=True
        )

        dfs_processados = []
        for _, grupo in self.df.groupby("id_equipamento"):
            grupo_temp = grupo.copy().set_index("timestamp")
            grupo_temp[colunas_numericas] = grupo_temp[
                colunas_numericas
            ].interpolate(method=metodo, limit_direction="both")
            dfs_processados.append(grupo_temp.reset_index())

        self.df = pd.concat(dfs_processados, ignore_index=True)
        return self.df