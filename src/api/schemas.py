from typing import List
from pydantic import BaseModel, Field


class SensorReadingInput(BaseModel):
    id_equipamento: str = Field(
        ...,
        examples=["TURB-101"],
        description="Identificador da bancada de ensaio",
    )
    temperatura_mancal_c: float = Field(
        ...,
        ge=0.0,
        le=150.0,
        description="Temperatura do mancal principal em °C",
    )
    vibracao_rms_mms: float = Field(
        ...,
        ge=0.0,
        le=25.0,
        description="Nível de vibração global do eixo em mm/s RMS",
    )
    pressao_oleo_bar: float = Field(
        ...,
        ge=0.5,
        le=10.0,
        description="Pressão do sistema de lubrificação em Bar",
    )
    rotacao_rpm: float = Field(
        ...,
        ge=500.0,
        le=5000.0,
        description="Velocidade angular do rotor em RPM",
    )
    emissao_acustica_db: float = Field(
        ...,
        ge=20.0,
        le=100.0,
        description="Ruído ultrassónico de fricção mecânica em dB",
    )
    corrente_eletrica_amp: float = Field(
        ...,
        ge=5.0,
        le=60.0,
        description="Corrente consumida pelo sistema de acionamento em A",
    )
    horas_operacao_acumuladas: float = Field(
        ...,
        ge=0.0,
        description="Tempo acumulado de funcionamento do motor em horas",
    )


class BatchSensorReadingInput(BaseModel):
    leituras: List[SensorReadingInput]


class PredictionOutput(BaseModel):
    id_equipamento: str
    classe_predita: int
    diagnostico: str
    probabilidades: dict
    anomalia_detectada: bool
    status_operacional: str


class HealthResponse(BaseModel):
    status: str
    versao_servico: str
    versao_modelo: str
    features_esperadas: int