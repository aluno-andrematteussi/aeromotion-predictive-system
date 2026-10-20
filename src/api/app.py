from typing import List
import os
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, status
from src.api.schemas import (
    BatchSensorReadingInput,
    HealthResponse,
    PredictionOutput,
    SensorReadingInput,
)

app = FastAPI(
    title="AeroMotion Predictive & Edge AI Service",
    description="Microsserviço de inferência para deteção precoce de falhas em turbinas industriais.",
    version="1.0.0",
)

# Mapeamento oficial de alvos de diagnóstico
MAPA_DIAGNOSTICO = {
    0: "Operação Normal",
    1: "Desgaste de Rolamento",
    2: "Desbalanceamento de Rotor",
    3: "Cavitação / Superaquecimento",
}

# Inicialização e carregamento dos modelos na memória
DIRETORIO_MODELOS = os.getenv("MODELS_DIR", "models")
try:
    classifier = joblib.load(
        os.path.join(DIRETORIO_MODELOS, "classifier_model.joblib")
    )
    isolation_forest = joblib.load(
        os.path.join(DIRETORIO_MODELOS, "isolation_forest.joblib")
    )
    scaler = joblib.load(os.path.join(DIRETORIO_MODELOS, "scaler.joblib"))
    feature_names = joblib.load(
        os.path.join(DIRETORIO_MODELOS, "feature_names.joblib")
    )
except Exception as e:
    raise RuntimeError(
        f"Falha ao carregar artefactos em '{DIRETORIO_MODELOS}': {str(e)}"
    )


def _processar_features_entrada(
    dados: SensorReadingInput,
) -> pd.DataFrame:
    """Deriva a Severidade Dinâmica e preenche médias móveis para o vetor de entrada."""
    registo = dados.model_dump()

    # Cálculo da Severidade Dinâmica
    rotacao_segura = (
        registo["rotacao_rpm"] if registo["rotacao_rpm"] > 0 else 1e-6
    )
    registo["severidade_dinamica"] = (
        registo["vibracao_rms_mms"] / rotacao_segura
    )

    # Para inferência instantânea sem buffer de histórico, as médias assumem o valor pontual
    registo["vibracao_rms_mms_rolling_mean_1h"] = registo[
        "vibracao_rms_mms"
    ]
    registo["vibracao_rms_mms_rolling_std_1h"] = 0.0
    registo["vibracao_rms_mms_rolling_mean_6h"] = registo[
        "vibracao_rms_mms"
    ]
    registo["vibracao_rms_mms_rolling_std_6h"] = 0.0
    registo["temperatura_mancal_c_rolling_mean_1h"] = registo[
        "temperatura_mancal_c"
    ]
    registo["temperatura_mancal_c_rolling_std_1h"] = 0.0
    registo["temperatura_mancal_c_rolling_mean_6h"] = registo[
        "temperatura_mancal_c"
    ]
    registo["temperatura_mancal_c_rolling_std_6h"] = 0.0

    df_entrada = pd.DataFrame([registo])
    df_selecionado = df_entrada[feature_names]
    return df_selecionado


@app.get(
    "/health", response_model=HealthResponse, tags=["Infraestrutura"]
)
def check_health():
    """Valida o estado de execução da API e dos modelos."""
    return {
        "status": "healthy",
        "versao_servico": "1.0.0",
        "versao_modelo": type(classifier).__name__,
        "features_esperadas": len(feature_names),
    }


@app.post(
    "/predict", response_model=PredictionOutput, tags=["Inferência"]
)
def predict_single(reading: SensorReadingInput):
    """Executa a inferência preditiva em tempo real para uma única leitura de bancada."""
    try:
        X_bruto = _processar_features_entrada(reading)
        X_escalado = pd.DataFrame(
            scaler.transform(X_bruto), columns=feature_names
        )

        classe_idx = int(classifier.predict(X_escalado)[0])
        probabilidades = {}
        if hasattr(classifier, "predict_proba"):
            probs = classifier.predict_proba(X_escalado)[0]
            probabilidades = {
                MAPA_DIAGNOSTICO[i]: round(float(p), 4)
                for i, p in enumerate(probs)
            }

        # Deteção não supervisionada de anomalia (-1 = anómalo, 1 = normal)
        anomalia = bool(isolation_forest.predict(X_escalado)[0] == -1)

        status_texto = (
            "ALERTA_INTERVENÇÃO"
            if (classe_idx != 0 or anomalia)
            else "NORMAL"
        )

        return {
            "id_equipamento": reading.id_equipamento,
            "classe_predita": classe_idx,
            "diagnostico": MAPA_DIAGNOSTICO.get(
                classe_idx, "Indeterminado"
            ),
            "probabilidades": probabilidades,
            "anomalia_detectada": anomalia,
            "status_operacional": status_texto,
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erro de inferência: {str(e)}",
        )


@app.post(
    "/predict-batch",
    response_model=List[PredictionOutput],
    tags=["Inferência"],
)
def predict_batch(batch: BatchSensorReadingInput):
    """Processa múltiplas leituras em lote simultaneamente."""
    respostas = []
    for leitura in batch.leituras:
        respostas.append(predict_single(leitura))
    return respostas