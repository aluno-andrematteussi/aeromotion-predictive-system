from fastapi.testclient import TestClient
from src.api.app import app

client = TestClient(app)


def test_health_check_status_e_modelo():
  """Valida se o serviço está ativo e reporta o modelo correto."""
  resposta = client.get("/health")
  assert resposta.status_code == 200
  dados = resposta.json()
  assert dados["status"] == "healthy"
  assert dados["versao_modelo"] == "LGBMClassifier"
  assert dados["features_esperadas"] == 16


def test_predict_regime_normal():
  """Valida inferência nominal com parâmetros calibrados no espaço do Isolation Forest."""
  payload = {
      "id_equipamento": "TURB-101",
      "temperatura_mancal_c": 62.5,
      "vibracao_rms_mms": 2.1,
      "pressao_oleo_bar": 4.2,
      "rotacao_rpm": 2850.0,
      "emissao_acustica_db": 52.0,
      "corrente_eletrica_amp": 28.0,
      "horas_operacao_acumuladas": 1250.0,
  }
  resposta = client.post("/predict", json=payload)
  assert resposta.status_code == 200
  dados = resposta.json()
  assert dados["classe_predita"] == 0
  assert dados["diagnostico"] == "Operação Normal"
  assert dados["anomalia_detectada"] is False
  assert dados["status_operacional"] == "NORMAL"


def test_predict_rejeicao_limite_fisico_pydantic():
  """Valida o bloqueio de requisições com violação de limites físicos (pressão < 0.5 bar)."""
  payload_invalido = {
      "id_equipamento": "TURB-102",
      "temperatura_mancal_c": 60.0,
      "vibracao_rms_mms": 2.0,
      "pressao_oleo_bar": 0.2,  # Violação de limite mínimo (ge=0.5)
      "rotacao_rpm": 2900.0,
      "emissao_acustica_db": 50.0,
      "corrente_eletrica_amp": 26.0,
      "horas_operacao_acumuladas": 1000.0,
  }
  resposta = client.post("/predict", json=payload_invalido)
  assert resposta.status_code == 422


def test_predict_batch():
  """Valida inferência em lote processando múltiplas bancadas."""
  payload_lote = {
      "leituras": [
          {
              "id_equipamento": "TURB-101",
              "temperatura_mancal_c": 62.0,
              "vibracao_rms_mms": 2.2,
              "pressao_oleo_bar": 4.2,
              "rotacao_rpm": 2800.0,
              "emissao_acustica_db": 52.0,
              "corrente_eletrica_amp": 28.0,
              "horas_operacao_acumuladas": 1200.0,
          },
          {
              "id_equipamento": "TURB-103",
              "temperatura_mancal_c": 110.0,
              "vibracao_rms_mms": 12.0,
              "pressao_oleo_bar": 2.0,
              "rotacao_rpm": 3400.0,
              "emissao_acustica_db": 85.0,
              "corrente_eletrica_amp": 45.0,
              "horas_operacao_acumuladas": 3500.0,
          },
      ]
  }
  resposta = client.post("/predict-batch", json=payload_lote)
  assert resposta.status_code == 200
  dados = resposta.json()
  assert len(dados) == 2
  assert dados[0]["id_equipamento"] == "TURB-101"
  assert dados[1]["id_equipamento"] == "TURB-103"