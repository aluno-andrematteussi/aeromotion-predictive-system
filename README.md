# AeroMotion Dynamics — IA e Análise de Dados em Sistemas Industriais
### Solução Preditiva Multimodal End-to-End, Visão Computacional e Deploy Edge para Turbomáquinas

<p align="left">
  <img src="https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/FastAPI-0.100-009688?logo=fastapi&logoColor=white" alt="FastAPI">
  <img src="https://img.shields.io/badge/Docker-Edge%20Ready-2496ED?logo=docker&logoColor=white" alt="Docker">
  <img src="https://img.shields.io/badge/Pytest-100%25%20Passed-brightgreen?logo=pytest&logoColor=white" alt="Pytest">
  <img src="https://img.shields.io/badge/OpenCV-Multimodal-5C3EE8?logo=opencv&logoColor=white" alt="OpenCV">
</p>

---

## 1. Contexto e Objetivos Industriais

A **AeroMotion Dynamics S.A.** (Divisão de Turbinas & Propulsão de Alta Precisão) opera quatro bancadas de testes de desenvolvimento instrumentadas (**TURB-101 a TURB-104**) com uma malha densa de sensores IoT coletando parâmetros de operação contínua a cada 10 minutos.

Este repositório consolida o desenvolvimento de uma solução de Inteligência Artificial de ponta a ponta (*End-to-End MLOps & Computer Vision*) projetada para:
* **Diagnóstico Precoce de Falhas Mecânicas**: Classificação preditiva multiclasse e detecção não supervisionada de anomalias com foco em Recall superior a 90% nas falhas críticas.
* **Inspeção Óptica e Térmica Multimodal em Tempo Real**: Extração de contornos e circularidade via Canny para validação de pás de turbina e delimitação de zonas térmicas críticas (*hotspots*) no espaço HSV.
* **Deploy para Edge Computing**: Microsserviço RESTful assíncrono em FastAPI validado por schemas Pydantic e conteinerizado em Docker para integração com redes industriais OT/IT.

---

## 2. Arquitetura do Repositório

O projeto adota uma arquitetura modular, orientada a componentes desacoplados e aderente aos padrões de engenharia de software e PEP 8:

```text
aeromotion-predictive-system/
├── data/
│   ├── raw/
│   │   ├── dados_sensores_aeromotion.csv    # 12.000 registros de telemetria das bancadas
│   │   └── vision/                          # Amostras para testes ópticos e térmicos
│   └── processed/
│       └── features_eng.parquet             # Base limpa com janelas móveis e severidade
│
├── notebooks/
│   ├── 01_eda_e_saneamento.ipynb            # Etapa 1: Auditoria temporal, filtros EMI e EDA
│   ├── 02_modelagem_supervisionada.ipynb    # Etapa 2: SMOTE, LightGBM, Isolation Forest e MLOps
│   └── 03_visao_computacional.ipynb         # Etapa 3: Inspeção geométrica (Canny) e térmica (HSV)
│
├── src/
│   ├── __init__.py
│   ├── data_pipeline/                       # Módulos de pré-processamento e engenharia de dados
│   │   ├── __init__.py
│   │   ├── preprocessor.py                  # Interpolação temporal e filtros de limites físicos
│   │   └── feature_engineering.py           # Estatísticas móveis (1h/6h) e Severidade Dinâmica
│   ├── ml/                                  # Treinamento preditivo e interpretabilidade
│   │   ├── __init__.py
│   │   ├── train.py                         # Orquestrador de treino (SMOTE, LightGBM, Isolation Forest)
│   │   └── evaluation.py                    # Matriz de confusão 4x4 e relevância física de features
│   ├── vision/                              # Processamento digital de imagens industriais
│   │   ├── __init__.py
│   │   ├── geometric_inspector.py           # Deteção de bordas com Canny, área e circularidade
│   │   └── thermal_inspector.py             # Segmentação em espaço HSV e hotspots críticos
│   └── api/                                 # Microsserviço de inferência de borda
│       ├── __init__.py
│       ├── app.py                           # Rotas FastAPI (/health, /predict, /predict-batch)
│       └── schemas.py                       # Contratos de dados e validações estritas Pydantic
│
├── models/                                  # Modelos serializados e transformadores ajustados
│   ├── classifier_model.joblib
│   ├── feature_names.joblib
│   ├── isolation_forest.joblib
│   └── scaler.joblib
│
├── docker/                                  # Configurações de conteinerização
│   ├── Dockerfile                           # Imagem otimizada baseada em Python 3.10-slim
│   └── docker-compose.yml                   # Orquestração do serviço no nó de Edge
│
├── docs/                                    # Documentação técnica e relatórios regulatórios
│   ├── relatorio_saneamento.pdf             # Relatório técnico de pré-processamento e EDA
│   ├── plano_mlops_drift.pdf                # Plano de monitoramento de Data e Concept Drift (12 meses)
│   └── parecer_tecnico_etico_regulatorio.pdf# Parecer de conformidade LGPD, EU AI Act e cibersegurança OT/IT
│
├── tests/                                   # Suíte de testes automatizados com Pytest
│   ├── __init__.py
│   └── test_api.py                          # Testes unitários e de integração dos endpoints
│
├── .dockerignore                            # Exclusão de artefatos pesados no build Docker
├── .gitignore                               # Isolamento de ambientes virtuais e caches
├── pytest.ini                               # Configuração de caminhos e descoberta de testes
├── requirements.txt                         # Dependências estritas do projeto
└── README.md
```

---

## 3. Matriz Tecnológica

| Categoria | Tecnologias | Finalidade no Projeto |
| :--- | :--- | :--- |
| **Linguagem & Ambiente** | Python 3.10+, VS Code, Jupyter | Desenvolvimento modular, prototipagem e scripts. |
| **Engenharia de Dados** | Pandas, NumPy, SciPy, PyArrow | Limpeza, interpolação temporal, filtros físicos e Parquet. |
| **EDA & Visualização** | Matplotlib, Seaborn | Matrizes de correlação multivariada e análise por regime. |
| **Machine Learning** | Scikit-Learn, LightGBM, Imbalanced-Learn | Modelagem multiclasse, balanceamento com SMOTE e Isolation Forest. |
| **Visão Computacional** | OpenCV (Headless) | Deteção de bordas (Canny) e segmentação radiométrica HSV. |
| **Microsserviço & Deploy** | FastAPI, Pydantic, Uvicorn, Docker | API assíncrona, validação de tipos e conteinerização Edge. |
| **Qualidade & Testes** | Pytest, HTTPX | Testes automatizados de integração e validação de contratos HTTP. |

---

## 4. Engenharia de Dados e Modelagem

### 4.1. Diagnóstico e Séries Temporais (Etapa 1)
* **Filtragem de Ruído Eletromagnético (EMI)**: Eliminação de leituras espúrias de saturação causadas pelo ambiente fabril (ex.: temperaturas de $999,9^\circ\text{C}$ e vibrações negativas de $-99,0\text{ mm/s}$).
* **Interpolação Temporal por Bancada**: Preenchimento de falhas de sinal respeitando a continuidade física de cada equipamento isoladamente (TURB-101 a TURB-104).
* **Rolling Features (1h e 6h)**: Médias e desvios móveis em janelas de 6 e 36 períodos de 10 minutos para vibração e temperatura, capturando dinâmicas transitórias e acúmulo de calor.
* **Severidade Dinâmica**: Indicador físico derivado que isola vibrações anormais da velocidade de rotação:
  $$\text{Severidade Dinâmica} = \frac{\text{Vibração RMS (mm/s)}}{\text{Rotação (RPM)}}$$

### 4.2. Inteligência Preditiva e Deteção de Anomalias (Etapa 2)
* **Particionamento Cronológico Estrito**: Divisão de 70% para treino e 30% para teste sem embaralhamento aleatório (*shuffle=False*).
* **Balanceamento com SMOTE**: Reamostragem aplicada exclusivamente sobre o conjunto de treino, corrigindo a assimetria da base (~12% de falhas reais).
* **Classificação Multiclasse Supervisionada**: Estimador `LightGBM` otimizado para maximizar o Recall (> 90%) nas 4 classes operacionais:
  * `0`: Operação Normal
  * `1`: Desgaste de Rolamento
  * `2`: Desbalanceamento de Rotor
  * `3`: Cavitação / Superaquecimento
* **Deteção Não Supervisionada (Isolation Forest)**: Ajustado unicamente com dados de operação nominal para acusar comportamentos atípicos e falhas ainda não documentadas.

### 4.3. Visão Computacional Multimodal (Etapa 3)
* **Inspeção Dimensional e Geométrica**: Suavização gaussiana, filtro Canny e extração de contornos para cálculo de circularidade ($C = \frac{4\pi \cdot \text{Área}}{\text{Perímetro}^2}$) e segregação de pás com defeitos de usinagem.
* **Inspeção Térmica Infravermelha**: Segmentação no espaço de cor HSV e limiarização com morfologia matemática para mapear a área acumulada de *hotspots* críticos.

---

## 5. Instalação e Execução Local

### 5.1. Configuração do Ambiente Local

```bash
git clone https://github.com/andre_matteussi/aeromotion-predictive-system.git
cd aeromotion-predictive-system
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 5.2. Execução dos Testes Automatizados
```bash
pytest tests/ -v
```

### 5.3. Inicialização da API FastAPI
```bash
uvicorn src.api.app:app --reload --port 8000
```
* **Swagger UI (Documentação Interativa)**: `http://localhost:8000/docs`
* **Redoc**: `http://localhost:8000/redoc`

---

## 6. Implantação Conteinerizada via Docker (Edge Computing)

Para implantar o microsserviço no ambiente de borda industrial via Docker Compose:

```bash
# Construção da imagem e inicialização do serviço em segundo plano
docker compose -f docker/docker-compose.yml up --build -d

# Verificação do status de prontidão da API
curl http://localhost:8000/health
```

---

## 7. Catálogo de Endpoints RESTful

### `GET /health`
Inspeciona a prontidão do serviço, a versão do classificador carregado e o total de variáveis exigidas.

### `POST /predict`
Executa inferência diagnóstica em tempo real para uma única leitura de telemetria.

**Exemplo de Payload:**
```json
{
  "id_equipamento": "TURB-101",
  "temperatura_mancal_c": 62.5,
  "vibracao_rms_mms": 2.1,
  "pressao_oleo_bar": 4.2,
  "rotacao_rpm": 2850.0,
  "emissao_acustica_db": 52.0,
  "corrente_eletrica_amp": 28.0,
  "horas_operacao_acumuladas": 1250.0
}
```

**Exemplo de Resposta (HTTP 200):**
```json
{
  "id_equipamento": "TURB-101",
  "classe_predita": 0,
  "diagnostico": "Operação Normal",
  "probabilidades": {
    "Operação Normal": 0.985,
    "Desgaste de Rolamento": 0.005,
    "Desbalanceamento de Rotor": 0.005,
    "Cavitação / Superaquecimento": 0.005
  },
  "anomalia_detectada": false,
  "status_operacional": "NORMAL"
}
```

### `POST /predict-batch`
Recebe uma coleção de registros de telemetria e executa predições simultâneas em lote.

---

## 8. Governança, Regulamentações e MLOps

O repositório disponibiliza na pasta `docs/` o detalhamento das diretrizes de governança e engenharia:
* **Plano de MLOps (`docs/plano_mlops_drift.pdf`)**: Protocolos de monitoramento contínuo em janela de 12 meses com testes de Kolmogorov-Smirnov e PSI para *Data Drift*, além de gatilhos automáticos de retreino para *Concept Drift* baseados no Recall por modo de falha.
* **Parecer Regulatório e Ético (`docs/parecer_tecnico_etico_regulatorio.pdf`)**: Análise de impactos ocupacionais e transição com *upskilling*, conformidade estrita com a LGPD via mascaramento facial ativo, enquadramento como Sistema de Alto Risco sob o *EU AI Act* e blindagem da rede OT/IT contra envenenamento de dados (*data poisoning*).