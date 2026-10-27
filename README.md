# AeroMotion Dynamics — IA e Análise de Dados em Sistemas Industriais
### Solução Preditiva Multimodal End-to-End, Visão Computacional e Deploy Edge para Turbomáquinas

<p align="left">
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/FastAPI-0.100%2B-009688?logo=fastapi&logoColor=white" alt="FastAPI">
  <img src="https://img.shields.io/badge/LightGBM-Multiclass-FF6F00?logo=lightgbm&logoColor=white" alt="LightGBM">
  <img src="https://img.shields.io/badge/OpenCV-Computer%20Vision-5C3EE8?logo=opencv&logoColor=white" alt="OpenCV">
  <img src="https://img.shields.io/badge/Docker-Edge%20Ready-2496ED?logo=docker&logoColor=white" alt="Docker">
  <img src="https://img.shields.io/badge/Pytest-Automated%20Tests-brightgreen?logo=pytest&logoColor=white" alt="Pytest">
  <img src="https://img.shields.io/badge/License-Proprietary-red" alt="License">
</p>

---

## 1. Visão Geral e Contexto Industrial

A **AeroMotion Dynamics S.A.** (Divisão de Turbinas & Propulsão de Alta Precisão) opera quatro bancadas instrumentadas de testes mecânicos e termodinâmicos (**TURB-101 a TURB-104**). Cada bancada conta com uma malha de sensores industriais IoT que registram leituras de telemetria a cada 10 minutos.

Este repositório consolida a arquitetura completa de Inteligência Artificial e Engenharia de Dados (*End-to-End Industrial MLOps & Computer Vision*), desenvolvida para atuar na prevenção proativa de falhas catastróficas, automação da inspeção de componentes e integração em tempo real com sistemas OT/IT fabris.

### Objetivos Estratégicos do Sistema
* **Diagnóstico Preditivo Precoce**: Classificação multiclasse supervisionada aliada a detector não supervisionado de anomalias, garantindo $\text{Recall} \ge 90\%$ nas falhas operacionais críticas.
* **Inspeção Multimodal Automatizada**: Processamento digital de imagens ópticas para detecção de deformações geométricas em pás de turbina (Canny e circularidade) e termografia infravermelha para delimitação de zonas térmicas críticas (*hotspots* no espaço HSV).
* **Microsserviço Resiliente para Borda (Edge Computing)**: API assíncrona em FastAPI com validação rigorosa via Pydantic v2, conteinerizada em Docker e otimizada para tempo de resposta sub-15ms por ciclo de telemetria.
* **Governança e Regulamentação**: Monitoramento contínuo de desvios (*Data & Concept Drift*) e aderência regulatória à LGPD, diretrizes de cibersegurança OT/IT e ao *EU AI Act* (sistemas industriais de alto risco).

---

## 2. Arquitetura e Estrutura do Repositório

O projeto segue boas práticas de engenharia de software (PEP 8, separação de responsabilidades e desacoplamento modular):

```text
aeromotion-predictive-system/
├── data/
│   ├── raw/
│   │   ├── dados_sensores_aeromotion.csv    # 12.000 registros de telemetria bruta das bancadas
│   │   └── vision/                          # Amostras para testes ópticos e termográficos
│   └── processed/
│       └── features_eng.parquet             # Base saneada, tratada e enriquecida com janelas móveis
│
├── notebooks/
│   ├── 01_eda_e_saneamento.ipynb            # Análise exploratória, auditoria temporal e filtros EMI
│   ├── 02_modelagem_supervisionada.ipynb    # Balanceamento SMOTE, LightGBM, Isolation Forest e validação
│   └── 03_visao_computacional.ipynb         # Inspeção dimensional (Canny) e termográfica (HSV)
│
├── src/
│   ├── __init__.py
│   └── dashboard/                                 # Aplicação via Streamlit do que contém no projeto
│   │   └── feature_engineering.py           # Janelas móveis (1h/6h) e Severidade Dinâmica
│   ├── __init__.py
│   ├── data_pipeline/                       # Pipeline de saneamento e enriquecimento de dados
│   │   ├── __init__.py
│   │   ├── preprocessor.py                  # Interpolação temporal isolada e remoção de ruídos EMI
│   │   └── feature_engineering.py           # Janelas móveis (1h/6h) e Severidade Dinâmica
│   ├── ml/                                  # Treinamento e avaliação de modelos preditivos
│   │   ├── __init__.py
│   │   ├── train.py                         # Orquestrador de treino (LightGBM + Isolation Forest)
│   │   └── evaluation.py                    # Matriz de confusão, métricas e interpretabilidade
│   ├── vision/                              # Processamento digital de imagens industriais
│   │   ├── __init__.py
│   │   └── generate_samples.py             # Exemplos de situações para validação das inspeções
│   │   ├── geometric_inspector.py           # Análise de contornos, perímetro, área e circularidade
│   │   └── thermal_inspector.py             # Limiarização HSV e quantificação de hotspots
│   └── api/                                 # Microsserviço de inferência para borda industrial
│       ├── __init__.py
│       ├── app.py                           # Rotas FastAPI (/health, /predict, /predict-batch, /vision/*)
│       └── schemas.py                       # Contratos de dados e validações estritas Pydantic
│
├── models/                                  # Artefatos serializados e transformadores ajustados
│   ├── classifier_model.joblib              # Modelo LightGBM multiclasse treinado
│   ├── isolation_forest.joblib              # Modelo Isolation Forest para anomalias
│   ├── scaler.joblib                        # StandardScaler ajustado para normalização
│   └── feature_names.joblib                 # Lista ordenada dos atributos esperados na inferência
│
├── docker/                                  # Configurações de conteinerização
│   ├── Dockerfile                           # Imagem enxuta baseada em Python 3.10-slim
│   └── docker-compose.yml                   # Orquestração do serviço no nó de Edge
│
├── docs/                                    # Documentação técnica e relatórios regulatórios
│   ├── relatorio_saneamento.pdf             # Relatório técnico de pré-processamento e EDA
│   ├── plano_mlops_drift.pdf                # Protocolo de monitoramento de Drift (KS-Test, PSI)
│   └── parecer_tecnico_etico_regulatorio.pdf# Conformidade LGPD, EU AI Act e cibersegurança OT/IT
│
├── tests/                                   # Suíte de testes automatizados com Pytest
│   ├── __init__.py
│   └── test_api.py                          # Testes unitários e de integração dos endpoints
│
├── .dockerignore                            # Filtro de exclusão de artefatos no build Docker
├── .gitignore                               # Isolamento de ambientes virtuais e binários locais
├── pytest.ini                               # Configurações de descoberta de testes e paths
├── requirements.txt                         # Dependências com versões estritas
└── README.md                                # Documentação técnica principal
```

---

## 3. Matriz Tecnológica

| Categoria | Tecnologias Principais | Finalidade no Ecossistema |
| :--- | :--- | :--- |
| **Linguagem & Runtime** | Python 3.10+, JupyterLab | Desenvolvimento modular, pipelines de dados e scripts operacionais. |
| **Manipulação de Dados** | Pandas, NumPy, SciPy, PyArrow | Limpeza temporal, filtros físicos, álgebra vetorial e serialização Parquet. |
| **Exploração & Gráficos** | Matplotlib, Seaborn | Inspeção de distribuições, correlação bivariada e matrizes de falha. |
| **Machine Learning** | Scikit-Learn, LightGBM, Imbalanced-Learn | Algoritmos supervisionados multiclasse, SMOTE e detecção de anomalias. |
| **Visão Computacional** | OpenCV (`opencv-python-headless`) | Filtragem Canny, morfologia matemática e segmentação térmica em HSV. |
| **Microsserviço de Borda** | FastAPI, Pydantic v2, Uvicorn | Endpoints assíncronos de baixa latência e validação estrita de tipos. |
| **Conteinerização** | Docker, Docker Compose | Empacotamento reproduzível e isolado para servidores de borda (*Edge*). |
| **Testes & Qualidade** | Pytest, HTTPX | Testes automatizados de contratos HTTP e integridade de inferência. |

---

## 4. Engenharia de Dados e Metodologia Científica

### 4.1. Saneamento e Tratamento Temporal (Etapa 1)
1. **Remoção de Ruído Eletromagnético (EMI)**:
   * Sensores em ambiente fabril sofrem espalhamento eletromagnético temporário gerando saturações espúrias (ex.: temperaturas de $999{,}9^\circ\text{C}$ e vibrações registradas como $-99{,}0\text{ mm/s}$).
   * Essas leituras são filtradas por limites físicos operacionais e marcadas para reconstituição contínua.
2. **Interpolação Temporal Particionada**:
   * A interpolação de dados faltantes é realizada estritamente dentro da série temporal de cada bancada individualmente (`TURB-101` a `TURB-104`), impedindo vazamento de dados entre equipamentos distintos.
3. **Engenharia de Atributos Móveis (Rolling Windows)**:
   * Como cada ciclo de telemetria corresponde a 10 minutos, foram computadas médias móveis e desvios-padrão móveis em duas janelas físicas:
     * **Janela Curta (1 hora = 6 períodos)**: Capta transientes rápidos e oscilações bruscas de carga.
     * **Janela Longa (6 horas = 36 períodos)**: Identifica acúmulo térmico progressivo e degradação mecânica lenta.
4. **Métrica Física de Severidade Dinâmica**:
   * Relaciona a energia vibratória transmitida com a rotação síncrona do eixo da turbina:
     $$\text{Severidade Dinâmica} = \frac{\text{Vibração RMS (mm/s)}}{\text{Rotação (RPM)}}$$

---

### 4.2. Estratégia de Modelagem Híbrida (Etapa 2)

```
                            [ Telemetria Bruta ]
                                     │
                         [ Pipeline de Features ]
                                     │
                     ┌───────────────┴───────────────┐
                     ▼                               ▼
         [ Classificador LightGBM ]        [ Isolation Forest ]
         (Modos de Falha Conhecidos)      (Comportamentos Atípicos)
                     │                               │
                     └───────────────┬───────────────┘
                                     ▼
                      [ Diagnóstico Operacional ]
```

1. **Separação Cronológica Estrita**:
   * Divisão em 70% treino e 30% teste respeitando a linha do tempo (*shuffle=False*), evitando contaminação temporal e sobreajuste irrealista.
2. **Tratamento de Desbalanceamento Severo**:
   * A operação nominal representa ~88% da base, enquanto falhas mecânicas compõem apenas ~12%. O balanceamento sintético via `SMOTE` foi aplicado **exclusivamente no conjunto de treinamento**.
3. **Classificação Multiclasse Supervisionada (`LightGBM`)**:
   * Otimizado via função de perda multiclasse com foco na maximização do *Recall* para mitigar falsos negativos nas classes:
     * **Classe 0**: Operação Normal
     * **Classe 1**: Desgaste de Rolamento
     * **Classe 2**: Desbalanceamento de Rotor
     * **Classe 3**: Cavitação / Superaquecimento
4. **Detecção de Anomalias Não Supervisionada (`Isolation Forest`)**:
   * Ajustado unicamente sobre dados rotulados como normais ($y = 0$). O modelo atua em paralelo como redundância diagnóstica para alertar comportamentos incomuns ou modos de falha emergentes ainda não catalogados.

---

### 4.3. Inspeção por Visão Computacional (Etapa 3)

1. **Inspeção Dimensional e Geométrica de Pás**:
   * Aplicação de filtro gaussiano para atenuação de ruído de alta frequência.
   * Detecção de bordas via detector adaptativo de Canny e extração de contornos fechados.
   * Extração da métrica de circularidade dimensional para triagem de pás trincadas ou deformadas:
     $$C = \frac{4\pi \cdot \text{Área}}{\text{Perímetro}^2}$$
     *(Pás nominais apresentam $C \approx 1{,}0$; reduções abaixo da tolerância indicam falha estrutural).*

2. **Termografia Radiométrica no Espaço HSV**:
   * Conversão das imagens térmicas da turbina do espaço RGB para HSV (Matiz, Saturação, Valor).
   * Isolamento do canal de matiz (*Hue*) calibrado para as faixas de radiação infravermelha correspondentes a temperaturas críticas ($H \in [0, 25]$ e saturações elevadas).
   * Quantificação automática da densidade de *hotspots* ($A_{\text{hotspot}} / A_{\text{total}}$) para detecção de superaquecimento antes do colapso de componentes cerâmicos.

---

## 5. Instalação e Execução Local

### 5.1. Pré-requisitos
* **Python 3.10** ou superior instalado.
* **Git** configurado.
* **Docker** e **Docker Compose** (opcional, para execução conteinerizada).

### 5.2. Clonagem e Configuração do Ambiente Virtual

```bash
# 1. Clonar o repositório oficial
git clone https://github.com/aluno-andrematteussi/aeromotion-predictive-system.git
cd aeromotion-predictive-system

# 2. Criar o ambiente virtual
python -m venv .venv

# 3. Ativar o ambiente virtual
# No Windows (PowerShell):
.\.venv\Scripts\Activate.ps1
# No Linux ou macOS:
source .venv/bin/activate

# 4. Atualizar o gerenciador de pacotes e instalar as dependências
pip install --upgrade pip
pip install -r requirements.txt
```

### 5.3. Execução dos Scripts e Treinamento do Modelo

Para reproduzir os pipelines de dados e atualizar os pesos dos modelos preditivos:

```bash
# Executar o saneamento temporal e engenharia de features
python -m src.data_pipeline.preprocessor
python -m src.data_pipeline.feature_engineering

# Treinar o classificador LightGBM e o detector Isolation Forest
python -m src.ml.train

# Executar a suíte de testes de visão computacional
python -m src.vision.geometric_inspector
python -m src.vision.thermal_inspector
```

### 5.4. Execução dos Testes Automatizados

A validação de integridade da API, schemas e previsões é executada via Pytest:

```bash
pytest tests/ -v
```

### 5.5. Inicialização da API em Modo Desenvolvimento

```bash
uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload
```

* **Documentação Interativa (Swagger UI)**: [http://localhost:8000/docs](http://localhost:8000/docs)
* **Documentação Alternativa (ReDoc)**: [http://localhost:8000/redoc](http://localhost:8000/redoc)

---

## 6. Implantação em Borda com Docker (Edge Computing)

Para implantar a solução no nó de computação em borda (servidor local ou gateway industrial):

```bash
# 1. Construir e inicializar o container em background
docker compose -f docker/docker-compose.yml up --build -d

# 2. Consultar os logs da aplicação
docker compose -f docker/docker-compose.yml logs -f

# 3. Testar a conectividade e prontidão do serviço
curl -i http://localhost:8000/health

# 4. Parar a execução do container
docker compose -f docker/docker-compose.yml down
```

---

## 7. Catálogo Completo da API RESTful

### 7.1. `GET /health`
Verifica o status de execução, a integridade dos artefatos serializados e a versão do serviço.

**Exemplo de Resposta (`200 OK`):**
```json
{
  "status": "HEALTHY",
  "versao_api": "1.0.0",
  "modelos_carregados": {
    "lightgbm_classifier": true,
    "isolation_forest": true,
    "scaler": true
  },
  "total_features_esperadas": 14
}
```

---

### 7.2. `POST /predict`
Executa o diagnóstico operacional e predição multiclasse para uma única leitura de telemetria.

**Exemplo de Requisição (`POST /predict`):**
```json
{
  "id_equipamento": "TURB-101",
  "temperatura_mancal_c": 82.4,
  "vibracao_rms_mms": 7.8,
  "pressao_oleo_bar": 2.1,
  "rotacao_rpm": 3200.0,
  "emissao_acustica_db": 74.5,
  "corrente_eletrica_amp": 36.2,
  "horas_operacao_acumuladas": 1850.0
}
```

**Exemplo de Resposta (`200 OK`):**
```json
{
  "id_equipamento": "TURB-101",
  "classe_predita": 1,
  "diagnostico": "Desgaste de Rolamento",
  "probabilidades": {
    "Operação Normal": 0.032,
    "Desgaste de Rolamento": 0.912,
    "Desbalanceamento de Rotor": 0.041,
    "Cavitação / Superaquecimento": 0.015
  },
  "anomalia_detectada": true,
  "score_anomalia": -0.184,
  "severidade_dinamica": 0.002437,
  "status_operacional": "ALERTA_MANUTENCAO"
}
```

---

### 7.3. `POST /predict-batch`
Processamento assíncrono em lote para ingestão de múltiplos registros acumulados de telemetria.

**Exemplo de Requisição (`POST /predict-batch`):**
```json
{
  "leituras": [
    {
      "id_equipamento": "TURB-101",
      "temperatura_mancal_c": 58.2,
      "vibracao_rms_mms": 1.9,
      "pressao_oleo_bar": 4.1,
      "rotacao_rpm": 2980.0,
      "emissao_acustica_db": 49.0,
      "corrente_eletrica_amp": 26.5,
      "horas_operacao_acumuladas": 1200.0
    },
    {
      "id_equipamento": "TURB-103",
      "temperatura_mancal_c": 94.1,
      "vibracao_rms_mms": 9.2,
      "pressao_oleo_bar": 1.8,
      "rotacao_rpm": 3410.0,
      "emissao_acustica_db": 82.1,
      "corrente_eletrica_amp": 41.0,
      "horas_operacao_acumuladas": 2400.0
    }
  ]
}
```

**Exemplo de Resposta (`200 OK`):**
```json
{
  "total_processado": 2,
  "predicoes": [
    {
      "id_equipamento": "TURB-101",
      "classe_predita": 0,
      "diagnostico": "Operação Normal",
      "probabilidade_confianca": 0.981,
      "anomalia_detectada": false,
      "status_operacional": "NORMAL"
    },
    {
      "id_equipamento": "TURB-103",
      "classe_predita": 3,
      "diagnostico": "Cavitação / Superaquecimento",
      "probabilidade_confianca": 0.934,
      "anomalia_detectada": true,
      "status_operacional": "PARADA_IMEDIATA"
    }
  ]
}
```

---

## 8. Governança, MLOps e Conformidade Regulatória

Os relatórios técnicos de suporte estão disponibilizados integralmente no diretório `docs/`:

1. **Plano de Monitoramento Contínuo e Drift (`docs/plano_mlops_drift.pdf`)**:
   * **Data Drift**: Teste bicaudal de Kolmogorov-Smirnov ($\alpha = 0{,}05$) aplicado a cada 30 dias para avaliar alterações nas distribuições de vibração e temperatura.
   * **Population Stability Index (PSI)**: Monitoramento contínuo das predições com gatilhos de alerta para $\text{PSI} \ge 0{,}10$ e parada para retreino obrigatório se $\text{PSI} \ge 0{,}25$.
   * **Concept Drift & Retreinamento**: Mecanismo de gatilho automático baseado na degradação do Recall da classe de falha mecânica crítica abaixo de $90\%$.

2. **Parecer Ético, Regulatório e Cibersegurança (`docs/parecer_tecnico_etico_regulatorio.pdf`)**:
   * **EU AI Act**: Enquadramento como componente de segurança em maquinário industrial crítico (*High-Risk AI System*), exigindo rastreabilidade de artefatos, log de auditoria técnica e supervisão humana ativa (*Human-in-the-Loop*).
   * **LGPD e Privacidade**: Proteção e anonimização de dados pessoais de operadores nas câmeras de chão de fábrica, com algoritmo de desfoque facial (*blur*) antes do armazenamento de imagens.
   * **Segurança OT/IT**: Isolamento do container em rede DMZ industrial restrita, comunicação criptografada via mTLS, execução em contêiner com usuário não-root (*least privilege*) e proteções ativas contra envenenamento de dados (*data poisoning*).

---

## 9. Autor e Contato

* **Responsável Técnico**: André Matteussi
* **Repositório GitHub**: [aluno-andrematteussi/aeromotion-predictive-system](https://github.com/aluno-andrematteussi/aeromotion-predictive-system)
* **Organização**: AeroMotion Dynamics S.A.