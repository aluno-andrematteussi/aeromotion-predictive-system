"""AeroMotion Predictive System — Painel de Engenharia, MLOps e Visão Computacional.

Sistema integrado de monitoramento preditivo e inspeção óptica/termográfica
para as bancadas de ensaio TURB-101 a TURB-104 da AeroMotion Dynamics S.A.

Responsável Técnico: André Matteussi
Curso: Aprendizagem Técnica em Cibersistemas para Automação — CentroWEG (Jaraguá do Sul - SC)
Unidade Curricular: Inteligência Artificial e Ferramentas para Análise de Dados em Sistemas Industriais
Docente Orientador: Prof. Fabiano da Silva Luiz
"""

import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

# Garante que a raiz do projeto esteja visível no PYTHONPATH
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.vision.geometric_inspector import GeometricInspector
from src.vision.thermal_inspector import ThermalInspector

plt.rcParams["axes.edgecolor"] = "#CBD5E1"
plt.rcParams["axes.linewidth"] = 0.8
plt.rcParams["font.sans-serif"] = "DejaVu Sans"

MAPA_CLASSES: Dict[int, str] = {
    0: "Classe 0 — Operação Nominal",
    1: "Classe 1 — Desgaste de Rolamento",
    2: "Classe 2 — Desbalanceamento de Rotor",
    3: "Classe 3 — Cavitação / Superaquecimento",
}

MODULOS_PROJETO: Dict[str, Dict[str, str]] = {
    "Visão Geral & Arquitetura": {
        "modulo_py": "src/dashboard/app.py",
        "etapa": "Visão Executiva & Bancadas",
        "icone": "bi-grid-1x2-fill",
        "tags": "visao geral sumario arquitetura bancadas turbina aeromotion",
    },
    "Saneamento & Engenharia (Etapa 1)": {
        "modulo_py": "src/data_pipeline/preprocessor.py & feature_engineering.py",
        "etapa": "Etapa 1 — Séries Temporais & EDA",
        "icone": "bi-sliders",
        "tags": "saneamento eda series temporais correlacao parquet emi severidade dinamica",
    },
    "Modelagem Preditiva & MLOps (Etapa 2)": {
        "modulo_py": "src/ml/train.py & evaluation.py",
        "etapa": "Etapa 2 — LightGBM & Isolation Forest",
        "icone": "bi-cpu-fill",
        "tags": "lightgbm smote isolation forest matriz confusao recall f1 importancia",
    },
    "Simulador de Inferência Edge (API)": {
        "modulo_py": "src/api/app.py & schemas.py",
        "etapa": "Etapa 3 — Microsserviço FastAPI /predict",
        "icone": "bi-broadcast-pin",
        "tags": "simulador tempo real api fastapi pydantic sensores predicao log",
    },
    "Visão Computacional (Etapa 3)": {
        "modulo_py": "src/vision/geometric_inspector.py & thermal_inspector.py",
        "etapa": "Etapa 3 — Canny Dimensional & Termografia HSV",
        "icone": "bi-camera-video-fill",
        "tags": "visao computacional opencv canny circularidade otsu termografia hsv hotspot upload",
    },
    "Documentação Técnica & Teoria": {
        "modulo_py": "docs/*.pdf & README.md",
        "etapa": "Manual de Engenharia & Conformidade",
        "icone": "bi-journal-code",
        "tags": "documentacao readme teoria justificativa mlops drift ks psi lgpd eu ai act ciberseguranca",
    },
}


# =============================================================================
# 1. FOLHA DE ESTILO CSS INDUSTRIAL (BOTÕES CUSTOMIZADOS E ACCENT AZUL)
# =============================================================================
def aplicar_estilo_industrial() -> None:
    """Aplica estilização CSS corporativa, convertendo inputs radio em azul industrial."""
    st.markdown(
        """
        <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.min.css">
        <style>
            /* Forçamento da cor primária em inputs de rádio e seletores */
            :root {
                --primary-color: #00579D !important;
                accent-color: #00579D !important;
            }
            input[type="radio"] {
                accent-color: #00579D !important;
            }
            .stApp {
                background-color: #F4F6F9;
                color: #102A43;
            }

            /* Estilização dos Botões de Ação Industriais */
            div.stButton > button {
                background: linear-gradient(135deg, #00579D 0%, #003B6F 100%) !important;
                color: #FFFFFF !important;
                border: 1px solid #002B49 !important;
                border-bottom: 3px solid #0090D4 !important;
                border-radius: 6px !important;
                padding: 0.55rem 1rem !important;
                font-weight: 600 !important;
                font-size: 0.86rem !important;
                letter-spacing: 0.3px !important;
                transition: all 0.2s ease-in-out !important;
                box-shadow: 0 2px 4px rgba(0, 43, 73, 0.12) !important;
            }
            div.stButton > button:hover {
                background: linear-gradient(135deg, #0066B8 0%, #004B8D 100%) !important;
                transform: translateY(-1px) !important;
                box-shadow: 0 4px 8px rgba(0, 87, 157, 0.25) !important;
            }

            /* Transforma os Radio Buttons da Sidebar em Blocos de Menu Executivos */
            section[data-testid="stSidebar"] div[role="radiogroup"] {
                gap: 6px !important;
            }
            section[data-testid="stSidebar"] div[role="radiogroup"] > label {
                background-color: #FFFFFF !important;
                border: 1px solid #D9E2EC !important;
                border-left: 4px solid #BCCCDC !important;
                border-radius: 6px !important;
                padding: 10px 14px !important;
                margin-bottom: 4px !important;
                transition: all 0.18s ease !important;
                cursor: pointer !important;
                width: 100% !important;
            }
            section[data-testid="stSidebar"] div[role="radiogroup"] > label:hover {
                border-left: 4px solid #00579D !important;
                background-color: #F0F7FF !important;
            }
            section[data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) {
                background-color: #002B49 !important;
                border-color: #002B49 !important;
                border-left: 4px solid #0090D4 !important;
                color: #FFFFFF !important;
                font-weight: 600 !important;
                box-shadow: 0 2px 5px rgba(0, 43, 73, 0.18) !important;
            }
            section[data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) p {
                color: #FFFFFF !important;
                font-weight: 600 !important;
            }

            /* Topbar Executiva AeroMotion */
            .aeromotion-header {
                background: linear-gradient(90deg, #002B49 0%, #00447C 100%);
                color: #FFFFFF;
                padding: 16px 24px;
                border-radius: 6px;
                border-bottom: 3px solid #0090D4;
                display: flex;
                justify-content: space-between;
                align-items: center;
                flex-wrap: wrap;
                gap: 12px;
                margin-bottom: 18px;
                box-shadow: 0 2px 8px rgba(0, 43, 73, 0.14);
            }
            .aeromotion-title {
                font-size: 1.15rem;
                font-weight: 700;
                letter-spacing: 0.6px;
                color: #FFFFFF;
                display: flex;
                align-items: center;
                gap: 10px;
            }
            .aeromotion-module-badge {
                font-family: 'Consolas', 'Courier New', monospace;
                font-size: 0.80rem;
                color: #E6F6FF;
                background-color: rgba(0, 144, 212, 0.25);
                border: 1px solid rgba(140, 210, 244, 0.4);
                padding: 4px 10px;
                border-radius: 4px;
            }
            .aeromotion-status-bar {
                font-family: 'Consolas', 'Courier New', monospace;
                font-size: 0.80rem;
                color: #D9E2EC;
                display: flex;
                gap: 16px;
                align-items: center;
                flex-wrap: wrap;
            }

            /* Cartões Estruturais */
            .corp-card {
                background-color: #FFFFFF;
                border: 1px solid #D9E2EC;
                border-top: 3px solid #00579D;
                border-radius: 6px;
                padding: 18px 20px;
                margin-bottom: 16px;
                box-shadow: 0 1px 3px rgba(16, 42, 67, 0.05);
            }
            .corp-card-title {
                font-size: 0.96rem;
                font-weight: 700;
                color: #00447C;
                margin-bottom: 12px;
                padding-bottom: 8px;
                border-bottom: 1px solid #E4E7EB;
                display: flex;
                align-items: center;
                gap: 8px;
                text-transform: uppercase;
                letter-spacing: 0.4px;
            }
            .corp-kpi-box {
                background-color: #FFFFFF;
                border: 1px solid #D9E2EC;
                border-left: 4px solid #00579D;
                border-radius: 6px;
                padding: 14px 16px;
                box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
            }
            .corp-kpi-label {
                font-size: 0.76rem;
                font-weight: 700;
                color: #486581;
                text-transform: uppercase;
                letter-spacing: 0.4px;
            }
            .corp-kpi-value {
                font-size: 1.50rem;
                font-weight: 700;
                color: #002B49;
                margin-top: 4px;
            }
            .corp-kpi-sub {
                font-size: 0.75rem;
                color: #00579D;
                margin-top: 3px;
                font-family: 'Consolas', monospace;
            }

            /* Teoria e Justificativa */
            .theory-box {
                background-color: #F0F7FF;
                border: 1px solid #B3D4FF;
                border-left: 4px solid #00579D;
                border-radius: 6px;
                padding: 14px 18px;
                margin-bottom: 16px;
                color: #102A43;
                font-size: 0.90rem;
                line-height: 1.6;
            }
            .math-pill {
                font-family: 'Cambria Math', 'Times New Roman', serif;
                font-weight: 700;
                background-color: #FFFFFF;
                border: 1px solid #CBD5E1;
                padding: 2px 8px;
                border-radius: 4px;
                color: #002B49;
            }

            /* Status Badges */
            .status-badge-ok {
                background-color: #DEF7EC;
                color: #03543F;
                border: 1px solid #84E1BC;
                border-left: 4px solid #0E9F6E;
                padding: 8px 14px;
                border-radius: 4px;
                font-weight: 700;
                font-size: 0.88rem;
                display: block;
            }
            .status-badge-crit {
                background-color: #FDE8E8;
                color: #9B1C1C;
                border: 1px solid #F8B4B4;
                border-left: 4px solid #C81E1E;
                padding: 8px 14px;
                border-radius: 4px;
                font-weight: 700;
                font-size: 0.88rem;
                display: block;
            }
            .status-badge-warn {
                background-color: #FEF3C7;
                color: #92400E;
                border: 1px solid #FCD34D;
                border-left: 4px solid #D97706;
                padding: 8px 14px;
                border-radius: 4px;
                font-weight: 700;
                font-size: 0.88rem;
                display: block;
            }

            /* Rodapé Simétrico em 3 Colunas */
            .aeromotion-footer {
                background-color: #002B49;
                color: #F0F4F8;
                border-radius: 6px;
                padding: 18px 24px;
                margin-top: 28px;
                border-top: 3px solid #0090D4;
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                gap: 20px;
                font-size: 0.82rem;
                line-height: 1.55;
            }
            .footer-col-title {
                color: #8CD2F4;
                font-weight: 700;
                font-size: 0.78rem;
                text-transform: uppercase;
                letter-spacing: 0.5px;
                margin-bottom: 6px;
                display: flex;
                align-items: center;
                gap: 6px;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def renderizar_kpi(coluna, titulo: str, valor: str, subtitulo: str, icone: str) -> None:
    """Renderiza um indicador chave de desempenho (KPI)."""
    coluna.markdown(
        f"""
        <div class="corp-kpi-box">
            <div class="corp-kpi-label"><i class="bi {icone}" style="color: #00579D; margin-right: 6px;"></i>{titulo}</div>
            <div class="corp-kpi-value">{valor}</div>
            <div class="corp-kpi-sub">{subtitulo}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def renderizar_cabecalho_sistema(meta_modulo: Dict[str, str]) -> None:
    """Renderiza o cabeçalho superior do AeroMotion Predictive System."""
    horario_local = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    st.markdown(
        f"""
        <div class="aeromotion-header">
            <div style="display: flex; align-items: center; flex-wrap: wrap; gap: 12px;">
                <span class="aeromotion-title">
                    <i class="bi bi-cpu-fill" style="color: #0090D4;"></i>
                    AEROMOTION PREDICTIVE SYSTEM
                </span>
                <span class="aeromotion-module-badge">
                    <i class="bi bi-file-earmark-code" style="margin-right: 4px;"></i>Módulo: {meta_modulo['modulo_py']}
                </span>
            </div>
            <div class="aeromotion-status-bar">
                <span><i class="bi bi-building-gear" style="color: #0090D4;"></i> AeroMotion Dynamics S.A.</span>
                <span><i class="bi bi-check2-circle" style="color: #31C48D;"></i> Edge FastAPI: Porta 8000</span>
                <span><i class="bi bi-clock" style="color: #8CD2F4;"></i> {horario_local}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def renderizar_rodape_sistema() -> None:
    """Renderiza o rodapé institucional simétrico em 3 colunas."""
    st.markdown(
        """
        <div class="aeromotion-footer">
            <div>
                <div class="footer-col-title"><i class="bi bi-hdd-network"></i> SISTEMA & ESCOPO INDUSTRIAL</div>
                <strong>AeroMotion Predictive System (v1.0.0-edge)</strong><br>
                <span style="color: #BCCCDC;">AeroMotion Dynamics S.A. — Divisão de Turbinas & Propulsão de Alta Precisão (Bancadas TURB-101 a TURB-104)</span>
            </div>
            <div>
                <div class="footer-col-title"><i class="bi bi-person-workspace"></i> AUTORIA & DESENVOLVIMENTO</div>
                <strong>André Matteussi</strong><br>
                <span style="color: #BCCCDC;">Aprendiz Técnico de Cibersistemas para Automação<br>Jaraguá do Sul — Santa Catarina (SC)</span>
            </div>
            <div>
                <div class="footer-col-title"><i class="bi bi-mortarboard"></i> REFERÊNCIA ACADÊMICA</div>
                <strong>Unidade Curricular:</strong> IA e Ferramentas para Análise de Dados em Sistemas Industriais<br>
                <span style="color: #BCCCDC;">Docente Orientador: Prof. Fabiano da Silva Luiz</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =============================================================================
# 2. CAMADA DE DADOS E DETECÇÃO AUTOMÁTICA DO ALVO
# =============================================================================
@st.cache_data
def carregar_dados() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Carrega os datasets bruto (.csv) e processado (.parquet)."""
    caminho_raw = ROOT_DIR / "data" / "raw" / "dados_sensores_aeromotion.csv"
    caminho_proc = ROOT_DIR / "data" / "processed" / "features_eng.parquet"

    df_raw = pd.read_csv(caminho_raw) if caminho_raw.exists() else pd.DataFrame()
    df_proc = pd.read_parquet(caminho_proc) if caminho_proc.exists() else pd.DataFrame()
    return df_raw, df_proc


@st.cache_resource
def carregar_artefatos_ml() -> Dict[str, Any]:
    """Carrega os modelos treinados e transformadores do diretório models/."""
    dir_models = ROOT_DIR / "models"
    return {
        "classifier": joblib.load(dir_models / "classifier_model.joblib"),
        "iso_forest": joblib.load(dir_models / "isolation_forest.joblib"),
        "scaler": joblib.load(dir_models / "scaler.joblib"),
        "features": joblib.load(dir_models / "feature_names.joblib"),
    }


def identificar_coluna_alvo(df: pd.DataFrame) -> str:
    """Localiza a coluna discreta de classe {0, 1, 2, 3} no DataFrame."""
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            valores_unicos = set(df[col].dropna().unique())
            if valores_unicos.issubset({0, 1, 2, 3, 0.0, 1.0, 2.0, 3.0}) and len(valores_unicos) >= 3:
                return col

    nomes_alvo = [
        "tipo_falha", "codigo_falha", "modo_falha", "status_falha",
        "falha", "target", "classe", "label", "estado_operacional",
    ]
    for nome in nomes_alvo:
        if nome in df.columns:
            return nome

    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]) and 2 <= df[col].nunique() <= 6:
            return col

    return df.columns[-1]


def alinhar_features_para_modelo(df: pd.DataFrame, feature_names: List[str]) -> pd.DataFrame:
    """Alinha o DataFrame de entrada exatamente com as features esperadas pelo modelo."""
    df_cod = pd.get_dummies(df, drop_first=False)
    return df_cod.reindex(columns=feature_names, fill_value=0.0).astype(float)


def obter_particao_teste_segura(
    df_proc: pd.DataFrame, col_alvo: str, estrategia: str
) -> pd.DataFrame:
    """Obtém a partição de teste de 30% garantindo suporte nas 4 classes."""
    y_int = df_proc[col_alvo].round().astype(int)

    if "Temporal" in estrategia and "id_equipamento" in df_proc.columns:
        fatias = []
        for _, grp in df_proc.groupby("id_equipamento", sort=False):
            corte = int(len(grp) * 0.70)
            fatias.append(grp.iloc[corte:])
        df_t = pd.concat(fatias, axis=0)
        if df_t[col_alvo].round().astype(int).nunique() == 4:
            return df_t

    contagens = y_int.value_counts()
    if (contagens >= 2).all():
        _, df_t = train_test_split(
            df_proc, test_size=0.30, stratify=y_int, random_state=42
        )
    else:
        _, df_t = train_test_split(df_proc, test_size=0.30, random_state=42)
    return df_t


# =============================================================================
# 3. CAMADA DE VISÃO COMPUTACIONAL
# =============================================================================
def carregar_imagem_upload(arquivo_upload) -> np.ndarray:
    """Lê o arquivo de imagem enviado e converte para matriz BGR 300x300."""
    img_pil = Image.open(arquivo_upload)
    if img_pil.mode in ("RGBA", "LA") or (img_pil.mode == "P" and "transparency" in img_pil.info):
        rgba = img_pil.convert("RGBA")
        fundo = Image.new("RGBA", rgba.size, (0, 0, 0, 255))
        img_rgb = Image.alpha_composite(fundo, rgba).convert("RGB")
    else:
        img_rgb = img_pil.convert("RGB")

    img_bgr = cv2.cvtColor(np.array(img_rgb), cv2.COLOR_RGB2BGR)
    return cv2.resize(img_bgr, (300, 300), interpolation=cv2.INTER_AREA)


def pre_processar_silhueta_inteligente(
    imagem_bgr: np.ndarray, inverter_manual: bool = False
) -> np.ndarray:
    """Isola a silhueta sólida com detecção automática de polaridade de fundo."""
    cinza = cv2.cvtColor(imagem_bgr, cv2.COLOR_BGR2GRAY)
    suavizada = cv2.GaussianBlur(cinza, (5, 5), 0)

    bordas_ext = np.concatenate([
        suavizada[0, :], suavizada[-1, :], suavizada[:, 0], suavizada[:, -1]
    ])
    fundo_claro = float(np.median(bordas_ext)) > 127.0
    inverter_real = not fundo_claro if inverter_manual else fundo_claro
    tipo_th = cv2.THRESH_BINARY_INV if inverter_real else cv2.THRESH_BINARY

    _, mascara = cv2.threshold(suavizada, 0, 255, tipo_th + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mascara_limpa = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, kernel, iterations=2)

    mascara_limpa[:6, :] = 0
    mascara_limpa[-6:, :] = 0
    mascara_limpa[:, :6] = 0
    mascara_limpa[:, -6:] = 0

    if cv2.countNonZero(mascara_limpa) < 200:
        return imagem_bgr
    return cv2.cvtColor(mascara_limpa, cv2.COLOR_GRAY2BGR)


def executar_inspecao_termica_segura(
    imagem_bgr: np.ndarray, limiar_area_px: float
) -> Dict[str, Any]:
    """Executa segmentação termográfica HSV com fallback nativo seguro."""
    try:
        inspetor = ThermalInspector(min_hotspot_area=limiar_area_px)
    except TypeError:
        inspetor = ThermalInspector()

    for attr in ("inspecionar_termografia", "analisar_termografia", "inspecionar"):
        fn = getattr(inspetor, attr, None)
        if callable(fn):
            res = fn(imagem_bgr)
            if isinstance(res, dict):
                area_c = float(res.get("area_critica_px", res.get("area_total_px", 0.0)))
                focos_c = int(res.get("focos", res.get("focos_detectados", 0)))
                masc = res.get("mascara_hsv", res.get("mascara_termica"))
                return {
                    "alerta": bool(area_c >= limiar_area_px),
                    "area_critica_px": round(area_c, 2),
                    "focos": focos_c,
                    "mascara_hsv": masc,
                }

    hsv = cv2.cvtColor(imagem_bgr, cv2.COLOR_BGR2HSV)
    m1 = cv2.inRange(hsv, np.array([0, 100, 100]), np.array([18, 255, 255]))
    m2 = cv2.inRange(hsv, np.array([160, 100, 100]), np.array([180, 255, 255]))
    masc_limpa = cv2.morphologyEx(
        cv2.bitwise_or(m1, m2),
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
    )
    conts, _ = cv2.findContours(masc_limpa, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    validos = [c for c in conts if cv2.contourArea(c) >= 15.0]
    area_tot = float(sum(cv2.contourArea(c) for c in validos))

    return {
        "alerta": bool(area_tot >= limiar_area_px),
        "area_critica_px": round(area_tot, 2),
        "focos": len(validos),
        "mascara_hsv": masc_limpa,
    }


def gerar_visualizacao_geom(
    imagem_original: np.ndarray, mascara_bordas: np.ndarray, aprovado: bool
) -> Tuple[np.ndarray, np.ndarray]:
    """Gera mapa de bordas Canny espesso e sobreposição colorida sobre a peça."""
    bordas_nitidas = cv2.dilate(mascara_bordas, np.ones((2, 2), dtype=np.uint8), iterations=1)
    overlay = imagem_original.copy()
    contornos, _ = cv2.findContours(
        mascara_bordas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cor_bgr = (110, 159, 14) if aprovado else (30, 30, 200)
    if contornos:
        maior_c = max(contornos, key=cv2.contourArea)
        cv2.drawContours(overlay, [maior_c], -1, cor_bgr, 3)

    return bordas_nitidas, cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)


# =============================================================================
# 4. TELAS DO PROJETO AERO-MOTION
# =============================================================================
def tela_visao_geral(df_raw: pd.DataFrame, df_proc: pd.DataFrame, artefatos: Dict[str, Any]) -> None:
    """Tela 01: Visão Geral Executiva, Arquitetura de Diretórios e Saúde das Bancadas."""
    col_alvo = identificar_coluna_alvo(df_proc)
    taxa_falha = (df_proc[col_alvo].round().astype(int) != 0).mean() * 100

    k1, k2, k3, k4 = st.columns(4)
    renderizar_kpi(k1, "Bancadas Instrumentadas", "04 Unidades", "TURB-101 a TURB-104 (IoT 10min)", "bi-hdd-rack-fill")
    renderizar_kpi(k2, "Série Temporal Saneada", f"{len(df_proc):,}".replace(",", "."), f"Base Bruta: {len(df_raw):,} leituras", "bi-database-fill-check")
    renderizar_kpi(k3, "Atributos Preditivos", f"{len(artefatos['features'])} Variáveis", "Rolling 1h/6h + Severidade", "bi-layers-fill")
    renderizar_kpi(k4, "Taxa de Falhas na Base", f"{taxa_falha:.1f}%", "Balanceamento Treino via SMOTE", "bi-exclamation-octagon-fill")

    st.markdown("<br>", unsafe_allow_html=True)

    c_left, c_right = st.columns([1.45, 1])
    with c_left:
        st.markdown(
            '<div class="corp-card"><div class="corp-card-title"><i class="bi bi-activity"></i> DIAGNÓSTICO CONSOLIDADO POR BANCADA DE TESTE</div>',
            unsafe_allow_html=True,
        )
        if "id_equipamento" in df_proc.columns:
            linhas_resumo = []
            for eq, grp in df_proc.groupby("id_equipamento"):
                cod_ult = int(round(grp.iloc[-1][col_alvo]))
                linhas_resumo.append({
                    "Bancada": eq,
                    "Registros": len(grp),
                    "Temp. Média (°C)": round(grp["temperatura_mancal_c"].mean(), 2),
                    "Vibração RMS (mm/s)": round(grp["vibracao_rms_mms"].mean(), 2),
                    "Pressão Óleo (Bar)": round(grp["pressao_oleo_bar"].mean(), 2),
                    "Rotação Média (RPM)": int(grp["rotacao_rpm"].mean()),
                    "Ocorrências de Falha": int((grp[col_alvo].round().astype(int) != 0).sum()),
                    "Último Status": MAPA_CLASSES.get(cod_ult, "Classe 0"),
                })
            st.dataframe(pd.DataFrame(linhas_resumo), use_container_width=True, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with c_right:
        st.markdown(
            """
            <div class="corp-card">
                <div class="corp-card-title"><i class="bi bi-folder2-open"></i> MAPEAMENTO DOS MÓDULOS DO PROJETO</div>
                <ul style="font-size: 0.86rem; line-height: 1.65; padding-left: 18px; margin-bottom: 0;">
                    <li><code>src/data_pipeline/</code>: Módulos <code>preprocessor.py</code> e <code>feature_engineering.py</code> para filtragem de EMI, interpolação temporal e janelas móveis.</li>
                    <li><code>src/ml/</code>: Módulos <code>train.py</code> e <code>evaluation.py</code> com treino supervisionado <code>LightGBM</code> e detector <code>Isolation Forest</code>.</li>
                    <li><code>src/vision/</code>: Módulos <code>geometric_inspector.py</code> (Canny/Otsu) e <code>thermal_inspector.py</code> (Segmentação HSV).</li>
                    <li><code>src/api/</code>: Microsserviço <code>app.py</code> e contratos <code>schemas.py</code> para inferência RESTful em Edge Computing.</li>
                    <li><code>docs/</code>: Relatórios formais em PDF cobrindo Saneamento, Plano de Drift (12 meses) e Parecer Ético/Regulatório.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )


def tela_engenharia_dados(df_raw: pd.DataFrame, df_proc: pd.DataFrame) -> None:
    """Tela 02: Etapa 1 — Saneamento Físico de Séries Temporais, Engenharia de Atributos e EDA."""
    st.markdown(
        """
        <div class="theory-box">
            <strong><i class="bi bi-journal-bookmark-fill" style="color: #00579D;"></i> FUNDAMENTAÇÃO DE ENGENHARIA DE DADOS (<code>src/data_pipeline/</code>)</strong><br>
            Leituras brutas de chão de fábrica apresentam picos espúrios de <strong>Interferência Eletromagnética (EMI)</strong>, como temperaturas de <span class="math-pill">999,9 &deg;C</span> e vibrações de <span class="math-pill">-99,0 mm/s</span>. O módulo <code>preprocessor.py</code> converte essas leituras em <code>NaN</code> e aplica interpolação linear agrupada por bancada, preservando a série temporal de 10 minutos. Em seguida, <code>feature_engineering.py</code> calcula janelas móveis de 1h (6 períodos) e 6h (36 períodos) e extrai a <strong>Severidade Dinâmica</strong>: <span class="math-pill">Severidade = Vibração RMS / Rotação RPM</span>.
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_alvo = identificar_coluna_alvo(df_proc)

    c1, c2, c3 = st.columns(3)
    equipamentos = sorted(df_proc["id_equipamento"].unique()) if "id_equipamento" in df_proc.columns else ["TURB-101"]
    turb_sel = c1.selectbox("Selecionar Bancada de Ensaio:", equipamentos)
    var_sel = c2.selectbox(
        "Variável Física Analisada:",
        ["temperatura_mancal_c", "vibracao_rms_mms", "pressao_oleo_bar", "rotacao_rpm", "severidade_dinamica"],
    )
    janela = c3.slider("Janela Cronológica (períodos de 10 min):", 60, 600, 200, 20)

    df_eq = df_proc[df_proc["id_equipamento"] == turb_sel].head(janela) if "id_equipamento" in df_proc.columns else df_proc.head(janela)

    g_left, g_right = st.columns(2)
    with g_left:
        st.markdown('<div class="corp-card"><div class="corp-card-title"><i class="bi bi-graph-up"></i> COMPORTAMENTO TEMPORAL & MÉDIAS MÓVEIS (1H / 6H)</div>', unsafe_allow_html=True)
        fig_t, ax_t = plt.subplots(figsize=(8, 4.0))
        ax_t.plot(df_eq.index, df_eq[var_sel], label="Leitura Saneada", color="#9FB3C8", linewidth=1.3)

        for suf, cor, lbl in [("_roll_mean_6", "#00579D", "Média Móvel 1h (6p)"), ("_roll_mean_36", "#0090D4", "Média Móvel 6h (36p)")]:
            col_m = f"{var_sel}{suf}"
            if col_m in df_eq.columns:
                ax_t.plot(df_eq.index, df_eq[col_m], label=lbl, color=cor, linewidth=2.0)

        ax_t.set_title(f"Série Temporal Saneada — {turb_sel} ({var_sel})", fontsize=10, fontweight="bold", color="#002B49")
        ax_t.set_xlabel("Índice Cronológico (Cadência: 10 min)")
        ax_t.spines["top"].set_visible(False)
        ax_t.spines["right"].set_visible(False)
        ax_t.legend(frameon=False, fontsize=8.5)
        ax_t.grid(True, linestyle="--", alpha=0.35)
        st.pyplot(fig_t)
        plt.close(fig_t)
        st.markdown("</div>", unsafe_allow_html=True)

    with g_right:
        st.markdown('<div class="corp-card"><div class="corp-card-title"><i class="bi bi-grid-3x3-gap-fill"></i> MATRIZ DE CORRELAÇÃO MULTIVARIADA DE PEARSON</div>', unsafe_allow_html=True)
        cols_corr = [
            c for c in [
                "temperatura_mancal_c", "vibracao_rms_mms", "pressao_oleo_bar",
                "rotacao_rpm", "emissao_acustica_db", "corrente_eletrica_amp",
                "severidade_dinamica", col_alvo,
            ] if c in df_proc.columns
        ]
        fig_c, ax_c = plt.subplots(figsize=(8, 4.0))
        sns.heatmap(
            df_proc[cols_corr].corr(),
            annot=True,
            fmt=".2f",
            cmap="YlGnBu",
            ax=ax_c,
            cbar=False,
            annot_kws={"size": 8},
        )
        ax_c.set_title("Correlação Linear entre Sensores e Modo de Falha", fontsize=10, fontweight="bold", color="#002B49")
        st.pyplot(fig_c)
        plt.close(fig_c)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="corp-card"><div class="corp-card-title"><i class="bi bi-table"></i> AMOSTRA DO DATASET PROCESSADO (data/processed/features_eng.parquet)</div>', unsafe_allow_html=True)
    st.dataframe(df_eq.head(12), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)


def tela_modelagem_ml(df_proc: pd.DataFrame, artefatos: Dict[str, Any]) -> None:
    """Tela 03: Etapa 2 — Avaliação do Modelo LightGBM e Isolation Forest."""
    st.markdown(
        """
        <div class="theory-box">
            <strong><i class="bi bi-journal-bookmark-fill" style="color: #00579D;"></i> FUNDAMENTAÇÃO DE MACHINE LEARNING (<code>src/ml/train.py &amp; evaluation.py</code>)</strong><br>
            Para mitigar a assimetria das classes (~12% de falhas na base bruta) sem causar vazamento temporal, a sobreamostragem <strong>SMOTE</strong> é aplicada restritamente ao conjunto de treino (70%). O modelo <code>LightGBM</code> é parametrizado para priorizar <strong>Recall &gt; 90%</strong> nas classes de falha crítica, operando em conjunto com o detector não supervisionado <code>Isolation Forest</code> para identificar anomalias inéditas.
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_alvo = identificar_coluna_alvo(df_proc)

    estrategia = st.radio(
        "Estratégia de Particionamento do Conjunto de Teste (30%):",
        [
            "Partição Estratificada Multiclasse (Preserva a proporção exata das 4 classes no teste)",
            "Partição Temporal por Bancada (Últimos 30% cronológicos de cada turbina)",
        ],
        horizontal=True,
    )

    df_teste = obter_particao_teste_segura(df_proc, col_alvo, estrategia)
    X_teste = alinhar_features_para_modelo(
        df_teste.drop(columns=[col_alvo], errors="ignore"), artefatos["features"]
    )
    y_teste = df_teste[col_alvo].round().astype(int).values

    X_teste_scaled = artefatos["scaler"].transform(X_teste)
    y_pred = artefatos["classifier"].predict(X_teste_scaled).round().astype(int)
    anom_pred = artefatos["iso_forest"].predict(X_teste_scaled)

    acc = accuracy_score(y_teste, y_pred)
    rec_macro = recall_score(y_teste, y_pred, average="macro", zero_division=0)
    prec_macro = precision_score(y_teste, y_pred, average="macro", zero_division=0)
    f1_macro = f1_score(y_teste, y_pred, average="macro", zero_division=0)
    taxa_iso = (anom_pred == -1).mean() * 100

    k1, k2, k3, k4 = st.columns(4)
    renderizar_kpi(k1, "Acurácia Global (Teste)", f"{acc * 100:.2f}%", f"Suporte Total: {len(y_teste)} amostras", "bi-bullseye")
    renderizar_kpi(k2, "Recall Macro Multiclasse", f"{rec_macro * 100:.2f}%", "Meta de Segurança: > 90%", "bi-shield-fill-check")
    renderizar_kpi(k3, "F1-Score / Precisão Macro", f"{f1_macro * 100:.2f}%", f"Precisão: {prec_macro * 100:.2f}%", "bi-bar-chart-line-fill")
    renderizar_kpi(k4, "Alertas Isolation Forest", f"{taxa_iso:.1f}%", f"Alvo Auditado: {col_alvo}", "bi-radar")

    st.markdown("<br>", unsafe_allow_html=True)

    c_mat, c_imp = st.columns(2)
    with c_mat:
        st.markdown('<div class="corp-card"><div class="corp-card-title"><i class="bi bi-border-all"></i> MATRIZ DE CONFUSÃO 4X4 (CONJUNTO DE TESTE)</div>', unsafe_allow_html=True)
        cm = confusion_matrix(y_teste, y_pred, labels=[0, 1, 2, 3])
        fig_cm, ax_cm = plt.subplots(figsize=(7, 4.5))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Normal (0)", "Rolamento (1)", "Rotor (2)", "Cavitação (3)"],
            yticklabels=["Normal (0)", "Rolamento (1)", "Rotor (2)", "Cavitação (3)"],
            ax=ax_cm,
            cbar=False,
            annot_kws={"size": 10, "weight": "bold"},
        )
        ax_cm.set_xlabel("Diagnóstico Predito pelo Classificador")
        ax_cm.set_ylabel("Estado Físico Real da Turbina")
        ax_cm.set_title("Matriz de Confusão Multiclasse", fontsize=10, fontweight="bold", color="#002B49")
        st.pyplot(fig_cm)
        plt.close(fig_cm)
        st.markdown("</div>", unsafe_allow_html=True)

    with c_imp:
        st.markdown('<div class="corp-card"><div class="corp-card-title"><i class="bi bi-sort-down"></i> EXPLICABILIDADE FÍSICA — TOP 12 VARIÁVEIS</div>', unsafe_allow_html=True)
        clf = artefatos["classifier"]
        if hasattr(clf, "feature_importances_"):
            imp = pd.Series(clf.feature_importances_, index=artefatos["features"]).sort_values(ascending=True).tail(12)
            fig_fi, ax_fi = plt.subplots(figsize=(7, 4.5))
            imp.plot(kind="barh", color="#00579D", ax=ax_fi)
            ax_fi.set_title("Ganho de Informação por Variável Física (LightGBM)", fontsize=10, fontweight="bold", color="#002B49")
            ax_fi.set_xlabel("Importância Relativa")
            ax_fi.spines["top"].set_visible(False)
            ax_fi.spines["right"].set_visible(False)
            ax_fi.grid(True, axis="x", linestyle="--", alpha=0.35)
            st.pyplot(fig_fi)
            plt.close(fig_fi)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="corp-card"><div class="corp-card-title"><i class="bi bi-clipboard2-data-fill"></i> MÉTRICAS DETALHADAS POR MODO DE FALHA</div>', unsafe_allow_html=True)
    rep = classification_report(
        y_teste,
        y_pred,
        labels=[0, 1, 2, 3],
        target_names=[MAPA_CLASSES[i] for i in range(4)],
        output_dict=True,
        zero_division=0,
    )
    df_rep = pd.DataFrame(rep).transpose().round(4)
    df_rep["support"] = df_rep["support"].astype(int)
    st.dataframe(df_rep, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)


def tela_simulador_edge(artefatos: Dict[str, Any]) -> None:
    """Tela 04: Simulador de Inferência Preditiva Edge e Contratos Pydantic."""
    st.markdown(
        """
        <div class="theory-box">
            <strong><i class="bi bi-journal-bookmark-fill" style="color: #00579D;"></i> FUNDAMENTAÇÃO DE DEPLOY EDGE (<code>src/api/app.py &amp; schemas.py</code>)</strong><br>
            Simula o processamento do nó de borda para leituras enviadas ao endpoint <code>POST /predict</code>. Os contratos <code>Pydantic</code> barram leituras fora dos limites físicos antes da transformação pelo <code>StandardScaler</code> e inferência simultânea nos modelos <code>classifier_model.joblib</code> e <code>isolation_forest.joblib</code>.
        </div>
        """,
        unsafe_allow_html=True,
    )

    if "preset_edge" not in st.session_state:
        st.session_state.preset_edge = "Normal"
    if "log_sessao" not in st.session_state:
        st.session_state.log_sessao = []

    b1, b2, b3, b4 = st.columns(4)
    if b1.button("Simular Classe 0: Operação Nominal", use_container_width=True):
        st.session_state.preset_edge = "Normal"
    if b2.button("Simular Classe 1: Desgaste Rolamento", use_container_width=True):
        st.session_state.preset_edge = "Rolamento"
    if b3.button("Simular Classe 2: Desbalanceamento Rotor", use_container_width=True):
        st.session_state.preset_edge = "Rotor"
    if b4.button("Simular Classe 3: Cavitação / Superaquec.", use_container_width=True):
        st.session_state.preset_edge = "Cavitacao"

    mapa_presets = {
        "Normal": (62.0, 2.0, 4.3, 2850.0, 51.0, 27.5, 1100.0),
        "Rolamento": (79.5, 6.2, 3.5, 2980.0, 68.0, 35.0, 3250.0),
        "Rotor": (71.0, 9.1, 4.0, 3520.0, 60.5, 39.5, 2600.0),
        "Cavitacao": (97.5, 7.4, 1.6, 3180.0, 74.0, 42.5, 4100.0),
    }
    p = mapa_presets[st.session_state.preset_edge]

    c_form1, c_form2, c_diag = st.columns([1, 1, 1.35])
    with c_form1:
        st.markdown('<div class="corp-card"><div class="corp-card-title"><i class="bi bi-sliders2"></i> SENSORES TERMOMECÂNICOS</div>', unsafe_allow_html=True)
        id_eq = st.selectbox("Equipamento:", ["TURB-101", "TURB-102", "TURB-103", "TURB-104"])
        temp_c = st.slider("Temperatura do Mancal (°C)", 20.0, 120.0, p[0], 0.5)
        vib_mms = st.slider("Vibração RMS (mm/s)", 0.1, 15.0, p[1], 0.1)
        press_bar = st.slider("Pressão do Óleo (Bar)", 0.5, 8.0, p[2], 0.1)
        st.markdown("</div>", unsafe_allow_html=True)

    with c_form2:
        st.markdown('<div class="corp-card"><div class="corp-card-title"><i class="bi bi-lightning-charge-fill"></i> SENSORES ELÉTRICOS E ACÚSTICOS</div>', unsafe_allow_html=True)
        rot_rpm = st.slider("Velocidade de Rotação (RPM)", 1000.0, 4500.0, p[3], 50.0)
        acust_db = st.slider("Emissão Acústica (dB)", 30.0, 90.0, p[4], 0.5)
        corr_amp = st.slider("Corrente Elétrica (A)", 10.0, 60.0, p[5], 0.5)
        horas_op = st.number_input("Horímetro Acumulado (h):", 0.0, 10000.0, p[6], 100.0)
        st.markdown("</div>", unsafe_allow_html=True)

    sev_din = vib_mms / max(rot_rpm, 1.0)
    vetor_in = {
        "temperatura_mancal_c": temp_c,
        "vibracao_rms_mms": vib_mms,
        "pressao_oleo_bar": press_bar,
        "rotacao_rpm": rot_rpm,
        "emissao_acustica_db": acust_db,
        "corrente_eletrica_amp": corr_amp,
        "horas_operacao_acumuladas": horas_op,
        "severidade_dinamica": sev_din,
        "temperatura_mancal_c_roll_mean_6": temp_c,
        "temperatura_mancal_c_roll_std_6": 0.4,
        "temperatura_mancal_c_roll_mean_36": temp_c,
        "temperatura_mancal_c_roll_std_36": 0.7,
        "vibracao_rms_mms_roll_mean_6": vib_mms,
        "vibracao_rms_mms_roll_std_6": 0.15,
        "vibracao_rms_mms_roll_mean_36": vib_mms,
        "vibracao_rms_mms_roll_std_36": 0.25,
        "temp_roll_mean_1h": temp_c,
        "temp_roll_std_1h": 0.4,
        "temp_roll_mean_6h": temp_c,
        "temp_roll_std_6h": 0.7,
        "vib_roll_mean_1h": vib_mms,
        "vib_roll_std_1h": 0.15,
        "vib_roll_mean_6h": vib_mms,
        "vib_roll_std_6h": 0.25,
    }

    df_s = alinhar_features_para_modelo(pd.DataFrame([vetor_in]), artefatos["features"])
    X_s = artefatos["scaler"].transform(df_s)
    classe_p = int(round(artefatos["classifier"].predict(X_s)[0]))
    probs = artefatos["classifier"].predict_proba(X_s)[0]
    anom_iso = bool(artefatos["iso_forest"].predict(X_s)[0] == -1)
    ts_str = datetime.now().strftime("%H:%M:%S")

    with c_diag:
        st.markdown('<div class="corp-card"><div class="corp-card-title"><i class="bi bi-cpu"></i> LAUDO DE INFERÊNCIA DO NÓ EDGE</div>', unsafe_allow_html=True)
        if classe_p == 0 and not anom_iso:
            st.markdown(f'<div class="status-badge-ok"><i class="bi bi-check-circle-fill"></i> {MAPA_CLASSES[classe_p]}</div>', unsafe_allow_html=True)
        elif classe_p == 0 and anom_iso:
            st.markdown('<div class="status-badge-warn"><i class="bi bi-exclamation-triangle-fill"></i> ALERTA ISOLATION FOREST: DESVIO NÃO MAPEADO</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="status-badge-crit"><i class="bi bi-x-octagon-fill"></i> FALHA CRÍTICA: {MAPA_CLASSES[classe_p]}</div>', unsafe_allow_html=True)

        st.markdown(
            f"""
            <div style="margin-top: 12px; font-size: 0.85rem; font-family: monospace; background: #F4F6F9; padding: 10px; border-radius: 4px; border: 1px solid #D9E2EC;">
                <strong>BANCADA:</strong> {id_eq} | <strong>TIMESTAMP:</strong> {ts_str}<br>
                <strong>SEVERIDADE DINÂMICA:</strong> {sev_din:.6f} mm/s/RPM<br>
                <strong>DETECTOR ISO-FOREST:</strong> {"ANOMALIA (-1)" if anom_iso else "NOMINAL (+1)"}
            </div>
            """,
            unsafe_allow_html=True,
        )

        fig_p, ax_p = plt.subplots(figsize=(5.5, 2.2))
        classes_curtas = ["Classe 0 (Normal)", "Classe 1 (Rolam.)", "Classe 2 (Rotor)", "Classe 3 (Cavit.)"]
        ax_p.barh(classes_curtas, probs * 100, color=["#0E9F6E", "#D97706", "#00579D", "#C81E1E"])
        ax_p.set_xlim(0, 100)
        ax_p.set_xlabel("Probabilidade (%)")
        ax_p.spines["top"].set_visible(False)
        ax_p.spines["right"].set_visible(False)
        st.pyplot(fig_p)
        plt.close(fig_p)

        if st.button("Salvar Leitura no Relatório de Sessão", use_container_width=True):
            st.session_state.log_sessao.insert(0, {
                "Horário": ts_str,
                "Bancada": id_eq,
                "Temp (°C)": temp_c,
                "Vib (mm/s)": vib_mms,
                "Pressão (Bar)": press_bar,
                "RPM": rot_rpm,
                "Diagnóstico": MAPA_CLASSES[classe_p],
                "Confiança (%)": round(float(np.max(probs)) * 100, 2),
                "IsoForest": "ANOMALIA" if anom_iso else "NORMAL",
            })
        st.markdown("</div>", unsafe_allow_html=True)

    if st.session_state.log_sessao:
        st.markdown('<div class="corp-card"><div class="corp-card-title"><i class="bi bi-journal-text"></i> HISTÓRICO DE TELEMETRIA AUDITADA NA SESSÃO</div>', unsafe_allow_html=True)
        st.dataframe(pd.DataFrame(st.session_state.log_sessao), use_container_width=True, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)


def tela_visao_computacional() -> None:
    """Tela 05: Etapa 3 — Visão Computacional Multimodal (Canny Dimensional e HSV Térmico)."""
    st.markdown(
        """
        <div class="theory-box">
            <strong><i class="bi bi-journal-bookmark-fill" style="color: #00579D;"></i> FUNDAMENTAÇÃO DE VISÃO COMPUTACIONAL (<code>src/vision/geometric_inspector.py &amp; thermal_inspector.py</code>)</strong><br>
            1. <strong>Módulo Dimensional (Canny + Otsu):</strong> Aplica filtro Gaussiana, binarização adaptativa de Otsu e extração de contornos para aferir o índice de <strong>Circularidade Matemática</strong>. O algoritmo avalia se o componente está dentro da tolerância geométrica nominal.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.latex(r"C = \frac{4\pi \cdot \text{Área}}{\text{Perímetro}^2}")
    st.markdown(
        """
        <div class="theory-box" style="margin-top: 10px;">
            2. <strong>Módulo Termográfico (Espaço HSV):</strong> Converte matrizes BGR para o espaço cromático <strong>HSV</strong> (Matiz, Saturação, Valor), isolando os espectros quentes e aplicando fechamento morfológico para quantificar a área térmica crítica em pixels.
        </div>
        """,
        unsafe_allow_html=True,
    )

    sub_rota = st.radio(
        "Selecione o Módulo de Inspeção Óptica:",
        [
            "Módulo Geométrico // Inspeção Dimensional de Rotores (Detector Canny + Circularidade)",
            "Módulo Termográfico // Inspeção Infravermelha de Mancais (Segmentação Morfológica HSV)",
        ],
        horizontal=True,
    )

    if "Geométrico" in sub_rota:
        c_in, c_out = st.columns([1, 2.2])
        with c_in:
            st.markdown('<div class="corp-card"><div class="corp-card-title"><i class="bi bi-gear-wide-connected"></i> PARÂMETROS GEOMÉTRICOS</div>', unsafe_allow_html=True)
            fonte_g = st.radio("Origem da Imagem:", ["Amostras de Bancada Padrão", "Upload de Arquivo Externo / IA"])
            min_circ = st.slider("Limiar de Circularidade Mínima:", 0.10, 0.95, 0.78, 0.02)
            min_area = st.slider("Limiar de Área Mínima (px):", 1000.0, 25000.0, 10000.0, 500.0)
            usar_otsu = st.checkbox("Isolar Silhueta (Filtro Otsu)", value=True)
            inv_manual = st.checkbox("Inverter Polaridade (Fundo Claro)", value=False)

            img_g = None
            nome_g = "Amostra Sintética"
            if "Upload" in fonte_g:
                arq = st.file_uploader("Selecione imagem (.png, .jpg, .jpeg)", type=["png", "jpg", "jpeg"])
                if arq is not None:
                    img_g = carregar_imagem_upload(arq)
                    nome_g = arq.name
            else:
                op = st.selectbox("Peça de Ensaio:", ["Amostra A: Disco Rotor Conforme", "Amostra B: Disco Rotor c/ Deformação e Trinca"])
                nome_g = op
                img_g = np.zeros((300, 300, 3), dtype=np.uint8)
                if "Conforme" in op:
                    cv2.circle(img_g, (150, 150), 75, (255, 255, 255), -1)
                else:
                    cv2.ellipse(img_g, (150, 150), (85, 45), 30, 0, 360, (255, 255, 255), -1)
                    cv2.rectangle(img_g, (140, 100), (160, 130), (0, 0, 0), -1)
            st.markdown("</div>", unsafe_allow_html=True)

        with c_out:
            st.markdown('<div class="corp-card"><div class="corp-card-title"><i class="bi bi-bounding-box"></i> RESULTADO DA INSPEÇÃO DIMENSIONAL</div>', unsafe_allow_html=True)
            if img_g is not None:
                img_proc = pre_processar_silhueta_inteligente(img_g, inverter_manual=inv_manual) if usar_otsu else img_g
                insp_g = GeometricInspector(blur_ksize=(5, 5), canny_low=50, canny_high=150, min_circularity=min_circ, min_area_px=min_area)
                res_g = insp_g.inspecionar_geometria(img_proc)
                if res_g.get("area_px", res_g.get("area_total", 0.0)) == 0.0 and usar_otsu:
                    res_g = insp_g.inspecionar_geometria(img_g)

                aprov = bool(res_g.get("aprovado", False))
                area_v = float(res_g.get("area_px", res_g.get("area_total", 0.0)))
                perim_v = float(res_g.get("perimetro_px", 0.0))
                circ_v = float(res_g.get("circularidade", 0.0))

                if aprov:
                    st.markdown(f'<div class="status-badge-ok"><i class="bi bi-check-circle-fill"></i> COMPONENTE APROVADO — {nome_g}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="status-badge-crit"><i class="bi bi-x-octagon-fill"></i> COMPONENTE REPROVADO (VIOLAÇÃO GEOMÉTRICA) — {nome_g}</div>', unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)
                m1, m2, m3 = st.columns(3)
                renderizar_kpi(m1, "Circularidade Calculada", f"{circ_v:.4f}", f"Exigido >= {min_circ:.2f}", "bi-circle-square")
                renderizar_kpi(m2, "Área do Contorno", f"{area_v:.1f} px", f"Exigido >= {min_area:.0f} px", "bi-aspect-ratio")
                renderizar_kpi(m3, "Perímetro Extraído", f"{perim_v:.1f} px", "Escala: 300x300 px", "bi-rulers")

                st.markdown("<br>", unsafe_allow_html=True)
                bordas_n, over_rgb = gerar_visualizacao_geom(img_g, res_g["mascara_bordas"], aprov)
                v1, v2, v3 = st.columns(3)
                v1.image(cv2.cvtColor(img_g, cv2.COLOR_BGR2RGB), caption="01. Imagem de Entrada (300x300)", use_container_width=True)
                v2.image(bordas_n, caption="02. Mapa de Bordas Canny", use_container_width=True)
                v3.image(over_rgb, caption="03. Vetor de Contorno Auditado", use_container_width=True)
            else:
                st.info("Envie um arquivo de imagem na coluna à esquerda para processar a inspeção.")
            st.markdown("</div>", unsafe_allow_html=True)

    else:
        c_tin, c_tout = st.columns([1, 2.2])
        with c_tin:
            st.markdown('<div class="corp-card"><div class="corp-card-title"><i class="bi bi-thermometer-high"></i> PARÂMETROS TERMOGRÁFICOS</div>', unsafe_allow_html=True)
            fonte_t = st.radio("Origem Termográfica:", ["Simulação de Bancada", "Upload de Termografia Externa"])
            lim_hot = st.slider("Área Crítica p/ Alarme de Hotspot (px):", 50.0, 5000.0, 300.0, 50.0)

            img_t = None
            nome_t = "Simulação Termográfica"
            if "Upload" in fonte_t:
                up_t = st.file_uploader("Selecione termografia (.png, .jpg)", type=["png", "jpg", "jpeg"])
                if up_t is not None:
                    img_t = carregar_imagem_upload(up_t)
                    nome_t = up_t.name
            else:
                cen = st.selectbox("Cenário Térmico:", ["Cenário 01: Regime Térmico Nominal Estável", "Cenário 02: Sobreaquecimento Crítico (2 Hotspots)"])
                nome_t = cen
                img_t = np.full((300, 300, 3), (180, 50, 30), dtype=np.uint8)
                if "Hotspots" in cen:
                    cv2.circle(img_t, (130, 160), 40, (0, 0, 255), -1)
                    cv2.circle(img_t, (200, 100), 20, (0, 20, 255), -1)
            st.markdown("</div>", unsafe_allow_html=True)

        with c_tout:
            st.markdown('<div class="corp-card"><div class="corp-card-title"><i class="bi bi-fire"></i> RESULTADO DA INSPEÇÃO TERMOGRÁFICA (HSV)</div>', unsafe_allow_html=True)
            if img_t is not None:
                res_t = executar_inspecao_termica_segura(img_t, lim_hot)
                alerta_t = res_t["alerta"]
                area_c = res_t["area_critica_px"]
                focos_c = res_t["focos"]
                masc_hsv = res_t["mascara_hsv"]

                if not alerta_t:
                    st.markdown(f'<div class="status-badge-ok"><i class="bi bi-check-circle-fill"></i> REGIME TÉRMICO ESTÁVEL — {nome_t}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="status-badge-crit"><i class="bi bi-exclamation-octagon-fill"></i> ALARME DE SOBREAQUECIMENTO ATIVO — {nome_t}</div>', unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)
                tk1, tk2, tk3 = st.columns(3)
                renderizar_kpi(tk1, "Estado Térmico", "CRÍTICO" if alerta_t else "NOMINAL", f"Limite: {lim_hot:.0f} px", "bi-thermometer-sun")
                renderizar_kpi(tk2, "Área Quente Isolada", f"{area_c:.1f} px", "Segmentação HSV Dupla", "bi-bounding-box-circles")
                renderizar_kpi(tk3, "Focos de Calor", f"{focos_c} Foco(s)", "Filtro Morfológico Ativo", "bi-bullseye")

                st.markdown("<br>", unsafe_allow_html=True)
                over_t = cv2.cvtColor(img_t, cv2.COLOR_BGR2RGB)
                if masc_hsv is not None:
                    cnts, _ = cv2.findContours(masc_hsv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    for c in cnts:
                        if cv2.contourArea(c) >= 15.0:
                            x, y, w, h = cv2.boundingRect(c)
                            cv2.rectangle(over_t, (x, y), (x + w, y + h), (255, 255, 0), 2)

                tv1, tv2, tv3 = st.columns(3)
                tv1.image(cv2.cvtColor(img_t, cv2.COLOR_BGR2RGB), caption="01. Imagem Termográfica (RGB)", use_container_width=True)
                if masc_hsv is not None:
                    tv2.image(masc_hsv, caption="02. Máscara Morfológica HSV", use_container_width=True)
                tv3.image(over_t, caption="03. Delimitação de Hotspots", use_container_width=True)
            else:
                st.info("Envie uma imagem termográfica na coluna à esquerda para processar a inspeção.")
            st.markdown("</div>", unsafe_allow_html=True)


def tela_documentacao_teorica() -> None:
    """Tela 06: Documentação Técnica Completa, Equações Formais e Governança."""
    st.markdown(
        """
        <div class="corp-card">
            <div class="corp-card-title"><i class="bi bi-book-half"></i> DOCUMENTAÇÃO TÉCNICA E FUNDAMENTAÇÃO TEÓRICA — AEROMOTION PREDICTIVE SYSTEM</div>
            <p style="font-size: 0.90rem; color: #334E68; margin-bottom: 0;">
                Manual interativo dos módulos de software, formulações matemáticas e diretrizes regulatórias implementadas para a AeroMotion Dynamics S.A.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    filtro_doc = st.text_input(
        "Filtrar Tópicos da Documentação (ex: SMOTE, Canny, Drift, LGPD, EU AI Act, Severidade):",
        placeholder="Digite uma palavra-chave para filtrar as seções abaixo...",
    ).strip().lower()

    if not filtro_doc or any(k in filtro_doc for k in ["1", "saneamento", "emi", "severidade", "rolling", "dados"]):
        with st.container():
            st.markdown('<div class="corp-card"><div class="corp-card-title"><i class="bi bi-database-gear"></i> 1. ENGENHARIA DE DADOS E SANEAMENTO FÍSICO (src/data_pipeline/)</div>', unsafe_allow_html=True)
            st.markdown(
                "- **`src/data_pipeline/preprocessor.py`**: Elimina leituras espúrias decorrentes de Interferência Eletromagnética (EMI) que violam limites físicos operacionais (ex.: temperaturas de **999,9 °C** e vibrações de **-99,0 mm/s**). Os registros corrompidos são convertidos em `NaN` e reconstruídos via interpolação temporal linear agrupada por `id_equipamento` (`TURB-101` a `TURB-104`).\n"
                "- **`src/data_pipeline/feature_engineering.py`**: Extrai estatísticas móveis (média e desvio padrão) em janelas de **1 hora (6 períodos de 10 min)** e **6 horas (36 períodos de 10 min)**, além do atributo de **Severidade Dinâmica**:"
            )
            st.latex(r"\text{Severidade Dinâmica} = \frac{\text{Vibração RMS (mm/s)}}{\text{Rotação (RPM)}}")
            st.markdown("</div>", unsafe_allow_html=True)

    if not filtro_doc or any(k in filtro_doc for k in ["2", "ml", "lightgbm", "smote", "isolation", "recall"]):
        with st.container():
            st.markdown('<div class="corp-card"><div class="corp-card-title"><i class="bi bi-cpu-fill"></i> 2. MODELAGEM PREDITIVA HÍBRIDA (src/ml/train.py & evaluation.py)</div>', unsafe_allow_html=True)
            st.markdown(
                "- **Balanceamento Estrito com SMOTE**: Como os modos de falha somam ~12% da base histórica, a técnica *Synthetic Minority Over-sampling Technique* (SMOTE) é aplicada exclusivamente na partição de treinamento (70%), evitando *data leakage* para o conjunto de teste (30%).\n"
                "- **Classificação Multiclasse (`LightGBM`)**: Otimizado para atingir **Recall > 90%** nas classes críticas (`0`: Operação Nominal, `1`: Desgaste de Rolamento, `2`: Desbalanceamento de Rotor, `3`: Cavitação / Superaquecimento).\n"
                "- **Detecção Não Supervisionada (`Isolation Forest`)**: Treinado com amostras de operação nominal para isolar desvios estocásticos e modos de falha inéditos em produção."
            )
            st.markdown("</div>", unsafe_allow_html=True)

    if not filtro_doc or any(k in filtro_doc for k in ["3", "visao", "canny", "otsu", "circularidade", "hsv", "termografia"]):
        with st.container():
            st.markdown('<div class="corp-card"><div class="corp-card-title"><i class="bi bi-camera-video-fill"></i> 3. VISÃO COMPUTACIONAL MULTIMODAL (src/vision/)</div>', unsafe_allow_html=True)
            st.markdown(
                "- **`src/vision/geometric_inspector.py`**: Executa suavização Gaussiana, limiarização adaptativa de Otsu com detecção automática de polaridade de fundo e detecção de bordas de Canny. A integridade estrutural da peça é validada pelo índice de **Circularidade de Cox**:"
            )
            st.latex(r"C = \frac{4\pi \cdot \text{Área}}{\text{Perímetro}^2}")
            st.markdown(
                "- **`src/vision/thermal_inspector.py`**: Converte matrizes radiométricas BGR para o espaço cromático **HSV**, aplicando máscara dupla nas extremidades do espectro vermelho e fechamento morfológico para contabilizar a área térmica crítica e o número de *hotspots*."
            )
            st.markdown("</div>", unsafe_allow_html=True)

    if not filtro_doc or any(k in filtro_doc for k in ["4", "mlops", "drift", "ks", "psi", "lgpd", "eu ai act", "seguranca"]):
        with st.container():
            st.markdown('<div class="corp-card"><div class="corp-card-title"><i class="bi bi-shield-lock-fill"></i> 4. GOVERNANÇA, PLANO DE DRIFT (12 MESES) E CIBERSEGURANÇA OT/IT</div>', unsafe_allow_html=True)
            st.markdown(
                "- **Monitoramento de Data Drift (`docs/plano_mlops_drift.pdf`)**: Execução quinzenal do teste não-paramétrico de **Kolmogorov-Smirnov (KS)** (alerta em *p-valor* < 0,01) e do **Population Stability Index (PSI)** (alerta crítico em PSI >= 0,25).\n"
                "- **Mitigação de Concept Drift**: Auditoria mensal do Recall em produção frente às ordens de manutenção. Quedas de Recall abaixo de **85%** disparam retreinamento automático com validação em *Shadow Mode*.\n"
                "- **LGPD, EU AI Act e Blindagem OT/IT (`docs/parecer_tecnico_etico_regulatorio.pdf`)**: Mascaramento facial ativo por filtro Gaussiano, supervisão humana (*Human-in-the-Loop*, Art. 14 do EU AI Act) e validação física via contratos `Pydantic` em `src/api/schemas.py` contra *Data Poisoning*."
            )
            st.markdown("</div>", unsafe_allow_html=True)


# =============================================================================
# 5. ORQUESTRADOR PRINCIPAL E MENU LATERAL
# =============================================================================
def main() -> None:
    """Inicializa o sistema e gerencia o roteamento de telas."""
    st.set_page_config(
        page_title="AeroMotion Predictive System",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    aplicar_estilo_industrial()

    with st.sidebar:
        st.markdown(
            """
            <div style="background-color: #002B49; color: #FFFFFF; padding: 14px 16px; border-radius: 6px; border-left: 4px solid #0090D4; margin-bottom: 14px;">
                <div style="font-size: 0.92rem; font-weight: 700; letter-spacing: 0.5px;">
                    <i class="bi bi-cpu" style="color: #0090D4; margin-right: 6px;"></i>AEROMOTION SYSTEM
                </div>
                <div style="font-size: 0.75rem; color: #9FB3C8; font-family: monospace; margin-top: 3px;">
                    AeroMotion Dynamics S.A.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        busca_modulo = st.text_input(
            "Pesquisar Módulo do Projeto:",
            placeholder="Ex: eda, mlops, vision, api, docs...",
        ).strip().lower()

        opcoes_filtradas = [
            nome for nome, meta in MODULOS_PROJETO.items()
            if not busca_modulo
            or busca_modulo in nome.lower()
            or busca_modulo in meta["modulo_py"].lower()
            or busca_modulo in meta["tags"]
        ]
        if not opcoes_filtradas:
            opcoes_filtradas = list(MODULOS_PROJETO.keys())

        modulo_selecionado = st.radio(
            "Módulos do Repositório:",
            opcoes_filtradas,
        )

        meta_atual = MODULOS_PROJETO[modulo_selecionado]

        st.divider()
        st.markdown(
            f"""
            <div style="background-color: #FFFFFF; border: 1px solid #D9E2EC; border-left: 3px solid #00579D; border-radius: 6px; padding: 12px 14px; font-size: 0.79rem; color: #334E68; line-height: 1.55;">
                <div style="font-weight: 700; color: #00447C; margin-bottom: 6px;">
                    <i class="bi bi-code-square"></i> ARQUIVO EM EXECUÇÃO
                </div>
                <div style="margin-bottom: 8px;"><code>{meta_atual['modulo_py']}</code></div>
                <div style="border-top: 1px solid #E4E7EB; padding-top: 6px;">
                    <strong>Autor:</strong> André Matteussi<br>
                    <strong>Docente:</strong> Prof. Fabiano da Silva Luiz
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    renderizar_cabecalho_sistema(meta_atual)

    df_raw, df_proc = carregar_dados()
    artefatos = carregar_artefatos_ml()

    if modulo_selecionado == "Visão Geral & Arquitetura":
        tela_visao_geral(df_raw, df_proc, artefatos)
    elif modulo_selecionado == "Saneamento & Engenharia (Etapa 1)":
        tela_engenharia_dados(df_raw, df_proc)
    elif modulo_selecionado == "Modelagem Preditiva & MLOps (Etapa 2)":
        tela_modelagem_ml(df_proc, artefatos)
    elif modulo_selecionado == "Simulador de Inferência Edge (API)":
        tela_simulador_edge(artefatos)
    elif modulo_selecionado == "Visão Computacional (Etapa 3)":
        tela_visao_computacional()
    elif modulo_selecionado == "Documentação Técnica & Teoria":
        tela_documentacao_teorica()

    renderizar_rodape_sistema()


if __name__ == "__main__":
    main()