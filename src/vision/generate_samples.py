"""Utilitário para geração de imagens sintéticas de bancada (.png).

Popula os diretórios data/raw/vision/geometric e data/raw/vision/thermal
com amostras de referência para validação via upload no Streamlit.
"""

from pathlib import Path
import cv2
import numpy as np


def gerar_amostras_visao() -> None:
    """Gera e salva imagens de teste geométricas e térmicas no disco."""
    base_dir = Path(__file__).resolve().parents[2]
    dir_geom = base_dir / "data" / "raw" / "vision" / "geometric"
    dir_therm = base_dir / "data" / "raw" / "vision" / "thermal"

    dir_geom.mkdir(parents=True, exist_ok=True)
    dir_therm.mkdir(parents=True, exist_ok=True)

    # 1. Amostra Geométrica Conforme (Rotor circular simétrico)
    geom_ok = np.zeros((300, 300, 3), dtype=np.uint8)
    cv2.circle(geom_ok, (150, 150), 75, (255, 255, 255), -1)
    cv2.imwrite(str(dir_geom / "amostra_rotor_conforme.png"), geom_ok)

    # 2. Amostra Geométrica com Defeito (Deformação elíptica + trinca na borda)
    geom_defeito = np.zeros((300, 300, 3), dtype=np.uint8)
    cv2.ellipse(geom_defeito, (150, 150), (85, 45), 30, 0, 360, (255, 255, 255), -1)
    cv2.rectangle(geom_defeito, (140, 100), (160, 130), (0, 0, 0), -1)
    cv2.imwrite(str(dir_geom / "amostra_rotor_trincado.png"), geom_defeito)

    # 3. Amostra Térmica Estável (Fundo frio homogêneo em BGR)
    therm_ok = np.full((300, 300, 3), (180, 50, 30), dtype=np.uint8)
    cv2.imwrite(str(dir_therm / "termografia_estavel.png"), therm_ok)

    # 4. Amostra Térmica Crítica (Presença de 2 hotspots severos em BGR)
    therm_critica = np.full((300, 300, 3), (180, 50, 30), dtype=np.uint8)
    cv2.circle(therm_critica, (130, 160), 40, (0, 0, 255), -1)
    cv2.circle(therm_critica, (200, 100), 20, (0, 20, 255), -1)
    cv2.imwrite(str(dir_therm / "termografia_hotspots.png"), therm_critica)

    print(f"Amostras salvas com sucesso em:\n- {dir_geom}\n- {dir_therm}")


if __name__ == "__main__":
    gerar_amostras_visao()