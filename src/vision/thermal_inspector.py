"""Módulo de inspeção termográfica infravermelha no espaço de cores HSV.

Segmenta zonas de sobreaquecimento (hotspots) em componentes rotativos e
mancais utilizando limiarização cromática e morfologia matemática.
"""

from typing import Any, Dict, Tuple
import cv2
import numpy as np


class ThermalInspector:
    """Detecta e quantifica focos críticos de calor (hotspots) em imagens térmicas."""

    def __init__(
        self,
        lower_hsv1: Tuple[int, int, int] = (0, 100, 100),
        upper_hsv1: Tuple[int, int, int] = (18, 255, 255),
        lower_hsv2: Tuple[int, int, int] = (160, 100, 100),
        upper_hsv2: Tuple[int, int, int] = (180, 255, 255),
        min_hotspot_area: float = 300.0,
    ) -> None:
        """Inicializa os intervalos de matiz quente no espaço HSV e o limiar crítico."""
        self.lower_hsv1 = np.array(lower_hsv1, dtype=np.uint8)
        self.upper_hsv1 = np.array(upper_hsv1, dtype=np.uint8)
        self.lower_hsv2 = np.array(lower_hsv2, dtype=np.uint8)
        self.upper_hsv2 = np.array(upper_hsv2, dtype=np.uint8)
        self.min_hotspot_area = min_hotspot_area

    def inspecionar_termografia(self, imagem_bgr: np.ndarray) -> Dict[str, Any]:
        """Processa a imagem BGR e retorna métricas de sobreaquecimento e máscara HSV."""
        hsv = cv2.cvtColor(imagem_bgr, cv2.COLOR_BGR2HSV)

        # Captura tons vermelhos, laranjas e amarelos intensos de alta temperatura
        mascara1 = cv2.inRange(hsv, self.lower_hsv1, self.upper_hsv1)
        mascara2 = cv2.inRange(hsv, self.lower_hsv2, self.upper_hsv2)
        mascara_combinada = cv2.bitwise_or(mascara1, mascara2)

        # Limpeza morfológica para consolidar os focos de calor
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mascara_limpa = cv2.morphologyEx(mascara_combinada, cv2.MORPH_OPEN, kernel)
        mascara_limpa = cv2.morphologyEx(mascara_limpa, cv2.MORPH_CLOSE, kernel)

        contornos, _ = cv2.findContours(
            mascara_limpa, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        focos_validos = [c for c in contornos if cv2.contourArea(c) >= 15.0]
        area_critica = float(sum(cv2.contourArea(c) for c in focos_validos))
        alerta = bool(area_critica >= self.min_hotspot_area)

        return {
            "alerta": alerta,
            "alerta_sobreaquecimento": alerta,
            "area_critica_px": round(area_critica, 2),
            "focos": len(focos_validos),
            "focos_detectados": len(focos_validos),
            "mascara_hsv": mascara_limpa,
            "mascara_termica": mascara_limpa,
        }

    def analisar_termografia(self, imagem_bgr: np.ndarray) -> Dict[str, Any]:
        """Alias de compatibilidade para chamadas externas."""
        return self.inspecionar_termografia(imagem_bgr)