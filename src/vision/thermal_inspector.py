from typing import Dict, Tuple
import cv2
import numpy as np


class ThermalInspector:
    """Segmenta assinaturas térmicas em espaço HSV para delimitar

    e alertar zonas críticas de sobreaquecimento (hotspots).
    """

    def __init__(
        self,
        faixa_hsv_critica: Tuple[Tuple[int, int, int], Tuple[int, int, int]] = (
            (0, 140, 140),
            (12, 255, 255),
        ),
        limiar_area_critica_px: float = 300.0,
    ):
        self.hsv_min = np.array(faixa_hsv_critica[0], dtype=np.uint8)
        self.hsv_max = np.array(faixa_hsv_critica[1], dtype=np.uint8)
        self.limiar_area_critica_px = limiar_area_critica_px

    def detetar_hotspots(
        self, imagem_termica_bgr: np.ndarray
    ) -> Dict[str, any]:
        """Identifica zonas térmicas anómalas através de segmentação HSV."""
        hsv = cv2.cvtColor(imagem_termica_bgr, cv2.COLOR_BGR2HSV)
        mascara_termica = cv2.inRange(hsv, self.hsv_min, self.hsv_max)

        # Operações morfológicas para eliminar ruído térmico impulsivo
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mascara_filtrada = cv2.morphologyEx(
            mascara_termica, cv2.MORPH_OPEN, kernel
        )

        contornos_quentes, _ = cv2.findContours(
            mascara_filtrada, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        area_total_hotspot = sum(
            cv2.contourArea(c) for c in contornos_quentes
        )
        sobreaquecimento_critico = (
            area_total_hotspot >= self.limiar_area_critica_px
        )

        return {
            "alerta_sobreaquecimento": bool(sobreaquecimento_critico),
            "area_critica_px": float(round(area_total_hotspot, 2)),
            "quantidade_focos": len(contornos_quentes),
            "mascara_termica": mascara_filtrada,
        }