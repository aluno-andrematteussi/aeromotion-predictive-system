from typing import Dict, Tuple
import cv2
import numpy as np


class GeometricInspector:
    """Valida tolerâncias dimensionais e circularidade de componentes

    mecânicos através de deteção de bordas e contornos.
    """

    def __init__(
        self,
        blur_ksize: Tuple[int, int] = (5, 5),
        canny_low: int = 50,
        canny_high: int = 150,
    ):
        self.blur_ksize = blur_ksize
        self.canny_low = canny_low
        self.canny_high = canny_high

    def inspecionar_geometria(
        self, imagem_bgr: np.ndarray
    ) -> Dict[str, any]:
        """Processa a imagem e retorna métricas dimensionais e de conformidade."""
        cinza = cv2.cvtColor(imagem_bgr, cv2.COLOR_BGR2GRAY)
        suavizada = cv2.GaussianBlur(cinza, self.blur_ksize, 0)
        bordas = cv2.Canny(
            suavizada, self.canny_low, self.canny_high
        )

        contornos, _ = cv2.findContours(
            bordas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contornos:
            return {
                "aprovado": False,
                "motivo": "Nenhum contorno identificado",
                "area_total": 0.0,
                "circularidade": 0.0,
                "mascara_bordas": bordas,
            }

        # Seleciona o contorno principal da pá ou componente
        maior_contorno = max(contornos, key=cv2.contourArea)
        area = float(cv2.contourArea(maior_contorno))
        perimetro = float(cv2.arcLength(maior_contorno, True))

        # Circularidade: C = (4 * pi * Area) / (Perimetro^2) -> 1.0 para círculo perfeito
        circularidade = 0.0
        if perimetro > 0:
            circularidade = (4 * np.pi * area) / (perimetro**2)

        # Critérios de tolerância dimensional de projeto
        tolerancia_area = area >= 1500.0
        tolerancia_circularidade = circularidade >= 0.40
        aprovado = bool(tolerancia_area and tolerancia_circularidade)

        return {
            "aprovado": aprovado,
            "area_px": round(area, 2),
            "perimetro_px": round(perimetro, 2),
            "circularidade": round(circularidade, 4),
            "total_contornos": len(contornos),
            "mascara_bordas": bordas,
        }