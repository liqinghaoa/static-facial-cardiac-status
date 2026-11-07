from __future__ import annotations

from dataclasses import dataclass


OUTPUT_WIDTH = 1024
OUTPUT_HEIGHT = 1280

FAILURE_CODES = (
    "missing_source_image",
    "image_decode_failed",
    "face_detection_failed",
    "facemesh_failed",
    "invalid_landmarks",
    "degenerate_eye_distance",
    "invalid_face_oval",
    "invalid_crop_geometry",
    "invalid_affine_matrix",
    "warp_failed",
    "parser_failed",
    "empty_skin_mask",
    "invalid_output_shape",
    "nonfinite_linear_rgb",
    "output_write_failed",
)

WARNING_CODES = (
    "large_roll",
    "high_hair_occlusion",
    "low_skin_fraction",
    "high_skin_fraction",
    "top_invalid_area",
    "bottom_invalid_area",
    "side_invalid_area",
    "face_near_canvas_border",
    "low_source_face_resolution",
    "possible_large_yaw",
    "possible_large_pitch",
)


class SampleFailure(RuntimeError):
    def __init__(self, code: str, detail: str):
        super().__init__(detail)
        self.code = code
        self.detail = detail


@dataclass(frozen=True)
class GeometryConfig:
    output_width: int = OUTPUT_WIDTH
    output_height: int = OUTPUT_HEIGHT
    side_margin_ratio: float = 0.03
    top_margin_ratio: float = 0.02
    bottom_margin_ratio: float = 0.025
    face_quantile_low: float = 0.02
    face_quantile_high: float = 0.98
    matrix_atol: float = 1.0e-4
    roundtrip_tolerance_px: float = 1.0e-3


@dataclass(frozen=True)
class SkinMaskConfig:
    eye_brow_lip_dilation_ratio: float = 0.006
    nostril_dilation_ratio: float = 0.003
    # Backward-compatible alias for initial pilot configs.
    exclusion_dilation_ratio: float | None = None
    min_component_area_ratio: float = 0.0002
    low_skin_fraction_warning: float = 0.08
    high_skin_fraction_warning: float = 0.65
    high_hair_fraction_warning: float = 0.12

    def normal_dilation_ratio(self) -> float:
        return (
            float(self.exclusion_dilation_ratio)
            if self.exclusion_dilation_ratio is not None
            else float(self.eye_brow_lip_dilation_ratio)
        )


@dataclass(frozen=True)
class DetectionConfig:
    min_detection_confidence: float = 0.5
    model_selection: int = 0
    expand_top_ratio: float = 0.10
    expand_bottom_ratio: float = 0.20
    expand_side_ratio: float = 0.20


@dataclass(frozen=True)
class ParserConfig:
    model: str = "bisenet"
    # Paths are resolved relative to the package's ``src`` root by the
    # whole-face preprocessing entry point.
    checkpoint: str = "../assets/face_parsing/79999_iter.pth"
    device: str = "auto"
