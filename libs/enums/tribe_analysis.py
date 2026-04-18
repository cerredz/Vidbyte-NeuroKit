from enum import StrEnum


class ComparisonMetric(StrEnum):
    ENGAGEMENT = "engagement"
    COGNITIVE_LOAD = "cognitive_load"
    LANGUAGE = "language"


class ComparisonWinner(StrEnum):
    A = "a"
    B = "b"
    TIE = "tie"


class DestrieuxRegion(StrEnum):
    G_FRONT_SUP = "G_front_sup"
    G_FRONT_MIDDLE = "G_front_middle"
    G_FRONT_INF_OPERCULAR = "G_front_inf-Opercular"
    G_FRONT_INF_TRIANGUL = "G_front_inf-Triangul"
    G_TEMP_SUP_G_T_TRANSV = "G_temp_sup-G_T_transv"
    G_TEMP_SUP_PLAN_TEMPO = "G_temp_sup-Plan_tempo"
    S_TEMPORAL_SUP = "S_temporal_sup"


class ExportFormat(StrEnum):
    JSON = "json"
    CSV = "csv"
    NIFTI = "nifti"


PFC_REGIONS: tuple[DestrieuxRegion, ...] = (
    DestrieuxRegion.G_FRONT_SUP,
    DestrieuxRegion.G_FRONT_MIDDLE,
    DestrieuxRegion.G_FRONT_INF_OPERCULAR,
    DestrieuxRegion.G_FRONT_INF_TRIANGUL,
)

LANGUAGE_REGIONS: tuple[DestrieuxRegion, ...] = (
    DestrieuxRegion.G_FRONT_INF_OPERCULAR,
    DestrieuxRegion.G_FRONT_INF_TRIANGUL,
    DestrieuxRegion.G_TEMP_SUP_G_T_TRANSV,
    DestrieuxRegion.G_TEMP_SUP_PLAN_TEMPO,
    DestrieuxRegion.S_TEMPORAL_SUP,
)
