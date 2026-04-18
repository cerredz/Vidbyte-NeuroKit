from enum import StrEnum


class TranslationOutputKey(StrEnum):
    TEMPORAL = "temporal"
    PEAK = "peak"
    REGIONS = "regions"
    COGNITIVE = "cognitive"
    LANGUAGE = "language"
    COMPARE = "compare"
    DIFF = "diff"
    NORMALIZE = "normalize"
    SEGMENT = "segment"
    EXPORT = "export"
