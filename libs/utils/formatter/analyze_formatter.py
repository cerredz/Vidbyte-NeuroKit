from libs.enums import DataFormat
from libs.utils.formatter.base_formatter import BaseFormatter


class AnalyzeFormatter(BaseFormatter):
    format_type = DataFormat.ANALYZE
    extensions = (".hdr", ".img")
