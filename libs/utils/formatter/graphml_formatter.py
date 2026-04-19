from libs.enums import DataFormat
from libs.utils.formatter.base_formatter import BaseFormatter


class GraphMlFormatter(BaseFormatter):
    format_type = DataFormat.GRAPHML
    extensions = (".graphml",)
