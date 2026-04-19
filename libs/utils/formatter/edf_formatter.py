from libs.enums import DataFormat
from libs.utils.formatter.base_formatter import BaseFormatter


class EdfFormatter(BaseFormatter):
    format_type = DataFormat.EDF
    extensions = (".edf",)
