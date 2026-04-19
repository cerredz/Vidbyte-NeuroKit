from libs.enums import DataFormat
from libs.utils.formatter.base_formatter import BaseFormatter


class FifFormatter(BaseFormatter):
    format_type = DataFormat.FIF
    extensions = (".fif",)
