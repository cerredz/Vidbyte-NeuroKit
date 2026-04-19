from libs.enums import DataFormat
from libs.utils.formatter.base_formatter import BaseFormatter


class MincFormatter(BaseFormatter):
    format_type = DataFormat.MINC
    extensions = (".mnc",)
