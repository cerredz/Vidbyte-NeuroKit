from libs.enums import DataFormat
from libs.utils.formatter.base_formatter import BaseFormatter


class GiftiFormatter(BaseFormatter):
    format_type = DataFormat.GIFTI
    extensions = (".gii",)
