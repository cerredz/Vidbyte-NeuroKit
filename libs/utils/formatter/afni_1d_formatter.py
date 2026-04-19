from libs.enums import DataFormat
from libs.utils.formatter.base_formatter import DelimitedTableFormatter


class Afni1DFormatter(DelimitedTableFormatter):
    format_type = DataFormat.AFNI_1D
    extensions = (".1d",)
    delimiter = " "
    read_delimiter = r"\s+"
    include_header = False
