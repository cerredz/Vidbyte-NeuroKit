from libs.enums import DataFormat
from libs.utils.formatter.base_formatter import DelimitedTableFormatter


class CsvFormatter(DelimitedTableFormatter):
    format_type = DataFormat.CSV
    extensions = (".csv",)
