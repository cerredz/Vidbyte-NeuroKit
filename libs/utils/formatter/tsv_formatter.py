from libs.enums import DataFormat
from libs.utils.formatter.base_formatter import DelimitedTableFormatter


class TsvFormatter(DelimitedTableFormatter):
    format_type = DataFormat.TSV
    extensions = (".tsv",)
    delimiter = "\t"
    read_delimiter = "\t"
