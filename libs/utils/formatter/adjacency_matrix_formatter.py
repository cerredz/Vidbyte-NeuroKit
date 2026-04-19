from libs.enums import DataFormat
from libs.utils.formatter.base_formatter import DelimitedTableFormatter


class AdjacencyMatrixFormatter(DelimitedTableFormatter):
    format_type = DataFormat.ADJACENCY_MATRIX
    extensions = (".adj.csv", ".adj.tsv", ".adj")
    include_header = False
