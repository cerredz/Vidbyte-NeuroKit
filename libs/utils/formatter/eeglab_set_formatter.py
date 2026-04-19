from libs.enums import DataFormat
from libs.utils.formatter.base_formatter import BaseFormatter


class EeglabSetFormatter(BaseFormatter):
    format_type = DataFormat.EEGLAB_SET
    extensions = (".set",)
