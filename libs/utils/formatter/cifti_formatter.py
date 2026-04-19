from libs.enums import DataFormat
from libs.utils.formatter.base_formatter import BaseFormatter


class CiftiFormatter(BaseFormatter):
    format_type = DataFormat.CIFTI
    extensions = (".dscalar.nii", ".dtseries.nii")
