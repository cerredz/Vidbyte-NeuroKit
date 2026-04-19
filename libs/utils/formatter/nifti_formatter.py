from libs.enums import DataFormat
from libs.utils.formatter.base_formatter import BaseFormatter


class NiftiFormatter(BaseFormatter):
    format_type = DataFormat.NIFTI
    extensions = (".nii.gz", ".nii")
