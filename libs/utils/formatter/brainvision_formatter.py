from libs.enums import DataFormat
from libs.utils.formatter.base_formatter import BaseFormatter


class BrainVisionFormatter(BaseFormatter):
    format_type = DataFormat.BRAINVISION
    extensions = (".vhdr", ".vmrk", ".eeg")
