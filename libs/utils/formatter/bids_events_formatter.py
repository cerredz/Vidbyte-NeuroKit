from libs.enums import DataFormat
from libs.utils.formatter.base_formatter import DelimitedTableFormatter


class BidsEventsFormatter(DelimitedTableFormatter):
    format_type = DataFormat.BIDS_EVENTS
    extensions = ("_events.tsv",)
    delimiter = "\t"
    read_delimiter = "\t"
