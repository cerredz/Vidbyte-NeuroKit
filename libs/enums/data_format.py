from enum import StrEnum


class DataFormat(StrEnum):
    NIFTI = "nifti"
    CIFTI = "cifti"
    ANALYZE = "analyze"
    MINC = "minc"
    BIDS = "bids"
    BIDS_EVENTS = "bids_events"
    AFNI_1D = "afni_1d"
    CSV = "csv"
    TSV = "tsv"
    HDF5 = "hdf5"
    GIFTI = "gifti"
    GRAPHML = "graphml"
    ADJACENCY_MATRIX = "adjacency_matrix"
    EDF = "edf"
    BRAINVISION = "brainvision"
    EEGLAB_SET = "eeglab_set"
    FIF = "fif"
