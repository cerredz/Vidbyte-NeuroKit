from __future__ import annotations

from enum import StrEnum


class AudioSuffix(StrEnum):
    WAV = ".wav"
    MP3 = ".mp3"
    FLAC = ".flac"
    OGG = ".ogg"


class VideoSuffix(StrEnum):
    MP4 = ".mp4"
    AVI = ".avi"
    MKV = ".mkv"
    MOV = ".mov"
    WEBM = ".webm"


class ImageSuffix(StrEnum):
    JPG = ".jpg"
    JPEG = ".jpeg"
    PNG = ".png"
    BMP = ".bmp"
    WEBP = ".webp"


class TextSuffix(StrEnum):
    MD = ".md"
    TXT = ".txt"


def _enum_values(enum_type: type[StrEnum]) -> frozenset[str]:
    return frozenset(member.value for member in enum_type)


AUDIO_SUFFIX_VALUES = _enum_values(AudioSuffix)
VIDEO_SUFFIX_VALUES = _enum_values(VideoSuffix)
IMAGE_SUFFIX_VALUES = _enum_values(ImageSuffix)
TEXT_SUFFIX_VALUES = _enum_values(TextSuffix)
SUPPORTED_SUFFIX_VALUES = AUDIO_SUFFIX_VALUES | VIDEO_SUFFIX_VALUES | IMAGE_SUFFIX_VALUES | TEXT_SUFFIX_VALUES
