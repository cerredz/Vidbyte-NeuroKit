from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from libs.dataclasses.prediction_result import PredictionResult
from libs.dataclasses.tribe_analysis import ComparisonResult


def _freeze_mapping(payload: Mapping[str, Any] | None) -> dict[str, Any]:
    return dict(payload or {})


@dataclass(frozen=True, slots=True)
class DownloadedProviderAsset:
    provider: str
    remote_id: str
    name: str
    local_path: Path
    mime_type: str | None = None
    source_url: str | None = None
    size_bytes: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "local_path", Path(self.local_path).expanduser().resolve())
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata))


@dataclass(frozen=True, slots=True)
class ProviderMetricResult:
    metric: str
    score: float | None
    result: Any


@dataclass(frozen=True, slots=True)
class ProviderAnalysisResult:
    provider: str
    asset: DownloadedProviderAsset
    prediction: PredictionResult
    metrics: tuple[ProviderMetricResult, ...]


@dataclass(frozen=True, slots=True)
class ProviderBatchAnalysisResult:
    provider: str
    items: tuple[ProviderAnalysisResult, ...]
    sorted_by: str | None = None


@dataclass(frozen=True, slots=True)
class ProviderComparisonResult:
    provider: str
    metric: str
    left: ProviderAnalysisResult
    right: ProviderAnalysisResult
    comparison: ComparisonResult


@dataclass(frozen=True, slots=True)
class VimeoAccount:
    user_id: str
    name: str
    uri: str
    account_type: str | None = None


@dataclass(frozen=True, slots=True)
class VimeoVideoFile:
    link: str
    public_name: str
    quality: str
    rendition: str
    type: str | None = None
    width: int | None = None
    height: int | None = None
    fps: float | None = None
    size_bytes: int | None = None
    source_link: str | None = None
    expires: str | None = None


@dataclass(frozen=True, slots=True)
class VimeoVideo:
    video_id: str
    uri: str
    name: str
    description: str | None
    duration_s: float | None
    width: int | None
    height: int | None
    created_time: str | None
    modified_time: str | None
    embed_link: str | None
    files: tuple[VimeoVideoFile, ...]
    download_links: tuple[str, ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata))


@dataclass(frozen=True, slots=True)
class GoogleDriveAccount:
    display_name: str | None
    email_address: str | None
    permission_id: str | None
    storage_quota_limit: int | None
    storage_quota_usage: int | None


@dataclass(frozen=True, slots=True)
class GoogleDriveFile:
    file_id: str
    name: str
    mime_type: str
    file_extension: str | None
    created_time: str | None
    modified_time: str | None
    size_bytes: int | None
    thumbnail_link: str | None
    web_view_link: str | None
    web_content_link: str | None
    can_download: bool | None
    is_google_workspace_document: bool
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata))


@dataclass(frozen=True, slots=True)
class DropboxAccount:
    account_id: str
    email: str | None
    display_name: str | None
    account_type: str | None


@dataclass(frozen=True, slots=True)
class DropboxEntry:
    entry_type: str
    entry_id: str
    name: str
    path_display: str | None
    path_lower: str | None
    server_modified: str | None
    size_bytes: int | None
    is_downloadable: bool | None
    content_hash: str | None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata))


@dataclass(frozen=True, slots=True)
class MetaAdAccount:
    id: str
    account_id: str | None
    name: str | None
    currency: str | None = None
    timezone_name: str | None = None


@dataclass(frozen=True, slots=True)
class MetaCampaign:
    campaign_id: str
    name: str
    status: str | None = None


@dataclass(frozen=True, slots=True)
class MetaCreativeAsset:
    asset_id: str
    asset_type: str
    url: str
    thumbnail_url: str | None = None
    mime_type: str | None = None
    title: str | None = None
    duration_s: float | None = None
    width: int | None = None
    height: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata))


@dataclass(frozen=True, slots=True)
class MetaCreative:
    creative_id: str
    name: str | None
    object_type: str | None
    title: str | None
    body: str | None
    image_url: str | None
    thumbnail_url: str | None
    video_id: str | None
    asset: MetaCreativeAsset | None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata))


@dataclass(frozen=True, slots=True)
class MetaAd:
    ad_id: str
    name: str | None
    effective_status: str | None
    creative: MetaCreative | None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata))


@dataclass(frozen=True, slots=True)
class SlackAccount:
    url: str | None
    team: str | None
    team_id: str | None
    user: str | None
    user_id: str | None
    bot_id: str | None = None


@dataclass(frozen=True, slots=True)
class SlackFile:
    file_id: str
    name: str
    title: str | None
    mimetype: str | None
    filetype: str | None
    pretty_type: str | None
    mode: str | None
    size_bytes: int | None
    created: int | None
    user_id: str | None
    is_public: bool | None
    permalink: str | None
    url_private: str | None
    url_private_download: str | None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata))
