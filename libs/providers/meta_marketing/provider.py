from __future__ import annotations

import mimetypes
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from libs.dataclasses.provider_models import (
    DownloadedProviderAsset,
    MetaAd,
    MetaAdAccount,
    MetaCampaign,
    MetaCreative,
    MetaCreativeAsset,
)
from libs.enums import ProviderBaseUrl
from libs.providers.http_client import ProviderHttpClient

from .credentials import MetaMarketingCredentials


class MetaMarketingProvider:
    def __init__(
        self,
        credentials: MetaMarketingCredentials | tuple[str, str],
        *,
        http_client: ProviderHttpClient | None = None,
        download_dir: str | Path | None = None,
        api_version: str = "v25.0",
    ) -> None:
        if isinstance(credentials, MetaMarketingCredentials):
            self.credentials = credentials
        else:
            token, account_id = credentials
            self.credentials = MetaMarketingCredentials(token=token, account_id=account_id)
        self.http_client = http_client or ProviderHttpClient()
        self.download_dir = Path(download_dir or Path.cwd() / "cache" / "providers" / "meta-ads").expanduser().resolve()
        self.api_version = api_version

    def get_account(self) -> MetaAdAccount:
        payload = self._get(self.credentials.account_id, fields="id,account_id,name,currency,timezone_name")
        return MetaAdAccount(
            id=str(payload.get("id", "")),
            account_id=payload.get("account_id"),
            name=payload.get("name"),
            currency=payload.get("currency"),
            timezone_name=payload.get("timezone_name"),
        )

    def list_campaigns(self, *, limit: int = 100) -> tuple[MetaCampaign, ...]:
        payload = self._get(f"{self.credentials.account_id}/campaigns", fields="id,name,status", limit=limit)
        return tuple(
            MetaCampaign(campaign_id=str(item.get("id", "")), name=str(item.get("name", "")), status=item.get("status"))
            for item in payload.get("data", [])
            if isinstance(item, dict)
        )

    def get_creative(self, creative_id: str) -> MetaCreative:
        payload = self._get(
            creative_id,
            fields="id,name,object_type,title,body,image_url,thumbnail_url,video_id,object_story_spec,asset_feed_spec",
        )
        return self._parse_creative(payload)

    def list_ads_for_campaign(self, campaign_id: str, *, limit: int = 100) -> tuple[MetaAd, ...]:
        payload = self._get(
            f"{campaign_id}/ads",
            fields="id,name,effective_status,creative{id,name,object_type,title,body,image_url,thumbnail_url,video_id,object_story_spec,asset_feed_spec}",
            limit=limit,
        )
        items: list[MetaAd] = []
        for raw_item in payload.get("data", []):
            if not isinstance(raw_item, dict):
                continue
            raw_creative = raw_item.get("creative")
            creative = self._parse_creative(raw_creative) if isinstance(raw_creative, dict) else None
            items.append(
                MetaAd(
                    ad_id=str(raw_item.get("id", "")),
                    name=raw_item.get("name"),
                    effective_status=raw_item.get("effective_status"),
                    creative=creative,
                    metadata=raw_item,
                )
            )
        return tuple(items)

    def fetch_media(self, creative_id: str) -> DownloadedProviderAsset:
        creative = self.get_creative(creative_id)
        if creative.asset is None:
            raise ValueError(f"Meta creative {creative_id} does not expose a downloadable image or video asset.")
        suffix = self._infer_suffix(creative.asset.url, creative.asset.mime_type)
        destination = self.download_dir / f"{creative.creative_id}{suffix}"
        self.http_client.download(creative.asset.url, destination)
        return DownloadedProviderAsset(
            provider="meta-ads",
            remote_id=creative.creative_id,
            name=creative.name or creative.creative_id,
            local_path=destination,
            mime_type=creative.asset.mime_type,
            source_url=creative.asset.url,
            metadata={"creative": creative},
        )

    def fetch_campaign_media(self, campaign_id: str) -> tuple[DownloadedProviderAsset, ...]:
        items: list[DownloadedProviderAsset] = []
        for ad in self.list_ads_for_campaign(campaign_id):
            if ad.creative is None or ad.creative.asset is None:
                continue
            items.append(self.fetch_media(ad.creative.creative_id))
        return tuple(items)

    def _get(self, path: str, **params: Any) -> dict[str, Any]:
        return self.http_client.get_json(
            f"{ProviderBaseUrl.META_GRAPH_API.value}/{self.api_version}/{path}",
            headers=self._headers(),
            params=params,
        )

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.credentials.token}"}

    def _parse_creative(self, payload: dict[str, Any]) -> MetaCreative:
        video_id = payload.get("video_id")
        image_url = payload.get("image_url") or payload.get("thumbnail_url") or self._extract_image_url(payload.get("object_story_spec"))
        asset = None
        if isinstance(video_id, str) and video_id:
            video_payload = self._get(video_id, fields="id,title,description,source,length,created_time,updated_time")
            source = video_payload.get("source")
            if isinstance(source, str) and source:
                asset = MetaCreativeAsset(
                    asset_id=str(video_payload.get("id", video_id)),
                    asset_type="video",
                    url=source,
                    mime_type="video/mp4",
                    title=video_payload.get("title"),
                    duration_s=self._coerce_float(video_payload.get("length")),
                    metadata=video_payload,
                )
        elif isinstance(image_url, str) and image_url:
            asset = MetaCreativeAsset(
                asset_id=str(payload.get("id", "")),
                asset_type="image",
                url=image_url,
                thumbnail_url=payload.get("thumbnail_url"),
                mime_type=self._guess_mime(image_url),
                title=payload.get("title"),
                metadata={"image_url": image_url},
            )
        return MetaCreative(
            creative_id=str(payload.get("id", "")),
            name=payload.get("name"),
            object_type=payload.get("object_type"),
            title=payload.get("title"),
            body=payload.get("body"),
            image_url=payload.get("image_url"),
            thumbnail_url=payload.get("thumbnail_url"),
            video_id=payload.get("video_id"),
            asset=asset,
            metadata=payload,
        )

    @staticmethod
    def _extract_image_url(spec: Any) -> str | None:
        if not isinstance(spec, dict):
            return None
        for key in ("photo_data", "link_data", "video_data"):
            value = spec.get(key)
            if isinstance(value, dict):
                if isinstance(value.get("image_url"), str):
                    return value["image_url"]
                if isinstance(value.get("picture"), str):
                    return value["picture"]
        return None

    @staticmethod
    def _guess_mime(url: str) -> str | None:
        guessed, _ = mimetypes.guess_type(url)
        return guessed

    @staticmethod
    def _infer_suffix(url: str, mime_type: str | None) -> str:
        path_suffix = Path(urlsplit(url).path).suffix
        if path_suffix:
            return path_suffix
        guessed = mimetypes.guess_extension(mime_type or "")
        return guessed or ""

    @staticmethod
    def _coerce_float(value: Any) -> float | None:
        return None if value in (None, "") else float(value)
