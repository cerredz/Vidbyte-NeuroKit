from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping
from urllib import parse, request
from urllib.error import HTTPError, URLError


class ProviderHttpError(RuntimeError):
    pass


class ProviderHttpClient:
    def __init__(self, timeout_seconds: float = 60.0) -> None:
        self.timeout_seconds = timeout_seconds

    def get_json(self, url: str, *, headers: Mapping[str, str] | None = None, params: Mapping[str, Any] | None = None) -> Any:
        response = self._request("GET", url, headers=headers, params=params)
        return json.loads(response.decode("utf-8"))

    def post_json(self, url: str, *, headers: Mapping[str, str] | None = None, params: Mapping[str, Any] | None = None, body: Mapping[str, Any] | None = None) -> Any:
        response = self._request("POST", url, headers=headers, params=params, body=body)
        return json.loads(response.decode("utf-8"))

    def request_bytes(
        self,
        method: str,
        url: str,
        *,
        headers: Mapping[str, str] | None = None,
        params: Mapping[str, Any] | None = None,
        body: Mapping[str, Any] | None = None,
        data: bytes | None = None,
    ) -> bytes:
        return self._request(method, url, headers=headers, params=params, body=body, data=data)

    def download(self, url: str, destination: str | Path, *, headers: Mapping[str, str] | None = None, params: Mapping[str, Any] | None = None) -> Path:
        resolved_destination = Path(destination).expanduser().resolve()
        resolved_destination.parent.mkdir(parents=True, exist_ok=True)
        payload = self._request("GET", url, headers=headers, params=params)
        resolved_destination.write_bytes(payload)
        return resolved_destination

    def _request(
        self,
        method: str,
        url: str,
        *,
        headers: Mapping[str, str] | None = None,
        params: Mapping[str, Any] | None = None,
        body: Mapping[str, Any] | None = None,
        data: bytes | None = None,
    ) -> bytes:
        request_url = self._build_url(url, params)
        request_headers = {"User-Agent": "tribe-provider-client/0.1", **dict(headers or {})}
        request_body = data
        if body is not None and request_body is not None:
            raise ValueError("Pass either body or data, not both.")
        if body is not None:
            request_body = json.dumps(body).encode("utf-8")
            request_headers.setdefault("Content-Type", "application/json")

        http_request = request.Request(request_url, data=request_body, headers=request_headers, method=method.upper())
        try:
            with request.urlopen(http_request, timeout=self.timeout_seconds) as response:
                return response.read()
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise ProviderHttpError(f"{method.upper()} {request_url} failed with HTTP {exc.code}: {detail}") from exc
        except URLError as exc:
            raise ProviderHttpError(f"{method.upper()} {request_url} failed: {exc.reason}") from exc

    @staticmethod
    def _build_url(url: str, params: Mapping[str, Any] | None) -> str:
        if not params:
            return url
        parsed = parse.urlsplit(url)
        current = parse.parse_qsl(parsed.query, keep_blank_values=True)
        for key, value in params.items():
            if value is None:
                continue
            if isinstance(value, (list, tuple)):
                for item in value:
                    current.append((key, str(item)))
            else:
                current.append((key, str(value)))
        return parse.urlunsplit((parsed.scheme, parsed.netloc, parsed.path, parse.urlencode(current), parsed.fragment))
