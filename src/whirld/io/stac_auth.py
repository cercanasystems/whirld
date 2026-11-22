"""Per-read STAC credential policy and scoped GDAL resource configuration."""

from __future__ import annotations

import os
import re
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any
from urllib.parse import urlparse

from ..errors import InvalidInputError


def http_origin(url: str) -> tuple[str, str, int] | None:
    """Normalize an HTTP origin; reject credentials/invalid ports in HTTP URLs."""
    try:
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            return None
        port = parsed.port
    except ValueError as exc:
        raise InvalidInputError("Invalid host or port in STAC URL.") from exc
    if port is None:
        port = 443 if parsed.scheme == "https" else 80
    if not parsed.hostname or parsed.username or parsed.password:
        raise InvalidInputError(
            "STAC HTTP URLs require a host and no user information."
        )
    return parsed.scheme, parsed.hostname.lower(), port


class StacAccessPolicy:
    """Own a single read's bearer token and explicitly authorized asset origins.

    Args:
        item_url: Original item location; its HTTP origin is authorized by default.
        token: Optional bearer credential, kept private and out of object reprs.
        token_origins: Additional direct asset origins explicitly authorized.

    Raises:
        InvalidInputError: Credentials or authorized-origin declarations are invalid.
    """

    def __init__(
        self,
        item_url: str,
        token: str | None = None,
        token_origins: tuple[str, ...] = (),
    ) -> None:
        if token and not re.fullmatch(r"[A-Za-z0-9._~+/-]+=*", token):
            raise InvalidInputError("STAC token must be a valid bearer token.")
        authorized = {http_origin(item_url)}
        for origin in token_origins:
            normalized = http_origin(origin)
            parsed = urlparse(origin)
            if (
                normalized is None
                or parsed.path not in ("", "/")
                or parsed.query
                or parsed.fragment
            ):
                raise InvalidInputError(
                    "token_origins must contain HTTP(S) origins only."
                )
            authorized.add(normalized)
        authorized.discard(None)
        self._authorized = frozenset(authorized)
        self._token = token

    def token_for(self, href: str) -> str | None:
        """Return credentials only for an explicitly authorized direct origin."""
        return self._token if http_origin(href) in self._authorized else None

    @contextmanager
    def asset_environment(self, href: str) -> Iterator[None]:
        """Scope GDAL credentials to an asset read, stripping them on redirects.

        Args:
            href: Direct asset URL whose origin determines credential eligibility.

        Yields:
            A GDAL environment restored on normal exit or read failure.

        Raises:
            InvalidInputError: Authenticated reads lack GDAL redirect protection.
        """
        import rasterio

        options: dict[str, Any] = {
            "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
            "VSI_CACHE": True,
            "GDAL_HTTP_HEADERS": "",
            # Empty string makes GDAL try opening an empty filename on every
            # HTTP read. A readable empty source clears inherited file headers
            # without warnings, and the Env restores the caller's path afterward.
            "GDAL_HTTP_HEADER_FILE": os.devnull,
            "CPL_VSIL_CURL_AUTHORIZATION_HEADER_ALLOWED_IF_REDIRECT": "NO",
        }
        token = self.token_for(href)
        if token:
            version = tuple(
                int(part) for part in rasterio.__gdal_version__.split(".")[:2]
            )
            if version < (3, 10):
                raise InvalidInputError(
                    "Authenticated STAC assets require GDAL >=3.10."
                )
            options["GDAL_HTTP_HEADERS"] = f"Authorization: Bearer {token}"
        with rasterio.Env(**options):
            yield
