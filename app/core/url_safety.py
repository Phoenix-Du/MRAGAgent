from __future__ import annotations

import ipaddress
from urllib.parse import urlparse


def is_blocked_hostname(hostname: str | None) -> bool:
    if not hostname:
        return True
    host = hostname.strip().rstrip(".").lower()
    if host in {"localhost"} or host.endswith(".localhost"):
        return True
    try:
        addr = ipaddress.ip_address(host)
    except ValueError:
        return False
    return (
        addr.is_private
        or addr.is_loopback
        or addr.is_link_local
        or addr.is_multicast
        or addr.is_reserved
        or addr.is_unspecified
    )


def is_safe_public_http_url(raw_url: str | None) -> bool:
    if not raw_url:
        return False
    parsed = urlparse(raw_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return False
    return not is_blocked_hostname(parsed.hostname)
