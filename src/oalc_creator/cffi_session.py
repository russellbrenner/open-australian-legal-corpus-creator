"""Shared Chrome-impersonating TLS session for sources that reject a plain
``aiohttp`` handshake (AustLII, and state registers fronted by a TLS-fingerprinting
WAF such as South Australia's).

Mixing :class:`CffiImpersonateMixin` into a :class:`~oalc_creator.scraper.Scraper`
overrides ``get`` to drive a ``curl_cffi`` session whose JA3/JA4 fingerprint matches
Chrome. A challenge or retryable status is treated as a transient condition and
backed off from with exponential backoff and jitter, mirroring the base ``Scraper.get``.
"""

import time
import random
import asyncio

from curl_cffi import CurlError
from curl_cffi.requests import AsyncSession

from .data import Request, Response
from .scraper import ParseError


class CffiImpersonateMixin:
    """Drive requests through a lazily-created Chrome-impersonating ``curl_cffi`` session."""

    #: The browser profile to impersonate.
    IMPERSONATE = 'chrome'

    #: Minimum seconds between requests (0 = unthrottled). Set per-source to honour
    #: a register's crawl-delay / avoid 429s under impersonation.
    CRAWL_DELAY = 0.0

    async def get(self, req: Request | str) -> Response:
        if isinstance(req, str):
            req = Request(req)

        # Lazily create the ``curl_cffi`` session inside the running event loop.
        if getattr(self, '_cffi_session', None) is None:
            self._cffi_session = AsyncSession(impersonate=self.IMPERSONATE)

        # Lazily create the crawl-delay gate.
        if self.CRAWL_DELAY and getattr(self, '_delay_lock', None) is None:
            self._delay_lock = asyncio.Lock()
            self._last_request = 0.0

        attempt = 0
        elapsed = 0

        while True:
            try:
                if self.CRAWL_DELAY:
                    async with self._delay_lock:
                        gap = self.CRAWL_DELAY - (time.monotonic() - self._last_request)
                        if gap > 0:
                            await asyncio.sleep(gap)
                        self._last_request = time.monotonic()
                async with self.semaphore:
                    response = await self._cffi_session.request(
                        req.method.upper(),
                        req.path,
                        data=dict(req.data) or None,
                        headers=dict(req.headers) or None,
                        timeout=60,
                    )

                # A Cloudflare/WAF challenge is treated as a retryable condition.
                challenged = (
                    response.headers.get('cf-mitigated') == 'challenge'
                    or (response.status_code in (403, 503) and b'Just a moment' in response.content)
                )

                if challenged or response.status_code in self.retry_statuses:
                    raise ParseError(f'Received a challenge or retryable status ({response.status_code}) when requesting {req.path}.')

                return Response(
                    response.content,
                    encoding=req.encoding,
                    type=(response.headers.get('content-type') or 'text/html').split(';')[0],
                    status=response.status_code,
                )

            except (ParseError, CurlError, asyncio.TimeoutError, OSError) as e:
                if elapsed > self.stop_after_waiting:
                    raise e

                attempt += 1

                wait = self.wait_base ** attempt / 2
                wait += random.uniform(0, wait)
                wait = min(wait, self.max_wait)
                wait += random.uniform(0, self.max_extra_jitter)

                await asyncio.sleep(wait)
                elapsed += wait

    async def _get_text(self, req: Request | str) -> str:
        """Retrieve content and decode it tolerantly (stray non-UTF-8 bytes would
        otherwise abort a whole crawl over a single defective document)."""

        resp = await self.get(req)

        return bytes(resp).decode(resp.encoding or 'utf-8', errors='replace')
