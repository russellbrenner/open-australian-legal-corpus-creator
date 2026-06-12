import re
import json
import asyncio

from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor

import aiohttp
import lxml.html

from inscriptis.css_profiles import CSS_PROFILES
from inscriptis.html_properties import Display
from inscriptis.model.html_element import HtmlElement

from ..data import Entry, Request, Document, make_doc
from ..helpers import log, warning
from ..scraper import Scraper, ParseError
from ..custom_mammoth import docx2html
from ..custom_inscriptis import CustomInscriptis, CustomParserConfig


class VictorianLegislation(Scraper):
    """A scraper for the Victorian Legislation database (legislation.vic.gov.au).

    The Victorian Legislation register is a Nuxt 3 application backed by a Tide/Drupal
    content store. There is no public document API, so documents are enumerated from the
    site's XML sitemap and each instrument's authorised file URLs are resolved from the
    Nuxt-rendered landing page and a small companion JSON endpoint:

      1. The CONTENT host sitemap (``content.legislation.vic.gov.au/sitemap.xml?page=N``)
         lists every page on the register. In-force primary legislation lives under
         ``site-6/in-force/acts/<slug>`` and in-force secondary legislation (statutory
         rules) under ``site-6/in-force/statutory-rules/<slug>``. These are the only
         paths kept; ``as-made`` and repealed pages are excluded so that the corpus
         holds only the currently in-force consolidation of each instrument.

      2. The WWW host landing page (``www.legislation.vic.gov.au/in-force/acts/<slug>``)
         is server-side rendered by Nuxt. Its embedded ``__NUXT_DATA__`` state array
         carries the current in-force version's GUID, point-in-time date, version number
         and status, which are read directly out of the HTML (no JavaScript execution
         required).

      3. The companion endpoint ``www.legislation.vic.gov.au/api/tide/in-force?id=<GUID>``
         returns the authorised DOCX (``version``) and authorised PDF
         (``authorisedVersion``) file URLs on the content host. The DOCX is downloaded
         and converted to text (the PDF is a fallback for the rare instrument with no
         DOCX), mirroring the Western Australian scraper's DOCX-first strategy, which
         yields cleaner text than the register's own HTML rendering.

    LICENCE NOTE: Victorian legislation is Crown copyright administered by the Victorian
    Government Printer and is NOT released under an open licence. It may be reproduced for
    private study and research, but it is not freely redistributable. Documents collected
    by this scraper are for LOCAL corpus builds only and must not be published as part of
    a redistributed corpus without the Government Printer's permission.

    The register asks crawlers (via robots.txt) to observe a crawl-delay of 2 seconds, so
    concurrency is held low and a >=2s spacing is honoured between requests.
    """

    # The sitemap is paginated. In-force acts appear on page 2 and in-force statutory
    # rules on pages 6 and 7 at the time of writing, but pages shift as the register
    # grows, so every page is fetched and filtered by path rather than hard-coding pages.
    SITEMAP_PAGES = range(1, 8)

    CONTENT_HOST = 'https://content.legislation.vic.gov.au'
    WWW_HOST = 'https://www.legislation.vic.gov.au'

    # The minimum spacing, in seconds, between requests (the register's stated crawl-delay).
    CRAWL_DELAY = 2

    def __init__(self,
                 indices_refresh_interval: bool | timedelta = None,
                 index_refresh_interval: bool | timedelta = None,
                 semaphore: asyncio.Semaphore = None,
                 session: aiohttp.ClientSession = None,
                 thread_pool_executor: ThreadPoolExecutor = None,
                 ocr_semaphore: asyncio.Semaphore = None,
                 ) -> None:
        super().__init__(
            source='victorian_legislation',
            indices_refresh_interval=indices_refresh_interval,
            index_refresh_interval=index_refresh_interval,
            # Hold concurrency to one in-flight request to honour the register's
            # crawl-delay; the spacing is enforced in `get`.
            semaphore=semaphore or asyncio.Semaphore(1),
            session=session,
            thread_pool_executor=thread_pool_executor,
            ocr_semaphore=ocr_semaphore,
        )

        self._jurisdiction = 'victoria'

        # A monotonic clock reading of the earliest time the next request may be made,
        # used to enforce the crawl-delay across concurrent callers.
        self._next_request_at = 0.0

        # Create a custom Inscriptis CSS profile mirroring the Western Australian scraper
        # (DOCX is converted via the same `docx2html` pipeline): render `p` as a block
        # without surrounding newlines and keep a single newline before headings.
        inscriptis_profile = CSS_PROFILES['strict'].copy()
        inscriptis_profile['p'] = HtmlElement(display=Display.block)
        inscriptis_profile |= dict.fromkeys(('h1', 'h2', 'h3', 'h4', 'h5'), HtmlElement(display=Display.block, margin_before=1))
        self._inscriptis_config = CustomParserConfig(inscriptis_profile)

    @log
    async def get(self, req: Request | str) -> 'aiohttp.ClientResponse':
        # Enforce the crawl-delay: block until the register's minimum spacing has elapsed
        # since the previous request before delegating to the base implementation.
        loop = asyncio.get_event_loop()

        async with self.semaphore:
            now = loop.time()

            if now < self._next_request_at:
                await asyncio.sleep(self._next_request_at - now)

            self._next_request_at = loop.time() + self.CRAWL_DELAY

        return await super().get(req)

    @log
    async def get_index_reqs(self) -> set[Request]:
        return {
            Request(f'{self.CONTENT_HOST}/sitemap.xml?page={page}')
            for page in self.SITEMAP_PAGES
        }

    @log
    async def get_index(self, req: Request) -> set[Entry]:
        # Retrieve a sitemap page.
        resp = (await self.get(req)).text

        entries = set()

        # Extract every URL in the sitemap and keep only in-force acts and statutory rules.
        for loc in re.findall(r'<loc>([^<]+)</loc>', resp):
            if '/site-6/in-force/acts/' in loc:
                type = 'primary_legislation'
                kind = 'acts'

            elif '/site-6/in-force/statutory-rules/' in loc:
                type = 'secondary_legislation'
                kind = 'statutory-rules'

            else:
                continue

            # The sitemap URL is on the content host; the human/Nuxt landing page (which
            # carries the version metadata) is the equivalent path on the www host.
            slug = loc.rsplit('/', 1)[-1]
            landing = f'{self.WWW_HOST}/in-force/{kind}/{slug}'

            entries.add(
                Entry(
                    request=Request(landing),
                    # The version id is finalised in `_get_doc` once the version number is
                    # known; for now the slug uniquely identifies the instrument.
                    version_id=f'{kind}/{slug}',
                    source=self.source,
                    type=type,
                    jurisdiction=self._jurisdiction,
                )
            )

        return entries

    def _parse_landing(self, html: str) -> dict | None:
        """Extract the current in-force version record from a landing page.

        The Nuxt SSR payload embeds a flat record of the form
        ``{...},"<guid>","<YYYY-MM-DD>","<version>","<status>","<url>"`` where the keys
        ``id``, ``date``, ``version``, ``title``, ``status`` and ``url`` precede the
        values in the serialised state array. Returns the GUID, date, version number and
        status, or `None` if no such record is present (eg, an instrument that is no
        longer in force and so carries no current version).
        """

        match = re.search(
            r'"id":\d+,"date":\d+,"version":\d+,"title":\d+,"status":\d+,"url":\d+\},'
            r'"([0-9a-f-]{36})","(\d{4}-\d{2}-\d{2})","([^"]+)","([^"]+)","([^"]+)"',
            html,
        )

        if not match:
            return None

        guid, date, version, status, url = match.groups()

        return {
            'guid': guid,
            'date': date,
            'version': version,
            'status': status,
            'url': url,
        }

    @log
    async def _get_doc(self, entry: Entry) -> Document | None:
        # Retrieve the Nuxt landing page.
        resp = await self.get(entry.request)

        if resp.status == 404:
            warning(f'Unable to retrieve {entry.request.path}. Error 404 (Not Found) encountered. Returning `None`.')
            return

        # Parse the current in-force version record from the embedded Nuxt state.
        record = self._parse_landing(resp.text)

        if record is None:
            warning(f'Unable to locate a current in-force version for {entry.request.path}. The instrument may no longer be in force. Returning `None`.')
            return

        # Only keep instruments whose current version is in force.
        if 'in force' not in record['status'].lower():
            warning(f"The current version of {entry.request.path} is '{record['status']}', not in force. Returning `None`.")
            return

        # Resolve the authorised file URLs from the companion Tide endpoint.
        tide = (await self.get(f'{self.WWW_HOST}/api/tide/in-force?id={record["guid"]}')).json

        # Prefer the authorised DOCX (cleaner text); fall back to the authorised PDF.
        docx_url = next((f['url'] for f in tide.get('version', []) if f.get('extension') == 'docx'), None)
        pdf_url = next((f['url'] for f in tide.get('authorisedVersion', []) if f.get('extension') == 'pdf'), None)

        if docx_url:
            # Download and convert the DOCX, mirroring the Western Australian scraper.
            stream = (await self.get(Request(docx_url))).stream
            html = docx2html(stream)
            etree = lxml.html.fromstring(html.value)
            text = CustomInscriptis(etree, self._inscriptis_config).get_text()
            mime = 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
            url = docx_url

        elif pdf_url:
            from ..ocr import pdf2txt
            stream = (await self.get(Request(pdf_url))).stream
            text = await pdf2txt(stream, self.ocr_batch_size, self.thread_pool_executor, self.ocr_semaphore)
            mime = 'application/pdf'
            url = pdf_url

        else:
            raise ParseError(f'No authorised DOCX or PDF file found for {entry.request.path}.')

        # The title is the page's <title>, with the ' | legislation.vic.gov.au' suffix
        # stripped. `format_citation` appends the ' (Vic)' jurisdiction suffix.
        title = re.search(r'<title>([^<|]+)', resp.text).group(1).strip()

        # Finalise the version id as '<kind>/<slug>/<version>' so a new consolidation of
        # the same instrument is treated as a distinct document.
        version_id = f'{entry.version_id}/{record["version"]}'

        return make_doc(
            version_id=version_id,
            type=entry.type,
            jurisdiction=entry.jurisdiction,
            source=entry.source,
            mime=mime,
            date=record['date'],
            citation=title,
            url=url,
            text=text,
        )
