import re
import asyncio
import itertools

from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor

import aiohttp
import lxml.html

from inscriptis.css_profiles import CSS_PROFILES
from inscriptis.html_properties import Display
from inscriptis.model.html_element import HtmlElement

from ..data import Entry, Request, Document, make_doc
from ..helpers import log, warning, format_date
from ..scraper import Scraper, ParseError
from ..custom_mammoth import docx2html
from ..custom_inscriptis import CustomInscriptis, CustomParserConfig


class ActLegislation(Scraper):
    """A scraper for the ACT Legislation Register (legislation.act.gov.au).

    The register is an ASP.NET site fronted by an F5 BIG-IP ASM web application firewall.
    The firewall sets a ``TS...`` session cookie on the first response that must be sent
    back on subsequent requests, so this scraper drives a single ``aiohttp`` session with
    a cookie jar (the shared session supplied by the ``Creator`` already persists cookies;
    when run standalone the scraper creates its own cookie-bearing session).

    Documents are enumerated and retrieved as follows:

      1. The robots-ALLOWED notification index (``/Notifications?category=cAct&year=YYYY``
         for Acts and ``category=cSub&year=YYYY`` for subordinate laws) is iterated over
         every year from 1912 (the earliest ACT enactment) to the current year. Each page
         links to instrument landing pages identified as ``/a/<year>-<n>`` (Acts) and
         ``/sl/<year>-<n>`` (subordinate laws).

      2. Each instrument's landing page (``/a/<id>/``) is fetched. An instrument that is
         currently in force exposes a "current" consolidation download
         (``/DownloadFile/a/<id>/current/DOCX/<id>.DOCX``); a repealed or expired
         instrument does not. The presence of that link is therefore used both as the
         in-force filter and as the source of the download URL. The page also states the
         consolidation's currency date as "(current) DD Month YYYY" and the instrument's
         full title in its ``<title>`` element.

      3. The current consolidation is downloaded as DOCX (cleaner text) and converted via
         the shared ``docx2html`` pipeline, with PDF as a fallback. RTF is also published
         but DOCX is preferred.

    The disallowed paths ``/results``, ``/View/`` and ``/html/`` are never requested; only
    ``/Notifications``, instrument landing pages and ``/DownloadFile`` (all robots-allowed)
    are fetched.

    LICENCE NOTE: ACT legislation is licensed under Creative Commons Attribution 4.0
    International (CC BY 4.0) per the ACT Justice and Community Safety Directorate's
    copyright policy. It is REDISTRIBUTABLE with attribution to the Australian Capital
    Territory.

    The register's robots.txt specifies a STRICT crawl-delay of 10 seconds, which this
    scraper honours by holding concurrency to one request and spacing requests >=10s apart.
    """

    BASE = 'https://www.legislation.act.gov.au'

    # The inclusive range of years over which the notification index is searched. The
    # earliest ACT ordinance dates to 1912; the upper bound is the current year.
    START_YEAR = 1912

    # The register's stated crawl-delay, in seconds.
    CRAWL_DELAY = 10

    def __init__(self,
                 indices_refresh_interval: bool | timedelta = None,
                 index_refresh_interval: bool | timedelta = None,
                 semaphore: asyncio.Semaphore = None,
                 session: aiohttp.ClientSession = None,
                 thread_pool_executor: ThreadPoolExecutor = None,
                 ocr_semaphore: asyncio.Semaphore = None,
                 ) -> None:
        super().__init__(
            source='act_legislation',
            indices_refresh_interval=indices_refresh_interval,
            index_refresh_interval=index_refresh_interval,
            # A single in-flight request; the 10s crawl-delay is enforced in `get`.
            semaphore=semaphore or asyncio.Semaphore(1),
            session=session,
            thread_pool_executor=thread_pool_executor,
            ocr_semaphore=ocr_semaphore,
        )

        self._jurisdiction = 'australian_capital_territory'

        # A monotonic clock reading of the earliest time the next request may be made.
        self._next_request_at = 0.0

        # A scraper-owned cookie-bearing session, created lazily when no shared session
        # is supplied, so that the F5 ASM `TS` cookie persists across requests.
        self._own_session: aiohttp.ClientSession | None = None

        # Create a custom Inscriptis CSS profile mirroring the Western Australian scraper.
        inscriptis_profile = CSS_PROFILES['strict'].copy()
        inscriptis_profile['p'] = HtmlElement(display=Display.block)
        inscriptis_profile |= dict.fromkeys(('h1', 'h2', 'h3', 'h4', 'h5'), HtmlElement(display=Display.block, margin_before=1))
        self._inscriptis_config = CustomParserConfig(inscriptis_profile)

    @log
    async def get(self, req: Request | str) -> 'aiohttp.ClientResponse':
        # Ensure a cookie-persisting session exists so the F5 ASM `TS` cookie is retained.
        if self.session is None and self._own_session is None:
            self._own_session = aiohttp.ClientSession()
            self.session = self._own_session

        # Enforce the crawl-delay before delegating to the base implementation.
        loop = asyncio.get_event_loop()

        async with self.semaphore:
            now = loop.time()

            if now < self._next_request_at:
                await asyncio.sleep(self._next_request_at - now)

            self._next_request_at = loop.time() + self.CRAWL_DELAY

        return await super().get(req)

    @log
    async def get_index_reqs(self) -> set[Request]:
        current_year = datetime.now().year

        return {
            Request(f'{self.BASE}/Notifications?category={category}&year={year}')
            for category, year in itertools.product(
                ('cAct', 'cSub'),
                range(self.START_YEAR, current_year + 1),
            )
        }

    @log
    async def get_index(self, req: Request) -> set[Entry]:
        # Acts are notified under `cAct`, subordinate laws under `cSub`.
        type = 'primary_legislation' if 'category=cAct' in req.path else 'secondary_legislation'

        resp = (await self.get(req)).text

        entries = set()

        # Each notified instrument links to its landing page, eg `/a/2001-14` (Acts) or
        # `/sl/2006-29` (subordinate laws). The link text is the instrument's short title.
        for path, title in re.findall(r'<a[^>]+href="(/(?:a|sl)/\d{4}-\d+)/?(?:#[^"]*)?"[^>]*>([^<]+)</a>', resp):
            doc_id = path.rsplit('/', 1)[-1]

            entries.add(
                Entry(
                    request=Request(f'{self.BASE}{path}/'),
                    version_id=doc_id, # eg `2001-14`; unique across the register.
                    source=self.source,
                    type=type,
                    jurisdiction=self._jurisdiction,
                    title=' '.join(title.split()),
                )
            )

        return entries

    @log
    async def _get_doc(self, entry: Entry) -> Document | None:
        # Retrieve the instrument's landing page.
        resp = await self.get(entry.request)

        if resp.status == 404:
            warning(f'Unable to retrieve {entry.request.path}. Error 404 (Not Found) encountered. Returning `None`.')
            return

        html = resp.text

        # The instrument's id, eg `2001-14`, parsed back out of the entry's version id.
        doc_id = entry.version_id.split(':', 1)[-1]

        # An in-force instrument exposes a "current" consolidation download. Its absence
        # marks a repealed or expired instrument, which is excluded from the corpus.
        type_prefix = 'a' if entry.type == 'primary_legislation' else 'sl'

        docx_path = f'/DownloadFile/{type_prefix}/{doc_id}/current/DOCX/{doc_id}.DOCX'
        pdf_path = f'/DownloadFile/{type_prefix}/{doc_id}/current/PDF/{doc_id}.PDF'

        if docx_path not in html and pdf_path not in html:
            warning(f'No current consolidation found for {entry.request.path}; the instrument is not in force. Returning `None`.')
            return

        # The consolidation's currency date, stated as "(current) DD Month YYYY".
        date = None

        if match := re.search(r'\(current\)\s*(\d{1,2} [A-Z][a-z]+ \d{4})', html):
            date = format_date(match.group(1))

        # Prefer the full title from the landing page's <title> (eg "Legislation Act
        # 2001"), falling back to the notification index's short title.
        title = entry.title

        if match := re.search(r'<title>([^|<]+)', html):
            title = match.group(1).strip() or title

        # Download the current consolidation, preferring DOCX over PDF.
        if docx_path in html:
            stream = (await self.get(Request(f'{self.BASE}{docx_path}'))).stream
            doc_html = docx2html(stream)
            etree = lxml.html.fromstring(doc_html.value)
            text = CustomInscriptis(etree, self._inscriptis_config).get_text()
            mime = 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
            url = f'{self.BASE}{docx_path}'

        else:
            from ..ocr import pdf2txt
            stream = (await self.get(Request(f'{self.BASE}{pdf_path}'))).stream
            text = await pdf2txt(stream, self.ocr_batch_size, self.thread_pool_executor, self.ocr_semaphore)
            mime = 'application/pdf'
            url = f'{self.BASE}{pdf_path}'

        # Incorporate the currency date into the version id so a fresh consolidation of
        # the same instrument is treated as a distinct document. The id is re-prefixed
        # with the source name to match the format of ids stored elsewhere in the corpus.
        version_id = Entry.format_id(f'{doc_id}/{date}' if date else doc_id, self.source)

        return make_doc(
            version_id=version_id,
            type=entry.type,
            jurisdiction=entry.jurisdiction,
            source=entry.source,
            mime=mime,
            date=date,
            citation=title,
            url=url,
            text=text,
        )
