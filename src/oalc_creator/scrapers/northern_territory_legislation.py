import re
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


class NorthernTerritoryLegislation(Scraper):
    """A scraper for the Northern Territory Legislation database (legislation.nt.gov.au).

    The NT register is a Sitecore site. Documents are enumerated and retrieved as follows:

      1. The by-title indexes (``/en/LegislationPortal/Acts/By-Title`` and
         ``/en/LegislationPortal/Subordinate-Legislation/By-Title``) list every
         instrument as a link to its record page (``/en/Legislation/<SLUG>``).

      2. Each record page carries a "View History Listing" link to an "As At Reprint
         History" page (``/Pages/Act History?itemId=<GUID>``). That history page is a
         table whose rows are the successive reprints of the instrument, each with a
         start date, an end date, a status, and Word/PDF download links of the form
         ``/api/sitecore/Act/Word_History?id=<N>`` (and ``.../PDF_History?id=<N>``).
         Both Acts and subordinate legislation use the ``Act/`` API path.

      3. The reprint covering today is selected (its start date is on or before today and
         its end date is on or after today or blank). Where no row strictly covers today
         (the register sometimes leaves the latest reprint's end date as the date it was
         superseded rather than blank), the most recent reprint is used. That reprint's
         DOCX is downloaded via ``Word_History`` (cleaner text), with ``PDF_History`` as a
         fallback, and converted through the shared ``docx2html`` pipeline.

    LICENCE NOTE: Northern Territory legislation is Crown copyright in right of the
    Northern Territory and is NOT released under any open licence. Documents collected by
    this scraper are for LOCAL corpus builds only and must not be redistributed.

    The register is crawled gently: a single in-flight request with ~1s spacing for index
    pages and ~2.5s spacing for document downloads.
    """

    BASE = 'https://legislation.nt.gov.au'

    BY_TITLE = {
        'primary_legislation': '/en/LegislationPortal/Acts/By-Title',
        'secondary_legislation': '/en/LegislationPortal/Subordinate-Legislation/By-Title',
    }

    # Request spacing, in seconds: a light delay for index pages and a heavier one for
    # the larger document downloads.
    INDEX_DELAY = 1.0
    DOC_DELAY = 2.5

    def __init__(self,
                 indices_refresh_interval: bool | timedelta = None,
                 index_refresh_interval: bool | timedelta = None,
                 semaphore: asyncio.Semaphore = None,
                 session: aiohttp.ClientSession = None,
                 thread_pool_executor: ThreadPoolExecutor = None,
                 ocr_semaphore: asyncio.Semaphore = None,
                 ) -> None:
        super().__init__(
            source='northern_territory_legislation',
            indices_refresh_interval=indices_refresh_interval,
            index_refresh_interval=index_refresh_interval,
            semaphore=semaphore or asyncio.Semaphore(1),
            session=session,
            thread_pool_executor=thread_pool_executor,
            ocr_semaphore=ocr_semaphore,
        )

        self._jurisdiction = 'northern_territory'

        # A monotonic clock reading of the earliest time the next request may be made.
        self._next_request_at = 0.0

        # Create a custom Inscriptis CSS profile mirroring the Western Australian scraper.
        inscriptis_profile = CSS_PROFILES['strict'].copy()
        inscriptis_profile['p'] = HtmlElement(display=Display.block)
        inscriptis_profile |= dict.fromkeys(('h1', 'h2', 'h3', 'h4', 'h5'), HtmlElement(display=Display.block, margin_before=1))
        self._inscriptis_config = CustomParserConfig(inscriptis_profile)

    async def _throttle(self, delay: float) -> None:
        """Block until at least `delay` seconds have elapsed since the previous request."""

        loop = asyncio.get_event_loop()

        async with self.semaphore:
            now = loop.time()

            if now < self._next_request_at:
                await asyncio.sleep(self._next_request_at - now)

            self._next_request_at = loop.time() + delay

    @log
    async def get(self, req: Request | str) -> 'aiohttp.ClientResponse':
        # Document downloads (the API endpoints) are spaced more heavily than index pages.
        path = req if isinstance(req, str) else req.path
        delay = self.DOC_DELAY if '/api/sitecore/' in path else self.INDEX_DELAY

        await self._throttle(delay)

        return await super().get(req)

    @log
    async def get_index_reqs(self) -> set[Request]:
        return {Request(f'{self.BASE}{path}') for path in self.BY_TITLE.values()}

    @log
    async def get_index(self, req: Request) -> set[Entry]:
        # Map the by-title path back to its document type.
        type = next(t for t, path in self.BY_TITLE.items() if path in req.path)

        resp = (await self.get(req)).text

        entries = set()

        # Each instrument links to its record page, eg `/en/Legislation/ABORIGINAL-LAND-ACT-1978`.
        for slug, title in re.findall(r'<a[^>]+href="/en/Legislation/([^"/?#]+)"[^>]*>([^<]+)</a>', resp):
            entries.add(
                Entry(
                    request=Request(f'{self.BASE}/en/Legislation/{slug}'),
                    version_id=slug, # The slug uniquely identifies the instrument.
                    source=self.source,
                    type=type,
                    jurisdiction=self._jurisdiction,
                    title=' '.join(title.split()),
                )
            )

        return entries

    def _select_reprint(self, history_html: str) -> dict | None:
        """Select the reprint covering today from an "As At Reprint History" table.

        Returns a dict with the chosen reprint's ``word_id`` (the numeric ``Word_History``
        id), ``pdf_id``, ``start`` date and ``end`` date (both `YYYY-MM-DD` or `None`), or
        `None` if the table contains no downloadable reprint.
        """

        etree = lxml.html.fromstring(history_html)

        today = datetime.now().strftime('%Y-%m-%d')

        rows = []

        for tr in etree.xpath('//table//tr'):
            cells = tr.xpath('./td')

            if len(cells) < 6:
                continue

            # Columns: Amendment History | Start Date | End Date | Type | Status | Download.
            start = self._parse_date(cells[1].text_content())
            end = self._parse_date(cells[2].text_content())

            # The download cell holds the Word and PDF history links.
            links = ' '.join(tr.xpath('.//a/@href'))

            word = re.search(r'Word_History\?id=(\d+)', links)
            pdf = re.search(r'PDF_History\?id=(\d+)', links)

            if not (word or pdf):
                continue

            rows.append({
                'word_id': word.group(1) if word else None,
                'pdf_id': pdf.group(1) if pdf else None,
                'start': start,
                'end': end,
            })

        if not rows:
            return None

        # Prefer the reprint whose date range covers today.
        for row in rows:
            if row['start'] and row['start'] <= today and (row['end'] is None or row['end'] >= today):
                return row

        # Otherwise fall back to the most recent reprint (the greatest start date). The
        # table is published newest-first, but sort defensively rather than relying on it.
        return max(rows, key=lambda r: r['start'] or '')

    @staticmethod
    def _parse_date(text: str) -> str | None:
        """Parse a 'DD/MM/YYYY' cell into 'YYYY-MM-DD', or `None` if absent."""

        match = re.search(r'(\d{2})/(\d{2})/(\d{4})', text)

        if not match:
            return None

        day, month, year = match.groups()

        return f'{year}-{month}-{day}'

    @log
    async def _get_doc(self, entry: Entry) -> Document | None:
        # Retrieve the record page.
        resp = await self.get(entry.request)

        if resp.status == 404:
            warning(f'Unable to retrieve {entry.request.path}. Error 404 (Not Found) encountered. Returning `None`.')
            return

        # Locate the "View History Listing" link to the reprint-history page.
        match = re.search(r'href="(/Pages/Act[%20 ]History\?itemId=[0-9a-f-]+)"', resp.text)

        if not match:
            warning(f'No history listing found for {entry.request.path}. The instrument may have no reprints. Returning `None`.')
            return

        history_path = match.group(1).replace(' ', '%20')

        # Retrieve the reprint-history page.
        history_html = (await self.get(f'{self.BASE}{history_path}')).text

        # Select the reprint covering today.
        reprint = self._select_reprint(history_html)

        if reprint is None:
            warning(f'No downloadable reprint found for {entry.request.path}. Returning `None`.')
            return

        # Download the reprint, preferring the Word (DOCX) version. Both Acts and
        # subordinate legislation use the `Act/` API path.
        if reprint['word_id']:
            url = f'{self.BASE}/api/sitecore/Act/Word_History?id={reprint["word_id"]}'
            stream = (await self.get(Request(url))).stream
            doc_html = docx2html(stream)
            etree = lxml.html.fromstring(doc_html.value)
            text = CustomInscriptis(etree, self._inscriptis_config).get_text()
            mime = 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'

        elif reprint['pdf_id']:
            from ..ocr import pdf2txt
            url = f'{self.BASE}/api/sitecore/Act/PDF_History?id={reprint["pdf_id"]}'
            stream = (await self.get(Request(url))).stream
            text = await pdf2txt(stream, self.ocr_batch_size, self.thread_pool_executor, self.ocr_semaphore)
            mime = 'application/pdf'

        else:
            raise ParseError(f'No Word or PDF reprint id for {entry.request.path}.')

        # Incorporate the reprint's start date into the version id so a new reprint of the
        # same instrument is treated as a distinct document.
        date = reprint['start']
        version_id = f'{entry.version_id}/{date}' if date else entry.version_id

        return make_doc(
            version_id=version_id,
            type=entry.type,
            jurisdiction=entry.jurisdiction,
            source=entry.source,
            mime=mime,
            date=date,
            citation=entry.title,
            url=url,
            text=text,
        )
