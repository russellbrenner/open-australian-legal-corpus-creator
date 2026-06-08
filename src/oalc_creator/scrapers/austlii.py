import re
import html
import random
import asyncio

from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor

import aiohttp
import lxml.html

from curl_cffi import CurlError
from curl_cffi.requests import AsyncSession
from inscriptis.css_profiles import CSS_PROFILES
from inscriptis.html_properties import Display
from inscriptis.model.html_element import HtmlElement

from ..data import Entry, Request, Response, Document, make_doc
from ..helpers import log
from ..scraper import Scraper, ParseError
from ..custom_inscriptis import CustomInscriptis, CustomParserConfig


class Austlii(Scraper):
    """A scraper for the family-law judgments of the Federal Circuit and Family Court of Australia (and its predecessor courts) as hosted on AustLII.

    AustLII sits behind Cloudflare. The challenge it serves is satisfied purely by a browser-like TLS (JA3/JA4) fingerprint, which `curl_cffi`'s Chrome impersonation supplies; no JavaScript execution, `cf_clearance` cookie or CAPTCHA solving is required. A plain `aiohttp`/`requests` handshake is challenged regardless of the user agent, which is why this scraper overrides `get` to use its own `curl_cffi` session instead of the shared `aiohttp` one.

    Should Cloudflare ever escalate to a persistent interactive (Turnstile) challenge under crawl load, the seam to add a fallback is `get`: solve the challenge once in a real browser, capture `cf_clearance` and the matching user agent, and pass them to the `curl_cffi` session via `cookies=`/`headers=`. That is intentionally not implemented because it is not currently needed.
    """

    BASE = 'https://www.austlii.edu.au'

    # AustLII database codes for the family-law jurisdiction, mapped to the inclusive range of years over which each court published decisions. An end year of `None` denotes a court that is still sitting, in which case the current year is used.
    # NOTE The general-federal-law databases (eg, `FedCFamC2G`, `FMCA`) are deliberately excluded so that the corpus remains family-law (parenting and property) only.
    COURTS: dict[str, tuple[int, int | None]] = {
        # Division 1 (and its predecessor, the Family Court of Australia). Family-law only by database.
        'FamCA'      : (1976, 2021), # Family Court of Australia.
        'FamCAFC'    : (2008, 2021), # Family Court of Australia (Full Court).
        'FedCFamC1A' : (2021, None), # FCFCOA (Division 1) Appellate Jurisdiction (the Full Court).
        'FedCFamC1F' : (2021, None), # FCFCOA (Division 1) First Instance.

        # Division 2 (and its predecessors, the Federal Magistrates Court and Federal Circuit Court).
        'FMCAfam'    : (2000, 2013), # Federal Magistrates Court of Australia (family-law database).
        'FCCA'       : (2013, 2021), # Federal Circuit Court of Australia. NOTE A mixed family/general-federal database with no family-only split, so it is filtered to family matters (see `FAMILY_FILTERED_COURTS`).
        'FedCFamC2F' : (2021, None), # FCFCOA (Division 2) Family Law.
    }

    # Courts whose database mixes family-law and general-federal-law decisions, and which must therefore be filtered down to family matters. (Every other court above is family-law only by database and needs no filtering.)
    FAMILY_FILTERED_COURTS: set[str] = {'FCCA'}

    def __init__(self,
                 indices_refresh_interval: bool | timedelta = None,
                 index_refresh_interval: bool | timedelta = None,
                 semaphore: asyncio.Semaphore = None,
                 session: aiohttp.ClientSession = None,
                 thread_pool_executor: ThreadPoolExecutor = None,
                 ocr_semaphore: asyncio.Semaphore = None,
                 min_year: int = None,
                 ) -> None:
        super().__init__(
            source='fcfcoa',
            indices_refresh_interval=indices_refresh_interval,
            index_refresh_interval=index_refresh_interval,
            # NOTE Keep concurrency deliberately low. Cloudflare rate-limits by request velocity even once the TLS-fingerprint challenge is satisfied, and AustLII is a free non-profit service. A low limit keeps us under the radar and is courteous.
            semaphore=semaphore or asyncio.Semaphore(2),
            session=session,
            thread_pool_executor=thread_pool_executor,
            ocr_semaphore=ocr_semaphore,
        )

        self._jurisdiction = 'commonwealth'
        self._type = 'decision'

        # An optional floor on the years to scrape. When set, each court's start year is clamped to this value, which is useful for bounding a run to recent decisions (eg, `min_year=2024` to scrape 2024 onwards). Courts whose final year predates `min_year` are skipped entirely.
        self._min_year = min_year

        # The `curl_cffi` session is created lazily inside the running event loop the first time `get` is called.
        self._cffi_session: AsyncSession | None = None

        # Create a custom Inscriptis CSS profile mirroring the NSW Caselaw scraper: render `p` elements as blocks without surrounding newlines and retain a single newline before (but not after) headings.
        inscriptis_profile = CSS_PROFILES['strict'].copy()
        inscriptis_profile['p'] = HtmlElement(display=Display.block)
        inscriptis_profile |= dict.fromkeys(('h1', 'h2', 'h3', 'h4', 'h5'), HtmlElement(display=Display.block, margin_before=1))
        self._inscriptis_config = CustomParserConfig(inscriptis_profile)

    @log
    async def get(self, req: Request | str) -> Response:
        """Retrieve content from AustLII via a Chrome-impersonating TLS session.

        A Cloudflare challenge (the TLS fingerprint being rejected, or an escalation to a JS/Turnstile challenge) is treated as a transient condition and backed off from with exponential backoff and jitter, exactly as the base `Scraper.get` treats retryable network errors.
        """

        # If the request is a string, convert it to a request object.
        if isinstance(req, str):
            req = Request(req)

        # Lazily create the `curl_cffi` session inside the event loop.
        if self._cffi_session is None:
            self._cffi_session = AsyncSession(impersonate='chrome')

        attempt = 0
        elapsed = 0

        while True:
            try:
                async with self.semaphore:
                    response = await self._cffi_session.request(
                        req.method.upper(),
                        req.path,
                        data=dict(req.data) or None,
                        headers=dict(req.headers) or None,
                        timeout=60,
                    )

                # Detect a Cloudflare challenge and treat it as a retryable condition.
                challenged = (
                    response.headers.get('cf-mitigated') == 'challenge'
                    or (response.status_code in (403, 503) and b'Just a moment' in response.content)
                )

                if challenged or response.status_code in self.retry_statuses:
                    raise ParseError(f'Received a Cloudflare challenge or retryable status ({response.status_code}) when requesting {req.path}.')

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

                # Implement exponential backoff with jitter (mirroring `Scraper.get`).
                wait = self.wait_base ** attempt / 2 # We divide by 2 so that `wait + jitter` is always <= `self.wait_base ** attempt`.
                wait += random.uniform(0, wait) # Add jitter between 0 and `wait`.
                wait = min(wait, self.max_wait) # Cap the wait time.
                wait += random.uniform(0, self.max_extra_jitter) # Add a little extra jitter to handle cases where `wait` has been capped.

                await asyncio.sleep(wait)
                elapsed += wait

    async def _get_text(self, req: Request | str) -> str:
        """Retrieve content and decode it as text tolerantly.

        AustLII serves pages as UTF-8 but some judgments contain stray non-UTF-8 bytes (eg, a lone Windows-1252 byte left in an otherwise UTF-8 document). A strict decode (as performed by `Response.text`) raises `UnicodeDecodeError` on these, which would abort an entire crawl over a single defective document, so we decode with `errors='replace'` and rely on `make_doc`'s `ftfy`-based text cleaning to tidy the result.
        """

        resp = await self.get(req)

        return bytes(resp).decode(resp.encoding or 'utf-8', errors='replace')

    @log
    async def get_index_reqs(self) -> set[Request]:
        # Generate a request for every court's every year of decisions. The base `Creator` removes from the Corpus any document of this source that does not reappear in the index, so the index must enumerate the court's entire back catalogue on every run, not merely recent years.
        current_year = datetime.now().year

        return {
            Request(f'{self.BASE}/cgi-bin/viewdb/au/cases/cth/{code}/{year}/')

            for code, (start, end) in self.COURTS.items()
            for year in range(max(start, self._min_year) if self._min_year else start, (end or current_year) + 1)
        }

    @log
    async def get_index(self, req: Request) -> set[Entry]:
        # Retrieve the year-listing page. A year with no decisions (eg, a court's establishment year before it began sitting) simply yields no matching links and therefore an empty set.
        resp = await self._get_text(req)

        # Determine whether this court's listing must be filtered to family matters.
        code_match = re.search(r'/au/cases/cth/(\w+)/', req.path)
        family_filter = bool(code_match) and code_match.group(1) in self.FAMILY_FILTERED_COURTS

        entries = set()

        # Each decision is a link to a `viewdoc` URL whose text is the case's citation (party names, medium-neutral citation and, usually, the delivery date in parentheses).
        for path, title in re.findall(r'href="(/cgi-bin/viewdoc/au/cases/cth/\w+/\d{4}/\d+\.html)"[^>]*>([^<]+)</a>', resp):
            # Normalise whitespace and unescape HTML entities in the title (AustLII renders the ampersand in anonymised party names as `&amp;`).
            title = ' '.join(html.unescape(title).split())

            # For mixed databases, keep only family-law matters. Anonymised family matters are styled "Surname & Surname"; general-federal matters (migration, bankruptcy, etc) are styled "Party v Respondent". Filtering on this connector selects family decisions from the listing alone, sparing us from fetching (and discarding) the larger volume of non-family judgments.
            if family_filter and not (' & ' in title and ' v ' not in f' {title} '):
                continue

            # Extract the delivery date from the trailing parenthesised date if present.
            date = None

            if match := re.search(r'\((\d{1,2} \w+ \d{4})\)$', title):
                try:
                    date = datetime.strptime(match.group(1), '%d %B %Y').strftime('%Y-%m-%d')

                except ValueError:
                    date = None

            entries.add(
                Entry(
                    request=Request(f'{self.BASE}{path}'),
                    version_id=path.removeprefix('/cgi-bin/viewdoc/'), # eg, `au/cases/cth/FamCA/2020/100.html`; unique across all courts and years.
                    source=self.source,
                    type=self._type,
                    jurisdiction=self._jurisdiction,
                    date=date,
                    title=title,
                )
            )

        return entries

    @log
    async def _get_doc(self, entry: Entry) -> Document | None:
        # Retrieve the document.
        resp = await self._get_text(entry.request)

        # Construct an etree from the response.
        etree = lxml.html.fromstring(resp)

        # Retrieve the `article` element containing the judgment if it exists, otherwise raise a `ParseError`. On AustLII the `article` element holds only the judgment (citation, court header and reasons); page navigation, citation context and feedback chrome all sit outside it.
        article = etree.xpath('//article')

        if not article:
            raise ParseError()

        article = article[0]

        # Extract the text of the judgment.
        text = CustomInscriptis(article, self._inscriptis_config).get_text()

        # On a minority of pages AustLII renders its print/download/citator/footer navigation inside the `article` element, as a trailing block of the form "Print / Print (pretty) / Print (eco-friendly) / Download / RTF format (N MB) / Signed PDF/A format / Cited By / LawCite records / NoteUp references / ...". Truncate at the earliest such marker. These phrases are site chrome that never occurs in judgment prose, so cutting here cannot remove substantive text.
        if chrome := re.search(r'Print \(pretty\)|Print \(eco-friendly\)|(?:RTF|Signed PDF/A) format|(?:Download|RTF format) \([\d.]+ ?[KMG]B\)|LawCite records|NoteUp references', text):
            # Cut from the start of the marker's line, then drop any dangling "Print"/"Download" footer header left on the preceding line.
            cut = text.rfind('\n', 0, chrome.start())
            text = text[:cut] if cut != -1 else text[:chrome.start()]
            text = re.sub(r'\n[ \t*]*(?:Print|Download)\s*$', '', text)

        # Skip AustLII placeholder pages for judgments whose text is withheld or not yet published (eg, "This judgment is currently unavailable"). These carry no substantive content.
        if 'judgment is currently unavailable' in text.lower():
            return None

        # Strip the trailing parenthesised delivery date from the citation (the date is preserved in the document's `date` field).
        citation = re.sub(r'\s*\(\d{1,2} \w+ \d{4}\)$', '', entry.title)

        # Create the document.
        return make_doc(
            version_id=entry.version_id,
            type=self._type,
            jurisdiction=self._jurisdiction,
            source=self.source,
            mime='text/html',
            date=entry.date,
            citation=citation,
            url=entry.request.path,
            text=text,
        )
