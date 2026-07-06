import re
import html
import asyncio

from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor

import aiohttp
import lxml.html

from curl_cffi import CurlError
from inscriptis.css_profiles import CSS_PROFILES
from inscriptis.html_properties import Display
from inscriptis.model.html_element import HtmlElement

from ..data import Entry, Request, Document, make_doc
from ..helpers import log
from ..scraper import Scraper, ParseError
from ..cffi_session import CffiImpersonateMixin
from ..custom_inscriptis import CustomInscriptis, CustomParserConfig


class AustliiCases(CffiImpersonateMixin, Scraper):
    """Base scraper for a state/territory jurisdiction's court and tribunal decisions as hosted on AustLII.

    This generalises the family-law :class:`Austlii` scraper to the medium-neutral
    case databases of any jurisdiction. Concrete subclasses set ``SOURCE``,
    ``JURISDICTION`` (the corpus' native jurisdiction string), ``JUR_PATH`` (the
    AustLII path segment, eg ``vic``) and ``COURTS`` (the AustLII database codes to
    crawl, eg ``VSC``).

    AustLII sits behind Cloudflare. The challenge it serves is satisfied purely by a
    browser-like TLS (JA3/JA4) fingerprint, which ``curl_cffi``'s Chrome impersonation
    supplies; no JavaScript execution, ``cf_clearance`` cookie or CAPTCHA solving is
    required. A plain ``aiohttp`` handshake is challenged regardless of the user agent,
    which is why this scraper overrides ``get`` to use its own ``curl_cffi`` session.

    LICENCE NOTE: AustLII content is made available for research under AustLII's own
    terms and is NOT released under an open licence. Rows from this source are flagged
    non-redistributable and excluded from published modules by default.
    """

    BASE = 'https://www.austlii.edu.au'

    # Subclass configuration.
    SOURCE: str = None
    JURISDICTION: str = None
    JUR_PATH: str = None
    COURTS: tuple[str, ...] = ()

    # When a court's database page advertises years separated by a gap larger than this,
    # the contiguous-from-current block is treated as the court's coverage and earlier
    # isolated year tokens (typically cited-case years in listing titles) are ignored.
    MAX_YEAR_GAP = 3

    # An absolute floor below which no year is ever enumerated, as a courtesy guard
    # against a malformed page producing an absurd range.
    ABSOLUTE_FLOOR = 1900

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
            source=self.SOURCE,
            indices_refresh_interval=indices_refresh_interval,
            index_refresh_interval=index_refresh_interval,
            # NOTE Keep concurrency deliberately low. Cloudflare rate-limits by request
            # velocity even once the TLS-fingerprint challenge is satisfied, and AustLII
            # is a free non-profit service. A low limit keeps us courteous.
            semaphore=semaphore or asyncio.Semaphore(2),
            session=session,
            thread_pool_executor=thread_pool_executor,
            ocr_semaphore=ocr_semaphore,
        )

        self._jurisdiction = self.JURISDICTION
        self._type = 'decision'

        # An optional floor on the years to scrape, clamping each court's discovered
        # start year (eg, ``min_year=2024`` to bound a run to recent decisions).
        self._min_year = min_year

        # Mirror the family-law scraper's Inscriptis profile: render ``p`` as blocks
        # without surrounding newlines, single newline before (not after) headings.
        inscriptis_profile = CSS_PROFILES['strict'].copy()
        inscriptis_profile['p'] = HtmlElement(display=Display.block)
        inscriptis_profile |= dict.fromkeys(('h1', 'h2', 'h3', 'h4', 'h5'), HtmlElement(display=Display.block, margin_before=1))
        self._inscriptis_config = CustomParserConfig(inscriptis_profile)

    def _coverage_floor(self, years: set[int], current_year: int) -> int | None:
        """Determine a court's earliest coverage year from the years advertised on its
        database page, walking back from the current year while gaps stay within
        ``MAX_YEAR_GAP`` so that isolated cited-case years are excluded."""

        present = {y for y in years if self.ABSOLUTE_FLOOR <= y <= current_year}

        if not present:
            return None

        # Anchor at the most recent advertised year (a court may have published nothing
        # in the calendar current year yet).
        floor = anchor = max(present)

        while True:
            nxt = next((y for y in range(floor - 1, anchor - 100, -1) if y in present), None)

            if nxt is None or (floor - nxt) > self.MAX_YEAR_GAP:
                break

            floor = nxt

        return floor

    @log
    async def get_index_reqs(self) -> set[Request]:
        """Enumerate one year-listing request per court per year of coverage.

        Each court's coverage span is discovered from its database landing page (the
        years it advertises) rather than hard-coded, so the back catalogue is captured
        in full without a brittle per-court year table. The base ``Creator`` removes any
        document of this source that does not reappear in the index, so the index must
        enumerate the entire back catalogue on every run.
        """

        current_year = datetime.now().year
        reqs: set[Request] = set()

        for code in self.COURTS:
            try:
                landing = await self._get_text(f'{self.BASE}/cgi-bin/viewdb/au/cases/{self.JUR_PATH}/{code}/')
            except (ParseError, CurlError, asyncio.TimeoutError, OSError):
                # A court whose landing page is unreachable is skipped; the next run retries.
                continue

            years = {int(y) for y in re.findall(r'\b(1[89]\d\d|20\d\d)\b', landing)}
            floor = self._coverage_floor(years, current_year)

            if floor is None:
                continue

            if self._min_year:
                floor = max(floor, self._min_year)

            for year in range(floor, current_year + 1):
                reqs.add(Request(f'{self.BASE}/cgi-bin/viewdb/au/cases/{self.JUR_PATH}/{code}/{year}/'))

        return reqs

    @log
    async def get_index(self, req: Request) -> set[Entry]:
        # The court code and year are encoded in the index URL.
        m = re.search(r'/au/cases/(\w+)/(\w+)/(\d{4})/', req.path)

        if not m:
            return set()

        jur_path, code, year = m.group(1), m.group(2), m.group(3)

        # A year with no decisions (eg, a court's establishment year before it began
        # sitting) simply yields no matching links and therefore an empty set.
        resp = await self._get_text(req)

        entries = set()

        # Each decision is a link to a ``viewdoc`` URL whose text is the case's citation
        # (party names, medium-neutral citation and, usually, the delivery date).
        for path, title in re.findall(rf'href="(/cgi-bin/viewdoc/au/cases/{jur_path}/{code}/{year}/\d+\.html)"[^>]*>([^<]+)</a>', resp):
            title = ' '.join(html.unescape(title).split())

            date = None

            if match := re.search(r'\((\d{1,2} \w+ \d{4})\)$', title):
                try:
                    date = datetime.strptime(match.group(1), '%d %B %Y').strftime('%Y-%m-%d')
                except ValueError:
                    date = None

            entries.add(
                Entry(
                    request=Request(f'{self.BASE}{path}'),
                    version_id=path.removeprefix('/cgi-bin/viewdoc/'),  # eg, ``au/cases/vic/VSC/2020/1.html``; unique across courts and years.
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
        resp = await self._get_text(entry.request)

        etree = lxml.html.fromstring(resp)

        # On AustLII the ``article`` element holds only the judgment (citation, court
        # header and reasons); navigation, citation context and chrome sit outside it.
        article = etree.xpath('//article')

        if not article:
            raise ParseError()

        article = article[0]

        text = CustomInscriptis(article, self._inscriptis_config).get_text()

        # Truncate the trailing print/download/citator footer chrome if AustLII rendered
        # it inside the ``article``. These phrases never occur in judgment prose.
        if chrome := re.search(r'Print \(pretty\)|Print \(eco-friendly\)|(?:RTF|Signed PDF/A) format|(?:Download|RTF format) \([\d.]+ ?[KMG]B\)|LawCite records|NoteUp references', text):
            cut = text.rfind('\n', 0, chrome.start())
            text = text[:cut] if cut != -1 else text[:chrome.start()]
            text = re.sub(r'\n[ \t*]*(?:Print|Download)\s*$', '', text)

        # Skip placeholder pages for judgments whose text is withheld or unpublished.
        if 'judgment is currently unavailable' in text.lower():
            return None

        # Strip the trailing parenthesised delivery date from the citation (preserved in ``date``).
        citation = re.sub(r'\s*\(\d{1,2} \w+ \d{4}\)$', '', entry.title)

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


class VictorianCourts(AustliiCases):
    """Victorian court and tribunal decisions on AustLII."""

    SOURCE = 'victorian_courts'
    JURISDICTION = 'victoria'
    JUR_PATH = 'vic'
    COURTS = ('VSC', 'VSCA', 'VCC', 'VMC', 'VCAT', 'VicCorC')


class QueenslandCourts(AustliiCases):
    """Queensland court and tribunal decisions on AustLII."""

    SOURCE = 'queensland_courts'
    JURISDICTION = 'queensland'
    JUR_PATH = 'qld'
    COURTS = ('QSC', 'QCA', 'QDC', 'QChC', 'QCAT', 'QCATA', 'QIRC', 'QLC', 'QLAC', 'QMHC', 'QPEC')


class SouthAustralianCourts(AustliiCases):
    """South Australian court and tribunal decisions on AustLII."""

    SOURCE = 'south_australian_courts'
    JURISDICTION = 'south_australia'
    JUR_PATH = 'sa'
    COURTS = ('SASC', 'SASCFC', 'SASCA', 'SADC', 'SAERDC', 'SACAT', 'SAIRC', 'SAWC')


class WesternAustralianCourts(AustliiCases):
    """Western Australian court and tribunal decisions on AustLII."""

    SOURCE = 'western_australian_courts'
    JURISDICTION = 'western_australia'
    JUR_PATH = 'wa'
    COURTS = ('WASC', 'WASCA', 'WADC', 'WASAT', 'WAMW', 'WAIRC')


class TasmanianCourts(AustliiCases):
    """Tasmanian court and tribunal decisions on AustLII."""

    SOURCE = 'tasmanian_courts'
    JURISDICTION = 'tasmania'
    JUR_PATH = 'tas'
    COURTS = ('TASSC', 'TASFC', 'TASCAT', 'TASMC')


class NorthernTerritoryCourts(AustliiCases):
    """Northern Territory court and tribunal decisions on AustLII."""

    SOURCE = 'northern_territory_courts'
    JURISDICTION = 'northern_territory'
    JUR_PATH = 'nt'
    COURTS = ('NTSC', 'NTCA', 'NTSCFC', 'NTLC', 'NTMC', 'NTCAT', 'NTWHC', 'NTYJC')


class AustralianCapitalTerritoryCourts(AustliiCases):
    """Australian Capital Territory court and tribunal decisions on AustLII."""

    SOURCE = 'australian_capital_territory_courts'
    JURISDICTION = 'australian_capital_territory'
    JUR_PATH = 'act'
    COURTS = ('ACTSC', 'ACTCA', 'ACTSCFC', 'ACTMC', 'ACAT', 'ACTAAT')


class CommonwealthTribunals(AustliiCases):
    """Commonwealth tribunal decisions on AustLII (/au/cases/cth/<CODE>/).

    These adjudicative tribunals are absent from the corpus (the family-law
    ``Austlii`` scraper covers only cth courts). Enterprise-agreement approvals
    (FWCA) are intentionally excluded as non-adjudicative bulk. Reuses the
    AustliiCases fetch/parse machinery, but overrides year discovery with an
    explicit (start, end) range per database: the base's landing-page
    ``_coverage_floor`` heuristic under-enumerates defunct tribunals (FWA/SCTA)
    whose data years sit behind a >3yr gap from spurious recent year tokens on
    the landing page. AustLII ToS: non-redistributable.
    """

    SOURCE = 'commonwealth_tribunals'
    JURISDICTION = 'commonwealth'
    JUR_PATH = 'cth'
    CRAWL_DELAY = 0.0   # owner-authorised crank; concurrency handled via a larger semaphore below

    def __init__(self, *args, semaphore=None, **kwargs):
        # Owner-authorised higher concurrency for this fresh large crawl (default base is 2).
        # AustLII rate-limits by velocity; monitor for 429/403 and dial back if they appear.
        super().__init__(*args, semaphore=semaphore or asyncio.Semaphore(8), **kwargs)

    # code -> (first year of decisions, last year | None for still-sitting). Ranges
    # are deliberately generous on the early bound; empty years yield no entries and
    # cost only one index fetch. Excluded: FWCA (non-adjudicative bulk); AFPDT/NST/
    # ATPT (no viewdb at /au/cases/cth/<code>/ — HTTP 500).
    COURT_YEARS: dict[str, tuple[int, int | None]] = {
        'AATA'  : (1976, 2024),  # Administrative Appeals Tribunal (→ ART 2024)
        'ARTA'  : (2024, None),  # Administrative Review Tribunal (AAT successor)
        'FWC'   : (2009, None),  # Fair Work Commission
        'FWCFB' : (2009, None),  # FWC Full Bench
        'FWCD'  : (2012, None),  # FWC General Manager & Delegates
        'FWA'   : (2009, 2012),  # Fair Work Australia (predecessor)
        'FWAFB' : (2009, 2012),  # Fair Work Australia Full Bench
        'FWAA'  : (2009, 2012),  # Fair Work Australia (agreements)
        'NNTTA' : (1994, None),  # National Native Title Tribunal
        'ATMO'  : (1982, None),  # Trade Marks Office
        'APO'   : (1981, None),  # Patent Office
        'ACompT': (1976, None),  # Australian Competition Tribunal
        'ACopyT': (1969, None),  # Copyright Tribunal
        'SCTA'  : (1994, 2021),  # Superannuation Complaints Tribunal (defunct 2021)
        'ADFDAT': (1985, None),  # Defence Force Discipline Appeal Tribunal
    }
    COURTS = tuple(COURT_YEARS)

    async def get_index_reqs(self) -> set:
        # Explicit ranges (no fragile landing-page year discovery). One year-listing
        # request per court per year; the base Creator drops docs absent from the index.
        from ..data import Request
        from datetime import datetime
        current_year = datetime.now().year
        reqs = set()
        for code, (start, end) in self.COURT_YEARS.items():
            lo = max(start, self._min_year) if self._min_year else start
            for year in range(lo, (end or current_year) + 1):
                reqs.add(Request(f'{self.BASE}/cgi-bin/viewdb/au/cases/{self.JUR_PATH}/{code}/{year}/'))
        return reqs
