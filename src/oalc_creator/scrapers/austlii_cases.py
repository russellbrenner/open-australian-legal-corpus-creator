import re
import os
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

    # When True and COURTS is empty, the database codes for JUR_PATH are discovered
    # from AustLII's master catalogue at run time (complete, self-maintaining coverage).
    DISCOVER_CATALOGUE: bool = False

    # Index-time dedup (opt-in). When True, index entries whose normalised neutral
    # citation is in the held set (env AUSTLII_HELD_NC) are dropped, so the crawl
    # fetches only not-yet-held documents. MUST only be enabled on scrapers with a
    # FRESH source name: the base Creator purges any doc of a source that is absent
    # from its index, so filtering an existing source's index would delete its corpus.
    DEDUP_HELD: bool = False
    HELD_NC_ENV = 'AUSTLII_HELD_NC'

    # An absolute floor below which no year is ever enumerated, as a courtesy guard
    # against a malformed page producing an absurd range. 1788 covers digitised
    # historical law-report series (eg NSWLawRp from 1856, ArgusLawRp from 1893).
    ABSOLUTE_FLOOR = 1788

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

    _NC_RE = re.compile(r'\[(\d{4})\]\s*([A-Za-z]+)\s*(\d+)')
    _held_nc = None  # class-level cache of the held-NC dedup set

    @classmethod
    def _load_held_nc(cls) -> frozenset:
        """Load (once) the normalised held-NC set for index-time dedup."""
        if AustliiCases._held_nc is None:
            path = os.environ.get(cls.HELD_NC_ENV)
            held = set()
            if path and os.path.exists(path):
                with open(path) as f:
                    held = {ln.strip() for ln in f if ln.strip()}
            AustliiCases._held_nc = frozenset(held)
        return AustliiCases._held_nc

    @classmethod
    def _norm_nc(cls, title: str) -> str | None:
        """Normalise a listing title's medium-neutral citation to '[YYYY] CODE N'."""
        m = cls._NC_RE.search(title or '')
        return f'[{m.group(1)}] {m.group(2).upper()} {m.group(3)}' if m else None

    async def _discover_courts(self) -> tuple[str, ...]:
        """Discover every case-database code for JUR_PATH from AustLII's catalogue."""
        cat = await self._get_text(Request(f'{self.BASE}/databases.html'))
        codes = sorted(set(re.findall(rf'/cgi-bin/viewdb/au/cases/{self.JUR_PATH}/([A-Za-z0-9]+)/', cat)))
        return tuple(codes)

    @log
    async def get_index_reqs(self) -> set[Request]:
        """Enumerate one year-listing request per court per advertised year.

        Codes are the subclass ``COURTS`` or, if empty and ``DISCOVER_CATALOGUE`` is
        set, the full AustLII catalogue for ``JUR_PATH``. Coverage years are every
        distinct year token advertised on each database's landing page (not a
        contiguous-from-current block: defunct/historical databases advertise their
        real years across a >3yr gap the old ``_coverage_floor`` heuristic wrongly
        truncated). Empty years yield an empty index at the cost of one cheap listing
        fetch. The base ``Creator`` drops any document absent from the index, so the
        index must enumerate the entire back catalogue on every run.
        """

        current_year = datetime.now().year
        reqs: set[Request] = set()

        courts = self.COURTS or (await self._discover_courts() if self.DISCOVER_CATALOGUE else ())

        for code in courts:
            try:
                landing = await self._get_text(f'{self.BASE}/cgi-bin/viewdb/au/cases/{self.JUR_PATH}/{code}/')
            except (ParseError, CurlError, asyncio.TimeoutError, OSError):
                # A database whose landing page is unreachable is skipped; next run retries.
                continue

            years = {int(y) for y in re.findall(r'\b(1[789]\d\d|20\d\d)\b', landing)}
            years = {y for y in years if self.ABSOLUTE_FLOOR <= y <= current_year}

            if self._min_year:
                years = {y for y in years if y >= self._min_year}

            for year in sorted(years):
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

            # Index-time dedup (opt-in, fresh-source scrapers only): skip documents
            # already held elsewhere in the corpus so only gaps are fetched.
            if self.DEDUP_HELD:
                held = self._load_held_nc()
                if held and (nc := self._norm_nc(title)) is not None and nc in held:
                    continue

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


class _AustliiComplete(AustliiCases):
    """Full-catalogue AustLII crawl for a jurisdiction, index-deduped against the held
    corpus so only not-yet-held documents are fetched. Uses a FRESH source name so the
    base Creator's index-removal pass is safe (the corpus only ever holds gap docs,
    which always reappear in the filtered index). Covers every case database AustLII
    lists for JUR_PATH (courts, tribunals, historical law-report series)."""

    DISCOVER_CATALOGUE = True
    DEDUP_HELD = True
    CRAWL_DELAY = 0.0  # owner-authorised crank; concurrency via the larger semaphore below

    def __init__(self, *args, semaphore=None, **kwargs):
        # Gentler concurrency for the long 597k full-catalogue crawl: semaphore 8 tripped
        # Cloudflare challenge-bursts ~50k in that exhausted the retry budget and crashed
        # the run. 5 keeps a courteous velocity.
        super().__init__(*args, semaphore=semaphore or asyncio.Semaphore(5), **kwargs)
        # Survive (don't crash on) sustained Cloudflare velocity blocks: keep retrying
        # with a long backoff so a block becomes a low-CPU PAUSE that resumes when it
        # clears, instead of exhausting the 15-min budget -> raise -> asyncio-shutdown
        # deadlock -> hung zombie. 4h ceiling; up to 5-min spacing between retries.
        self.stop_after_waiting = 4 * 60 * 60
        self.max_wait = 5 * 60


class CommonwealthCasesComplete(_AustliiComplete):
    SOURCE = 'commonwealth_austlii'
    JURISDICTION = 'commonwealth'
    JUR_PATH = 'cth'


class NewSouthWalesCasesComplete(_AustliiComplete):
    SOURCE = 'new_south_wales_austlii'
    JURISDICTION = 'new_south_wales'
    JUR_PATH = 'nsw'


class VictoriaCasesComplete(_AustliiComplete):
    SOURCE = 'victoria_austlii'
    JURISDICTION = 'victoria'
    JUR_PATH = 'vic'


class QueenslandCasesComplete(_AustliiComplete):
    SOURCE = 'queensland_austlii'
    JURISDICTION = 'queensland'
    JUR_PATH = 'qld'


class SouthAustraliaCasesComplete(_AustliiComplete):
    SOURCE = 'south_australia_austlii'
    JURISDICTION = 'south_australia'
    JUR_PATH = 'sa'


class WesternAustraliaCasesComplete(_AustliiComplete):
    SOURCE = 'western_australia_austlii'
    JURISDICTION = 'western_australia'
    JUR_PATH = 'wa'


class TasmaniaCasesComplete(_AustliiComplete):
    SOURCE = 'tasmania_austlii'
    JURISDICTION = 'tasmania'
    JUR_PATH = 'tas'


class NorthernTerritoryCasesComplete(_AustliiComplete):
    SOURCE = 'northern_territory_austlii'
    JURISDICTION = 'northern_territory'
    JUR_PATH = 'nt'


class AustralianCapitalTerritoryCasesComplete(_AustliiComplete):
    SOURCE = 'australian_capital_territory_austlii'
    JURISDICTION = 'australian_capital_territory'
    JUR_PATH = 'act'


class NorfolkIslandCasesComplete(_AustliiComplete):
    SOURCE = 'norfolk_island_austlii'
    JURISDICTION = 'norfolk_island'
    JUR_PATH = 'nf'
