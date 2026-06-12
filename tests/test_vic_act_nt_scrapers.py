"""Offline unit tests for the Victorian, ACT and Northern Territory legislation scrapers.

Every test runs entirely against captured fixtures (``tests/fixtures``); no network access
is made. Each scraper's ``get`` is monkeypatched to return a fixture-backed ``Response``,
so index parsing, in-force/version selection and DOCX text extraction are exercised exactly
as they would be against the live sites, minus the HTTP and crawl-delay layers.
"""

import asyncio
from pathlib import Path

import pytest

from oalc_creator.data import Entry, Request, Response
from oalc_creator.scrapers import (
    ActLegislation,
    VictorianLegislation,
    NorthernTerritoryLegislation,
)

FIXTURES = Path(__file__).parent / 'fixtures'


def _response(body: bytes, content_type: str = 'text/html', status: int = 200) -> Response:
    return Response(body, encoding='utf-8', type=content_type, status=status)


def _text_response(path: Path, content_type: str = 'text/html') -> Response:
    return _response(path.read_bytes(), content_type=content_type)


def _run(coro):
    return asyncio.run(coro)


# --------------------------------------------------------------------------- #
# Victorian Legislation
# --------------------------------------------------------------------------- #

class TestVictorianLegislation:
    def _scraper(self, routes: dict[str, Response]) -> VictorianLegislation:
        scraper = VictorianLegislation()

        async def fake_get(req):
            path = req if isinstance(req, str) else req.path
            # Match the most specific (longest) registered key that is a substring of the
            # request path, so that eg a `/DownloadFile/a/2001-14/...` request is not
            # shadowed by a `/a/2001-14/` landing-page route.
            best = max((k for k in routes if k in path), key=len, default=None)
            if best is not None:
                return routes[best]
            raise AssertionError(f'Unexpected request: {path}')

        scraper.get = fake_get
        return scraper

    def test_index_filters_to_in_force_acts(self):
        scraper = self._scraper({
            'sitemap.xml?page=2': _text_response(FIXTURES / 'vic_sitemap_p2.xml', 'application/xml'),
        })

        entries = _run(scraper.get_index(Request('https://content.legislation.vic.gov.au/sitemap.xml?page=2')))

        assert entries, 'expected in-force acts on sitemap page 2'
        # Every entry must be an in-force act on the www host with the act type.
        for entry in entries:
            assert entry.type == 'primary_legislation'
            assert entry.jurisdiction == 'victoria'
            assert entry.request.path.startswith('https://www.legislation.vic.gov.au/in-force/acts/')
            assert entry.version_id.startswith('victorian_legislation:acts/')

    def test_index_filters_to_in_force_statutory_rules(self):
        scraper = self._scraper({
            'sitemap.xml?page=7': _text_response(FIXTURES / 'vic_sitemap_p7.xml', 'application/xml'),
        })

        entries = _run(scraper.get_index(Request('https://content.legislation.vic.gov.au/sitemap.xml?page=7')))

        assert entries
        for entry in entries:
            assert entry.type == 'secondary_legislation'
            assert '/in-force/statutory-rules/' in entry.request.path

    def test_index_ignores_as_made_and_other_pages(self):
        # A sitemap page with no in-force content yields no entries.
        scraper = self._scraper({
            'sitemap.xml?page=2': _response(
                b'<urlset><url><loc>https://content.legislation.vic.gov.au/site-6/as-made/acts/foo</loc></url></urlset>',
                'application/xml',
            ),
        })
        entries = _run(scraper.get_index(Request('https://content.legislation.vic.gov.au/sitemap.xml?page=2')))
        assert entries == set()

    def test_parse_landing_extracts_version_record(self):
        scraper = VictorianLegislation()
        html = (FIXTURES / 'vic_act_payload.json').read_text(encoding='utf-8', errors='replace')

        record = scraper._parse_landing(html)

        assert record is not None
        assert record['guid'] == '5c6824f9-7356-4c09-bb6b-be6b89306c78'
        assert record['date'] == '2026-04-15'
        assert record['version'] == '055'
        assert record['status'] == 'In force'

    def test_parse_landing_handles_statutory_rule(self):
        scraper = VictorianLegislation()
        html = (FIXTURES / 'vic_sr_page.html').read_text(encoding='utf-8', errors='replace')

        record = scraper._parse_landing(html)

        assert record is not None
        assert record['guid'] == '50682d75-8176-4cec-913e-84cfb41705bc'
        assert record['version'] == '004'
        assert record['status'] == 'In force'

    def test_get_doc_extracts_text_from_docx(self):
        scraper = self._scraper({
            '/in-force/acts/eastlink-project-act-2004': _text_response(FIXTURES / 'vic_act_payload.json'),
            '/api/tide/in-force': _text_response(FIXTURES / 'vic_tide_act.json', 'application/json'),
            '04-39a055.docx': _response(
                (FIXTURES / 'vic_act.docx').read_bytes(),
                'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            ),
        })

        entry = Entry(
            request=Request('https://www.legislation.vic.gov.au/in-force/acts/eastlink-project-act-2004'),
            version_id='acts/eastlink-project-act-2004',
            source='victorian_legislation',
            type='primary_legislation',
            jurisdiction='victoria',
        )

        doc = _run(scraper._get_doc(entry))

        assert doc is not None
        assert doc.citation == 'EastLink Project Act 2004 (Vic)'
        assert doc.jurisdiction == 'victoria'
        assert doc.type == 'primary_legislation'
        assert doc.date == '2026-04-15'
        assert doc.mime == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
        assert doc.version_id == 'victorian_legislation:acts/eastlink-project-act-2004/055'
        assert doc.url.endswith('04-39a055.docx')
        # The DOCX text must contain substantive legislative content.
        assert 'EastLink' in doc.text
        assert len(doc.text) > 1000


# --------------------------------------------------------------------------- #
# ACT Legislation
# --------------------------------------------------------------------------- #

class TestActLegislation:
    def _scraper(self, routes: dict[str, Response]) -> ActLegislation:
        scraper = ActLegislation()

        async def fake_get(req):
            path = req if isinstance(req, str) else req.path
            # Match the most specific (longest) registered key that is a substring of the
            # request path, so that eg a `/DownloadFile/a/2001-14/...` request is not
            # shadowed by a `/a/2001-14/` landing-page route.
            best = max((k for k in routes if k in path), key=len, default=None)
            if best is not None:
                return routes[best]
            raise AssertionError(f'Unexpected request: {path}')

        scraper.get = fake_get
        return scraper

    def test_index_parses_act_and_subordinate_ids(self):
        scraper = self._scraper({
            'Notifications': _text_response(FIXTURES / 'act_notifications_2001.html'),
        })

        entries = _run(scraper.get_index(Request('https://www.legislation.act.gov.au/Notifications?category=cAct&year=2001')))

        assert entries
        version_ids = {e.version_id for e in entries}
        # The Legislation Act 2001 is notified as a/2001-14.
        assert 'act_legislation:2001-14' in version_ids
        for entry in entries:
            assert entry.type == 'primary_legislation'
            assert entry.jurisdiction == 'australian_capital_territory'
            assert entry.request.path.endswith('/')

    def test_index_reqs_cover_acts_and_subordinate(self):
        scraper = ActLegislation()
        reqs = _run(scraper.get_index_reqs())
        paths = {r.path for r in reqs}
        assert any('category=cAct&year=1912' in p for p in paths)
        assert any('category=cSub&year=1912' in p for p in paths)

    def test_get_doc_in_force_extracts_docx(self):
        scraper = self._scraper({
            '/a/2001-14/': _text_response(FIXTURES / 'act_landing.html'),
            'current/DOCX/2001-14.DOCX': _response(
                (FIXTURES / 'act_2001-14.docx').read_bytes(),
                'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            ),
        })

        entry = Entry(
            request=Request('https://www.legislation.act.gov.au/a/2001-14/'),
            version_id='2001-14',
            source='act_legislation',
            type='primary_legislation',
            jurisdiction='australian_capital_territory',
            title='Legislation Act',
        )

        doc = _run(scraper._get_doc(entry))

        assert doc is not None
        assert doc.citation == 'Legislation Act 2001 (ACT)'
        assert doc.date == '2026-02-23'
        assert doc.mime == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
        assert doc.version_id == 'act_legislation:2001-14/2026-02-23'
        assert doc.url.endswith('current/DOCX/2001-14.DOCX')
        assert len(doc.text) > 1000

    def test_get_doc_skips_not_in_force(self):
        # A landing page with no current consolidation link must yield None.
        scraper = self._scraper({
            '/a/1900-1/': _response(b'<html><title>Repealed Act 1900</title><body>Repealed</body></html>'),
        })
        entry = Entry(
            request=Request('https://www.legislation.act.gov.au/a/1900-1/'),
            version_id='1900-1',
            source='act_legislation',
            type='primary_legislation',
            jurisdiction='australian_capital_territory',
            title='Repealed Act',
        )
        assert _run(scraper._get_doc(entry)) is None


# --------------------------------------------------------------------------- #
# Northern Territory Legislation
# --------------------------------------------------------------------------- #

class TestNorthernTerritoryLegislation:
    def _scraper(self, routes: dict[str, Response]) -> NorthernTerritoryLegislation:
        scraper = NorthernTerritoryLegislation()

        async def fake_get(req):
            path = req if isinstance(req, str) else req.path
            # Match the most specific (longest) registered key that is a substring of the
            # request path, so that eg a `/DownloadFile/a/2001-14/...` request is not
            # shadowed by a `/a/2001-14/` landing-page route.
            best = max((k for k in routes if k in path), key=len, default=None)
            if best is not None:
                return routes[best]
            raise AssertionError(f'Unexpected request: {path}')

        scraper.get = fake_get
        return scraper

    def test_index_parses_record_links(self):
        scraper = self._scraper({
            'Acts/By-Title': _text_response(FIXTURES / 'nt_acts_bytitle.html'),
        })

        entries = _run(scraper.get_index(Request('https://legislation.nt.gov.au/en/LegislationPortal/Acts/By-Title')))

        assert entries
        slugs = {e.version_id.split(':', 1)[-1] for e in entries}
        assert 'ADOPTION-OF-CHILDREN-ACT-1994' in slugs
        for entry in entries:
            assert entry.type == 'primary_legislation'
            assert entry.jurisdiction == 'northern_territory'
            assert entry.request.path.startswith('https://legislation.nt.gov.au/en/Legislation/')

    def test_index_reqs_cover_acts_and_subordinate(self):
        scraper = NorthernTerritoryLegislation()
        reqs = _run(scraper.get_index_reqs())
        paths = {r.path for r in reqs}
        assert any('Acts/By-Title' in p for p in paths)
        assert any('Subordinate-Legislation/By-Title' in p for p in paths)

    def test_select_reprint_picks_current_row(self):
        scraper = NorthernTerritoryLegislation()
        html = (FIXTURES / 'nt_history_current.html').read_text()

        reprint = scraper._select_reprint(html)

        # The open-ended row (01/01/2025 -> blank) covers today.
        assert reprint is not None
        assert reprint['word_id'] == '30001'
        assert reprint['start'] == '2025-01-01'
        assert reprint['end'] is None

    def test_select_reprint_falls_back_to_most_recent(self):
        scraper = NorthernTerritoryLegislation()
        html = (FIXTURES / 'nt_history_adoption.html').read_text()

        reprint = scraper._select_reprint(html)

        # No row covers today (all end dates are in the past), so the most recent reprint
        # (start 2018-06-20, the greatest start date) is chosen.
        assert reprint is not None
        assert reprint['word_id'] == '22417'
        assert reprint['start'] == '2018-06-20'

    def test_parse_date(self):
        assert NorthernTerritoryLegislation._parse_date('20/06/2018') == '2018-06-20'
        assert NorthernTerritoryLegislation._parse_date('') is None

    def test_get_doc_traces_record_history_and_extracts_text(self):
        # The record page points at the history listing; the history listing's chosen
        # reprint points at the Word_History DOCX (reuse the VIC DOCX bytes as a stand-in
        # for the binary, since NT's binary DOCX could not be captured offline).
        scraper = self._scraper({
            '/en/Legislation/ADOPTION-OF-CHILDREN-ACT-1994': _text_response(FIXTURES / 'nt_record_adoption.html'),
            '/Pages/Act%20History': _text_response(FIXTURES / 'nt_history_current.html'),
            '/api/sitecore/Act/Word_History': _response(
                (FIXTURES / 'vic_act.docx').read_bytes(),
                'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            ),
        })

        entry = Entry(
            request=Request('https://legislation.nt.gov.au/en/Legislation/ADOPTION-OF-CHILDREN-ACT-1994'),
            version_id='ADOPTION-OF-CHILDREN-ACT-1994',
            source='northern_territory_legislation',
            type='primary_legislation',
            jurisdiction='northern_territory',
            title='ADOPTION OF CHILDREN ACT 1994',
        )

        doc = _run(scraper._get_doc(entry))

        assert doc is not None
        assert doc.citation == 'ADOPTION OF CHILDREN ACT 1994 (NT)'
        assert doc.jurisdiction == 'northern_territory'
        assert doc.mime == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
        assert doc.url.endswith('Word_History?id=30001')
        assert doc.version_id == 'northern_territory_legislation:ADOPTION-OF-CHILDREN-ACT-1994/2025-01-01'
        assert len(doc.text) > 1000

    def test_get_doc_skips_record_without_history(self):
        scraper = self._scraper({
            '/en/Legislation/NO-HISTORY': _response(b'<html><body>No history listing here</body></html>'),
        })
        entry = Entry(
            request=Request('https://legislation.nt.gov.au/en/Legislation/NO-HISTORY'),
            version_id='NO-HISTORY',
            source='northern_territory_legislation',
            type='primary_legislation',
            jurisdiction='northern_territory',
            title='No History Act',
        )
        assert _run(scraper._get_doc(entry)) is None
