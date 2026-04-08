from tools.hltv.parser import (
    parse_results_page,
    parse_match_page,
    get_map_from_dem_filename,
)

# ── Fixtures ──────────────────────────────────────────────────────────────

RESULTS_HTML = """
<html><body>
<div class="results-sublist">
  <div class="result-con">
    <a class="a-reset plain" href="/matches/2380145/navi-vs-faze-starladder-budapest-major-2025">
      <div class="result-teamnames text-ellipsis">
        <span class="team-name">Natus Vincere</span>
        <span class="team-name">FaZe Clan</span>
      </div>
      <div class="event-name">StarLadder Budapest Major 2025</div>
      <div class="date-cell"><div data-unix="1733500000000"></div></div>
    </a>
  </div>
  <div class="result-con">
    <a class="a-reset plain" href="/matches/2380200/spirit-vs-vitality-iem-cologne-2025">
      <div class="result-teamnames text-ellipsis">
        <span class="team-name">Team Spirit</span>
        <span class="team-name">Team Vitality</span>
      </div>
      <div class="event-name">IEM Cologne 2025</div>
      <div class="date-unix"><div data-unix="1720000000000"></div></div>
    </a>
  </div>
</div>
</body></html>
"""

MATCH_HTML = """
<html><body>
<div class="standard-box veto-box">
  <div class="mapholder">
    <div class="mapname">Mirage</div>
    <div class="results-center-half-score">
      <span class="ct">CT</span>
      <span class="t">T</span>
    </div>
    <div class="results-left">
      <div class="team">Natus Vincere</div>
    </div>
    <div class="results-right">
      <div class="team">FaZe Clan</div>
    </div>
  </div>
</div>
<div class="streams-and-vods">
  <div class="stream-box">
    <a href="/download/demo/654321">Download Demo</a>
  </div>
</div>
<div class="lineups">
  <div class="lineup standard-box">
    <div class="teamName">Natus Vincere</div>
    <div class="players">
      <div class="player">
        <a href="/player/7998/s1mple">s1mple</a>
      </div>
      <div class="player">
        <a href="/player/8918/electronic">electronic</a>
      </div>
    </div>
  </div>
  <div class="lineup standard-box">
    <div class="teamName">FaZe Clan</div>
    <div class="players">
      <div class="player">
        <a href="/player/10394/karrigan">karrigan</a>
      </div>
      <div class="player">
        <a href="/player/11816/ropz">ropz</a>
      </div>
    </div>
  </div>
</div>
</body></html>
"""

# ── Tests ─────────────────────────────────────────────────────────────────

def test_parse_results_page_returns_matches():
    matches = parse_results_page(RESULTS_HTML)
    assert len(matches) == 2


def test_parse_results_page_match_fields():
    matches = parse_results_page(RESULTS_HTML)
    m = matches[0]
    assert m["match_id"] == "2380145"
    assert "navi-vs-faze" in m["url"]
    assert m["event"] == "StarLadder Budapest Major 2025"
    assert m["date"] == "2024-12-06"  # unix 1733500000 → 2024-12-06


def test_parse_results_page_empty():
    assert parse_results_page("<html><body></body></html>") == []


def test_parse_match_page_demo_url():
    result = parse_match_page(MATCH_HTML)
    assert result is not None
    assert result["demo_url"] == "/download/demo/654321"


def test_parse_match_page_teams():
    result = parse_match_page(MATCH_HTML)
    assert "Natus Vincere" in (result["team_ct"], result["team_t"])
    assert "FaZe Clan" in (result["team_ct"], result["team_t"])


def test_parse_match_page_no_demo_returns_none():
    result = parse_match_page("<html><body><p>No demo here</p></body></html>")
    assert result is None


def test_get_map_from_dem_filename_standard():
    assert get_map_from_dem_filename("navi-vs-faze-m1-de_mirage.dem") == "de_mirage"


def test_get_map_from_dem_filename_no_prefix():
    assert get_map_from_dem_filename("match-inferno.dem") == "de_inferno"


def test_get_map_from_dem_filename_unknown():
    assert get_map_from_dem_filename("match-vertigo.dem") is None
