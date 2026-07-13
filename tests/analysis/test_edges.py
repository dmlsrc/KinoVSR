"""Persistent edge-analysis tests."""

import mlx.core as mx


def _sanitize_samples(junk_bottom=False, bar_rows=0):
    mx.random.seed(7)
    out = []
    for _ in range(5):
        fr = mx.clip(mx.random.uniform(shape=(48, 64, 3)) * 0.5 + 0.35, 0, 1)
        if junk_bottom:
            fr[-1:] = fr[-2:-1] * 0.3   # dark junk line, tracks content
        if bar_rows:
            fr[:bar_rows] = 0.02        # constant near-black bar at the top
            fr[-bar_rows:] = 0.02
        out.append(fr)
    return out


def test_detect_junk_edges_finds_dark_row():
    from kinovsr.analysis.edges import detect_junk_edges

    edges, notices = detect_junk_edges(_sanitize_samples(junk_bottom=True))
    assert edges == (0, 1, 0, 0)
    assert notices == []


def test_detect_junk_edges_leaves_clean_content_untouched():
    from kinovsr.analysis.edges import detect_junk_edges

    edges, notices = detect_junk_edges(_sanitize_samples())
    assert edges == (0, 0, 0, 0)


def test_detect_junk_edges_reports_letterbox_without_filling():
    from kinovsr.analysis.edges import detect_junk_edges

    edges, notices = detect_junk_edges(_sanitize_samples(bar_rows=12))
    assert edges == (0, 0, 0, 0)
    assert any("letterbox-class" in n for n in notices)


def test_detect_junk_edges_ignores_blank_samples():
    from kinovsr.analysis.edges import detect_junk_edges

    blanks = [mx.full((32, 32, 3), 0.02) for _ in range(5)]
    edges, notices = detect_junk_edges(blanks)
    assert edges == (0, 0, 0, 0)
    assert any("too few" in n for n in notices)


def test_detect_bars_finds_letterbox_and_pillarbox():
    from kinovsr.analysis.edges import detect_bars

    mx.random.seed(11)
    letter, pillar = [], []
    for _ in range(5):
        fr = mx.clip(mx.random.uniform(shape=(48, 64, 3)) * 0.5 + 0.35, 0, 1)
        lb = fr[:]
        lb[:12] = 0.02
        lb[-12:] = 0.02
        letter.append(lb)
        pb_ = fr[:]
        pb_[:, :22] = 0.02
        pb_[:, -22:] = 0.02
        pillar.append(pb_)
    assert detect_bars(letter) == (12, 12, 0, 0)
    assert detect_bars(pillar) == (0, 0, 22, 22)

    # even-ization: an 11-row top bar leaves 37 rows -> bottom bumped by 1
    odd = []
    for _ in range(5):
        fr = mx.clip(mx.random.uniform(shape=(48, 64, 3)) * 0.5 + 0.35, 0, 1)
        fr[:11] = 0.02
        odd.append(fr)
    assert detect_bars(odd) == (11, 1, 0, 0)

    # clean content: no bars
    clean = [mx.clip(mx.random.uniform(shape=(48, 64, 3)) * 0.5 + 0.35, 0, 1)
             for _ in range(5)]
    assert detect_bars(clean) == (0, 0, 0, 0)
