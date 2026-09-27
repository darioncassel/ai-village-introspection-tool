"""Shared test setup."""

import pytest

from village_introspect import cc_lib as cc


@pytest.fixture(autouse=True)
def _no_real_village_days():
    """Unit tests never read the real village-transcript.json (or write its cache into the user's state
    dir): village days are empty, so every timestamp gets day None, unless a test installs its own
    ranges with cc.set_day_ranges()."""
    cc.set_day_ranges([])
    yield
    cc.set_day_ranges(None)
