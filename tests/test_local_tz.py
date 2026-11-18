"""System zone detection for `mem serve --tz` (TZ, else /etc/localtime, else UTC)."""

import pytest

from mem.common import local_tz


def _zoneinfo_link(tmp_path, key):
    target = tmp_path / "usr" / "share" / "zoneinfo" / key
    target.parent.mkdir(parents=True)
    target.write_bytes(b"")
    link = tmp_path / "localtime"
    link.symlink_to(target)
    return link


@pytest.mark.parametrize("value", ["Europe/Lisbon", ":Europe/Lisbon"])
def test_tz_env_names_the_zone(value, tmp_path):
    assert local_tz({"TZ": value}, tmp_path / "missing") == "Europe/Lisbon"


def test_localtime_symlink_names_the_zone_when_tz_is_unset(tmp_path):
    assert local_tz({}, _zoneinfo_link(tmp_path, "Asia/Tokyo")) == "Asia/Tokyo"


def test_unknown_tz_falls_through_to_localtime(tmp_path):
    assert (
        local_tz({"TZ": "Nowhere/Atlantis"}, _zoneinfo_link(tmp_path, "Asia/Tokyo")) == "Asia/Tokyo"
    )


def test_utc_when_nothing_names_a_zone(tmp_path):
    plain = tmp_path / "localtime"
    plain.write_bytes(b"TZif")  # a copied file, not a symlink: no zone name to read
    assert local_tz({}, plain) == "UTC"
    assert local_tz({"TZ": ""}, tmp_path / "missing") == "UTC"
