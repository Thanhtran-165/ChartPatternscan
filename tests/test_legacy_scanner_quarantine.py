from __future__ import annotations

import pytest

from scanner.legacy_guard import ALLOW_LEGACY_ENV, require_legacy_enabled


def test_legacy_scanner_is_blocked_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(ALLOW_LEGACY_ENV, raising=False)

    with pytest.raises(RuntimeError, match="quarantined legacy scanner logic"):
        require_legacy_enabled("legacy-test")


def test_legacy_scanner_can_be_enabled_for_historical_comparison(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ALLOW_LEGACY_ENV, "1")

    require_legacy_enabled("legacy-test")


def test_legacy_report_entrypoints_are_blocked_before_argparse(monkeypatch: pytest.MonkeyPatch) -> None:
    from scanner.audit_kpi import main as audit_kpi_main
    from scanner.report_bulkowski import main as report_bulkowski_main

    monkeypatch.delenv(ALLOW_LEGACY_ENV, raising=False)

    with pytest.raises(RuntimeError, match="scanner/report_bulkowski.py is quarantined"):
        report_bulkowski_main()
    with pytest.raises(RuntimeError, match="scanner/audit_kpi.py is quarantined"):
        audit_kpi_main()
