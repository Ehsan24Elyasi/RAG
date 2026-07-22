import pytest

from app.security import WidgetTokenError, create_widget_token, verify_widget_token


def test_widget_token_round_trip():
    token = create_widget_token(
        "secret",
        workspace_id="workspace-1",
        external_user_id="customer-1",
        audience="support-widget",
        display_name="Customer",
        now=1_000,
        expires_in_seconds=60,
    )

    identity = verify_widget_token(
        token,
        "secret",
        audience="support-widget",
        clock_skew_seconds=0,
        now=1_030,
    )

    assert identity.workspace_id == "workspace-1"
    assert identity.external_user_id == "customer-1"
    assert identity.display_name == "Customer"


def test_widget_token_rejects_tampering_and_expiry():
    token = create_widget_token(
        "secret",
        workspace_id="workspace-1",
        external_user_id="customer-1",
        now=1_000,
        expires_in_seconds=10,
    )

    with pytest.raises(WidgetTokenError):
        verify_widget_token(f"{token[:-1]}x", "secret", now=1_005)
    with pytest.raises(WidgetTokenError):
        verify_widget_token(token, "secret", clock_skew_seconds=0, now=1_011)
