import os
import secrets

from fastapi import Header, HTTPException


def require_admin(x_admin_key: str | None = Header(default=None)) -> None:
    """Guards admin endpoints. The key comes only from the SOLAR_ADMIN_KEY environment
    variable; with no key configured the admin API is disabled rather than open."""
    expected = os.environ.get("SOLAR_ADMIN_KEY")
    if not expected:
        raise HTTPException(status_code=503, detail="Admin API is disabled: SOLAR_ADMIN_KEY is not configured on the server.")
    if not x_admin_key or not secrets.compare_digest(x_admin_key.encode(), expected.encode()):
        raise HTTPException(status_code=401, detail="Invalid or missing admin key.")
