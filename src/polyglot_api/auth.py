import hmac
import os
from typing import Optional

from fastapi import Header, HTTPException, status


API_TOKEN_HEADER = "X-Polyglot-Api-Key"


def require_api_token(x_polyglot_api_key: Optional[str] = Header(default=None)) -> bool:
    expected = os.getenv("POLYGLOT_API_TOKEN", "").strip()
    if not expected:
        return True
    provided = (x_polyglot_api_key or "").strip()
    if hmac.compare_digest(provided, expected):
        return True
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Missing or invalid Polyglot API token",
    )
