import os
from fastapi import Security, HTTPException, status, Depends
from fastapi.security import APIKeyHeader

API_KEY_NAME = "X-API-Key"
API_KEY_ENV_VAR = "MOCHI_API_KEY"

api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=True)

# Load the expected API key from environment variable
EXPECTED_API_KEY = os.getenv(API_KEY_ENV_VAR)

async def get_api_key(api_key: str = Security(api_key_header)):
    """
    Dependency function to validate the API key.
    
    Raises HTTPException 401 if the key is invalid or 503 if not configured.
    """
    if not EXPECTED_API_KEY:
        # Service unavailable if not configured
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="API Key not configured on server. Set MOCHI_API_KEY environment variable."
        )
    
    # Use a secure comparison if this were production (though constant time isn't crucial here)
    if api_key != EXPECTED_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API Key",
        )
    # Return the key or a confirmation value if needed by endpoints
    return api_key 