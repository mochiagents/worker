from fastapi import FastAPI, Depends

from .routers import tasks, control, conversations
from .security import get_api_key

app = FastAPI(
    title="Mochi Worker API", 
    version="0.1.0",
    description="External API for controlling the Mochi worker agent and managing tasks."
)

# Include routers with API Key Security
app.include_router(
    tasks.router, 
    prefix="/tasks", 
    tags=["Tasks"],
    dependencies=[Depends(get_api_key)]
)
app.include_router(
    control.router, 
    prefix="/control", 
    tags=["Control"],
    dependencies=[Depends(get_api_key)]
)
app.include_router(
    conversations.router, 
    prefix="/conversations", 
    tags=["Conversations"],
    dependencies=[Depends(get_api_key)]
)

@app.get("/", tags=["Status"])
async def read_root():
    """Basic status check endpoint."""
    return {"status": "Mochi Worker API running"}
