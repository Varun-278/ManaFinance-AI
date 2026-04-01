from fastapi import APIRouter, HTTPException, Query

from backend.services import financial_assistant_legacy as legacy

router = APIRouter(prefix="/api/compare", tags=["compare"])


@router.get("/returns")
def compare_returns(symbols: str = Query(...), start: int | None = Query(None), end: int | None = Query(None)):
    parsed = [s.strip().upper() for s in symbols.split(",") if s.strip()]
    if len(parsed) < 2:
        raise HTTPException(status_code=400, detail="Provide at least two symbols (comma-separated)")
    if (start is None) ^ (end is None):
        raise HTTPException(status_code=400, detail="Provide both start and end years together")
    if start is not None and end is not None and start > end:
        raise HTTPException(status_code=400, detail="start year must be <= end year")

    summary = legacy.compare_returns(parsed, start, end)
    return {
        "symbols": parsed,
        "start": start,
        "end": end,
        "summary": summary,
    }
