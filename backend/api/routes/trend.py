from fastapi import APIRouter, HTTPException, Query

from backend.services import financial_assistant_legacy as legacy

router = APIRouter(prefix="/api/trend", tags=["trend"])


@router.get("")
def trend(symbol: str = Query(...), start: int = Query(...), end: int = Query(...)):
    symbol = symbol.upper()
    if start > end:
        raise HTTPException(status_code=400, detail="start year must be <= end year")
    summary = legacy.trend_analysis(symbol, start, end)
    return {"symbol": symbol, "start": start, "end": end, "summary": summary}
