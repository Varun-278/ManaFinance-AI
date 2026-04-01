from fastapi import APIRouter, HTTPException

from backend.services import financial_assistant_legacy as legacy

router = APIRouter(prefix="/api/stocks", tags=["stocks"])


@router.get("/{symbol}/date/{date}")
def stock_on_date(symbol: str, date: str):
    symbol = symbol.upper()
    summary = legacy.structured_answer(symbol, date)
    if not summary:
        raise HTTPException(status_code=404, detail="No record found for symbol/date")
    return {"symbol": symbol, "date": date, "summary": summary}
