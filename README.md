# ManaFinance-AI

ManaFinance-AI is an AI-powered financial assistant for NIFTY50 analysis that combines:
- structured stock data (OHLCV)
- multi-year trend and return comparison
- RAG-based finance Q&A using Groq + FAISS

## Project Structure

```text
backend/
  api/routes/
  models/
  services/
  main.py
frontend/
  src/components/
  src/lib/
  src/App.jsx
  src/main.jsx
data/
src/
```

## Run Backend

```bash
pip install -r backend/requirements.txt
uvicorn backend.main:app --reload
```

Backend URL: `http://127.0.0.1:8000`

## Run Frontend

```bash
cd frontend
npm install
npm run dev
```

Frontend URL: `http://127.0.0.1:5173`

## Legacy Gradio UI

```bash
python src/financial_assistant.py
```
