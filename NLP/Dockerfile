FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY sentiment_api.py .
COPY bilstm_sentiment_model.keras .
COPY tokenizer.pkl .

EXPOSE 8000

CMD ["uvicorn", "sentiment_api:app", "--host", "0.0.0.0", "--port", "8000"]