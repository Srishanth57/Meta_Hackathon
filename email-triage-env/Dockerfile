FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV API_BASE_URL="https://api.openai.com/v1"
ENV MODEL_NAME="gpt-4.1-mini"

CMD ["python", "inference.py"]
