FROM python:3.14-slim
WORKDIR /app

COPY . .
COPY .env .env

RUN apt-get update && apt-get install -y curl

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "app.py"]