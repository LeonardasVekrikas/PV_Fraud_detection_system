FROM python:3.9

WORKDIR /app

COPY . /app

COPY model_api/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

CMD ["uvicorn", "model_api.main:app", "--host", "0.0.0.0", "--port", "8000"]
