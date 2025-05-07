FROM python:3.9-slim

WORKDIR /home/app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY model/ model/
COPY api.py .
COPY speech_predict.py .

EXPOSE 10000

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "10000"]
