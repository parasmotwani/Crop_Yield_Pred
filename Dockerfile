FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY models /app/models
COPY yield_df.csv /app/yield_df.csv
COPY . .
EXPOSE 5050
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5050", "app:app"]
