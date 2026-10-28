FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=7860

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Utilisateur non privilégié (exigé par certains hébergeurs, bonne pratique sinon)
RUN useradd -m -u 1000 appuser && chown -R appuser /app
USER appuser

EXPOSE 7860
# 1 worker : le modèle est chargé une seule fois ; 4 threads pour les requêtes simultanées
CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:${PORT} --workers 1 --threads 4 --timeout 120 app:app"]
