FROM python:3.10-slim

RUN pip install pipenv 

WORKDIR /app 

COPY ["Pipfile", "Pipfile.lock", "./"]

RUN pipenv install --system --deploy

COPY ["web_service.py", "model.model", "dv.pkl", "./" ]

EXPOSE 9696 

ENTRYPOINT [ "gunicorn", "--bind=0.0.0.0:9696", "web_service:app" ]