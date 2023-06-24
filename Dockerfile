FROM python:3.10.11

EXPOSE 8000

WORKDIR /code

RUN pip install --upgrade pip

RUN pip install poetry

COPY . /code

# чтобы не использовалось виртуальное окружение, все равно это образ
RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi --without test

CMD poetry run gunicorn src.main:app ----workers 2 --worker-class uvicorn.workers.UvicornWorker  --bind=8000:8000 --timeout 420
