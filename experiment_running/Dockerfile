FROM python:3.10

WORKDIR /app
RUN pip install pymongo requests pandas mongomock tiktoken aiohttp transformers
COPY . .
#RUN mkdir ./model_tasks
#COPY ../model_tasks/ .
