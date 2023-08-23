FROM python:3.10

WORKDIR /app
RUN pip install pymongo pandas django mongomock
COPY . .

EXPOSE 8000

RUN python manage.py makemigrations && python3 manage.py migrate

#ENTRYPOINT [ "executable" ]
