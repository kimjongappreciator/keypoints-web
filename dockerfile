# Set base image (host OS)
FROM python:3.11.9

EXPOSE 5000/tcp

WORKDIR /app

COPY requirements.txt .

RUN pip3 install --upgrade pip

RUN pip3 install -r requirements.txt

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

COPY my_model3.keras .

COPY templates ./templates

COPY database.py .

COPY app.py .

# Specify the command to run on container start
CMD [ "python", "./app.py" ]