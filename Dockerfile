FROM python:3.10-slim-bullseye

RUN apt-get update && apt-get install -y \
    build-essential cmake libgl1 libglib2.0-0 libx11-6 ffmpeg && \
    rm -rf /var/lib/apt/lists/*

COPY . /app
WORKDIR /app

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

EXPOSE 5000
CMD ["python", "app.py"]
