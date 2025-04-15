FROM continuumio/miniconda3

# Create conda environment
RUN conda create -n faceenv python=3.10 -y
SHELL ["conda", "run", "-n", "faceenv", "/bin/bash", "-c"]

# Activate conda and install packages
RUN conda install -y cmake dlib face_recognition opencv numpy flask matplotlib \
    && pip install ultralytics

COPY . /app
WORKDIR /app

EXPOSE 5000
CMD ["conda", "run", "-n", "faceenv", "python", "app.py"]
