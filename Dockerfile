FROM continuumio/miniconda3

ENV DEBIAN_FRONTEND=noninteractive
ENV PATH /opt/conda/bin:$PATH

# Add missing dependencies for OpenCV and Dlib
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libxext6 \
    libxrender1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Create conda environment
RUN conda create -n faceenv python=3.10 -y
SHELL ["conda", "run", "-n", "faceenv", "/bin/bash", "-c"]

# Use conda-forge for Dlib and Face Recognition
RUN conda config --add channels conda-forge && \
    conda install -y cmake && \
    conda install -y dlib && \
    conda install -y face_recognition && \
    conda install -y opencv numpy flask matplotlib

# Use pip for Ultralytics YOLO
RUN pip install --no-cache-dir ultralytics

# Copy project files
COPY . /app
WORKDIR /app

EXPOSE 5000
CMD ["conda", "run", "-n", "faceenv", "python", "app.py"]
