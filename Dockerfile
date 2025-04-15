# Use a Conda-based image
FROM continuumio/miniconda3

# Avoid interactive prompts during builds
ENV DEBIAN_FRONTEND=noninteractive
ENV PATH /opt/conda/bin:$PATH

# Create Conda env
RUN conda create -n faceenv python=3.10 -y
SHELL ["conda", "run", "-n", "faceenv", "/bin/bash", "-c"]

# Use conda-forge for access to dlib + face_recognition
RUN conda config --add channels conda-forge && \
    conda install -y cmake && \
    conda install -y dlib && \
    conda install -y face_recognition && \
    conda install -y opencv numpy flask matplotlib

# Install YOLO from pip
RUN pip install ultralytics

# Copy project into container
COPY . /app
WORKDIR /app

# Flask app entry point
EXPOSE 5000
CMD ["conda", "run", "-n", "faceenv", "python", "app.py"]
