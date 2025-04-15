FROM continuumio/miniconda3

# Set env for non-interactive conda
ENV DEBIAN_FRONTEND=noninteractive
ENV PATH /opt/conda/bin:$PATH

# Create conda env
RUN conda create -n faceenv python=3.10 -y
SHELL ["conda", "run", "-n", "faceenv", "/bin/bash", "-c"]

# Add conda-forge and install packages step by step
RUN conda config --add channels conda-forge && \
    conda install -y cmake && \
    conda install -y dlib && \
    conda install -y face_recognition && \
    conda install -y opencv numpy flask matplotlib

# Use pip for ultralytics
RUN pip install --no-cache-dir ultralytics

COPY . /app
WORKDIR /app

EXPOSE 5000
CMD ["conda", "run", "-n", "faceenv", "python", "app.py"]
