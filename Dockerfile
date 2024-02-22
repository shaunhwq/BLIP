FROM python:3.8

# Install OpenCV dependencies
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

ADD requirements.txt .
RUN pip install -r requirements.txt

RUN pip install python-multipart==0.0.9 fastapi==0.109.2 uvicorn==0.27.1
RUN pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1
RUN pip install opencv-python==4.9.0.80

# Copy server related files to docker container
COPY ./server ./server
COPY ./models ./models
ADD ./utils ./utils
COPY ./configs ./configs
ADD run.py .
EXPOSE 4050
CMD [ "python3", "run.py"]