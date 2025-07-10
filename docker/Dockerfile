# Sử dụng Triton Server image
FROM nvcr.io/nvidia/tritonserver:25.06-py3

# Cài đặt các thư viện Python cần thiết cho client/service
RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir numpy soundfile librosa tritonclient[all]

# Copy code service vào container
WORKDIR /app
COPY . /app

COPY ./requirements.txt /requirements.txt
RUN pip install -r /requirements.txt

# Copy model repository vào container
COPY ./models /models

# Mặc định chạy Triton Server
CMD ["tritonserver", "--model-repository=/models"]
