FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .

# Install CPU-only PyTorch
RUN pip install --no-cache-dir \
    torch==2.7.1 torchvision==0.22.1 \
    --index-url https://download.pytorch.org/whl/cpu

# Install the remaining dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

RUN pip install --no-cache-dir PyYAML==6.0.2
COPY src/ ./src/
COPY data/ ./data/
COPY plot.py .
COPY experiment_runner.py .

CMD ["python", "src/server/fedavg.py", "--help"]