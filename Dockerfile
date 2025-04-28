FROM python:3.12-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml requirements.txt ./
COPY conf ./conf
COPY src ./src

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir "kedro-datasets[pandas.CSVDataset,pandas.ParquetDataset,matplotlib.MatplotlibWriter,json.JSONDataset]>=1.0"

# Expose port for Kedro Viz
EXPOSE 4141

# Command to run Kedro
CMD ["kedro", "run"]
