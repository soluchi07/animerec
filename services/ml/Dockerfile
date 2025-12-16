# Use a Python base image, preferably one for data science (slim is often enough)
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# --- START: New System Dependency Install ---
# libgomp1 is required by the 'implicit' library for parallel processing (OpenMP)
# libblas3 and liblapack3 are also commonly required dependencies for optimized NumPy/SciPy operations
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libgomp1 \
    libblas3 \
    postgresql-client \
    liblapack3 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*
# --- END: New System Dependency Install ---

# Copy the requirements file into the working directory
COPY requirements.txt .

# Install the necessary Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Entrypoint is not strictly necessary as we are using 'tail -f /dev/null'