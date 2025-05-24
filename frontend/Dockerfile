FROM python:3.9

# Set working directory
WORKDIR /app

# Copy all source code
COPY . /app

# Install system dependencies (e.g., sqlite3)
RUN apt-get update && apt-get install -y sqlite3 && apt-get clean

# Install Python dependencies
RUN pip install --no-cache-dir \
    streamlit \
    pandas \
    sqlalchemy \
    matplotlib \
    joblib \
    requests \
    scikit-learn \
    openai>=1.0.0

# Run the Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.enableCORS=false"]

