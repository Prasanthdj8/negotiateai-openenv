FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY *.py .
COPY openenv.yaml .

# HuggingFace Spaces runs on port 7860
EXPOSE 7860

# Environment defaults (override in HF Spaces secrets)
ENV OPENAI_BASE_URL=""
ENV OPENAI_API_KEY=""
ENV SUPPLIER_LLM_MODEL="gpt-4o-mini"
ENV RIVAL_LLM_MODEL="gpt-4o-mini"
ENV INFERENCE_MODEL="gpt-4o-mini"
ENV ENV_URL="http://localhost:7860"

CMD ["python", "main.py"]
