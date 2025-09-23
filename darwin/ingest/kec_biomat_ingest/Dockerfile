# Dockerfile for the KEC Biomaterials ingestion job
FROM python:3.11-slim

# Set the working directory
WORKDIR /usr/src/app

# Install git for dependency resolution if needed, and clean up
RUN apt-get update && apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/*

# Copy all the ingestion pipeline files
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create the output directory
RUN mkdir -p build/chroma

# Define the command to run the full pipeline.
# This runs the data ingestion and then the indexing.
CMD ["/bin/bash", "-c", "python cross_ref.py --input seed/dois.txt --outdir build --emit-bibtex --emit-csv --qmap seed/journal_quartiles.csv && python rag_load.py --jsonl build/references.jsonl --db build/chroma --collection kec_biomat_refs"]
