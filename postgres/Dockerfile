FROM postgres:17

# Download pgvector
RUN apt-get update && \
    apt-get install -y postgresql-17-pgvector && \
    rm -rf /var/lib/apt/lists/*

# Postgres is already running, no CMD start needed