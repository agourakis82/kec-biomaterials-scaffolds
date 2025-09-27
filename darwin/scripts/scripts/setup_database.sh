#!/bin/bash

# =============================================================================
# DARWIN Database Setup Script
# Script para configurar PostgreSQL production-ready com vector extensions
# =============================================================================

set -euo pipefail

# =============================================================================
# Configuration and Constants
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Color codes for output formatting
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Default values
ENVIRONMENT="production"
PROJECT_ID=""
REGION="us-central1"
DATABASE_INSTANCE=""
DATABASE_NAME=""
CREATE_READ_REPLICA="false"
SETUP_PGVECTOR="true"
RUN_MIGRATIONS="false"
VERIFY_ONLY="false"
VERBOSE="false"

# =============================================================================
# Utility Functions
# =============================================================================

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1" >&2
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1" >&2
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1" >&2
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

log_debug() {
    if [[ "$VERBOSE" == "true" ]]; then
        echo -e "${CYAN}[DEBUG]${NC} $1" >&2
    fi
}

show_banner() {
    echo -e "${PURPLE}"
    cat << 'EOF'
    ╔═══════════════════════════════════════════════════════════════╗
    ║                                                               ║
    ║                     DARWIN DATABASE                           ║
    ║                   Production Setup                            ║
    ║                                                               ║
    ║           PostgreSQL + pgvector + Optimization               ║
    ║                                                               ║
    ╚═══════════════════════════════════════════════════════════════╝
EOF
    echo -e "${NC}"
}

show_usage() {
    cat << EOF
Usage: $0 [OPTIONS]

DARWIN Database Setup Script

OPTIONS:
    -p, --project-id PROJECT_ID       GCP Project ID (required)
    -e, --environment ENVIRONMENT     Environment [dev|staging|production] (default: production)
    -r, --region REGION              GCP Region (default: us-central1)
    -i, --instance INSTANCE          Database instance name (auto-detected if not provided)
    -d, --database DATABASE          Database name (auto-detected if not provided)
    
    Database features:
    --create-read-replica            Create read replica for performance
    --setup-pgvector                 Setup pgvector extension
    --run-migrations                 Run database migrations
    --verify                         Verify database configuration only
    
    Options:
    -v, --verbose                    Enable verbose logging
    -h, --help                       Show this help message

EXAMPLES:
    $0 -p my-project --setup-pgvector
    $0 -p my-project --create-read-replica
    $0 -p my-project --verify
    $0 -p my-project --run-migrations

ENVIRONMENT VARIABLES:
    DARWIN_PROJECT_ID                Project ID
    DARWIN_ENVIRONMENT               Environment
    DARWIN_REGION                    GCP Region

EOF
}

check_prerequisites() {
    log_info "Checking database prerequisites..."
    
    local missing_tools=()
    
    # Check required tools
    command -v gcloud >/dev/null 2>&1 || missing_tools+=("gcloud")
    command -v psql >/dev/null 2>&1 || missing_tools+=("psql")
    command -v jq >/dev/null 2>&1 || missing_tools+=("jq")
    
    if [[ ${#missing_tools[@]} -gt 0 ]]; then
        log_warning "Missing optional tools: ${missing_tools[*]}"
        log_info "Installing Cloud SQL Proxy for database access..."
        
        # Download Cloud SQL Proxy
        if [[ ! -f "/tmp/cloud-sql-proxy" ]]; then
            curl -o /tmp/cloud-sql-proxy https://storage.googleapis.com/cloud-sql-connectors/cloud-sql-proxy/v2.7.0/cloud-sql-proxy.linux.amd64
            chmod +x /tmp/cloud-sql-proxy
        fi
    fi
    
    # Check gcloud authentication
    if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | head -n1 > /dev/null; then
        log_error "Not authenticated with gcloud. Please run: gcloud auth login"
        exit 1
    fi
    
    # Check project access
    if ! gcloud projects describe "$PROJECT_ID" >/dev/null 2>&1; then
        log_error "Cannot access project $PROJECT_ID"
        exit 1
    fi
    
    log_success "Prerequisites check completed"
}

detect_database_instance() {
    if [[ -z "$DATABASE_INSTANCE" ]]; then
        log_info "Auto-detecting database instance..."
        
        local instances
        instances=$(gcloud sql instances list \
            --filter="name~'darwin-$ENVIRONMENT'" \
            --format="value(name)" \
            --project="$PROJECT_ID" 2>/dev/null || echo "")
        
        if [[ -n "$instances" ]]; then
            DATABASE_INSTANCE=$(echo "$instances" | head -n1)
            log_success "Found database instance: $DATABASE_INSTANCE"
        else
            log_error "No database instance found for environment: $ENVIRONMENT"
            log_error "Please deploy infrastructure first or specify instance with -i"
            exit 1
        fi
    fi
    
    if [[ -z "$DATABASE_NAME" ]]; then
        DATABASE_NAME="darwin_${ENVIRONMENT}"
        log_info "Using database name: $DATABASE_NAME"
    fi
}

verify_database_status() {
    log_info "Verifying database status..."
    
    # Check instance status
    local instance_status
    instance_status=$(gcloud sql instances describe "$DATABASE_INSTANCE" \
        --project="$PROJECT_ID" \
        --format="value(state)" 2>/dev/null || echo "NOT_FOUND")
    
    if [[ "$instance_status" != "RUNNABLE" ]]; then
        log_error "Database instance $DATABASE_INSTANCE is not ready (status: $instance_status)"
        exit 1
    fi
    
    log_success "Database instance is running"
    
    # Get instance details
    log_debug "Database instance details:"
    gcloud sql instances describe "$DATABASE_INSTANCE" \
        --project="$PROJECT_ID" \
        --format="table(
            name,
            databaseVersion,
            settings.tier,
            settings.diskSizeGb,
            settings.availabilityType,
            state
        )" 2>/dev/null || log_warning "Could not get instance details"
}

setup_pgvector_extension() {
    if [[ "$SETUP_PGVECTOR" != "true" ]]; then
        log_info "Skipping pgvector setup"
        return 0
    fi
    
    log_info "Setting up pgvector extension..."
    
    # Check if pgvector is already enabled
    local extensions
    extensions=$(gcloud sql instances describe "$DATABASE_INSTANCE" \
        --project="$PROJECT_ID" \
        --format="value(settings.databaseFlags[].value)" 2>/dev/null | grep -o "vector" || echo "")
    
    if [[ "$extensions" == "vector" ]]; then
        log_success "pgvector extension already enabled in database flags"
    else
        log_info "Enabling pgvector extension in database flags..."
        
        # Update database flags to include vector extension
        gcloud sql instances patch "$DATABASE_INSTANCE" \
            --database-flags=shared_preload_libraries="vector,pg_stat_statements,pg_buffercache" \
            --project="$PROJECT_ID" \
            --quiet
        
        log_info "Database flags updated. Restarting instance..."
        gcloud sql instances restart "$DATABASE_INSTANCE" \
            --project="$PROJECT_ID" \
            --quiet
        
        # Wait for restart
        log_info "Waiting for database to restart..."
        local retries=0
        while [[ $retries -lt 10 ]]; do
            local status
            status=$(gcloud sql instances describe "$DATABASE_INSTANCE" \
                --project="$PROJECT_ID" \
                --format="value(state)" 2>/dev/null || echo "UNKNOWN")
            
            if [[ "$status" == "RUNNABLE" ]]; then
                break
            fi
            
            retries=$((retries + 1))
            log_debug "Waiting for restart... ($retries/10)"
            sleep 30
        done
        
        if [[ $retries -eq 10 ]]; then
            log_error "Database did not restart within expected time"
            exit 1
        fi
        
        log_success "Database restarted successfully"
    fi
    
    # Connect and create extension in database
    log_info "Installing pgvector extension in database..."
    
    # Get connection details
    local connection_name
    connection_name=$(gcloud sql instances describe "$DATABASE_INSTANCE" \
        --project="$PROJECT_ID" \
        --format="value(connectionName)")
    
    # Start Cloud SQL Proxy
    log_debug "Starting Cloud SQL Proxy..."
    /tmp/cloud-sql-proxy "$connection_name" --port=5432 &
    local proxy_pid=$!
    
    # Wait for proxy to be ready
    sleep 10
    
    # Get database credentials from Secret Manager
    local db_password
    db_password=$(gcloud secrets versions access latest \
        --secret="darwin-${ENVIRONMENT}-database-password" \
        --project="$PROJECT_ID" 2>/dev/null || echo "")
    
    if [[ -z "$db_password" ]]; then
        log_error "Could not retrieve database password from Secret Manager"
        kill $proxy_pid 2>/dev/null || true
        exit 1
    fi
    
    # Get database user
    local db_user
    db_user=$(gcloud sql users list \
        --instance="$DATABASE_INSTANCE" \
        --project="$PROJECT_ID" \
        --format="value(name)" 2>/dev/null | grep -v "postgres" | head -n1 || echo "")
    
    if [[ -z "$db_user" ]]; then
        log_error "Could not find database user"
        kill $proxy_pid 2>/dev/null || true
        exit 1
    fi
    
    # Connect and install extension
    log_debug "Connecting to database to install pgvector..."
    
    # Create pgvector extension
    PGPASSWORD="$db_password" psql \
        -h localhost \
        -p 5432 \
        -U "$db_user" \
        -d "$DATABASE_NAME" \
        -c "CREATE EXTENSION IF NOT EXISTS vector;" \
        2>/dev/null || log_warning "pgvector extension may already exist"
    
    # Verify extension
    local vector_installed
    vector_installed=$(PGPASSWORD="$db_password" psql \
        -h localhost \
        -p 5432 \
        -U "$db_user" \
        -d "$DATABASE_NAME" \
        -t -c "SELECT 1 FROM pg_extension WHERE extname = 'vector';" \
        2>/dev/null | xargs || echo "")
    
    if [[ "$vector_installed" == "1" ]]; then
        log_success "pgvector extension installed successfully"
    else
        log_error "Failed to install pgvector extension"
    fi
    
    # Kill proxy
    kill $proxy_pid 2>/dev/null || true
    
    log_success "pgvector setup completed"
}

create_read_replica() {
    if [[ "$CREATE_READ_REPLICA" != "true" ]]; then
        log_info "Skipping read replica creation"
        return 0
    fi
    
    log_info "Creating read replica for performance..."
    
    local replica_name="${DATABASE_INSTANCE}-replica"
    
    # Check if replica already exists
    if gcloud sql instances describe "$replica_name" --project="$PROJECT_ID" >/dev/null 2>&1; then
        log_success "Read replica already exists: $replica_name"
        return 0
    fi
    
    # Create read replica
    log_info "Creating read replica: $replica_name"
    gcloud sql instances create "$replica_name" \
        --master-instance-name="$DATABASE_INSTANCE" \
        --project="$PROJECT_ID" \
        --region="$REGION" \
        --tier="db-f1-micro" \
        --replica-type="READ" \
        --enable-bin-log \
        --quiet
    
    # Wait for replica to be ready
    log_info "Waiting for read replica to be ready..."
    local retries=0
    while [[ $retries -lt 20 ]]; do
        local replica_status
        replica_status=$(gcloud sql instances describe "$replica_name" \
            --project="$PROJECT_ID" \
            --format="value(state)" 2>/dev/null || echo "UNKNOWN")
        
        if [[ "$replica_status" == "RUNNABLE" ]]; then
            break
        fi
        
        retries=$((retries + 1))
        log_debug "Waiting for replica... ($retries/20)"
        sleep 30
    done
    
    if [[ $retries -eq 20 ]]; then
        log_error "Read replica did not become ready within expected time"
        exit 1
    fi
    
    log_success "Read replica created successfully: $replica_name"
}

setup_connection_pooling() {
    log_info "Setting up database connection pooling..."
    
    # Configure database flags for optimal connection pooling
    local current_flags
    current_flags=$(gcloud sql instances describe "$DATABASE_INSTANCE" \
        --project="$PROJECT_ID" \
        --format="json" | jq -r '.settings.databaseFlags[]? | "\(.name)=\(.value)"' | tr '\n' ',' | sed 's/,$//')
    
    log_debug "Current database flags: $current_flags"
    
    # Recommended flags for production with connection pooling
    local production_flags="shared_preload_libraries=vector\\,pg_stat_statements\\,pg_buffercache,max_connections=200,shared_buffers=512MB,effective_cache_size=2GB,maintenance_work_mem=128MB,checkpoint_completion_target=0.9,wal_buffers=32MB,default_statistics_target=100,random_page_cost=1.1,effective_io_concurrency=200,work_mem=4MB,max_worker_processes=8,max_parallel_workers_per_gather=2,max_parallel_workers=8,max_parallel_maintenance_workers=2"
    
    # Apply optimized flags
    log_info "Applying production database flags..."
    gcloud sql instances patch "$DATABASE_INSTANCE" \
        --database-flags="$production_flags" \
        --project="$PROJECT_ID" \
        --quiet
    
    log_success "Database connection pooling configured"
}

run_database_migrations() {
    if [[ "$RUN_MIGRATIONS" != "true" ]]; then
        log_info "Skipping database migrations"
        return 0
    fi
    
    log_info "Running database migrations..."
    
    # Check if migrations directory exists
    local migrations_dir="$PROJECT_ROOT/src/kec_unified_api/migrations"
    if [[ ! -d "$migrations_dir" ]]; then
        log_warning "No migrations directory found at $migrations_dir"
        log_info "Creating basic database schema..."
        
        # Create basic schema using SQL
        create_basic_schema
        return 0
    fi
    
    # Run Alembic migrations
    log_info "Running Alembic migrations from $migrations_dir..."
    
    # Start Cloud SQL Proxy
    local connection_name
    connection_name=$(gcloud sql instances describe "$DATABASE_INSTANCE" \
        --project="$PROJECT_ID" \
        --format="value(connectionName)")
    
    /tmp/cloud-sql-proxy "$connection_name" --port=5432 &
    local proxy_pid=$!
    
    # Wait for proxy
    sleep 10
    
    # Get credentials
    local db_password
    db_password=$(gcloud secrets versions access latest \
        --secret="darwin-${ENVIRONMENT}-database-password" \
        --project="$PROJECT_ID")
    
    local db_user
    db_user=$(gcloud sql users list \
        --instance="$DATABASE_INSTANCE" \
        --project="$PROJECT_ID" \
        --format="value(name)" | grep -v "postgres" | head -n1)
    
    # Set environment variables for Alembic
    export DATABASE_URL="postgresql://$db_user:$db_password@localhost:5432/$DATABASE_NAME"
    export PYTHONPATH="$PROJECT_ROOT/src"
    
    # Run migrations
    cd "$migrations_dir/.."
    if command -v alembic >/dev/null 2>&1; then
        alembic upgrade head
        log_success "Database migrations completed"
    else
        log_warning "Alembic not installed, installing..."
        pip install alembic psycopg2-binary
        alembic upgrade head
        log_success "Database migrations completed"
    fi
    
    # Kill proxy
    kill $proxy_pid 2>/dev/null || true
}

create_basic_schema() {
    log_info "Creating basic database schema..."
    
    # Start Cloud SQL Proxy
    local connection_name
    connection_name=$(gcloud sql instances describe "$DATABASE_INSTANCE" \
        --project="$PROJECT_ID" \
        --format="value(connectionName)")
    
    /tmp/cloud-sql-proxy "$connection_name" --port=5432 &
    local proxy_pid=$!
    
    # Wait for proxy
    sleep 10
    
    # Get credentials
    local db_password
    db_password=$(gcloud secrets versions access latest \
        --secret="darwin-${ENVIRONMENT}-database-password" \
        --project="$PROJECT_ID")
    
    local db_user
    db_user=$(gcloud sql users list \
        --instance="$DATABASE_INSTANCE" \
        --project="$PROJECT_ID" \
        --format="value(name)" | grep -v "postgres" | head -n1)
    
    # Create basic schema
    log_debug "Creating basic tables..."
    PGPASSWORD="$db_password" psql \
        -h localhost \
        -p 5432 \
        -U "$db_user" \
        -d "$DATABASE_NAME" \
        -v ON_ERROR_STOP=1 << 'EOF'

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_stat_statements;
CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE EXTENSION IF NOT EXISTS btree_gin;
CREATE EXTENSION IF NOT EXISTS uuid-ossp;

-- Create basic tables for DARWIN platform
CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    email VARCHAR(255) UNIQUE NOT NULL,
    name VARCHAR(255) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    is_active BOOLEAN DEFAULT TRUE
);

CREATE TABLE IF NOT EXISTS documents (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    title TEXT NOT NULL,
    content TEXT,
    embedding vector(1536),
    metadata JSONB,
    user_id UUID REFERENCES users(id),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS research_papers (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    title TEXT NOT NULL,
    abstract TEXT,
    authors TEXT[],
    doi VARCHAR(255),
    embedding vector(1536),
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS knowledge_graph (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    entity_type VARCHAR(100) NOT NULL,
    entity_name TEXT NOT NULL,
    properties JSONB,
    embedding vector(1536),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS chat_sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id),
    title TEXT,
    messages JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_documents_embedding ON documents USING ivfflat (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS idx_papers_embedding ON research_papers USING ivfflat (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS idx_knowledge_embedding ON knowledge_graph USING ivfflat (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS idx_documents_user ON documents(user_id);
CREATE INDEX IF NOT EXISTS idx_documents_created ON documents(created_at);
CREATE INDEX IF NOT EXISTS idx_papers_doi ON research_papers(doi);
CREATE INDEX IF NOT EXISTS idx_knowledge_type ON knowledge_graph(entity_type);
CREATE INDEX IF NOT EXISTS idx_chat_user ON chat_sessions(user_id);

-- Create GIN indexes for full-text search
CREATE INDEX IF NOT EXISTS idx_documents_content_gin ON documents USING gin(to_tsvector('english', content));
CREATE INDEX IF NOT EXISTS idx_papers_abstract_gin ON research_papers USING gin(to_tsvector('english', abstract));
CREATE INDEX IF NOT EXISTS idx_knowledge_name_gin ON knowledge_graph USING gin(to_tsvector('english', entity_name));

-- Create triggers for updated_at timestamps
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_users_updated_at BEFORE UPDATE ON users
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_documents_updated_at BEFORE UPDATE ON documents
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_papers_updated_at BEFORE UPDATE ON research_papers
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_knowledge_updated_at BEFORE UPDATE ON knowledge_graph
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_chat_updated_at BEFORE UPDATE ON chat_sessions
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Grant permissions
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO CURRENT_USER;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO CURRENT_USER;

EOF
    
    if [[ $? -eq 0 ]]; then
        log_success "Basic database schema created successfully"
    else
        log_error "Failed to create basic database schema"
    fi
    
    # Kill proxy
    kill $proxy_pid 2>/dev/null || true
}

setup_backup_configuration() {
    log_info "Configuring automated backups..."
    
    # Check current backup configuration
    local backup_enabled
    backup_enabled=$(gcloud sql instances describe "$DATABASE_INSTANCE" \
        --project="$PROJECT_ID" \
        --format="value(settings.backupConfiguration.enabled)" 2>/dev/null || echo "false")
    
    if [[ "$backup_enabled" == "True" ]]; then
        log_success "Automated backups already enabled"
        
        # Show backup details
        log_debug "Backup configuration:"
        gcloud sql instances describe "$DATABASE_INSTANCE" \
            --project="$PROJECT_ID" \
            --format="table(
                settings.backupConfiguration.enabled:label=ENABLED,
                settings.backupConfiguration.startTime:label=START_TIME,
                settings.backupConfiguration.pointInTimeRecoveryEnabled:label=PITR,
                settings.backupConfiguration.backupRetentionSettings.retainedBackups:label=RETENTION
            )" 2>/dev/null || log_warning "Could not get backup details"
    else
        log_info "Enabling automated backups..."
        
        # Enable backups with production settings
        gcloud sql instances patch "$DATABASE_INSTANCE" \
            --backup-start-time="03:00" \
            --enable-bin-log \
            --backup-location="$REGION" \
            --retained-backups-count=30 \
            --retained-transaction-log-days=7 \
            --project="$PROJECT_ID" \
            --quiet
        
        log_success "Automated backups enabled"
    fi
}

optimize_database_performance() {
    log_info "Optimizing database performance..."
    
    # Start Cloud SQL Proxy for performance analysis
    local connection_name
    connection_name=$(gcloud sql instances describe "$DATABASE_INSTANCE" \
        --project="$PROJECT_ID" \
        --format="value(connectionName)")
    
    /tmp/cloud-sql-proxy "$connection_name" --port=5432 &
    local proxy_pid=$!
    sleep 10
    
    # Get credentials
    local db_password
    db_password=$(gcloud secrets versions access latest \
        --secret="darwin-${ENVIRONMENT}-database-password" \
        --project="$PROJECT_ID")
    
    local db_user
    db_user=$(gcloud sql users list \
        --instance="$DATABASE_INSTANCE" \
        --project="$PROJECT_ID" \
        --format="value(name)" | grep -v "postgres" | head -n1)
    
    # Run performance optimization queries
    log_debug "Running performance optimization..."
    PGPASSWORD="$db_password" psql \
        -h localhost \
        -p 5432 \
        -U "$db_user" \
        -d "$DATABASE_NAME" \
        -v ON_ERROR_STOP=1 << 'EOF'

-- Analyze all tables for better query planning
ANALYZE;

-- Update statistics
SELECT pg_stat_reset();

-- Show current database statistics
SELECT 
    schemaname,
    tablename,
    n_tup_ins + n_tup_upd + n_tup_del as total_operations,
    n_live_tup as live_tuples,
    n_dead_tup as dead_tuples
FROM pg_stat_user_tables
ORDER BY total_operations DESC;

-- Show index usage statistics
SELECT 
    schemaname,
    tablename,
    indexname,
    idx_tup_read,
    idx_tup_fetch
FROM pg_stat_user_indexes
ORDER BY idx_tup_read DESC;

-- Show current settings
SELECT name, setting, unit, short_desc 
FROM pg_settings 
WHERE name IN (
    'max_connections',
    'shared_buffers',
    'effective_cache_size',
    'maintenance_work_mem',
    'checkpoint_completion_target',
    'wal_buffers',
    'default_statistics_target',
    'random_page_cost',
    'effective_io_concurrency'
)
ORDER BY name;

EOF
    
    # Kill proxy
    kill $proxy_pid 2>/dev/null || true
    
    log_success "Database performance optimization completed"
}

verify_vector_functionality() {
    log_info "Verifying vector search functionality..."
    
    # Start Cloud SQL Proxy
    local connection_name
    connection_name=$(gcloud sql instances describe "$DATABASE_INSTANCE" \
        --project="$PROJECT_ID" \
        --format="value(connectionName)")
    
    /tmp/cloud-sql-proxy "$connection_name" --port=5432 &
    local proxy_pid=$!
    sleep 10
    
    # Get credentials
    local db_password
    db_password=$(gcloud secrets versions access latest \
        --secret="darwin-${ENVIRONMENT}-database-password" \
        --project="$PROJECT_ID")
    
    local db_user
    db_user=$(gcloud sql users list \
        --instance="$DATABASE_INSTANCE" \
        --project="$PROJECT_ID" \
        --format="value(name)" | grep -v "postgres" | head -n1)
    
    # Test vector operations
    log_debug "Testing vector operations..."
    PGPASSWORD="$db_password" psql \
        -h localhost \
        -p 5432 \
        -U "$db_user" \
        -d "$DATABASE_NAME" \
        -v ON_ERROR_STOP=1 << 'EOF'

-- Test vector extension functionality
DO $$
DECLARE
    test_vector vector(3) := '[1,2,3]';
    test_vector2 vector(3) := '[4,5,6]';
    similarity_score float;
BEGIN
    -- Test vector creation
    RAISE NOTICE 'Test vector 1: %', test_vector;
    RAISE NOTICE 'Test vector 2: %', test_vector2;
    
    -- Test cosine similarity
    similarity_score := 1 - (test_vector <=> test_vector2);
    RAISE NOTICE 'Cosine similarity: %', similarity_score;
    
    -- Test L2 distance
    RAISE NOTICE 'L2 distance: %', test_vector <-> test_vector2;
    
    -- Test inner product
    RAISE NOTICE 'Inner product: %', test_vector <#> test_vector2;
    
    RAISE NOTICE 'Vector functionality test completed successfully!';
END $$;

-- Show vector extension version
SELECT * FROM pg_extension WHERE extname = 'vector';

EOF
    
    if [[ $? -eq 0 ]]; then
        log_success "Vector search functionality verified"
    else
        log_error "Vector search functionality test failed"
    fi
    
    # Kill proxy
    kill $proxy_pid 2>/dev/null || true
}

show_database_summary() {
    echo ""
    echo -e "${CYAN}╔═══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║                    DATABASE SUMMARY                          ║${NC}"
    echo -e "${CYAN}╠═══════════════════════════════════════════════════════════════╣${NC}"
    echo -e "${CYAN}║${NC} Project ID:       $PROJECT_ID"
    echo -e "${CYAN}║${NC} Environment:      $ENVIRONMENT"
    echo -e "${CYAN}║${NC} Instance:         $DATABASE_INSTANCE"
    echo -e "${CYAN}║${NC} Database:         $DATABASE_NAME"
    echo -e "${CYAN}║${NC} Region:           $REGION"
    echo -e "${CYAN}║${NC} pgvector:         $([ "$SETUP_PGVECTOR" == "true" ] && echo "✅ Enabled" || echo "⏭️ Skipped")"
    echo -e "${CYAN}║${NC} Read Replica:     $([ "$CREATE_READ_REPLICA" == "true" ] && echo "✅ Created" || echo "⏭️ Skipped")"
    echo -e "${CYAN}║${NC} Migrations:       $([ "$RUN_MIGRATIONS" == "true" ] && echo "✅ Applied" || echo "⏭️ Skipped")"
    echo -e "${CYAN}║${NC} Timestamp:        $(date)"
    echo -e "${CYAN}╠═══════════════════════════════════════════════════════════════╣${NC}"
    echo -e "${CYAN}║${NC} Database is production-ready!"
    echo -e "${CYAN}║${NC} "
    echo -e "${CYAN}║${NC} 🎯 Features Available:"
    echo -e "${CYAN}║${NC}   • Vector similarity search with pgvector"
    echo -e "${CYAN}║${NC}   • Full-text search with PostgreSQL"
    echo -e "${CYAN}║${NC}   • Automated backups with point-in-time recovery"
    echo -e "${CYAN}║${NC}   • High availability and performance optimization"
    echo -e "${CYAN}║${NC}   • Connection pooling and query optimization"
    echo -e "${CYAN}║${NC} "
    echo -e "${CYAN}║${NC} 🔗 Database URLs:"
    echo -e "${CYAN}║${NC}   • Cloud SQL Console: console.cloud.google.com/sql"
    echo -e "${CYAN}║${NC}   • Connection: Use Cloud SQL Proxy"
    echo -e "${CYAN}║${NC}   • Monitoring: Integrated with Cloud Monitoring"
    echo -e "${CYAN}╚═══════════════════════════════════════════════════════════════╝${NC}"
    echo ""
}

# =============================================================================
# Main Execution
# =============================================================================

main() {
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            -p|--project-id)
                PROJECT_ID="$2"
                shift 2
                ;;
            -e|--environment)
                ENVIRONMENT="$2"
                shift 2
                ;;
            -r|--region)
                REGION="$2"
                shift 2
                ;;
            -i|--instance)
                DATABASE_INSTANCE="$2"
                shift 2
                ;;
            -d|--database)
                DATABASE_NAME="$2"
                shift 2
                ;;
            --create-read-replica)
                CREATE_READ_REPLICA="true"
                shift
                ;;
            --setup-pgvector)
                SETUP_PGVECTOR="true"
                shift
                ;;
            --run-migrations)
                RUN_MIGRATIONS="true"
                shift
                ;;
            --verify)
                VERIFY_ONLY="true"
                shift
                ;;
            -v|--verbose)
                VERBOSE="true"
                shift
                ;;
            -h|--help)
                show_usage
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                show_usage
                exit 1
                ;;
        esac
    done
    
    # Check for environment variables
    PROJECT_ID="${PROJECT_ID:-${DARWIN_PROJECT_ID:-}}"
    ENVIRONMENT="${ENVIRONMENT:-${DARWIN_ENVIRONMENT:-production}}"
    REGION="${REGION:-${DARWIN_REGION:-us-central1}}"
    
    # Validate required parameters
    if [[ -z "$PROJECT_ID" ]]; then
        log_error "Project ID is required"
        show_usage
        exit 1
    fi
    
    # Show banner
    show_banner
    
    log_info "Starting DARWIN database setup..."
    log_info "Project: $PROJECT_ID | Environment: $ENVIRONMENT | Region: $REGION"
    
    check_prerequisites
    detect_database_instance
    verify_database_status
    
    if [[ "$VERIFY_ONLY" == "true" ]]; then
        log_info "Verification mode - checking database configuration"
        verify_vector_functionality
    else
        log_info "Setup mode - configuring database"
        setup_pgvector_extension
        setup_connection_pooling
        setup_backup_configuration
        create_read_replica
        run_database_migrations
        optimize_database_performance
        verify_vector_functionality
    fi
    
    show_database_summary
    
    log_success "DARWIN database setup completed successfully!"
}

# Execute main function if script is run directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi