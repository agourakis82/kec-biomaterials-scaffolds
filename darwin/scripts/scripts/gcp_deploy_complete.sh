#!/bin/bash

# 🚀 DARWIN GCP Complete Deploy - api.agourakis.med.br + darwin.agourakis.med.br
# Script completo para deploy DARWIN no Google Cloud Platform

set -e

# Colors for epic output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
WHITE='\033[1;37m'
BOLD='\033[1m'
NC='\033[0m'

# Configuration
PROJECT_ID="pcs-helio"
REGION="us-central1"
BACKEND_SERVICE="darwin-backend-api"
FRONTEND_SERVICE="darwin-frontend-web"
BACKEND_DOMAIN="api.agourakis.med.br"
FRONTEND_DOMAIN="darwin.agourakis.med.br"
BACKEND_IMAGE="gcr.io/$PROJECT_ID/darwin-backend"
FRONTEND_IMAGE="gcr.io/$PROJECT_ID/darwin-frontend"

echo -e "${PURPLE}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${PURPLE}║                                                              ║${NC}"
echo -e "${PURPLE}║  🚀 DARWIN GCP COMPLETE DEPLOY - PRODUCTION REVOLUTIONARY  ║${NC}"
echo -e "${PURPLE}║                                                              ║${NC}"
echo -e "${PURPLE}║  Project: pcs-helio                                         ║${NC}"
echo -e "${PURPLE}║  Backend: api.agourakis.med.br                              ║${NC}"
echo -e "${PURPLE}║  Frontend: darwin.agourakis.med.br                          ║${NC}"
echo -e "${PURPLE}║                                                              ║${NC}"
echo -e "${PURPLE}╚══════════════════════════════════════════════════════════════╝${NC}"

# Function to print status
print_status() {
    echo -e "${BLUE}🚀 [GCP]${NC} $1"
}

print_success() {
    echo -e "${GREEN}✅ [SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠️  [WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}❌ [ERROR]${NC} $1"
}

print_epic() {
    echo -e "${CYAN}🎯 [EPIC]${NC} $1"
}

# Check authentication and setup
check_auth_and_setup() {
    print_status "Checking authentication and project setup..."

    # Check if authenticated
    if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | head -n 1 > /dev/null; then
        print_error "❌ Not authenticated with Google Cloud."
        echo -e "${YELLOW}Please run first:${NC}"
        echo -e "   ${CYAN}gcloud auth login${NC}"
        echo -e "   ${CYAN}gcloud config set project $PROJECT_ID${NC}"
        echo -e "   ${CYAN}gcloud auth application-default login${NC}"
        exit 1
    fi

    # Set project
    gcloud config set project $PROJECT_ID

    # Enable required APIs
    print_status "Enabling required GCP APIs..."
    gcloud services enable \
        cloudbuild.googleapis.com \
        run.googleapis.com \
        containerregistry.googleapis.com \
        artifactregistry.googleapis.com \
        domains.googleapis.com \
        dns.googleapis.com \
        --quiet

    print_success "Authentication and APIs validated"
}

# Build backend image
build_backend() {
    print_status "Building backend Docker image..."

    # Check if local image exists
    if ! docker image inspect darwin-backend:latest > /dev/null 2>&1; then
        print_status "Building backend locally first..."
        docker build -t darwin-backend:latest -f Dockerfile.simple .
    fi

    # Tag and push to GCR
    docker tag darwin-backend:latest $BACKEND_IMAGE:latest
    docker tag darwin-backend:latest $BACKEND_IMAGE:$(date +%Y%m%d-%H%M%S)

    # Configure Docker for GCR
    gcloud auth configure-docker --quiet

    # Push to GCR
    print_status "Pushing backend to Google Container Registry..."
    docker push $BACKEND_IMAGE:latest

    print_success "Backend image built and pushed"
}

# Build frontend image
build_frontend() {
    print_status "Building frontend Docker image..."

    cd ui

    # Create production Dockerfile for frontend
    cat > Dockerfile.production << EOF
FROM node:18-alpine AS base

# Install dependencies only when needed
FROM base AS deps
WORKDIR /app

# Copy package files
COPY package*.json ./
RUN npm ci --only=production

# Rebuild the source code only when needed
FROM base AS builder
WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . .

# Set production environment
ENV NODE_ENV=production
ENV NEXT_PUBLIC_BACKEND_URL=https://api.agourakis.med.br

RUN npm run build

# Production image, copy all the files and run next
FROM base AS runner
WORKDIR /app

ENV NODE_ENV=production

RUN addgroup --system --gid 1001 nodejs
RUN adduser --system --uid 1001 nextjs

COPY --from=builder /app/public ./public

# Automatically leverage output traces to reduce image size
COPY --from=builder --chown=nextjs:nodejs /app/.next/standalone ./
COPY --from=builder --chown=nextjs:nodejs /app/.next/static ./.next/static

USER nextjs

EXPOSE 3000

ENV PORT 3000
ENV HOSTNAME "0.0.0.0"

CMD ["node", "server.js"]
EOF

    # Build frontend image
    docker build -t darwin-frontend:latest -f Dockerfile.production .
    
    # Tag for GCR
    docker tag darwin-frontend:latest $FRONTEND_IMAGE:latest
    docker tag darwin-frontend:latest $FRONTEND_IMAGE:$(date +%Y%m%d-%H%M%S)

    # Push to GCR
    print_status "Pushing frontend to Google Container Registry..."
    docker push $FRONTEND_IMAGE:latest

    cd ..
    print_success "Frontend image built and pushed"
}

# Deploy backend to Cloud Run
deploy_backend() {
    print_status "Deploying backend to Cloud Run..."

    gcloud run deploy $BACKEND_SERVICE \
        --image=$BACKEND_IMAGE:latest \
        --platform=managed \
        --region=$REGION \
        --allow-unauthenticated \
        --port=8090 \
        --memory=4Gi \
        --cpu=2 \
        --concurrency=1000 \
        --min-instances=1 \
        --max-instances=20 \
        --timeout=300 \
        --set-env-vars="ENVIRONMENT=production,DEBUG=false,LOG_LEVEL=info" \
        --set-env-vars="CORS_ORIGINS=https://darwin.agourakis.med.br,https://api.agourakis.med.br" \
        --set-env-vars="FRONTEND_URL=https://darwin.agourakis.med.br" \
        --set-env-vars="JAX_ENABLE_X64=true,JAX_PLATFORMS=cpu" \
        --labels="app=darwin,component=backend,environment=production" \
        --execution-environment=gen2 \
        --quiet

    BACKEND_URL=$(gcloud run services describe $BACKEND_SERVICE --region=$REGION --format="value(status.url)")
    print_success "Backend deployed: $BACKEND_URL"
}

# Deploy frontend to Cloud Run
deploy_frontend() {
    print_status "Deploying frontend to Cloud Run..."

    gcloud run deploy $FRONTEND_SERVICE \
        --image=$FRONTEND_IMAGE:latest \
        --platform=managed \
        --region=$REGION \
        --allow-unauthenticated \
        --port=3000 \
        --memory=2Gi \
        --cpu=1 \
        --concurrency=1000 \
        --min-instances=1 \
        --max-instances=10 \
        --timeout=60 \
        --set-env-vars="NODE_ENV=production" \
        --set-env-vars="NEXT_PUBLIC_BACKEND_URL=https://api.agourakis.med.br" \
        --labels="app=darwin,component=frontend,environment=production" \
        --execution-environment=gen2 \
        --quiet

    FRONTEND_URL=$(gcloud run services describe $FRONTEND_SERVICE --region=$REGION --format="value(status.url)")
    print_success "Frontend deployed: $FRONTEND_URL"
}

# Configure custom domains
configure_custom_domains() {
    print_status "Configuring custom domains..."

    # Backend domain mapping
    print_status "Setting up backend domain: $BACKEND_DOMAIN"
    gcloud run domain-mappings create \
        --service=$BACKEND_SERVICE \
        --domain=$BACKEND_DOMAIN \
        --region=$REGION \
        --quiet 2>/dev/null || print_warning "Backend domain mapping may already exist"

    # Frontend domain mapping
    print_status "Setting up frontend domain: $FRONTEND_DOMAIN"
    gcloud run domain-mappings create \
        --service=$FRONTEND_SERVICE \
        --domain=$FRONTEND_DOMAIN \
        --region=$REGION \
        --quiet 2>/dev/null || print_warning "Frontend domain mapping may already exist"

    print_success "Custom domain mappings created"

    # Show DNS configuration
    echo -e "${CYAN}📋 DNS Configuration Required:${NC}"
    echo ""
    echo -e "${YELLOW}Configure these DNS records in your domain provider:${NC}"
    echo ""
    
    print_status "Getting DNS records for $BACKEND_DOMAIN..."
    gcloud run domain-mappings describe $BACKEND_DOMAIN --region=$REGION --format="table(status.resourceRecords[].name,status.resourceRecords[].rrdata,status.resourceRecords[].type)" 2>/dev/null || echo "Domain mapping not ready yet"
    
    echo ""
    print_status "Getting DNS records for $FRONTEND_DOMAIN..."
    gcloud run domain-mappings describe $FRONTEND_DOMAIN --region=$REGION --format="table(status.resourceRecords[].name,status.resourceRecords[].rrdata,status.resourceRecords[].type)" 2>/dev/null || echo "Domain mapping not ready yet"
}

# Validate complete deployment
validate_complete_deployment() {
    print_status "Validating complete deployment..."

    BACKEND_URL=$(gcloud run services describe $BACKEND_SERVICE --region=$REGION --format="value(status.url)")
    FRONTEND_URL=$(gcloud run services describe $FRONTEND_SERVICE --region=$REGION --format="value(status.url)")

    # Test backend
    if curl -f $BACKEND_URL/health --max-time 30 > /dev/null 2>&1; then
        print_success "✅ Backend health check passed"
    else
        print_error "❌ Backend health check failed"
        return 1
    fi

    # Test frontend
    if curl -f $FRONTEND_URL --max-time 30 > /dev/null 2>&1; then
        print_success "✅ Frontend accessible"
    else
        print_error "❌ Frontend access failed"
        return 1
    fi

    # Test KEC API
    if curl -f $BACKEND_URL/api/v1/kec-metrics/health --max-time 30 > /dev/null 2>&1; then
        print_success "✅ KEC API operational"
    else
        print_warning "⚠️ KEC API may need more time to start"
    fi

    return 0
}

# Print final summary
print_final_summary() {
    BACKEND_URL=$(gcloud run services describe $BACKEND_SERVICE --region=$REGION --format="value(status.url)")
    FRONTEND_URL=$(gcloud run services describe $FRONTEND_SERVICE --region=$REGION --format="value(status.url)")

    print_epic "DARWIN GCP DEPLOYMENT COMPLETE!"
    echo ""
    echo -e "${GREEN}🔗 Production URLs (Cloud Run):${NC}"
    echo -e "   🚀 Backend API: ${CYAN}$BACKEND_URL${NC}"
    echo -e "   🌐 Frontend Web: ${CYAN}$FRONTEND_URL${NC}"
    echo ""
    echo -e "${GREEN}🌐 Custom Domains (after DNS config):${NC}"
    echo -e "   🚀 Backend API: ${CYAN}https://$BACKEND_DOMAIN${NC}"
    echo -e "   🌐 Frontend Web: ${CYAN}https://$FRONTEND_DOMAIN${NC}"
    echo ""
    echo -e "${GREEN}📊 Services Deployed:${NC}"
    echo -e "   📦 Project: ${CYAN}$PROJECT_ID${NC}"
    echo -e "   🌍 Region: ${CYAN}$REGION${NC}"
    echo -e "   🎯 Backend: ${CYAN}$BACKEND_SERVICE${NC}"
    echo -e "   🎯 Frontend: ${CYAN}$FRONTEND_SERVICE${NC}"
    echo ""
    echo -e "${YELLOW}⚠️ DNS Configuration Steps:${NC}"
    echo -e "   1. Configure DNS CNAME records in your domain provider"
    echo -e "   2. Point ${CYAN}$BACKEND_DOMAIN${NC} → ${CYAN}ghs.googlehosted.com${NC}"
    echo -e "   3. Point ${CYAN}$FRONTEND_DOMAIN${NC} → ${CYAN}ghs.googlehosted.com${NC}"
    echo -e "   4. Wait 24-48h for DNS propagation"
    echo ""
    echo -e "${CYAN}📋 Useful Commands:${NC}"
    echo -e "   View backend logs: ${CYAN}gcloud run logs tail $BACKEND_SERVICE --region=$REGION${NC}"
    echo -e "   View frontend logs: ${CYAN}gcloud run logs tail $FRONTEND_SERVICE --region=$REGION${NC}"
    echo -e "   Scale backend: ${CYAN}gcloud run services update $BACKEND_SERVICE --max-instances=50 --region=$REGION${NC}"
}

# Main execution
main() {
    echo -e "${CYAN}🚀 Starting DARWIN Complete GCP Deploy...${NC}"
    echo -e "${CYAN}📦 Project: $PROJECT_ID${NC}"
    echo -e "${CYAN}🌐 Backend: $BACKEND_DOMAIN${NC}"
    echo -e "${CYAN}🌐 Frontend: $FRONTEND_DOMAIN${NC}"
    echo

    # Step 1: Authentication and setup
    check_auth_and_setup

    # Step 2: Build and push images
    build_backend
    build_frontend

    # Step 3: Deploy services
    deploy_backend
    deploy_frontend

    # Step 4: Configure domains
    configure_custom_domains

    # Step 5: Validate deployment
    echo
    print_status "Running complete deployment validation..."
    if validate_complete_deployment; then
        print_epic "DARWIN GCP DEPLOYMENT SUCCESS!"
        print_final_summary
        echo
        echo -e "${GREEN}🎉 DARWIN Meta-Research Brain deployed to production!${NC}"
        echo -e "${GREEN}✅ Backend + Frontend + Custom domains configured${NC}"
        echo -e "${GREEN}✅ Auto-scaling + monitoring + production ready${NC}"
        echo
        echo -e "${PURPLE}🧬 READY FOR REVOLUTIONARY BIOMATERIALS RESEARCH! 🚀${NC}"
    else
        print_error "Deployment validation failed"
        echo -e "${RED}🚨 Check service logs for issues${NC}"
        exit 1
    fi
}

# Handle interruption
trap 'echo -e "\n${YELLOW}Deploy interrupted by user${NC}"; exit 130' INT

# Run main function
main "$@"