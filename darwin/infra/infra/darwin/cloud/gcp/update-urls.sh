#!/usr/bin/env bash
set -euo pipefail

# Script to update plugin and OpenAPI URLs after deployment
RUN_URL="${1:-}"

if [ -z "$RUN_URL" ]; then
  echo "Usage: $0 <RUN_URL>"
  echo "Example: $0 https://darwin-rag-abc123-uc.a.run.app"
  exit 1
fi

echo "🔄 Updating URLs to: $RUN_URL"

# Update ai-plugin.json
echo "📝 Updating ai-plugin.json..."
sed -i "s|https://darwin-rag-[^/]*/|${RUN_URL}/|g" api/.well-known/ai-plugin.json

# Update openapi.yaml
echo "📝 Updating openapi.yaml..."
sed -i "s|- url: https://darwin-rag-[^/]*|- url: ${RUN_URL}|g" api/openapi.yaml

echo "✅ URLs updated successfully!"
echo "📋 Updated files:"
echo "   - api/.well-known/ai-plugin.json"
echo "   - api/openapi.yaml"
