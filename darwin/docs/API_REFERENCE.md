# DARWIN API Reference

**Documentação Completa da API DARWIN - JAX-Powered Scientific Research Platform**

---

## 📋 Visão Geral

### Base URLs
```
Production:  https://api.agourakis.med.br
Staging:     https://api-staging.agourakis.med.br
Development: https://api-dev.agourakis.med.br
```

### Características
- **Protocol:** HTTPS only (TLS 1.2+)
- **Format:** JSON (application/json)
- **Authentication:** JWT Bearer tokens
- **Rate Limiting:** 1000 requests/minute per user
- **Documentation:** OpenAPI 3.0 at `/docs`

---

## 🔐 Autenticação

### JWT Bearer Token

```http
Authorization: Bearer eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9...
```

### Login

**POST** `/api/auth/login`

```json
{
  "email": "user@example.com",
  "password": "secure_password"
}
```

**Response:**
```json
{
  "access_token": "eyJhbGciOiJSUzI1NiIs...",
  "token_type": "bearer",
  "expires_in": 86400,
  "refresh_token": "eyJhbGciOiJSUzI1NiIs..."
}
```

---

## 🏥 Core Endpoints

### Health Check

**GET** `/health`

```json
{
  "status": "healthy",
  "timestamp": "2025-09-22T14:15:00Z",
  "version": "1.0.0",
  "environment": "production"
}
```

### Detailed Health

**GET** `/api/health`

```json
{
  "status": "healthy",
  "dependencies": {
    "database": {"status": "connected", "response_time_ms": 12},
    "redis": {"status": "connected", "response_time_ms": 3},
    "storage": {"status": "accessible", "response_time_ms": 45}
  }
}
```

---

## 🔍 Vector Search

### Search Documents

**POST** `/api/search/documents`

```json
{
  "query": "biomaterial scaffolds for tissue engineering",
  "limit": 10,
  "similarity_threshold": 0.8
}
```

**Response:**
```json
{
  "results": [
    {
      "id": "doc_123",
      "title": "Advanced Porous Scaffolds",
      "similarity_score": 0.95,
      "metadata": {
        "authors": ["Smith, J."],
        "doi": "10.1234/example.doi"
      }
    }
  ],
  "total_results": 25,
  "search_time_ms": 145
}
```

---

## 🤖 Multi-AI Chat

### Start Session

**POST** `/api/multi-ai/chat/start`

```json
{
  "models": ["gpt-4", "claude-3", "gemini-pro"],
  "system_prompt": "Scientific research assistant"
}
```

**Response:**
```json
{
  "session_id": "sess_1234567890",
  "models": ["gpt-4", "claude-3", "gemini-pro"],
  "status": "active"
}
```

### Send Message

**POST** `/api/multi-ai/chat/{session_id}/message`

```json
{
  "message": "What are bioactive glass properties?",
  "stream": false
}
```

**Response:**
```json
{
  "responses": {
    "gpt-4": {
      "content": "Bioactive glass scaffolds possess...",
      "response_time_ms": 1250,
      "tokens_used": 180
    },
    "claude-3": {
      "content": "The primary characteristics include...",
      "response_time_ms": 1100,
      "tokens_used": 165
    }
  }
}
```

---

## 📊 Status Codes

| Code | Description |
|------|-------------|
| 200 | Success |
| 201 | Created |
| 400 | Bad Request |
| 401 | Unauthorized |
| 403 | Forbidden |
| 404 | Not Found |
| 429 | Rate Limited |
| 500 | Server Error |

---

## 💡 Usage Examples

### Python
```python
import requests

headers = {"Authorization": "Bearer YOUR_TOKEN"}
response = requests.post(
    "https://api.agourakis.med.br/api/search/documents",
    headers=headers,
    json={"query": "biomaterials", "limit": 10}
)
```

### JavaScript
```javascript
const response = await fetch('https://api.agourakis.med.br/api/search/documents', {
  method: 'POST',
  headers: {
    'Authorization': 'Bearer YOUR_TOKEN',
    'Content-Type': 'application/json'
  },
  body: JSON.stringify({query: 'biomaterials', limit: 10})
});
```

### curl
```bash
curl -X POST "https://api.agourakis.med.br/api/search/documents" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"query": "biomaterials", "limit": 10}'
```

---

## 🔗 Resources

- **Interactive Docs:** https://api.agourakis.med.br/docs
- **OpenAPI Spec:** https://api.agourakis.med.br/openapi.json
- **Support:** api-support@agourakis.med.br

**Version:** 1.0.0 | **Last Updated:** 2025-09-22