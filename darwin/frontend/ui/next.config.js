/** @type {import('next').NextConfig} */
const nextConfig = {
  output: 'standalone',
  reactStrictMode: true,
  experimental: {
    typedRoutes: false,
  },
  
  // DARWIN Backend Proxy Configuration
  async rewrites() {
    return [
      // DARWIN API routes redirect to port 8090
      {
        source: '/api/darwin/:path*',
        destination: 'http://localhost:8090/api/v1/:path*'
      },
      
      // AutoGen Research Team endpoints
      {
        source: '/api/research-team/:path*',
        destination: 'http://localhost:8090/research-team/:path*'
      },
      
      // JAX Ultra-Performance endpoints
      {
        source: '/api/ultra-performance/:path*',
        destination: 'http://localhost:8090/ultra-performance/:path*'
      },
      
      // Health check redirect
      {
        source: '/api/health',
        destination: 'http://localhost:8090/api/v1/health'
      },
      
      // RAG endpoints redirect
      {
        source: '/api/rag/:path*',
        destination: 'http://localhost:8090/api/v1/rag-plus/:path*'
      },
      
      // Multi-AI Hub redirect
      {
        source: '/api/multi-ai/:path*',
        destination: 'http://localhost:8090/api/v1/multi-ai/:path*'
      },
      
      // KEC Metrics redirect
      {
        source: '/api/kec-metrics/:path*',
        destination: 'http://localhost:8090/api/v1/kec-metrics/:path*'
      },
      
      // Tree Search redirect
      {
        source: '/api/tree-search/:path*',
        destination: 'http://localhost:8090/api/v1/tree-search/:path*'
      },
      
      // Knowledge Graph redirect
      {
        source: '/api/knowledge-graph/:path*',
        destination: 'http://localhost:8090/api/v1/knowledge-graph/:path*'
      },
      
      // Discovery redirect
      {
        source: '/api/discovery/:path*',
        destination: 'http://localhost:8090/api/v1/discovery/:path*'
      },
      
      // Contracts redirect
      {
        source: '/api/contracts/:path*',
        destination: 'http://localhost:8090/api/v1/contracts/:path*'
      },
      
      // PUCT redirect (legacy compatibility)
      {
        source: '/api/puct',
        destination: 'http://localhost:8090/api/v1/tree-search/puct'
      }
    ]
  },

  // Headers for CORS and API calls
  async headers() {
    return [
      {
        source: '/api/:path*',
        headers: [
          { key: 'Access-Control-Allow-Origin', value: '*' },
          { key: 'Access-Control-Allow-Methods', value: 'GET,POST,PUT,DELETE,OPTIONS' },
          { key: 'Access-Control-Allow-Headers', value: 'X-Requested-With,content-type,Authorization,X-API-KEY' },
        ],
      },
    ]
  }
}

module.exports = nextConfig