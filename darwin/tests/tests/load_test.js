// =============================================================================
// DARWIN Load Testing Script
// K6 script for load testing DARWIN platform
// =============================================================================

import http from 'k6/http';
import { check, sleep } from 'k6';
import { Rate, Trend } from 'k6/metrics';

// Custom metrics
const errorRate = new Rate('errors');
const responseTime = new Trend('response_time');

// Configuration
const API_URL = __ENV.API_URL || 'https://api-staging.agourakis.med.br';
const FRONTEND_URL = __ENV.FRONTEND_URL || 'https://darwin-staging.agourakis.med.br';

// Test configuration
export const options = {
  stages: [
    // Ramp up
    { duration: '2m', target: 5 },    // Ramp up to 5 users over 2 minutes
    { duration: '5m', target: 5 },    // Stay at 5 users for 5 minutes
    { duration: '2m', target: 10 },   // Ramp up to 10 users over 2 minutes
    { duration: '5m', target: 10 },   // Stay at 10 users for 5 minutes
    { duration: '2m', target: 20 },   // Ramp up to 20 users over 2 minutes
    { duration: '5m', target: 20 },   // Stay at 20 users for 5 minutes
    { duration: '2m', target: 0 },    // Ramp down to 0 users
  ],
  
  thresholds: {
    // Error rate should be less than 5%
    errors: ['rate<0.05'],
    
    // 95% of requests should be below 5 seconds
    http_req_duration: ['p(95)<5000'],
    
    // Failed requests should be less than 2%
    http_req_failed: ['rate<0.02'],
    
    // Average response time should be below 2 seconds
    response_time: ['avg<2000'],
    
    // Check throughput
    http_reqs: ['rate>10'],  // At least 10 requests per second
  },
  
  // User agent
  userAgent: 'DARWIN-LoadTest/1.0',
};

// Test scenarios
export default function () {
  // Test API endpoints
  testAPIEndpoints();
  
  // Test frontend endpoints
  testFrontendEndpoints();
  
  // Sleep between iterations
  sleep(Math.random() * 2 + 1); // 1-3 seconds
}

function testAPIEndpoints() {
  const endpoints = [
    { path: '/health', weight: 30 },
    { path: '/api/health', weight: 20 },
    { path: '/metrics', weight: 10 },
    { path: '/docs', weight: 5 },
    { path: '/openapi.json', weight: 5 },
    { path: '/api/core/status', weight: 30 },
  ];
  
  // Select endpoint based on weight (simplified)
  const randomNum = Math.random() * 100;
  let cumulativeWeight = 0;
  let selectedEndpoint = endpoints[0];
  
  for (const endpoint of endpoints) {
    cumulativeWeight += endpoint.weight;
    if (randomNum <= cumulativeWeight) {
      selectedEndpoint = endpoint;
      break;
    }
  }
  
  // Make request
  const startTime = new Date().getTime();
  const response = http.get(`${API_URL}${selectedEndpoint.path}`, {
    headers: {
      'User-Agent': 'DARWIN-LoadTest/1.0',
      'Accept': 'application/json',
    },
    timeout: '30s',
  });
  const endTime = new Date().getTime();
  
  // Record metrics
  const duration = endTime - startTime;
  responseTime.add(duration);
  
  // Check response
  const checkResult = check(response, {
    'status is 200': (r) => r.status === 200,
    'response time < 5000ms': (r) => r.timings.duration < 5000,
    'response has body': (r) => r.body && r.body.length > 0,
  });
  
  if (!checkResult) {
    errorRate.add(1);
  }
  
  // Log errors for debugging
  if (response.status !== 200) {
    console.error(`API Error: ${selectedEndpoint.path} returned ${response.status}`);
  }
}

function testFrontendEndpoints() {
  // Randomly test frontend endpoints (less frequently)
  if (Math.random() < 0.3) { // 30% chance
    const endpoints = [
      '/',
      '/api/health',
    ];
    
    const endpoint = endpoints[Math.floor(Math.random() * endpoints.length)];
    const response = http.get(`${FRONTEND_URL}${endpoint}`, {
      headers: {
        'User-Agent': 'DARWIN-LoadTest/1.0',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
      },
      timeout: '30s',
    });
    
    check(response, {
      'frontend status is 200': (r) => r.status === 200,
      'frontend response time < 3000ms': (r) => r.timings.duration < 3000,
    }) || errorRate.add(1);
  }
}

// Setup function (runs once per VU)
export function setup() {
  console.log('🚀 Starting DARWIN load test...');
  console.log(`API URL: ${API_URL}`);
  console.log(`Frontend URL: ${FRONTEND_URL}`);
  
  // Verify services are accessible before starting load test
  const healthCheck = http.get(`${API_URL}/health`, { timeout: '10s' });
  if (healthCheck.status !== 200) {
    console.error('❌ API health check failed before load test');
    throw new Error('API not accessible');
  }
  
  console.log('✅ Pre-load test health check passed');
  return { startTime: new Date().getTime() };
}

// Teardown function (runs once after all VUs finish)
export function teardown(data) {
  const endTime = new Date().getTime();
  const duration = (endTime - data.startTime) / 1000;
  
  console.log('🏁 Load test completed');
  console.log(`Duration: ${duration} seconds`);
  
  // Final health check
  const healthCheck = http.get(`${API_URL}/health`, { timeout: '10s' });
  if (healthCheck.status === 200) {
    console.log('✅ Post-load test health check passed');
  } else {
    console.error('❌ Post-load test health check failed');
  }
}

// Handle summary (called after test execution)
export function handleSummary(data) {
  const summary = {
    test_start: data.state.testRunDurationMs,
    metrics: {},
    checks: {},
  };
  
  // Extract key metrics
  for (const [name, metric] of Object.entries(data.metrics)) {
    if (metric.values) {
      summary.metrics[name] = {
        avg: metric.values.avg,
        max: metric.values.max,
        min: metric.values.min,
        p95: metric.values['p(95)'],
        count: metric.values.count || metric.values.rate,
      };
    }
  }
  
  // Extract check results
  for (const [name, check] of Object.entries(data.checks)) {
    summary.checks[name] = {
      passes: check.passes,
      fails: check.fails,
      rate: check.passes / (check.passes + check.fails),
    };
  }
  
  return {
    'stdout': textSummary(data, { indent: ' ', enableColors: false }),
    'load-test-results.json': JSON.stringify(summary, null, 2),
    'load-test-detailed.json': JSON.stringify(data, null, 2),
  };
}