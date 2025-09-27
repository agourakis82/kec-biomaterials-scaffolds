# 1. Base Image: Use the official Node.js 18 image.
FROM node:18-alpine AS base

# 2. Build Stage: Install dependencies and build the Next.js app
FROM base AS builder
WORKDIR /app
COPY package*.json ./
RUN npm install
COPY . .
RUN npm run build

# 3. Production Stage: Create a small production image
FROM base AS runner
WORKDIR /app

ENV NODE_ENV=production

# Create a non-root user and group
RUN addgroup -g 1001 -S nodejs
RUN adduser -S nextjs -u 1001

# Copy the standalone output from the builder stage
COPY --from=builder /app/public ./public
COPY --from=builder --chown=nextjs:nodejs /app/.next/standalone ./
COPY --from=builder --chown=nextjs:nodejs /app/.next/static ./.next/static

# Run the app as a non-root user
USER nextjs

EXPOSE 3000
ENV PORT 3000

CMD ["node", "server.js"]
