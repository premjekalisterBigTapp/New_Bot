# BigTapp Agentic Chatbot - Infrastructure Setup Guide

This document provides complete instructions for setting up the BigTapp Agentic Chatbot infrastructure using Docker.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Quick Start](#quick-start)
3. [Service Overview](#service-overview)
4. [Security Configuration](#security-configuration)
5. [Environment Variables](#environment-variables)
6. [Starting Services](#starting-services)
7. [Initialization Scripts](#initialization-scripts)
8. [Accessing Services](#accessing-services)
9. [Monitoring with Grafana](#monitoring-with-grafana)
10. [Troubleshooting](#troubleshooting)
11. [Production Deployment](#production-deployment)

---

## Prerequisites

- **Docker**: v20.10 or later
- **Docker Compose**: v2.0 or later
- **Python**: 3.11+ (for running the application)
- **Disk Space**: Minimum 10GB for data volumes

### Verify Installation

```bash
docker --version
docker compose version
```

---

## Quick Start

```bash
# 1. Navigate to project directory
cd d:\agentic

# 2. Update passwords in .env (search for DOCKER INFRASTRUCTURE section)
# Change default passwords before production!

# 3. Start all infrastructure services
docker compose up -d

# 4. Verify all services are healthy
docker compose ps

# 5. Run initialization scripts
python scripts/init_redis.py
python scripts/init_mongodb.py

# 6. Start the application
python main.py
```

---

## Service Overview

| Service | Port | Purpose | Authentication |
|---------|------|---------|----------------|
| **Redis** | 6379 | Session & checkpoint storage | Password |
| **MongoDB** | 27017 | Conversation history | Username/Password |
| **Weaviate** | 8080, 50051 | Vector database (RAG) | API Key |
| **Prometheus** | 9090 | Metrics collection | None (internal) |
| **Grafana** | 3000 | Metrics visualization | Username/Password |

---

## Security Configuration

### Default Credentials (CHANGE IN PRODUCTION!)

All credentials are stored in the main `.env` file under the `DOCKER INFRASTRUCTURE` section:

```env
# Redis
REDIS_PASSWORD=agentic_redis_2025

# MongoDB (root user for Docker init)
MONGO_ROOT_USER=admin
MONGO_ROOT_PASSWORD=agentic_mongo_2025

# Weaviate
WEAVIATE_API_KEY=agentic_weaviate_2025

# Grafana
GRAFANA_ADMIN_USER=admin
GRAFANA_ADMIN_PASSWORD=agentic_grafana_2025
```

**Note:** The application uses a separate MongoDB user (`agentic_app`) created by `mongo-init.js` for better security.



---

## Environment Variables

### Required for Docker Compose

| Variable | Default | Description |
|----------|---------|-------------|
| `REDIS_PASSWORD` | `agentic_redis_2025` | Redis authentication password |
| `MONGO_ROOT_USER` | `admin` | MongoDB root username |
| `MONGO_ROOT_PASSWORD` | `agentic_mongo_2025` | MongoDB root password |
| `MONGO_DB_NAME` | `bigtapp` | MongoDB database name |
| `WEAVIATE_API_KEY` | `agentic_weaviate_2025` | Weaviate API key |
| `GRAFANA_ADMIN_USER` | `admin` | Grafana admin username |
| `GRAFANA_ADMIN_PASSWORD` | `agentic_grafana_2025` | Grafana admin password |

---

## Starting Services

### Start All Services

```bash
docker compose up -d
```

### Start Specific Services

```bash
# Start only Redis and MongoDB
docker compose up -d redis mongodb

# Start monitoring stack
docker compose up -d prometheus grafana
```

### View Logs

```bash
# All services
docker compose logs -f

# Specific service
docker compose logs -f mongodb
docker compose logs -f weaviate
```

### Check Health Status

```bash
docker compose ps
```

Expected output:
```
NAME                STATUS                   PORTS
agentic-grafana     running (healthy)        0.0.0.0:3000->3000/tcp
agentic-mongodb     running (healthy)        0.0.0.0:27017->27017/tcp
agentic-prometheus  running (healthy)        0.0.0.0:9090->9090/tcp
agentic-redis       running (healthy)        0.0.0.0:6379->6379/tcp
agentic-weaviate    running (healthy)        0.0.0.0:8080->8080/tcp, 0.0.0.0:50051->50051/tcp
```

---

## Initialization Scripts

### 1. Initialize Redis

```bash
python scripts/init_redis.py
```

Options:
- `--clear`: Clear all existing agentic data

### 2. Initialize MongoDB

```bash
python scripts/init_mongodb.py
```

This script:
- Connects to MongoDB
- Creates required indexes
- Verifies write access

### 3. Health Check

```bash
python scripts/healthcheck.py
```

This script checks:
- Redis connectivity
- MongoDB connectivity
- Weaviate connectivity
- LLM availability

---

## Accessing Services

### Redis CLI

```bash
# Connect to Redis container
docker exec -it agentic-redis redis-cli -a your_password

# Common commands
127.0.0.1:6379> KEYS agentic:*
127.0.0.1:6379> INFO memory
```

### MongoDB Shell

```bash
# Connect to MongoDB container
docker exec -it agentic-mongodb mongosh -u admin -p your_password --authenticationDatabase admin

# Switch to application database
use bigtapp

# View collections
show collections

# Query conversation history
db.agentic_conversation_history.find().limit(5)
```

### Weaviate REST API

```bash
# Check if ready
curl http://localhost:8080/v1/.well-known/ready

# Get schema (with API key)
curl -H "Authorization: Bearer your_weaviate_api_key" \
     http://localhost:8080/v1/schema
```

### Prometheus UI

Open in browser: http://localhost:9090

Example queries:
- `agentic_latency_seconds_sum / agentic_latency_seconds_count` - Average latency
- `rate(agentic_messages_total[5m])` - Messages per second
- `agentic_bg_log_queue_size` - Background logger queue

### Grafana Dashboard

Open in browser: http://localhost:3000/grafana/

1. Login with admin credentials
2. Go to **Dashboards** → **New** → **Import**
3. Create panels with Prometheus queries

---

## Monitoring with Grafana

### Recommended Panels

| Panel | Query | Type |
|-------|-------|------|
| Request Rate | `rate(agentic_messages_total[1m])` | Graph |
| Avg Latency | `agentic_latency_seconds_sum / agentic_latency_seconds_count` | Stat |
| Active Sessions | `agentic_active_sessions` | Gauge |
| Queue Size | `agentic_bg_log_queue_size` | Gauge |
| Intent Distribution | `sum by (intent) (agentic_intent_classification_total)` | Pie |
| LLM Latency | `agentic_llm_latency_seconds_sum / agentic_llm_latency_seconds_count` | Graph |

### Sample Alert Rules

```yaml
# Prometheus alerting rules (add to config/prometheus/rules.yml)
groups:
  - name: agentic
    rules:
      - alert: HighLatency
        expr: agentic_latency_seconds_sum / agentic_latency_seconds_count > 10
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High response latency"
      
      - alert: QueueBacklog
        expr: agentic_bg_log_queue_size > 100
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "Background logger queue backlog"
```

---

## Troubleshooting

### Redis Connection Refused

```bash
# Check if Redis is running
docker compose ps redis

# Check Redis logs
docker compose logs redis

# Verify password
docker exec -it agentic-redis redis-cli -a wrong_password PING
```

### MongoDB Authentication Failed

```bash
# Check MongoDB logs
docker compose logs mongodb

# Reset and reinitialize (WARNING: deletes data)
docker compose down -v
docker compose up -d mongodb
```

### Weaviate Not Ready

```bash
# Check Weaviate logs
docker compose logs weaviate

# Check health
curl http://localhost:8080/v1/.well-known/ready
```

### Prometheus Not Scraping

```bash
# Check Prometheus targets
curl http://localhost:9090/api/v1/targets

# Verify application metrics endpoint
curl http://localhost:8000/metrics
```

---

## Production Deployment

### Security Checklist

- [ ] Change all default passwords
- [ ] Use secrets management (AWS Secrets Manager, HashiCorp Vault)
- [ ] Enable TLS for all services
- [ ] Configure firewall rules
- [ ] Set up backup for MongoDB and Redis
- [ ] Enable log aggregation (ELK, CloudWatch)

### Resource Recommendations

| Service | CPU | Memory | Disk |
|---------|-----|--------|------|
| Redis | 0.5 | 512MB | 1GB |
| MongoDB | 1.0 | 2GB | 20GB |
| Weaviate | 2.0 | 4GB | 50GB |
| Prometheus | 0.5 | 1GB | 10GB |
| Grafana | 0.25 | 256MB | 1GB |

### Backup Commands

```bash
# Backup MongoDB
docker exec agentic-mongodb mongodump --out /backup --db bigtapp

# Backup Redis
docker exec agentic-redis redis-cli -a $REDIS_PASSWORD BGSAVE

# Backup Weaviate
# Use Weaviate backup API: https://weaviate.io/developers/weaviate/configuration/backups
```

---

## Directory Structure

```
agentic/
├── docker-compose.yml          # Docker infrastructure
├── .env                        # All configuration (app + docker)
├── config/
│   ├── prometheus.yml          # Prometheus config
│   └── grafana/
│       └── provisioning/
│           └── datasources/
│               └── datasources.yml
├── scripts/
│   ├── init_mongodb.py         # MongoDB indexes + verification
│   ├── init_redis.py           # Redis verification
│   ├── mongo-init.js           # MongoDB Docker entrypoint (creates app user)
│   └── healthcheck.py          # Infrastructure health check
└── docs/
    └── INFRASTRUCTURE_SETUP.md # This document
```

---

## Support

For issues related to:
- **Redis**: Check `docker compose logs redis`
- **MongoDB**: Check `docker compose logs mongodb`
- **Weaviate**: Check `docker compose logs weaviate`
- **Prometheus**: Check http://localhost:9090/targets
- **Grafana**: Check http://localhost:3000/grafana/

For application issues, check:
- Application logs: `python main.py` output
- Health endpoint: http://localhost:8000/ready
- Metrics endpoint: http://localhost:8000/metrics
