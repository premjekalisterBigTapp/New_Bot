# BigTapp Agentic Chatbot - Production Deployment Guide

**Server:** Ubuntu, 8 cores, 62GB RAM  
**Recommended Workers:** 17 (2 × 8 + 1)

---

## Current State

You have 3 containers running (dev setup - insecure):
```
mongo      - 0.0.0.0:27017 (exposed to network)
redis      - 0.0.0.0:6379  (exposed to network)
weaviate   - 0.0.0.0:8080  (exposed to network)
```

**Goal:** Migrate to secure docker-compose with localhost-only binding.

---

## Unified Password

Use this password for all BigTapp services (change in production!):

```
bigtapp_secure_2025
```

| Service | Username | Password |
|---------|----------|----------|
| Redis | - | `bigtapp_secure_2025` |
| MongoDB | `bigtapp_admin` | `bigtapp_secure_2025` |
| Weaviate | - | `bigtapp_secure_2025` (API key) |
| Grafana | `admin` | `bigtapp_secure_2025` |

---

## Step 1: Pull Latest Code

```bash
cd ~/agentic
git pull origin main
```

---

## Step 2: Stop Old Containers

```bash
# Stop the old containers
sudo docker stop mongo redis weaviate

# Verify they're stopped
sudo docker ps
```

---

## Step 3: Update .env

```bash
nano .env
```

Update these values:

```env
# ==============================================================================
# 2. DATA STORES
# ==============================================================================

# Redis
REDIS_PASSWORD=bigtapp_secure_2025
REDIS_URL=redis://:bigtapp_secure_2025@localhost:6379/0

# MongoDB (use your existing user OR create new one)
MONGO_URI="mongodb://bigtapp_admin:bigtapp_secure_2025@localhost:27017/bigtapp?authSource=admin"
DB_NAME="bigtapp"
AGENTIC_HISTORY_COLLECTION="agentic_conversation_history"

# Weaviate
WEAVIATE_API_KEY=bigtapp_secure_2025

# ==============================================================================
# 6. DOCKER INFRASTRUCTURE
# ==============================================================================

# MongoDB root (for Docker init only)
MONGO_ROOT_USER=bigtapp_admin
MONGO_ROOT_PASSWORD=bigtapp_secure_2025

# Grafana
GRAFANA_ADMIN_USER=admin
GRAFANA_ADMIN_PASSWORD=bigtapp_secure_2025

# Logging
LOG_LEVEL=WARNING
```

---

## Step 4: Start Infrastructure

```bash
cd ~/agentic

# Start all services
sudo docker compose up -d

# Wait for health checks
sleep 15

# Verify all healthy
sudo docker compose ps
```

Expected output:
```
NAME                STATUS
agentic-redis       running (healthy)
agentic-mongodb     running (healthy)
agentic-weaviate    running (healthy)
agentic-prometheus  running (healthy)
agentic-grafana     running (healthy)
```

---

## Step 5: Initialize MongoDB

```bash
# Activate Python environment
source ~/bigtapp/bin/activate

# Run MongoDB init (creates indexes)
cd ~/agentic
python scripts/init_mongodb.py

# Verify Redis
python scripts/init_redis.py
```

---

## Step 6: Test Application

```bash
# Test run
python main.py

# In another terminal, test endpoints
curl http://localhost:8000/ready
curl http://localhost:8000/health
```

---

## Step 7: Create Systemd Service

```bash
sudo nano /etc/systemd/system/agentic.service
```

Paste:
```ini
[Unit]
Description=BigTapp Agentic Chatbot
After=network.target docker.service
Requires=docker.service

[Service]
Type=simple
User=vendor
Group=vendor
WorkingDirectory=/home/vendor/agentic
Environment="PATH=/home/vendor/bigtapp/bin:/usr/local/bin:/usr/bin"
ExecStart=/home/vendor/bigtapp/bin/gunicorn main:app \
    --workers 17 \
    --worker-class uvicorn.workers.UvicornWorker \
    --bind 0.0.0.0:8000 \
    --timeout 120 \
    --keep-alive 5 \
    --access-logfile /var/log/agentic/access.log \
    --error-logfile /var/log/agentic/error.log
ExecReload=/bin/kill -HUP $MAINPID
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

```bash
# Create log directory
sudo mkdir -p /var/log/agentic
sudo chown vendor:vendor /var/log/agentic

# Enable and start
sudo systemctl daemon-reload
sudo systemctl enable agentic
sudo systemctl start agentic

# Check status
sudo systemctl status agentic
```

---

## Step 8: Remove Old Containers

After verifying everything works:

```bash
# Remove old containers
sudo docker rm mongo redis weaviate
```

---

## Quick Reference Commands

| Task | Command |
|------|---------|
| Start infrastructure | `sudo docker compose up -d` |
| Stop infrastructure | `sudo docker compose down` |
| View container logs | `sudo docker compose logs -f <service>` |
| Start app | `sudo systemctl start agentic` |
| Stop app | `sudo systemctl stop agentic` |
| Restart app | `sudo systemctl restart agentic` |
| View app logs | `sudo journalctl -u agentic -f` |
| Deploy updates | `git pull && sudo systemctl restart agentic` |

---

## Access Services

| Service | URL | Credentials |
|---------|-----|-------------|
| App Health | http://localhost:8000/ready | - |
| App Metrics | http://localhost:8000/metrics | - |
| Prometheus | http://localhost:9090 | - |
| Grafana | http://localhost:3000 | admin / bigtapp_secure_2025 |

**Note:** Services are bound to localhost only. Access via SSH tunnel:
```bash
ssh -L 3000:localhost:3000 -L 9090:localhost:9090 vendor@server-ip
```

---

## Rollback Plan

If something goes wrong:

```bash
# Stop new setup
sudo docker compose down
sudo systemctl stop agentic

# Start old containers
sudo docker start mongo redis weaviate

# Run old application
python main.py
```

---

## Architecture

```
WhatsApp → Meta Cloud → HAProxy (SSL) → Gunicorn:8000 → FastAPI App
                                              ↓
                         ┌──────────────────────────────────────┐
                         │           localhost only             │
                         │  Redis:6379  MongoDB:27017  Weaviate │
                         │  Prometheus:9090  Grafana:3000       │
                         └──────────────────────────────────────┘
```

**No nginx required** - HAProxy handles SSL termination.

---

## Security Checklist

- [ ] Changed `bigtapp_secure_2025` to your own secure password
- [ ] Verified ports are localhost-only: `sudo netstat -tlnp | grep LISTEN`
- [ ] Firewall allows only port 8000 from HAProxy
- [ ] Logs are being written
