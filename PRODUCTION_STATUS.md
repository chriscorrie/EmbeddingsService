# EmbeddingsService Production Setup - Complete

## 🎯 **Current Status: PRODUCTION READY**

### ✅ **Setup Complete (July 24, 2025)**
- **Location**: `/home/chris/Projects/EmbeddingsService/`
- **Repository**: `https://github.com/chriscorrie/EmbeddingsService`
- **Service**: Running and enabled for auto-start
- **Dependencies**: Milvus database running via Docker

---

## 🏗️ **Production Architecture**

### **Main Components**:
```
🌐 Production API Service
├── 🐍 Python: production_rest_api_service_v3.py
├── 🧠 PyTorch: 2.9.0.dev20250724+cu129 with CUDA 12.9
├── 🗄️ Vector DB: Milvus v2.3.0 (via Docker)
└── 📊 Performance: 14,539 sentences/second
```

### **Service Configuration**:
- **Name**: `document-embedding-api-v3.service`
- **Auto-start**: ✅ Enabled
- **Dependencies**: Docker service
- **Port**: 5000
- **User**: chris

---

## 🚀 **Service Management**

### **Quick Commands**:
```bash
# Check overall status
cd /home/chris/Projects/EmbeddingsService
./scripts/status_check.sh

# Service control
sudo systemctl status document-embedding-api-v3.service
sudo systemctl restart document-embedding-api-v3.service
sudo systemctl stop document-embedding-api-v3.service

# View logs
sudo journalctl -u document-embedding-api-v3.service -f

# Management script
./manage_service_v3.sh status|logs|restart
```

### **API Endpoints**:
- **Health Check**: `http://localhost:5000/health`
- **API Documentation**: `http://localhost:5000/docs/`
- **API Base**: `http://localhost:5000/api/v1/`
- **External Access**: `http://192.168.15.206:5000/`

---

## 🐳 **Docker Dependencies**

### **Milvus Containers** (Auto-managed):
```bash
# Check containers
sudo docker ps

# Milvus containers should show:
# - milvus-standalone (port 19530)
# - milvus-minio (ports 9000, 9001)  
# - milvus-etcd (port 2379)
```

### **Manual Container Management** (if needed):
```bash
cd /home/chris/Projects/EmbeddingsService
docker-compose up -d      # Start containers
docker-compose down       # Stop containers
docker-compose restart    # Restart containers
```

---

## 🔧 **Boot Sequence**

### **Auto-Start Order**:
1. **Docker Service** (enabled) → Starts Docker daemon
2. **Milvus Containers** (via docker-compose) → Vector database
3. **EmbeddingsService** (enabled) → Waits for dependencies, then starts API

### **Verification After Reboot**:
```bash
# Check all services
cd /home/chris/Projects/EmbeddingsService
./scripts/status_check.sh

# Should show:
# ✅ Milvus containers running
# ✅ document-embedding-api-v3.service: active  
# ✅ API responding at http://localhost:5000
```

---

## 🛠️ **Troubleshooting**

### **Common Issues**:

#### **API Not Starting**:
```bash
# Check dependencies
sudo docker ps | grep milvus
sudo systemctl status docker

# Check service logs
sudo journalctl -u document-embedding-api-v3.service -n 50

# Restart sequence
sudo systemctl restart document-embedding-api-v3.service
```

#### **Milvus Connection Issues**:
```bash
# Check Milvus containers
sudo docker ps | grep milvus

# Restart Milvus
cd /home/chris/Projects/EmbeddingsService
docker-compose restart

# Test connection
source venv/bin/activate
python -c "from pymilvus import connections; connections.connect('default', host='localhost', port='19530'); print('✅ Connected')"
```

#### **Performance Issues**:
```bash
# Check GPU status
nvidia-smi

# Check service resources
sudo systemctl status document-embedding-api-v3.service

# Run performance test
source venv/bin/activate
python debug/gpu_performance_test.py
```

---

## 📊 **Performance Specifications**

### **Current Performance**:
- **Processing Speed**: 14,539 sentences/second
- **GPU**: NVIDIA GeForce RTX 5060 Ti (15.5GB VRAM)
- **Memory Usage**: ~1GB (200MB peak GPU)
- **CUDA**: 12.9
- **PyTorch**: 2.9.0.dev (latest nightly)

### **Capacity Estimates**:
| Document Count | Processing Time |
|---------------|----------------|
| 1 Million     | 1.1 minutes    |
| 10 Million    | 11.5 minutes   |  
| 100 Million   | 1.9 hours      |
| 1 Billion     | 19.3 hours     |

---

## 🔐 **Security & Permissions**

### **Service Security**:
- ✅ `NoNewPrivileges=yes`
- ✅ `PrivateTmp=yes`
- ✅ `ProtectSystem=strict`
- ✅ Limited file access to project directory

### **Docker Permissions**:
- User `chris` added to `docker` group
- Containers run with appropriate permissions
- Data stored on dedicated NVME drives

---

## 📁 **File Structure**

```
/home/chris/Projects/EmbeddingsService/
├── production_rest_api_service_v3.py    # Main API service
├── document-embedding-api-v3.service    # Systemd service file
├── docker-compose.yml                   # Milvus container config
├── scripts/                             # Utility scripts
│   ├── manage_service_v3.sh             # Service management
│   ├── status_check.sh                  # Status checker
│   └── startup.sh                       # Startup script
├── venv/                                # Python virtual environment
├── requirements.txt                     # Python dependencies
└── [performance & config files]         # Additional components
```

---

## 🎯 **Next Steps After Server Restart**

1. **Verify Services**: Run `./scripts/status_check.sh`
2. **Test API**: Check `http://localhost:5000/health`
3. **Monitor Performance**: Check logs and resource usage
4. **Backup**: Ensure critical data is backed up

---

**✅ Production Status: FULLY OPERATIONAL**  
**🔄 Auto-restart: ENABLED**  
**📈 Performance: OPTIMIZED**  
**🛡️ Security: CONFIGURED**

*Last Updated: July 24, 2025*
