#!/bin/bash

# Cleanup script to remove old MinIO data from PCIE drive (NVME4)
# Run this ONLY after you've verified the migration was successful

echo "This script will remove the old MinIO data from NVME4 (PCIE drive)"
echo "New MinIO data is now on NVME0 (motherboard drive)"
echo ""
echo "Current usage:"
echo "- NVME0 (new location): $(du -sh /mnt/NVME_0/milvus_data/minio 2>/dev/null | cut -f1)"
echo "- NVME4 (old location): $(du -sh /mnt/NVME_4/milvus_data/minio 2>/dev/null | cut -f1)"
echo ""

# Safety check - ensure containers are running with new location
if sudo docker ps | grep -q milvus-minio; then
    echo "✅ MinIO container is running with new configuration"
else
    echo "❌ MinIO container is not running. Please start containers first."
    exit 1
fi

# Prompt for confirmation
read -p "Are you sure you want to delete the old MinIO data from NVME4? (yes/no): " confirm

if [ "$confirm" = "yes" ]; then
    echo "Removing old MinIO data from /mnt/NVME_4/milvus_data/minio..."
    sudo rm -rf /mnt/NVME_4/milvus_data/
    echo "✅ Old MinIO data removed from NVME4"
    echo "NVME4 is now available for other uses or can be physically removed"
else
    echo "Cleanup cancelled. Old data remains on NVME4."
fi
