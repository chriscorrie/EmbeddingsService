#!/bin/bash

# Script to restore PCIE drives (NVME4 and NVME5) after reinstalling the PCIE card
# Run this script if you need to remount the PCIE drives

echo "=== PCIE Drive Restoration Script ==="
echo ""
echo "This script will restore the PCIE drives (NVME4 and NVME5) to their mount points"
echo "Run this ONLY after you have reinstalled the PCIE card with the drives"
echo ""

# Check if the drives are detected
if [ ! -b /dev/nvme4n1p1 ] || [ ! -b /dev/nvme5n1p1 ]; then
    echo "❌ PCIE drives not detected. Please ensure:"
    echo "   1. PCIE card is properly installed"
    echo "   2. System has been rebooted"
    echo "   3. Drives are properly connected"
    exit 1
fi

echo "✅ PCIE drives detected"
echo ""

# Restore fstab entries
echo "Restoring /etc/fstab entries..."
sudo sed -i 's|^#UUID=53451fb5-0ca7-42e3-9477-598295423f1d|UUID=53451fb5-0ca7-42e3-9477-598295423f1d|' /etc/fstab
sudo sed -i 's|^#UUID=acfc36da-001a-4f34-b829-1bb44c4f427e|UUID=acfc36da-001a-4f34-b829-1bb44c4f427e|' /etc/fstab
echo "✅ fstab entries restored"

# Mount the drives
echo ""
echo "Mounting PCIE drives..."
sudo mount /mnt/NVME_4
sudo mount /mnt/NVME_5
echo "✅ PCIE drives mounted"

echo ""
echo "Verification:"
mount | grep NVME

echo ""
echo "✅ PCIE drives successfully restored!"
echo ""
echo "Drive locations:"
echo "- NVME4: /mnt/NVME_4 (PCIE slot)"
echo "- NVME5: /mnt/NVME_5 (PCIE slot)"
