#!/bin/bash
# Database Reset Wrapper Script
# Automatically activates virtual environment and runs ResetDatabase.py

echo "🔧 Database Reset Wrapper"
echo "=========================="
echo "Activating virtual environment..."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment 'venv' not found!"
    echo "Please run: python3 -m venv venv && pip install -r requirements.txt"
    exit 1
fi

# Activate virtual environment and run the script
source venv/bin/activate
echo "✅ Virtual environment activated"
echo "🚀 Running ResetDatabase.py..."
echo ""

python ResetDatabase.py "$@"

# Deactivate when done
deactivate
echo ""
echo "✅ Script completed, virtual environment deactivated"
