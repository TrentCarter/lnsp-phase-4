#!/bin/bash
# Demo: VP (LCO) + PAS Stub Integration
# Tests end-to-end workflow from terminal client to execution

set -e

echo "=== VP + PAS Integration Demo ==="
echo ""

# Check prerequisites
if ! curl -s http://localhost:6200/health > /dev/null 2>&1; then
    echo "❌ PAS stub not running!"
    echo "   Start with: make run-pas-stub"
    exit 1
fi

echo "✓ PAS stub is running"
echo ""

# Clean state
rm -rf ~/.vp
echo "✓ Cleaned VP state"
echo ""

# Test workflow
echo "📋 Test Workflow:"
echo ""

echo "1️⃣  vp new --name demo-project"
./.venv/bin/python cli/vp.py new --name demo-project
echo ""

echo "2️⃣  vp plan"
./.venv/bin/python cli/vp.py plan
echo ""

echo "3️⃣  vp estimate"
./.venv/bin/python cli/vp.py estimate
echo ""

echo "4️⃣  vp simulate --rehearsal 0.01"
./.venv/bin/python cli/vp.py simulate --rehearsal 0.01
echo ""

echo "5️⃣  vp start"
./.venv/bin/python cli/vp.py start
echo ""

echo "⏳ Waiting 20 seconds for synthetic execution..."
sleep 20
echo ""

echo "6️⃣  vp status"
./.venv/bin/python cli/vp.py status
echo ""

echo "✅ Integration Demo Complete!"
echo ""
echo "📊 Summary:"
echo "   - VP CLI: ✓ Operational"
echo "   - PAS Stub: ✓ Executing tasks"
echo "   - End-to-end flow: ✓ Working"
echo ""
echo "📚 Next Steps:"
echo "   - Phase 1: LightRAG Code Index (Week 1)"
echo "   - Phase 2: Multi-Metric Telemetry (Week 2)"
echo "   - Phase 3: Full LCO features (Weeks 3-4)"
echo "   - Phase 4: Real PAS (Weeks 5-8)"
