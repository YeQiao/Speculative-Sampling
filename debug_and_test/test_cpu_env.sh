#!/bin/bash
# Test different CPU threading configurations for optimal performance

echo "Testing CPU performance with different threading configurations"
echo "================================================================"

# Test 1: Default (current)
echo ""
echo "Test 1: Default settings"
python debug_cpu_performance.py 2>&1 | grep -A 3 "TEST 1:"

# Test 2: Set OMP_NUM_THREADS
echo ""
echo "Test 2: OMP_NUM_THREADS=64"
OMP_NUM_THREADS=64 python debug_cpu_performance.py 2>&1 | grep -A 3 "TEST 1:"

# Test 3: More conservative threading
echo ""
echo "Test 3: OMP_NUM_THREADS=32"
OMP_NUM_THREADS=32 python debug_cpu_performance.py 2>&1 | grep -A 3 "TEST 1:"

# Test 4: Even more conservative
echo ""
echo "Test 4: OMP_NUM_THREADS=16"
OMP_NUM_THREADS=16 python debug_cpu_performance.py 2>&1 | grep -A 3 "TEST 1:"

# Test 5: Match available cores
echo ""
echo "Test 5: OMP_NUM_THREADS=8"
OMP_NUM_THREADS=8 python debug_cpu_performance.py 2>&1 | grep -A 3 "TEST 1:"

echo ""
echo "================================================================"
echo "Check which configuration gives best throughput"
