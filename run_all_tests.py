import os
import subprocess
import sys

def run_test(test_file, test_name):
    print(f"\n--------------------------------------------------")
    print(f"Running {test_name}...")
    print(f"--------------------------------------------------\n")
    result = subprocess.run([sys.executable, test_file])
    return result.returncode == 0  # Return True if test passed

def main():
    tests = [
        ('test_environment.py', 'Environment Test'),
        ('test_pytorch.py', 'PyTorch Installation Test'),
        ('test_spikingjelly.py', 'SpikingJelly Installation Test'),
        ('test_opencv.py', 'OpenCV Installation Test'),
        ('tests/test_optical_flow.py', 'Optical Flow Test'),  # New test added
    ]
    
    passed = 0
    failed = 0
    
    for test_file, test_name in tests:
        if run_test(test_file, test_name):
            passed += 1
            print(f"✅ {test_name} PASSED")
        else:
            failed += 1
            print(f"❌ {test_name} FAILED")
    
    print(f"\n--------------------------------------------------")
    print(f"Test Summary: {passed} passed, {failed} failed")
    print(f"--------------------------------------------------")
    
    return failed == 0  # Return True if all tests passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)  # Exit with 0 if all tests passed, 1 otherwise 