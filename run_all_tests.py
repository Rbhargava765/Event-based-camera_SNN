import os
import subprocess

test_files = [
    "test_environment.py",
    "test_pytorch.py",
    "test_spikingjelly.py",
    "test_opencv.py",
    "test_snn_model.py",
    "test_obstacle_avoidance.py"
]

print("RUNNING ALL SYSTEM TESTS\n" + "="*30)

for test_file in test_files:
    print(f"\nRunning {test_file}...")
    print("-" * 40)
    subprocess.run(["python", test_file])
    print("-" * 40)

print("\nAll tests completed!") 