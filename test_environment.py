import sys
import platform

print(f"Python version: {sys.version}")
print(f"Platform: {platform.platform()}")
print(f"64-bit: {platform.architecture()[0]}") 