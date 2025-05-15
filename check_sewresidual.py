import sys

print("Checking for SEWResidual in SpikingJelly...")

try:
    # Try the new module structure first
    from spikingjelly.activation_based import layer
    print("Successfully imported spikingjelly.activation_based.layer")
    
    # Print all available attributes in the layer module
    print("Available attributes in layer module:")
    for attr in dir(layer):
        print(f"- {attr}")
    
    if hasattr(layer, 'SEWResidual'):
        print("\nSEWResidual is AVAILABLE")
    else:
        print("\nSEWResidual is NOT AVAILABLE")
except ImportError as e:
    print(f"Error importing spikingjelly.activation_based.layer: {e}")
    
    # Try the old module structure as fallback
    try:
        from spikingjelly.clock_driven import layer
        print("Successfully imported spikingjelly.clock_driven.layer")
        
        # Print all available attributes in the layer module
        print("Available attributes in layer module:")
        for attr in dir(layer):
            print(f"- {attr}")
        
        if hasattr(layer, 'SEWResidual'):
            print("\nSEWResidual is AVAILABLE")
        else:
            print("\nSEWResidual is NOT AVAILABLE")
    except ImportError as e:
        print(f"Error importing spikingjelly.clock_driven.layer: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
    
print("Check complete.") 