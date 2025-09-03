#!/usr/bin/env python3
"""
Test script to verify all required dependencies are available
"""

def test_imports():
    """Test importing all required packages"""
    
    try:
        import sys
        print(f"Python version: {sys.version}")
        print("Testing imports...")
        
        # Core scientific computing
        import numpy as np
        print(f"+ numpy {np.__version__}")
        
        import scipy
        print(f"+ scipy {scipy.__version__}")
        
        import pandas as pd
        print(f"+ pandas {pd.__version__}")
        
        # Machine learning
        import sklearn
        print(f"+ scikit-learn {sklearn.__version__}")
        
        # Data visualization
        import matplotlib
        print(f"+ matplotlib {matplotlib.__version__}")
        
        import seaborn as sns
        print(f"+ seaborn {sns.__version__}")
        
        # Data I/O
        import h5py
        print(f"+ h5py {h5py.__version__}")
        
        # Network analysis
        import networkx as nx
        print(f"+ networkx {nx.__version__}")
        
        # Built-in modules used in the project
        import os
        import warnings
        from collections import defaultdict
        from itertools import combinations
        from pathlib import Path
        from datetime import datetime
        
        print("\n+ All dependencies successfully imported!")
        print("The environment is ready for MAP2 neural data analysis.")
        
        return True
        
    except ImportError as e:
        print(f"- Import error: {e}")
        return False
    except Exception as e:
        print(f"- Unexpected error: {e}")
        return False

if __name__ == "__main__":
    success = test_imports()
    exit(0 if success else 1)