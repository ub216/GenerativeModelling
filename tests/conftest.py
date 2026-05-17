import os
import sys

# Ensure repo root is on the path so imports like `from models.diffusion import ...` work
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
