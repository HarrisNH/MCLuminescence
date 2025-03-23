# paths.py
import os

# For example, define your project root and common subdirectories:
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
CONFIG_DIR = os.path.join(PROJECT_ROOT, "conf")