
from pathlib import Path
from environ import Env
import os

# Initialise l'environnement
env = Env()

# Build paths inside the project like this: BASE_DIR / 'subdir'.
BASE_DIR = Path(__file__).resolve().parent.parent

# Directory to export the model
EXPORT_DIR = os.path.join(BASE_DIR, 'data')

# Take environment variables from .env file
Env.read_env(os.path.join(BASE_DIR, '.env'))

