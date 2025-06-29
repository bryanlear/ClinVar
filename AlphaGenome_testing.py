from alphagenome.data import gene_annotation
from alphagenome.data import genome
from alphagenome.data import transcript as transcript_utils
from alphagenome.interpretation import ism
from alphagenome.models import dna_client
from alphagenome.models import variant_scorers
from alphagenome.visualization import plot_components
import matplotlib.pyplot as plt
import pandas as pd
import os

try:
    from dotenv import load_dotenv
    load_dotenv()
    print("Loaded .env file")
except ImportError:
    print("missing dotenv")
    print("Loading environment variables directly from .env file...")

    if os.path.exists('.env'):
        with open('.env', 'r') as f:
            for line in f:
                if '=' in line and not line.startswith('#'):
                    key, value = line.strip().split('=', 1)
                    key = key.strip()
                    value = value.strip().strip('"').strip("'")
                    os.environ[key] = value

ALPHA_GENOME_API_KEY = os.getenv('ALPHA_GENOME_API_KEY')

if not ALPHA_GENOME_API_KEY:
    raise ValueError("ALPHA_GENOME_API_KEY not found, get one")

print(f"API key loaded: {ALPHA_GENOME_API_KEY[:10]}...")


os.environ['GOOGLE_API_KEY'] = ALPHA_GENOME_API_KEY

client = dna_client.DnaClient(channel= 'gRPC')

