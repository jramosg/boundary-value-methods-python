"""Execute the learning notebooks and save their outputs; fail on errors."""
from pathlib import Path
import sys

import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager

root = Path(__file__).resolve().parents[1]
for path in sorted((root/'notebooks').glob('*.ipynb')):
    nb = nbformat.read(path, as_version=4)
    manager = KernelManager(kernel_name='python3')
    manager.kernel_spec.argv[0] = sys.executable
    NotebookClient(nb, km=manager, timeout=180,
                   resources={'metadata': {'path': str(root)}}).execute()
    nbformat.write(nb, path)
    print(f'Executed {path.name}', flush=True)
