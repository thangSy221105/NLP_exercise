from pathlib import Path
import nbformat
from nbclient import NotebookClient
p=Path(__file__).resolve().parent
f=p/"transformer_classification.ipynb"
nb=nbformat.read(f,as_version=4)
try:
 NotebookClient(nb,timeout=7200,resources={"metadata":{"path":str(p)}}).execute()
finally:
 nbformat.write(nb,f)
