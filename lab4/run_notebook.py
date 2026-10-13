"""Run Lab 4; keep outputs embedded in the same notebook."""
import asyncio
import json
import os
from pathlib import Path
import sys
import nbformat
from nbclient import NotebookClient
from preprocessing import SEED

sys.stdout.reconfigure(encoding="utf-8")
if sys.platform=="win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
folder=Path(__file__).resolve().parent
kernel=folder/".jupyter"/"kernels"/"lab4-python"
kernel.mkdir(parents=True,exist_ok=True)
(kernel/"kernel.json").write_text(json.dumps({
    "argv":[sys.executable,"-m","ipykernel_launcher","-f","{connection_file}"],
    "display_name":"Python (Lab 4)","language":"python"}),encoding="utf-8")
os.environ["JUPYTER_PATH"]=str(folder/".jupyter")
os.environ["PYTHONHASHSEED"]=str(SEED)
os.environ["MPLBACKEND"]="module://matplotlib_inline.backend_inline"
for name in ["OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"]:
    os.environ[name]="1"
path=folder/"classification.ipynb"
nb=nbformat.read(path,as_version=4)
def started(cell,cell_index,**kwargs):
    if cell.cell_type=="code":
        print(f"Running cell {cell_index}: {cell.source.splitlines()[0][:90]}",flush=True)
def finished(cell,cell_index,**kwargs):
    if cell.cell_type=="code":
        nbformat.write(nb,path)
        print(f"Completed cell {cell_index}",flush=True)
client=NotebookClient(nb,kernel_name="lab4-python",timeout=1800,
    resources={"metadata":{"path":str(folder)}},on_cell_start=started,on_cell_executed=finished)
try:
    client.execute()
finally:
    nbformat.write(nb,path)
print("Notebook execution finished.",flush=True)
