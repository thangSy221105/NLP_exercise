"""Execute Lab 5 and preserve outputs in the same notebook."""
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
kernel=folder/".jupyter"/"kernels"/"lab5-python"
kernel.mkdir(parents=True,exist_ok=True)
(kernel/"kernel.json").write_text(json.dumps({
 "argv":[sys.executable,"-m","ipykernel_launcher","-f","{connection_file}"],
 "display_name":"Python (Lab 5)","language":"python"}),encoding="utf-8")
os.environ["JUPYTER_PATH"]=str(folder/".jupyter")
os.environ["PYTHONHASHSEED"]=str(SEED)
os.environ["MPLBACKEND"]="module://matplotlib_inline.backend_inline"
for name in ["OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"]:
    os.environ[name]="4"
path=folder/"sequence_classification.ipynb"
nb=nbformat.read(path,as_version=4)
def started(cell,cell_index,**kwargs):
    if cell.cell_type=="code":
        print(f"Running cell {cell_index}: {cell.source.splitlines()[0][:85]}",flush=True)
def finished(cell,cell_index,**kwargs):
    if cell.cell_type=="code":
        nbformat.write(nb,path)
        print(f"Completed cell {cell_index}",flush=True)
class ProgressClient(NotebookClient):
    def process_message(self,msg,cell,cell_index):
        if msg.get("msg_type")=="stream":
            for line in msg.get("content",{}).get("text","").splitlines():
                if " epoch " in line or line.startswith("Loaded completed"):
                    print(line,flush=True)
        return super().process_message(msg,cell,cell_index)
client=ProgressClient(nb,kernel_name="lab5-python",timeout=3600,
    resources={"metadata":{"path":str(folder)}},on_cell_start=started,on_cell_executed=finished)
try:
    client.execute()
finally:
    nbformat.write(nb,path)
print("Lab 5 notebook complete.",flush=True)
