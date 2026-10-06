"""Execute Lab 3 and embed outputs in the same notebook."""
import json
import os
from pathlib import Path
import sys
import nbformat
from nbclient import NotebookClient

sys.stdout.reconfigure(encoding="utf-8")
folder = Path(__file__).resolve().parent
kernel = folder / ".jupyter" / "kernels" / "lab3-python"
kernel.mkdir(parents=True, exist_ok=True)
(kernel / "kernel.json").write_text(json.dumps({
    "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
    "display_name": "Python (Lab 3)", "language": "python"
}), encoding="utf-8")
os.environ["JUPYTER_PATH"] = str(folder / ".jupyter")
os.environ["PYTHONHASHSEED"] = "42"
os.environ["MPLBACKEND"] = "module://matplotlib_inline.backend_inline"
for variable in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ[variable] = "1"
path = folder / "word_embedding.ipynb"
nb = nbformat.read(path, as_version=4)

def started(cell, cell_index, **kwargs):
    if cell.cell_type == "code":
        print(f"Running code cell {cell_index}: {cell.source.splitlines()[0][:90]}", flush=True)

def finished(cell, cell_index, **kwargs):
    if cell.cell_type == "code":
        nbformat.write(nb, path)
        print(f"Completed cell {cell_index}", flush=True)

class ProgressClient(NotebookClient):
    def process_message(self, msg, cell, cell_index):
        content = msg.get("content", {})
        if msg.get("msg_type") == "stream" and "Trained " in content.get("text", ""):
            for line in content["text"].splitlines():
                if line.startswith("Trained "):
                    print(line, flush=True)
        return super().process_message(msg, cell, cell_index)

client = ProgressClient(nb, kernel_name="lab3-python", timeout=1800,
    resources={"metadata": {"path": str(folder)}},
    on_cell_start=started, on_cell_executed=finished)
try:
    client.execute()
finally:
    nbformat.write(nb, path)
print("Notebook execution finished.", flush=True)
