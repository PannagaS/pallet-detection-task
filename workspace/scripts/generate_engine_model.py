import sys
import os
from ultralytics import YOLO

def convert_pt_to_engine(models_dir=None, set_half=False, set_int8=True):
    """
    1. Load `best.pt` from `models_dir`.
    2. Export it as a TensorRT `.engine` file.
    3. Save the final `best.engine` in the same directory.

    If `models_dir` is None, we assume ../models relative to this script.
    """
    if models_dir is None:
        # Get the directory where THIS script lives
        script_dir = os.path.dirname(os.path.abspath(__file__))
        models_dir = os.path.join(script_dir, "..", "models")
        models_dir = os.path.abspath(models_dir)
        print("Model path: {}".format(models_dir))

    pt_path = os.path.join(models_dir, "best.pt")
    if not os.path.isfile(pt_path):
        raise FileNotFoundError(f"Model file not found: {pt_path}")

    # Load the .pt model
    model = YOLO(pt_path)
    print("================Loaded best.pt==================")

    # Export model to TensorRT engine
    model.export(
        format="engine",   # TensorRT format
        device=0,          # GPU device (e.g. "0")
        half=set_half,     # use FP16
        project=models_dir,
        int8 = set_int8,    # use int8 quantization
        exist_ok=True
    )

    print("===============Exported TensorRT engine===============")

if __name__ == "__main__":
    models_dir = sys.argv[1] if len(sys.argv) > 1 else None
    set_half = sys.argv[2].lower() == "true" if len(sys.argv) > 2 else False
    set_int8 = sys.argv[3].lower() == "false" if len(sys.argv) > 3 else True

    convert_pt_to_engine(models_dir=models_dir, set_half=set_half, set_int8=set_int8)
