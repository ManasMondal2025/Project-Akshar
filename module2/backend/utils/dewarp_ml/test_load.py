import os
import torch
import sys

# Add backend to path so imports work
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from utils.dewarp_ml.model import DewarpTextlineMaskGuide

def test_load():
    model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../pretrained_models/30.pt"))
    print(f"Loading from {model_path}")
    
    model = DewarpTextlineMaskGuide(image_size=224)
    state_dict = torch.load(model_path, map_location="cpu")
    
    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
        
    try:
        model.load_state_dict(state_dict, strict=True)
        print("SUCCESS! Model loaded perfectly with strict=True.")
    except Exception as e:
        print("FAILED to load model:")
        print(e)

if __name__ == "__main__":
    test_load()
