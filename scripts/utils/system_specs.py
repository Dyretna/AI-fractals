import os
import platform

import torch


def get_system_specs_str():
    header = "\n" + "=" * 80 + "\n System Specs \n" + "=" * 80
    rows = [header]

    rows.append(f"OS: {platform.system()} {platform.release()}")
    rows.append(f"Machine: {platform.machine()}")
    rows.append(f"CPU: {platform.processor()}")
    rows.append(f"Cores: {os.cpu_count()}")
    rows.append(f"Python: {platform.python_version()}")
    rows.append(f"PyTorch: {torch.__version__}")
    rows.append(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        rows.append(f"  CUDA version: {torch.version.cuda}")
        rows.append(f"  GPU: {torch.cuda.get_device_name(0)}")
        rows.append(f"  GPU capability: {torch.cuda.get_device_capability(0)}")
        mem = torch.cuda.get_device_properties(0).total_memory // (1024**2)
        rows.append(f"  GPU memory: {mem} MB")
    rows.append("=" * 80)

    return "\n".join(rows)
