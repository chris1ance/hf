"""Utility functions for GPU memory management and hardware information retrieval.

This module provides utility functions for managing GPU memory and querying GPU hardware
information. It includes functions for clearing GPU memory and retrieving VRAM information
from NVIDIA GPUs.

Functions
---------
clear_gpu_mem() -> None
    Clear GPU memory by collecting garbage and emptying the cache.

Example:
    >>> clear_gpu_mem()  # Frees up GPU memory

get_nvidia_gpu_vram() -> float | None
    Get total available VRAM across all NVIDIA GPUs in GiB.

Example:
    >>> vram = get_nvidia_gpu_vram()
    >>> print(f"Available VRAM: {vram} GiB")
    Available VRAM: 8.45 GiB

"""

import gc
import re
import subprocess

import torch

########################################################################################


def clear_gpu_mem() -> None:
    """Clear GPU memory by collecting garbage and emptying the cache."""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def get_nvidia_gpu_vram() -> float | None:
    """Retrieve the total available VRAM of NVIDIA GPUs in the system.

    This function executes the 'nvidia-smi' command to query the free memory of each NVIDIA GPU installed in the system.
    It then calculates and returns the sum of the available VRAM across all detected GPUs. The VRAM values are
    converted from MiB to GiB with two decimal places of precision.

    Returns:
    -------
        float | None: The total available VRAM in GiB across all NVIDIA GPUs in the system. If 'nvidia-smi' fails to
        execute, or if there are no NVIDIA GPUs present, the function returns None.

    Raises:
    ------
        subprocess.CalledProcessError: If the 'nvidia-smi' command fails to execute, this error is raised, and a
        message is printed to the console. It indicates that the NVIDIA drivers may not be installed or 'nvidia-smi'
        is not in the system's PATH.

    Note:
    ----
        This function requires NVIDIA drivers and the 'nvidia-smi' tool to be installed on the host system.

    """
    try:
        # Run the nvidia-smi command to get GPU details
        nvidia_smi_output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader"], encoding="utf-8"
        )

        # Extract VRAM values for each GPU
        vram_values = re.findall(r"\d+", nvidia_smi_output)
        vram_available_per_gpu = [round(float(vram) / 1024, 2) for vram in vram_values]
        total_vram_available = sum(vram_available_per_gpu)

        return total_vram_available
    except subprocess.CalledProcessError:
        print(
            "Failed to execute nvidia-smi. Make sure NVIDIA drivers are installed and nvidia-smi is in your PATH."
        )
        return None
