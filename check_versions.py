#!/usr/bin/env python3
"""
Check installed package versions for ResNet-50 ImageNet training.
This script verifies that all required packages are installed and displays their versions.
"""

import sys
import importlib.metadata
from typing import Dict, Optional


def get_version(package_name: str) -> Optional[str]:
    """Get version of an installed package."""
    try:
        return importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        return None


def check_packages() -> Dict[str, Dict]:
    """Check all required packages and their versions."""

    # Core packages with minimum required versions
    packages = {
        "Core PyTorch": {
            "torch": "2.5.0",
            "torchvision": "0.20.0",
            "torchaudio": "2.5.0",
        },
        "Fast Data Loading": {
            "ffcv": "1.0.0",
            "numba": "0.60.0",
        },
        "Datasets": {
            "datasets": "3.2.0",
            "huggingface_hub": "0.26.0",
            "transformers": "4.46.0",
            "pyarrow": "18.0.0",
        },
        "Image Processing": {
            "opencv-python": "4.10.0",
            "Pillow": "11.0.0",
            "albumentations": "1.4.0",
        },
        "Numerical Computing": {
            "numpy": "1.26.4",
            "scipy": "1.14.1",
        },
        "Training Utilities": {
            "tqdm": "4.67.0",
            "matplotlib": "3.9.2",
            "torchinfo": "1.8.0",
            "pyyaml": "6.0.2",
        },
        "Experiment Tracking (Optional)": {
            "wandb": "0.18.0",
            "tensorboard": "2.18.0",
            "tensorboardX": "2.6.2",
        },
        "Distributed Training (Optional)": {
            "mpi4py": "3.1.6",
        },
    }

    results = {}

    for category, pkgs in packages.items():
        results[category] = {}
        for pkg_import, min_version in pkgs.items():
            # Handle special cases for package names
            if pkg_import == "opencv-python":
                pkg_name = "opencv-python"
                import_name = "cv2"
            elif pkg_import == "Pillow":
                pkg_name = "Pillow"
                import_name = "PIL"
            elif pkg_import == "pyyaml":
                pkg_name = "PyYAML"
                import_name = "yaml"
            else:
                pkg_name = pkg_import
                import_name = pkg_import

            version = get_version(pkg_name)

            # Try to import the package
            can_import = False
            try:
                __import__(import_name)
                can_import = True
            except ImportError:
                pass

            results[category][pkg_import] = {
                "installed_version": version,
                "min_version": min_version,
                "can_import": can_import,
                "status": "OK"
                if version and can_import
                else ("Not Installed" if not version else "Import Error"),
            }

    return results


def print_results(results: Dict[str, Dict]):
    """Print the package check results in a formatted table."""
    print("\n" + "=" * 80)
    print(" ResNet-50 ImageNet Training - Package Version Check")
    print("=" * 80 + "\n")

    # Check Python version
    python_version = sys.version.split()[0]
    print(f"Python Version: {python_version}")
    if sys.version_info >= (3, 9):
        print("✅ Python version OK (3.9+ required)\n")
    else:
        print("⚠️ Python 3.9+ recommended for best compatibility\n")

    # Check CUDA availability
    try:
        import torch

        if torch.cuda.is_available():
            print(f"CUDA Available: ✅ Yes")
            print(f"CUDA Version: {torch.version.cuda}")
            print(f"GPU Count: {torch.cuda.device_count()}")
            print(f"Current GPU: {torch.cuda.get_device_name(0)}\n")
        else:
            print("CUDA Available: ❌ No (CPU only mode)\n")
    except:
        print("CUDA Check: ⚠️ Could not check (PyTorch not installed)\n")

    # Print package status
    for category, packages in results.items():
        print(f"\n{category}:")
        print("-" * 75)
        print(f"{'Package':<25} {'Installed':<15} {'Required':<15} {'Status':<10}")
        print("-" * 75)

        for pkg_name, info in packages.items():
            version = info["installed_version"] or "Not Found"
            min_version = info["min_version"]
            status = info["status"]

            # Add status emoji
            if status == "OK":
                status_icon = "✅"
            elif status == "Not Installed":
                status_icon = "❌"
            else:
                status_icon = "⚠️"

            print(
                f"{pkg_name:<25} {version:<15} {'>=' + min_version:<15} {status_icon} {status:<10}"
            )

    print("\n" + "=" * 80)

    # Summary and recommendations
    print("\n📋 Summary:")

    all_ok = True
    missing_required = []
    missing_optional = []

    for category, packages in results.items():
        for pkg_name, info in packages.items():
            if info["status"] != "OK":
                if "Optional" in category:
                    missing_optional.append(pkg_name)
                else:
                    missing_required.append(pkg_name)
                    all_ok = False

    if all_ok:
        print("✅ All required packages are installed and working!")
    else:
        if missing_required:
            print(f"\n❌ Missing required packages: {', '.join(missing_required)}")
            print("\nTo install missing required packages:")
            print("pip install -r requirements.txt")

        if missing_optional:
            print(f"\n⚠️ Missing optional packages: {', '.join(missing_optional)}")
            print("These are not required but can enhance functionality.")

    print("\n💡 Installation Tips:")
    print(
        "1. For GPU support: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124"
    )
    print("2. For FFCV: pip install ffcv")
    print("3. For all packages: pip install -r requirements.txt")
    print("4. For max performance: pip install cupy-cuda12x triton")

    print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    results = check_packages()
    print_results(results)
