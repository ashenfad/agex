"""Test the Modal image build to verify all dependencies are installed.

Builds the same image that _build_function() would create for the funcy
example agent, then runs a verification command inside it.

Usage:
    uv run python scripts/test_modal_image.py
"""

import sys
from pathlib import Path

import modal


def build_test_image():
    """Replicate the image build from Modal._build_function()."""
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
    image = modal.Image.debian_slim(python_version=python_version)

    # Step 1: cloudpickle
    image = image.pip_install("cloudpickle")

    # Step 2: agex from local source (same as _build_function dev mode)
    import agex

    agex_path = Path(agex.__file__).parent
    repo_path = agex_path.parent

    print(f"agex package: {agex_path}")
    print(f"repo root:    {repo_path}")
    print(
        f"pyproject:    {repo_path / 'pyproject.toml'} exists={(repo_path / 'pyproject.toml').exists()}"
    )

    if (repo_path / "pyproject.toml").exists():
        print("\n--- Dev mode: add_local_dir + pip install /agex_src ---")
        image = image.add_local_dir(str(repo_path), remote_path="/agex_src", copy=True)
        image = image.run_commands("pip install /agex_src")
    else:
        print("\n--- Production mode: pip install agex ---")
        image = image.pip_install("agex")

    # Step 3: extra packages (LLM provider)
    image = image.pip_install("google-genai")

    return image


def main():
    image = build_test_image()

    # Run a verification script inside the built image
    verify_code = """\
import sys
print(f"Python {sys.version}")
print()

# Check all expected packages
packages = [
    "agex", "kvgit", "sandtrap", "termish", "reprobate", "monkeyfs",
    "tiktoken", "diskcache", "pydantic", "pygments",
    "cloudpickle", "google.genai",
]

ok = True
for pkg in packages:
    try:
        mod = __import__(pkg)
        version = getattr(mod, "__version__", "?")
        loc = getattr(mod, "__file__", "?")
        print(f"  OK  {pkg:20s} {version:10s}  {loc}")
    except ImportError as e:
        print(f"  FAIL {pkg:20s} {e}")
        ok = False

print()

# Try the actual import chain that fails
try:
    from agex.host.base import Host
    print("  OK  agex.host.base.Host imported successfully")
except Exception as e:
    print(f"  FAIL agex.host.base: {e}")
    ok = False

print()
if ok:
    print("ALL CHECKS PASSED")
else:
    print("SOME CHECKS FAILED")
    sys.exit(1)
"""

    app = modal.App("agex-image-test")
    verify_fn = app.function(image=image, serialized=True)(lambda: exec(verify_code))

    print("\nBuilding image and running verification...\n")
    with app.run():
        verify_fn.remote()


if __name__ == "__main__":
    main()
