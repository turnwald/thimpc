# Environment Strategy: Venv, Dev Container, Codespaces

WSL/Ubuntu with a local `.venv` is the primary day-to-day and live-teaching
workflow. It avoids relying on container rebuilds, extension downloads, image
availability, or VS Code server startup during class.

The VS Code Dev Container remains the reproducible validation environment. It
uses the same `requirements.txt` as local venv setup so the two paths stay
aligned.

GitHub Codespaces is the browser fallback. It should use the same devcontainer
configuration and should be validated in a fresh Codespace before public
release.

No conda, Poetry, PDM, wheelhouse, or hidden environment manager is introduced.
The optional `tools/setup_venv.sh` script is intentionally a small wrapper
around Python's built-in `venv`, pip, and ipykernel registration.
