# Repository Guidelines

## Project Structure & Module Organization
- `mujoco_viewer.py`: PyQt5 application, MuJoCo rendering, and controls. Entrypoint is `main()`.
- `README.md`: usage and dependency notes.
- Runtime config: the app stores the last opened model path at `~/.mujoco_viewer_last_path.txt` (not tracked).
- Tests (when added): place under `tests/` with `test_*.py` files.

## Build, Run, and Development
- Python 3.9+ recommended. Create a virtual env:
  ```bash
  python -m venv .venv && source .venv/bin/activate
  pip install mujoco pyqt5 numpy
  # Linux GUI deps (if missing):
  sudo apt-get install -y libxcb-xinerama0 libxcb-icccm4 libxcb-image0 \
      libxcb-keysyms1 libxcb-render-util0 libxcb-xkb1 libxkbcommon-x11-0
  ```
- Run locally:
  ```bash
  python mujoco_viewer.py
  ```

## Coding Style & Naming Conventions
- Follow PEP 8. Use 4-space indentation and type hints for new/modified code.
- Naming: `PascalCase` for classes (e.g., `MujocoViewer`, `BaseControlWidget`), `snake_case` for functions (e.g., `get_object_pose_mujoco`).
- Keep UI logic thin; isolate math/model utilities for testability.
- Formatting/linting (if available locally):
  ```bash
  black mujoco_viewer.py && ruff check .
  ```

## Testing Guidelines
- Prefer `pytest`. Place tests in `tests/`, named `test_*.py`.
- Unit-test non-GUI logic (e.g., pose/quaternion helpers). GUI interactions can use `pytest-qt` or be marked as slow.
- Commands:
  ```bash
  pytest -q
  # optional coverage
  pytest --cov=. --cov-report=term-missing
  ```

## Commit & Pull Request Guidelines
- Git history is informal; adopt Conventional Commits going forward:
  - `feat:`, `fix:`, `docs:`, `refactor:`, `chore:`
  - Example: `feat(ui): add reset posture button`
- PRs should include:
  - Summary, rationale, and before/after notes
  - Linked issues, OS info, MuJoCo/PyQt versions
  - Screenshots/GIFs of the viewer where relevant

## Security & Configuration Tips
- Headless/CI: `MUJOCO_GL=egl` and `QT_QPA_PLATFORM=offscreen` (or run via `xvfb-run`).
- Wayland issues: try `QT_QPA_PLATFORM=xcb`.
- Do not commit local IDE files (e.g., `.idea/`) or secrets.
