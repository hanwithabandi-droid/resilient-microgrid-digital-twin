"""
STEP 5: Launch the Interactive Streamlit Digital Twin Dashboard.
Click the Play button (top right in VS Code) to run this script.
"""

import os
import sys
import subprocess

print("Step 5: Launching Streamlit dashboard...")

python_exe = sys.executable
app_path = os.path.join(os.path.dirname(__file__), "app", "app.py")

print(f"Running: {app_path}")
print("Opening at http://localhost:8501 ...")
print("Press Ctrl+C to stop.\n")

# Launch Streamlit with UTF-8 encoding
env = os.environ.copy()
env["PYTHONUTF8"] = "1"
env["PYTHONIOENCODING"] = "utf-8"

subprocess.run([
    python_exe, "-m", "streamlit", "run", app_path, "--server.port", "8501"
], env=env)
