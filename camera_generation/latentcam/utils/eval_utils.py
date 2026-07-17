import subprocess
from pathlib import Path

def run_command_in_dir(command, work_dir):
    result = subprocess.run(
        command,
        cwd=Path(work_dir),
        capture_output=True,
        text=True
    )

    return result.stdout, result.stderr, result.returncode