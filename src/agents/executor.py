"""
CodeExecutor: runs Python code in a subprocess with a configurable timeout.

The code is written to a NamedTemporaryFile so the subprocess receives a
real path (no stdin piping required).  The working directory is set to
``working_dir`` so that relative CSV paths inside the generated code resolve
correctly.
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path


class CodeExecutor:
    """Execute arbitrary Python code in a subprocess and capture its output."""

    def __init__(self, timeout: int = 30) -> None:
        self.timeout = timeout

    def run(self, code: str, working_dir: Path) -> tuple[str, str]:
        """
        Write *code* to a temp file, execute it with ``python``, and return
        ``(stdout, stderr)`` both stripped of leading/trailing whitespace.

        On ``TimeoutExpired`` the process is killed and ``stderr`` is set to a
        descriptive ``TimeoutError`` string.  The temp file is always removed.
        """
        tmp_path: str | None = None
        try:
            # Write to a named temp file (suffix ensures Python recognises it)
            with tempfile.NamedTemporaryFile(
                mode="w",
                suffix=".py",
                delete=False,
                encoding="utf-8",
            ) as tmp:
                tmp.write(code)
                tmp_path = tmp.name

            env = os.environ.copy()

            result = subprocess.run(
                [sys.executable, tmp_path],
                capture_output=True,
                text=True,
                cwd=str(working_dir),
                env=env,
                timeout=self.timeout,
            )
            return result.stdout.strip(), result.stderr.strip()

        except subprocess.TimeoutExpired:
            return "", f"TimeoutError: code execution exceeded {self.timeout}s"

        finally:
            if tmp_path is not None:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
