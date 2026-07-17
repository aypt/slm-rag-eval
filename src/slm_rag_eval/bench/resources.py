"""Sample GPU and memory use while a leg runs, so R4 is measured rather than remembered.

Peak VRAM is the number that supports "this deploys on constrained hardware", and it can
only be read while the model is resident. Asking an operator to watch `nvidia-smi` during a
five-hour run is how that number ends up estimated after the fact; a background sampler
records it without anyone watching.

Degrades to nothing useful rather than failing: no `nvidia-smi`, no GPU, or no `/proc` all
leave the fields empty instead of raising, because the same code has to run on the laptop
where the pipeline is rehearsed.
"""

from __future__ import annotations

import shutil
import subprocess
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import TracebackType


@dataclass
class ResourceUsage:
    """Peak resource use observed across one leg."""

    samples: int = 0
    duration_s: float = 0.0
    peak_gpu_memory_mib: dict[str, int] = field(default_factory=dict)
    """Highest used VRAM per GPU index, in MiB."""
    peak_process_rss_mib: int | None = None
    """Highest resident set size of this process. Excludes the model server."""
    peak_system_used_mib: int | None = None
    """Highest system-wide memory in use, which is what a local model actually occupies."""
    gpu_names: dict[str, str] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        """Plain dictionary for the manifest."""
        return asdict(self)

    def render(self) -> str:
        """One line for the run log."""
        if self.peak_gpu_memory_mib:
            gpu = ", ".join(
                f"GPU{index} {value} MiB"
                for index, value in sorted(self.peak_gpu_memory_mib.items())
            )
        else:
            gpu = "no GPU detected"
        system = (
            f", system peak {self.peak_system_used_mib} MiB" if self.peak_system_used_mib else ""
        )
        return f"peak {gpu}{system} over {self.samples} samples in {self.duration_s:.0f}s"


def _nvidia_smi_available() -> bool:
    return shutil.which("nvidia-smi") is not None


def read_gpu_memory() -> tuple[dict[str, int], dict[str, str]]:
    """Used VRAM per GPU right now, plus the GPU names. Empty on a CPU-only machine."""
    if not _nvidia_smi_available():
        return {}, {}
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return {}, {}

    used: dict[str, int] = {}
    names: dict[str, str] = {}
    for line in completed.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 3:
            continue
        index, name, memory = parts[0], parts[1], parts[2]
        try:
            used[index] = int(memory)
        except ValueError:
            continue
        names[index] = name
    return used, names


def read_process_rss_mib() -> int | None:
    """This process's resident set size. Note the judge usually runs in another process."""
    try:
        with Path("/proc/self/status").open(encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) // 1024
    except (OSError, ValueError, IndexError):
        return None
    return None


def read_system_used_mib() -> int | None:
    """System memory in use — total minus available, which is what a resident model shows up in."""
    total = available = None
    try:
        with Path("/proc/meminfo").open(encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("MemTotal:"):
                    total = int(line.split()[1])
                elif line.startswith("MemAvailable:"):
                    available = int(line.split()[1])
                if total is not None and available is not None:
                    break
    except (OSError, ValueError, IndexError):
        return None
    if total is None or available is None:
        return None
    return (total - available) // 1024


class ResourceSampler:
    """Context manager that records peak usage in a background thread.

    Used as `with ResourceSampler() as sampler: ...` around a leg; read `sampler.usage`
    afterwards. The thread is a daemon and the interval is coarse, because this must not
    perturb the latency it sits beside.
    """

    def __init__(self, interval_s: float = 5.0) -> None:
        self._interval_s = interval_s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.usage = ResourceUsage()
        self._started_at = 0.0

    def __enter__(self) -> ResourceSampler:
        self._started_at = time.perf_counter()
        self._sample_once()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="resource-sampler")
        self._thread.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self._interval_s + 5)
        self._sample_once()
        self.usage.duration_s = time.perf_counter() - self._started_at

    def _loop(self) -> None:
        while not self._stop.wait(self._interval_s):
            self._sample_once()

    def _sample_once(self) -> None:
        gpu, names = read_gpu_memory()
        for index, value in gpu.items():
            if value > self.usage.peak_gpu_memory_mib.get(index, -1):
                self.usage.peak_gpu_memory_mib[index] = value
        self.usage.gpu_names.update(names)

        rss = read_process_rss_mib()
        if rss is not None and rss > (self.usage.peak_process_rss_mib or -1):
            self.usage.peak_process_rss_mib = rss

        system = read_system_used_mib()
        if system is not None and system > (self.usage.peak_system_used_mib or -1):
            self.usage.peak_system_used_mib = system

        self.usage.samples += 1


def ollama_resident_models(base_url: str) -> list[dict[str, object]]:
    """What the local Ollama currently holds in memory, via `ollama ps`.

    Reported alongside peak VRAM because the two answer different questions: VRAM is what
    the hardware had to provide, and this is which artifact was occupying it.
    """
    if shutil.which("ollama") is None:
        return []
    try:
        completed = subprocess.run(
            ["ollama", "ps"], capture_output=True, text=True, check=True, timeout=15
        )
    except (OSError, subprocess.SubprocessError):
        return []

    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if len(lines) < 2:
        return []
    resident: list[dict[str, object]] = []
    for line in lines[1:]:
        parts = line.split()
        if len(parts) >= 3:
            resident.append({"name": parts[0], "id": parts[1], "size": " ".join(parts[2:4])})
    return resident
