"""Low-overhead process and accelerator resource measurements for training fits."""

from __future__ import annotations

from dataclasses import dataclass
from threading import Event, Lock, Thread

import psutil  # type: ignore[import-untyped]
import torch


@dataclass(frozen=True)
class TrainingResourceUsage:
    """Observed resource usage between monitor start and stop."""

    baseline_rss_bytes: int
    peak_rss_bytes: int
    cpu_user_seconds: float
    cpu_system_seconds: float
    peak_mps_tensor_bytes: int
    peak_mps_driver_bytes: int
    peak_cuda_allocated_bytes: int
    peak_cuda_reserved_bytes: int

    @property
    def rss_increase_bytes(self) -> int:
        return max(0, self.peak_rss_bytes - self.baseline_rss_bytes)


class TrainingResourceMonitor:
    """Sample process RSS and MPS memory while one fit is training."""

    def __init__(self, device: torch.device, *, interval_seconds: float = 0.05) -> None:
        if interval_seconds <= 0:
            raise ValueError("resource monitoring interval must be positive")
        self.device = device
        self.interval_seconds = interval_seconds
        self.process = psutil.Process()
        self._stop = Event()
        self._lock = Lock()
        self._thread: Thread | None = None
        self._started = False
        self._baseline_rss = 0
        self._peak_rss = 0
        self._peak_mps_tensor = 0
        self._peak_mps_driver = 0
        self._cpu_user_start = 0.0
        self._cpu_system_start = 0.0

    def _sample(self) -> None:
        rss = self.process.memory_info().rss
        mps_tensor = 0
        mps_driver = 0
        if self.device.type == "mps":
            mps_tensor = int(torch.mps.current_allocated_memory())
            mps_driver = int(torch.mps.driver_allocated_memory())
        with self._lock:
            self._peak_rss = max(self._peak_rss, rss)
            self._peak_mps_tensor = max(self._peak_mps_tensor, mps_tensor)
            self._peak_mps_driver = max(self._peak_mps_driver, mps_driver)

    def _sample_until_stopped(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            self._sample()

    def start(self) -> None:
        """Start monitoring; may be called once."""
        if self._started:
            raise RuntimeError("resource monitor has already started")
        self._started = True
        cpu = self.process.cpu_times()
        self._cpu_user_start = float(cpu.user)
        self._cpu_system_start = float(cpu.system)
        self._baseline_rss = self.process.memory_info().rss
        self._peak_rss = self._baseline_rss
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        self._sample()
        self._thread = Thread(target=self._sample_until_stopped, daemon=True)
        self._thread.start()

    def stop(self) -> TrainingResourceUsage:
        """Stop monitoring and return the observed usage."""
        if not self._started:
            raise RuntimeError("resource monitor has not started")
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        self._sample()
        cpu = self.process.cpu_times()
        with self._lock:
            peak_rss = self._peak_rss
            peak_mps_tensor = self._peak_mps_tensor
            peak_mps_driver = self._peak_mps_driver
        peak_cuda_allocated = 0
        peak_cuda_reserved = 0
        if self.device.type == "cuda":
            peak_cuda_allocated = int(torch.cuda.max_memory_allocated(self.device))
            peak_cuda_reserved = int(torch.cuda.max_memory_reserved(self.device))
        return TrainingResourceUsage(
            baseline_rss_bytes=self._baseline_rss,
            peak_rss_bytes=peak_rss,
            cpu_user_seconds=max(0.0, float(cpu.user) - self._cpu_user_start),
            cpu_system_seconds=max(0.0, float(cpu.system) - self._cpu_system_start),
            peak_mps_tensor_bytes=peak_mps_tensor,
            peak_mps_driver_bytes=peak_mps_driver,
            peak_cuda_allocated_bytes=peak_cuda_allocated,
            peak_cuda_reserved_bytes=peak_cuda_reserved,
        )
