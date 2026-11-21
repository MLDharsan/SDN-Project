"""Start, stop and monitor the os-ken controller processes.

Simulator controller ids map to processes as:
    0        master  -> the Python orchestrator itself (never owns switches)
    1..5     slaves  -> one osken-manager process each, TCP port base_port + id - 1

So with the default base port 6653: c1=6653, c2=6654, c3=6655, c4=6656, c5=6657.
Starting a process takes no root privileges (it only listens on a TCP port).
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from mininet_integration._paths import OSKEN_APP_PATH


def find_osken_manager() -> str:
    local = Path(sys.executable).parent / "osken-manager"
    if local.exists():
        return str(local)
    found = shutil.which("osken-manager") or shutil.which("ryu-manager")
    if not found:
        raise FileNotFoundError("osken-manager not found; install with `pip install os-ken`")
    return found


def port_is_open(host: str, port: int, timeout: float = 0.2) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(timeout)
        return sock.connect_ex((host, port)) == 0


class EventLogReader:
    """Incrementally reads a controller's JSONL event log."""

    def __init__(self, path: Path):
        self.path = path
        self.offset = 0
        self.events: List[Dict] = []

    def poll(self) -> List[Dict]:
        if not self.path.exists():
            return []
        new: List[Dict] = []
        with open(self.path, "r", encoding="utf-8") as handle:
            handle.seek(self.offset)
            while True:
                line = handle.readline()
                if not line or not line.endswith("\n"):
                    break
                self.offset = handle.tell()
                try:
                    new.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        self.events.extend(new)
        return new

    def find(self, event: str, since: float, dpid: Optional[int] = None) -> Optional[Dict]:
        self.poll()
        for record in self.events:
            if record.get("event") != event or record.get("t", 0) < since:
                continue
            if dpid is not None and int(record.get("dpid", -1)) != dpid:
                continue
            return record
        return None


@dataclass
class ControllerProcess:
    controller_id: int
    port: int
    process: subprocess.Popen
    event_log: Path
    stdout_log: Path
    started_at: float


class ControllerManager:
    def __init__(
        self,
        spec_path: Path,
        log_dir: Path,
        listen_host: str = "127.0.0.1",
        base_port: int = 6653,
        startup_timeout_s: float = 20.0,
        app_path: Path = OSKEN_APP_PATH,
    ):
        self.spec_path = Path(spec_path)
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.listen_host = listen_host
        self.base_port = base_port
        self.startup_timeout_s = startup_timeout_s
        self.app_path = Path(app_path)
        self.osken_manager = find_osken_manager()
        self.processes: Dict[int, ControllerProcess] = {}
        self.readers: Dict[int, EventLogReader] = {}

    def port_for(self, controller_id: int) -> int:
        if controller_id < 1:
            raise ValueError("Controller 0 is the master/orchestrator and has no OpenFlow port")
        return self.base_port + controller_id - 1

    def target_for(self, controller_id: int) -> str:
        return f"tcp:{self.listen_host}:{self.port_for(controller_id)}"

    def event_log_path(self, controller_id: int) -> Path:
        return self.log_dir / f"controller_{controller_id}_events.jsonl"

    def reader(self, controller_id: int) -> EventLogReader:
        if controller_id not in self.readers:
            self.readers[controller_id] = EventLogReader(self.event_log_path(controller_id))
        return self.readers[controller_id]

    def is_running(self, controller_id: int) -> bool:
        proc = self.processes.get(controller_id)
        return proc is not None and proc.process.poll() is None

    def pid(self, controller_id: int) -> Optional[int]:
        proc = self.processes.get(controller_id)
        return proc.process.pid if proc and proc.process.poll() is None else None

    def running_pids(self) -> Dict[int, int]:
        return {cid: p.process.pid for cid, p in self.processes.items() if p.process.poll() is None}

    def start(self, controller_id: int) -> Dict:
        """Start a controller and wait until its OpenFlow port accepts connections."""
        if self.is_running(controller_id):
            return {"controller_id": controller_id, "success": True, "already_running": True,
                    "startup_ms": 0.0}
        port = self.port_for(controller_id)
        if port_is_open(self.listen_host, port):
            raise RuntimeError(
                f"Port {port} is already in use by another process; stop it before running "
                "the Mininet experiment (try `python -m mininet_integration.cleanup`)."
            )
        stdout_log = self.log_dir / f"controller_{controller_id}_stdout.log"
        env = dict(os.environ)
        env.update(
            {
                "MININET_SDN_SPEC": str(self.spec_path),
                "MININET_SDN_CONTROLLER_ID": str(controller_id),
                "MININET_SDN_EVENT_LOG": str(self.event_log_path(controller_id)),
                "PYTHONDONTWRITEBYTECODE": "1",
            }
        )
        t0 = time.time()
        with open(stdout_log, "a", encoding="utf-8") as out:
            process = subprocess.Popen(
                [
                    self.osken_manager,
                    "--ofp-listen-host", self.listen_host,
                    "--ofp-tcp-listen-port", str(port),
                    str(self.app_path),
                ],
                stdout=out,
                stderr=subprocess.STDOUT,
                env=env,
                start_new_session=True,
            )
        self.processes[controller_id] = ControllerProcess(
            controller_id, port, process, self.event_log_path(controller_id), stdout_log, t0
        )
        deadline = t0 + self.startup_timeout_s
        while time.time() < deadline:
            if process.poll() is not None:
                tail = stdout_log.read_text(errors="replace")[-2000:]
                raise RuntimeError(f"Controller {controller_id} exited during startup:\n{tail}")
            if port_is_open(self.listen_host, port):
                return {"controller_id": controller_id, "success": True,
                        "startup_ms": (time.time() - t0) * 1000.0, "port": port}
            time.sleep(0.05)
        self.stop(controller_id)
        return {"controller_id": controller_id, "success": False,
                "startup_ms": (time.time() - t0) * 1000.0, "error": "startup timeout"}

    def stop(self, controller_id: int, timeout_s: float = 5.0) -> Dict:
        proc = self.processes.pop(controller_id, None)
        if proc is None or proc.process.poll() is not None:
            return {"controller_id": controller_id, "success": True, "stop_ms": 0.0,
                    "already_stopped": True}
        t0 = time.time()
        try:
            os.killpg(proc.process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            proc.process.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            os.killpg(proc.process.pid, signal.SIGKILL)
            proc.process.wait(timeout=timeout_s)
        return {"controller_id": controller_id, "success": True,
                "stop_ms": (time.time() - t0) * 1000.0}

    def stop_all(self) -> None:
        for controller_id in list(self.processes):
            self.stop(controller_id)
