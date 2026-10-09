"""Thin client used by ``app.py`` to talk to the resident server."""

from __future__ import annotations

import json
import os
import socket
import time

from .timing import Deadline


class ServerUnavailable(RuntimeError):
    """The server could not be reached or did not answer in time."""


def _roundtrip(socket_path: str, payload: dict, deadline: Deadline) -> dict:
    remaining = deadline.remaining()
    if remaining <= 0:
        raise ServerUnavailable("deadline reached before connecting")
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
            sock.settimeout(remaining)
            sock.connect(socket_path)
            sock.sendall((json.dumps(payload) + "\n").encode("utf-8"))
            buffer = b""
            while not buffer.endswith(b"\n"):
                sock.settimeout(max(0.05, deadline.remaining()))
                chunk = sock.recv(65536)
                if not chunk:
                    break
                buffer += chunk
    except (OSError, TimeoutError) as exc:
        raise ServerUnavailable(f"{type(exc).__name__}: {exc}") from exc
    try:
        return json.loads(buffer.decode("utf-8"))
    except ValueError as exc:
        raise ServerUnavailable("server sent an invalid reply") from exc


def ping(socket_path: str, timeout_s: float = 2.0) -> dict | None:
    """Return the server's ping reply, or ``None`` if it is not reachable."""
    try:
        return _roundtrip(socket_path, {"op": "ping"}, Deadline(timeout_s))
    except ServerUnavailable:
        return None


def wait_for_socket(socket_path: str, deadline: Deadline, poll_s: float = 0.1) -> bool:
    """Block until the socket file appears or the deadline passes."""
    while not os.path.exists(socket_path):
        if deadline.expired():
            return False
        time.sleep(poll_s)
    return True


def predict(socket_path: str, image_path: str, deadline: Deadline) -> dict:
    """Ask the server to read ``image_path``; raises :class:`ServerUnavailable`."""
    reply = _roundtrip(
        socket_path,
        {"op": "predict", "image": image_path, "deadline_s": deadline.remaining()},
        deadline,
    )
    if not reply.get("ok"):
        raise ServerUnavailable(str(reply.get("error", "server error")))
    return reply
