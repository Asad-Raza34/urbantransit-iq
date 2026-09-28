"""Diagnostic: emulate Spark's JVM-side python worker handshake so the worker's
real stderr reaches the terminal instead of vanishing into the JVM."""
import os
import socket
import subprocess
import sys
import threading

PY = sys.executable
SECRET = "diagnostic-secret"

from pyspark.serializers import read_int, write_int  # noqa: E402


def read_int_f(sock):
    return read_int(sock)


def main():
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    port = srv.getsockname()[1]
    print(f"[diag] fake JVM factory listening on 127.0.0.1:{port}", flush=True)

    env = dict(os.environ)
    env["PYTHON_WORKER_FACTORY_PORT"] = str(port)
    env["PYTHON_WORKER_FACTORY_SECRET"] = SECRET
    env["PYTHONUNBUFFERED"] = "1"

    proc = subprocess.Popen(
        [PY, "-m", "pyspark.worker"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
    )

    def pump(stream, tag):
        for line in iter(stream.readline, b""):
            print(f"[worker {tag}] {line.decode('utf-8', 'replace').rstrip()}", flush=True)

    threading.Thread(target=pump, args=(proc.stdout, "out"), daemon=True).start()
    threading.Thread(target=pump, args=(proc.stderr, "err"), daemon=True).start()

    srv.settimeout(20)
    try:
        conn, _ = srv.accept()
    except socket.timeout:
        print("[diag] TIMEOUT: worker never connected back", flush=True)
        proc.kill()
        return

    print("[diag] worker connected; doing auth handshake", flush=True)
    conn.settimeout(20)
    # Spark uses a buffered file object over the socket, not the socket itself.
    sock_file = conn.makefile("rwb", int(os.environ.get("SPARK_BUFFER_SIZE", 65536)))
    try:
        n = read_int_f(sock_file)
        secret = sock_file.read(n)
        print(f"[diag] worker sent secret of len {n}: {secret!r}", flush=True)
        write_int(0, sock_file)
        sock_file.flush()
        pid = read_int_f(sock_file)
        print(f"[diag] worker pid = {pid}", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[diag] handshake failed: {type(exc).__name__}: {exc}", flush=True)
        proc.kill()
        return

    # The worker now waits for task commands (-1 == EOF == shut down).
    write_int(-1, sock_file)
    sock_file.flush()
    conn.close()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
    print(f"[diag] worker exit code: {proc.returncode}", flush=True)
    print("[diag] if the worker imported cleanly, this is a Spark-side launch issue", flush=True)


if __name__ == "__main__":
    main()
