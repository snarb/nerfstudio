"""Quiet detached watchdog; genuine completion or explicit review stops it."""
from pathlib import Path
import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
from watcher import read, sha, write


def identity(pid):
    p = Path('/proc') / str(pid)
    values = (p / 'stat').read_text().split(') ', 1)[1].split()
    return dict(pid=pid, start=values[19], boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip())


def main(root):
    lock = (root / 'watchdog.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    config = read(root / 'config.json')
    if config.get('authorized') is not True:
        raise ValueError('Unreleased passive watcher')
    binding = sha(root / 'config.json')
    write(root / 'watchdog_identity.json', dict(**identity(os.getpid()), config_sha256=binding))
    while True:
        if sha(root / 'config.json') != binding:
            raise ValueError('Live watcher configuration changed')
        with (root / 'controller_stdout.log').open('ab') as log:
            process = subprocess.Popen([sys.executable, str(root / 'watcher.py'), '--root', str(root)],
                                       stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
            write(root / 'controller_identity.json', dict(**identity(process.pid), config_sha256=binding))
            code = process.wait()
        if code in (0, 2):
            write(root / 'watchdog_terminal.json', dict(returncode=code, time=time.time()))
            return code
        time.sleep(max(30, config['poll_seconds']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    raise SystemExit(main(parser.parse_args().root))
