"""Keep the scheduling ledger available after an evacuated workstation disappears."""
import json
import shlex
import time
from pathlib import Path

from campaign import Host
from common import atomic, read


class RetiringHost(Host):
    def __init__(self, row, campaign_root):
        super().__init__(row)
        self.enabled = row['name'] in ('ws07', 'ws08')
        self.cache = Path(campaign_root) / 'evacuated_hosts' / row['name']
        self.backup = Path(campaign_root).parent.parent / 'emergency_workstations_20261111'
        self.offline = self.enabled and (self.cache / 'offline.json').exists()

    def cached_path(self, path):
        relative = Path(path).relative_to(self.root)
        return self.cache / 'files' / relative

    def activate(self):
        if self.offline:
            return
        queue_path = self.cached_path(self.agent_root / 'queue.json')
        state_path = self.cached_path(self.agent_root / 'agent_state.json')
        if not queue_path.exists() or not state_path.exists():
            raise RuntimeError('Workstation observation cache is incomplete')
        verified = set()
        folder = self.backup / self.row['name']
        receipt = folder / 'checkpoint_verification.json'
        inventory = folder / 'source_inventory.json'
        if receipt.exists() and inventory.exists():
            check = read(receipt)
            records = read(inventory)['records']
            if not check['errors'] and check['verified'] == len(records):
                verified = {r['sha256'] for r in records if '/model/' in r['path']}
        state = read(state_path)
        retained = {}
        for key, value in state.get('jobs', {}).items():
            if value.get('status') != 'complete':
                continue
            completion = self.cached_path(value['job']['completion'])
            if completion.exists() and read(completion).get('checkpoint_sha256') in verified:
                retained[key] = value
        state['jobs'] = retained
        state['active'] = None
        atomic(state_path, state)
        self.offline = True
        atomic(self.cache / 'offline.json', dict(host=self.row['name'], policy='Requeue unfinished or unbacked work'))

    def get(self, path, default=None):
        if not self.enabled:
            return super().get(path, default)
        if not self.offline:
            value = super().get(path, default)
            if value is not None:
                atomic(self.cached_path(path), value)
                return value
            if Path(path) == self.agent_root / 'queue.json':
                self.activate()
            else:
                return default
        cached = self.cached_path(path)
        return read(cached) if cached.exists() else default

    def put(self, path, value):
        if self.offline:
            atomic(self.cached_path(path), value)
        else:
            super().put(path, value)
            if self.enabled:
                atomic(self.cached_path(path), value)

    def command(self, command, input=None, timeout=60):
        if self.offline:
            if command.startswith('rm -f '):
                path = shlex.split(command)[2]
                self.cached_path(path).unlink(missing_ok=True)
                return ''
            raise RuntimeError('Execution requested on an evacuated workstation')
        try:
            result = super().command(command, input=input, timeout=timeout)
            if self.enabled:
                (self.cache / 'connection_failures.json').unlink(missing_ok=True)
            return result
        except RuntimeError as error:
            connection_errors = ('Connection refused', 'Connection timed out', 'Connection closed',
                                 'Connection reset', 'No route to host', 'Could not resolve',
                                 'connect to host', 'kex_exchange_identification')
            if self.enabled and any(message in str(error) for message in connection_errors):
                failures = self.cache / 'connection_failures.json'
                history = read(failures) if failures.exists() else dict(first=time.time(), count=0)
                history['count'] += 1
                atomic(failures, history)
                if history['count'] < 3 or time.time() - history['first'] < 120:
                    raise
                self.activate()
                return self.command(command, input=input, timeout=timeout)
            raise
