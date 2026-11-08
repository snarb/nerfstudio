"""Producer entry: durable isolated native work plus legacy MAT input aliases."""
from pathlib import Path
import argparse, json, shlex, subprocess, sys
import marshawn_sam3_assist as io


def run(V, frame):
    release = io.read(V / 'operations/body_g1_v140/entry_release.json')
    for path, digest in release['files_sha256'].items():
        if io.sha(path) != digest:
            raise ValueError('V5 producer entry source changed: ' + path)
    subprocess.run([sys.executable, release['native_cli'], '--campaign', str(V), '--frame', str(frame)], check=True)
    frame_root = V / 'operations/sam3_assist/v5' / ('frame%06d' % frame)
    native_ready = frame_root / 'all66_controller/all66_review_ready.json'
    recipe = io.read(V / 'operations/sam3_assist/v5/full_release.json')
    remote = Path(recipe['remote_root']); remote_code = remote / 'operations/sam3_assist/code'
    subprocess.run(['rsync', '-rt', release['alias_helper'], recipe['remote_host'] + ':' + str(remote_code) + '/'], check=True)
    subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8', recipe['remote_host'],
                    shlex.join([recipe['remote_python'], str(remote_code / 'marshawn_sam3_v5_props_alias.py'),
                                '--campaign', str(remote), '--frame', str(frame)])], check=True, stdout=subprocess.DEVNULL)
    alias = frame_root / 'props_alias_ready.json'
    subprocess.run(['rsync', '-rt', recipe['remote_host'] + ':' + str(remote / 'operations/sam3_assist/v5' / ('frame%06d' % frame) / alias.name), str(alias)], check=True)
    receipt = io.read(alias)
    if receipt['frame'] != frame or len(receipt['records']) != 66 or receipt['helper_sha256'] != io.sha(release['alias_helper']):
        raise ValueError('V5 MAT alias receipt differs')
    output = frame_root / 'native_inputs_ready.json'
    io.write(output, dict(frame=frame, all66_review_ready=str(native_ready), all66_review_ready_sha256=io.sha(native_ready),
                         props_alias_ready=str(alias), props_alias_ready_sha256=io.sha(alias),
                         isolated_campaign=str(frame_root / 'isolated_campaign'),
                         prepared_publication_allowed=False, live_install_owned_by_producer_source_guard=True))
    print(str(output), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--campaign', type=Path, required=True); p.add_argument('--frame', type=int, required=True)
    a = p.parse_args(); run(a.campaign.resolve(), a.frame)
