"""Render-only frame-generation contract; never relocate scientific artifacts."""
from pathlib import Path
import hashlib
import json

LOCAL = Path('/fsx/tmp/marshawn_video_1660_2015_24fps_140k_20261107')
SILVER = Path('/cache/ubuntu/lookcloser/campaigns') / LOCAL.name
GEN1 = Path('/fsx/tmp/marshawn_video_1660_2015_24fps_140k_gcrole_g1_20261108')
SGEN1 = SILVER.parent / GEN1.name
REPLACED = {1695, 1700}
GOOD_ADOPTED = {1660, 1665, 1670, 1675, 1680, 1685, 1690}
SEAL = 'd873cd61ad5c00f9501f2f6c477cb42f962d4d15591f5785830e0aeae381ec14'
GPU_SHA = '6603aa06dc8ef2df2cd89cada087729eaa40bd81be5b5c981e2e3e2369539eba'
CPU_SHA = '51239b1e3c2d969388bec12b7af907a2822b35454f08d0a7476ecb626bedc4bf'
PROFILE_SHA = '2bd9bca54f43bb5da2920a5f95124677ee2754762bb8ad5551163d9dbf06eb81'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def bound(path, expected):
    path = Path(path)
    if sha(path) != expected:
        raise ValueError('Generation-bound input changed: ' + str(path))
    return path


def expected_campaign(frame, host):
    if host not in ('olive', 'dev3', 'silver'):
        raise ValueError('Unknown physical render host')
    if int(frame) in REPLACED:
        return SGEN1 if host == 'silver' else GEN1
    return SILVER if host == 'silver' else LOCAL


def validate_claim(frame, claim, generation):
    """Validate JSON only, before touching or loading any checkpoint."""
    frame = int(frame)
    host = claim['host']
    root = expected_campaign(frame, host)
    mg = int(generation['model_generation'])
    sg = int(generation['source_generation'])
    if mg != int(frame in REPLACED) or sg != int(frame >= 1695):
        raise ValueError('Retired model/source generation selected')
    if Path(generation['model_campaign_by_host'][host]) != root:
        raise ValueError('Plan model campaign violates frame-generation policy')
    if Path(claim['host_root']) != root:
        raise ValueError('Assignment points to retired/foreign model campaign')
    # The seven original claims predate explicit generation fields. Only those
    # immutable claims may omit them, and only the manifest builder adopts them.
    for key, value in [('model_generation', mg), ('source_generation', sg)]:
        if key in claim and int(claim[key]) != value:
            raise ValueError('Claim and plan generation differ')
        if key not in claim and frame not in GOOD_ADOPTED:
            raise ValueError('New claim lacks explicit generation')
    if sg and Path(claim['source_root']) != Path(generation['source_campaign']):
        raise ValueError('Claim source generation root differs')
    return root


def manifest_row(manifest_path, index, *, spec=None, check_local_inputs=True):
    manifest = read(manifest_path)
    if manifest.get('kind') != 'FRAME_GENERATION_NATIVE_RENDER_MANIFEST_V1':
        raise ValueError('Unknown render generation manifest')
    if spec is not None:
        if Path(spec['generation_manifest']) != Path(manifest_path):
            raise ValueError('Spec selects another generation manifest')
        bound(manifest_path, spec['generation_manifest_sha256'])
    row = manifest['records'][str(int(index))]
    if int(row['index']) != int(index):
        raise ValueError('Generation index mismatch')
    frame = int(row['frame_id'])
    root = validate_claim(frame, row['assignment'], row['generation'])
    if str(root) != row['native_campaign']:
        raise ValueError('Manifest native campaign differs')
    if check_local_inputs:
        bound(row['assignment_path'], row['assignment_sha256'])
        for path, digest in row['local_source_inputs_sha256'].items():
            bound(path, digest)
    return manifest, row


def validate_native(row, *, hash_checkpoint=True):
    """Native-host checks. The original loader adds YAML/dataparser checks."""
    frame = int(row['frame_id'])
    root = validate_claim(frame, row['assignment'], row['generation'])
    directory = root / 'frames' / ('%06d' % frame)
    for relative, digest in row['native_inputs_sha256'].items():
        bound(directory / relative, digest)
    bound(root / 'code_seal.json', SEAL)
    completion = read(directory / 'frame_completion.json')
    result = read(directory / 'train/result.json')
    if completion.get('technical') is not False or result.get('technical_smoke') is not False:
        raise ValueError('Technical model cannot enter final export')
    if result['hard_max_iterations'] != 140000 or result['completed_iterations'] != 140000:
        raise ValueError('Final render requires genuine 140000-update completion')
    if result['final_step'] != 139999 or completion['final_step'] != 139999:
        raise ValueError('Terminal step/update accounting differs')
    if Path(completion['code']) != root / 'code':
        raise ValueError('Foreign model code path')
    choice = completion['selectors']['actor_lpips']
    if choice['checkpoint_sha256'] != row['checkpoint_sha256']:
        raise ValueError('Another selected checkpoint')
    checkpoint = Path(choice['checkpoint']).resolve()
    checkpoint.relative_to((directory / 'train/checkpoints').resolve())
    if hash_checkpoint:
        bound(checkpoint, row['checkpoint_sha256'])
    if int(row['generation']['source_generation']) == 1:
        publication = row['generation_source_binding']
        if (publication['model_generation'] != int(frame in REPLACED)
                or publication['source_generation'] != 1
                or Path(publication['host_root']) != root
                or publication['source_ready_sha256'] != row['native_inputs_sha256']['data/ready.json']
                or publication['assignment_sha256'] != row['assignment_sha256']
                or publication['assignment_nonce'] != row['assignment']['assignment_nonce']
                or publication['input_manifest_sha256'] != row['assignment']['input_manifest_sha256']
                or publication.get('no_model_or_learned_state_staged') is not True
                or publication.get('regular_owned_sameframe_DATA_materialized') is not True
                or not publication.get('frozen_source_support_verified')
                or publication.get('dense_all66_prerequisite_passed') is not True):
            raise ValueError('Current source publication is not qualified for this model')
        if read(directory / 'dispatch_ready.json') != row['native_dispatch']:
            raise ValueError('Current native input dispatch differs')
        if row['native_dispatch'] != publication:
            raise ValueError('Native marker is not the original source publication')
    if frame in REPLACED:
        cold = read(directory / 'train/cold_setup_audit.json')
        if (cold.get('cold') is not True or cold.get('resume') is not None
                or cold.get('trainer_start_step') != 0 or cold.get('occupied_cells') != 0
                or cold.get('nonzero_ema_cells') != 0
                or any(cold.get('optimizer_state_sizes', {}).values())):
            raise ValueError('Replacement generation was not trained from zero')
        if completion.get('cold_provenance_sha256') != row['cold_provenance_sha256']:
            raise ValueError('Cold provenance differs from approved manifest')
        bound(directory / 'train/cold_provenance.json', row['cold_provenance_sha256'])
    return directory


def geometry_check(directory, row, request, path_sha):
    """Same check on native host and after float-only gather."""
    receipt = read(directory / 'geometry_complete.json')
    binding, checkpoint = receipt['binding'], receipt['checkpoint_binding']
    if (receipt.get('kind') != 'NATIVE_MODEL_BOUND_FLOAT_GEOMETRY_V1'
            or binding['path_sha256'] != path_sha
            or binding['record'] != request['records'][int(row['index'])]
            or binding['worker_sha256'] != GPU_SHA
            or binding['data_ready_sha256'] != row['native_inputs_sha256']['data/ready.json']
            or binding['transforms_sha256'] != row['native_inputs_sha256']['data/transforms.json']
            or binding['frame_completion_sha256'] != row['native_inputs_sha256']['frame_completion.json']
            or checkpoint['production_campaign'] != row['native_campaign']
            or checkpoint['checkpoint_sha256'] != row['checkpoint_sha256']
            or checkpoint['completed_iterations'] != 140000
            or receipt['pipeline_tensor_SHA_before'] != receipt['pipeline_tensor_SHA_after']
            or receipt['native_HC_parity']['max_RGB_byte_difference'] > 1):
        raise ValueError('Foreign generation, model, source or shader geometry')
    for name, digest in receipt['files_sha256'].items():
        target = (directory / name).resolve()
        target.relative_to(directory.resolve())
        bound(target, digest)
    return dict(verified=True, geometry_receipt_sha256=sha(directory / 'geometry_complete.json'),
                checkpoint_sha256=row['checkpoint_sha256'], native_campaign=row['native_campaign'])

