"""Seal final immutable native-campaign choices from the qualified F2 claims."""
from pathlib import Path
import hashlib
import json
import subprocess
import shlex
from policy import (GOOD_ADOPTED, LOCAL, SEAL, bound, expected_campaign,
                    read, sha, validate_claim)

LEGACY_ASSIGNMENTS = Path('/fsx/tmp/marshawn_video_1660_2015_24fps_20261107/operations/fleet10/assignments')


def claim(plan, state, frame):
    path = Path(plan['assignments']) / ('%06d.json' % frame)
    adopted = False
    if not path.exists():
        if frame not in GOOD_ADOPTED:
            raise FileNotFoundError('Genuine generation assignment not ready: ' + str(path))
        path = LEGACY_ASSIGNMENTS / path.name
        adopted = True
    value = read(path)
    bound(path, state['frames'][str(frame)]['assignment_sha256'])
    return path, value, adopted


def build(fleet_root, request, collector, *, call, limit=None):
    """`call(host,args)` transports metadata; it never copies models or YAMLs."""
    fleet_root = Path(fleet_root)
    plan_path = fleet_root / 'video_plan.json'
    plan = read(plan_path)
    state = read(fleet_root / 'state.json')
    # Exact frozen marshawn_tuning_controller.digest serialization (its default
    # JSON separators include spaces; the file SHA is a separate binding).
    plan_digest = hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()
    if state['manifest_sha256'] != plan_digest:
        raise ValueError('F2 scientific state belongs to another canonical plan digest')
    adoption = read(fleet_root / 'adoption.json')
    if set(adoption['seven_good_completed_frames']) != GOOD_ADOPTED:
        raise ValueError('Legacy completion adoption set differs')
    records = {}
    for index, record in enumerate(request['records'][:72]):
        if limit is not None and index >= limit:
            break
        frame = int(record['frame_id'])
        current = state['frames'][str(frame)]
        current_job = state['jobs'][current['name']]
        if current_job.get('status') != 'complete':
            raise ValueError('Current frame generation is not fleet-complete: ' + str(frame))
        path, assignment, adopted = claim(plan, state, frame)
        generation = plan['frame_generations'][str(frame)]
        native = validate_claim(frame, assignment, generation)
        host = plan['hosts'][assignment['host']]
        command = [host['python'], str(collector(assignment['host'])),
                   '--root', str(native), '--frame', str(frame)]
        proof = json.loads(call(host, command).stdout)
        completion, result = proof['completion'], proof['result']
        if proof['code_seal_sha256'] != SEAL:
            raise ValueError('Unqualified native code seal')
        if proof['recipe_sha256'] != sha(LOCAL / 'recipe.json'):
            raise ValueError('Native model recipe differs from the selected frozen recipe')
        if result['completed_iterations'] != 140000 or result['final_step'] != 139999:
            raise ValueError('Frame not genuinely terminal at the selected hard cap')
        local_root = expected_campaign(frame, 'olive')
        local_data = local_root / 'frames' / ('%06d' % frame) / 'data'
        local_inputs = {}
        for name in ['ready.json', 'transforms.json']:
            bound(local_data / name, proof['inputs_sha256']['data/' + name])
            local_inputs[str(local_data / name)] = sha(local_data / name)
        row = dict(index=index, frame_id=record['frame_id'],
                   fleet_current_job=current['name'], fleet_current_job_status='complete',
                   assignment=assignment, assignment_path=str(path),
                   assignment_sha256=sha(path), adopted_original_claim=adopted,
                   generation=generation, native_campaign=str(native),
                   native_inputs_sha256=proof['inputs_sha256'],
                   checkpoint_sha256=completion['selectors']['actor_lpips']['checkpoint_sha256'],
                   selected_step=completion['selectors']['actor_lpips']['step'],
                   completed_iterations=result['completed_iterations'],
                   cold_provenance_sha256=completion['cold_provenance_sha256'],
                   local_data_template=str(local_root / 'frames/{frame_id}/data'),
                   local_source_inputs_sha256=local_inputs,
                   native_dispatch=proof['dispatch'],
                   native_metadata=proof)
        if int(generation['source_generation']) == 1:
            authorization_path = Path(assignment['source_generation_authorization'])
            bound(authorization_path, assignment['source_generation_authorization_sha256'])
            authorization = read(authorization_path)
            if (authorization.get('authorized') is not True
                    or authorization.get('source_generation') != 1
                    or Path(authorization['source_campaign']) != Path(generation['source_campaign'])
                    or authorization.get('model_code_seal_sha256', authorization.get('code_seal_sha256')) != SEAL
                    or not (authorization.get('frame') == frame or frame in authorization.get('frames', []))):
                raise ValueError('Current generation source lacks immutable ROOT authorization')
            if frame in (1695, 1700) and authorization_path.name != '%06d.json' % frame:
                raise ValueError('Cold replacement requires its own per-frame ROOT authorization')
            row['source_authorization'] = authorization
            row['local_source_inputs_sha256'][str(authorization_path)] = sha(authorization_path)
            publication_path = Path(generation['source_campaign']) / 'operations/advanced_inputs' / ('%06d' % frame) / 'host_route_publication.json'
            publication = read(publication_path)
            row['generation_source_binding'] = publication
            row['generation_source_publication_path'] = str(publication_path)
            row['generation_source_publication_sha256'] = sha(publication_path)
            row['local_source_inputs_sha256'][str(publication_path)] = sha(publication_path)
        records[str(index)] = row
    return dict(kind='FRAME_GENERATION_NATIVE_RENDER_MANIFEST_V1',
                fleet_plan_path=str(plan_path), fleet_plan_sha256=sha(plan_path),
                fleet_adoption_sha256=sha(fleet_root / 'adoption.json'),
                expected_neural_records=72, records=records,
                partial_qualification_only=limit is not None,
                retired_model_frames_replaced=[1695, 1700],
                actual_GT_records_have_no_model_or_perframe_DATA=True,
                models_YAML_and_scientific_inputs_not_relocated=True)
