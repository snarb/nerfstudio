"""OPS-only final encoder override around the qualified durable render route."""
from pathlib import Path
import argparse,hashlib,importlib.util,json,sys,time
O=Path(__file__).parent;C=O.parents[1];OLD=C/'operations/video_geometry_cpu_durable_v3'
sys.path[:0]=[str(O),str(OLD),str(C/'code'),str(C/'code/LookCloser/scripts')]
from encoder import install,AwaitEncodingOwnership,sha,read

def main(args):
    release_path=O/'root_release.json';release=read(release_path)
    if release.get('approved')is not True:raise ValueError('Encoding ROOT release missing')
    spec=importlib.util.spec_from_file_location('unchanged_durable_v3',OLD/'durable_route_v3.py')
    previous=importlib.util.module_from_spec(spec);spec.loader.exec_module(previous)
    import marshawn_video_render as renderer
    original=renderer.encode;install(renderer,release_path)
    try:
        while True:
            try:return previous.main(args)
            except AwaitEncodingOwnership as error:
                previous.write(O/'ownership_wait.json',dict(time=time.time(),reason=str(error),retry_not_consumed=True))
                time.sleep(5)
    finally:renderer.encode=original

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--fleet',type=Path,required=True)
    parser.add_argument('--export-plan',type=Path,required=True);parser.add_argument('--cpu-workers',type=int,default=3)
    main(parser.parse_args())
