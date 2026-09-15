"""Six fresh-process workers, rendering only the126 needed raw mesh frames."""
import argparse
import supervise_large_motion_choices as lifecycle
from cinematic_pushin_live_ending import BASE,VARIANTS,CANARIES,RAW_IDS

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--canary',action='store_true');a=p.parse_args()
    lifecycle.BASE=BASE;lifecycle.VARIANTS=VARIANTS;lifecycle.CANARIES=CANARIES
    verified=lifecycle.verify_request
    def queue_request(root):
        request=verified(root)
        # In-memory queue view only: never alter the complete150-time request.
        return dict(request,ordered_frame_ids=RAW_IDS)
    lifecycle.verify_request=queue_request
    lifecycle.supervise(canary=a.canary)
