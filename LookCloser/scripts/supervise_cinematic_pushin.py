"""Reuse verified fresh-process lifecycle; never install twice in a worker."""
import argparse
import supervise_large_motion_choices as lifecycle
from cinematic_pushin_beauty import BASE,VARIANTS,CANARIES

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--canary',action='store_true');p.add_argument('--ending-only',action='store_true');a=p.parse_args()
    lifecycle.BASE=BASE;lifecycle.VARIANTS=VARIANTS;lifecycle.CANARIES=CANARIES
    if a.ending_only:lifecycle.CANARIES=['001123','001151','001197']
    lifecycle.supervise(canary=a.canary or a.ending_only)
