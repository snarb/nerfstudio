from pathlib import Path
import sys,tempfile,json
V=Path('/fsx/tmp/marshawn_video_1660_2015_24fps_20261107');sys.path[:0]=[str(V/'operations/sam3_assist/code'),str(V/'code/LookCloser/scripts')]
import marshawn_native_route_v111 as r
with tempfile.TemporaryDirectory()as td:
 t=Path(td);(t/'operations/native_pack_cpu8_v2').mkdir(parents=True);c=t/'operations/sam3_assist/code';c.mkdir(parents=True)
 for n in ['marshawn_mask_assist_full_v7.py','marshawn_mask_assist_full_v111.py']:(c/n).write_text(n)
 root=t/'operations/sam3_assist/v5/frame001700';root.mkdir(parents=True);(root/'existing').write_text('unchanged')
 calls=[];r.subprocess.run=lambda argv,check:calls.append(argv)
 r.run(t,1700);r.run(t,1705)
 nr=t/'operations/sam3_assist/v5/frame001705';nr.mkdir();(nr/'now_started').write_text('same owner');r.run(t,1705)
 assert [Path(a[1]).name for a in calls]==['marshawn_mask_assist_full_v7.py','marshawn_mask_assist_full_v111.py','marshawn_mask_assist_full_v111.py']
 print('PASS old-owner preserved; next-unstarted CPU8; persisted per-frame choice adopted')
