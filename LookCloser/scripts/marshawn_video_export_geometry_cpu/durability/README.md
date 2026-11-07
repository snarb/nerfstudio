# External export durability

These files archive the deployed OPS-only wrapper in
`C/operations/video_geometry_cpu_durable_v3`. Both qualified shaders and the
parameter profile remain unchanged. Deployment requires the campaign's
ROOT-owned release JSON; it is not copied into this source package.

The wrapper reloads the same `state.json` that the frozen fleet writes before
SSH dispatch. CPU launches have a source-bound prelaunch intent and nonce;
a child takes an execution lock, persists PID/start/boot before shader exec,
and retains its execution/capacity locks across exec. A restarting controller
adopts that owner without consuming retry budget for a busy lock. Controls
with identical bytes are not rewritten beneath active workers. Orphaned GPU
results require the existing native-model/data/camera/file-hash proof.

The bridge handles neural records only. The final actual-GT records use the
frozen `one.py --cpu` path, which requires the source-visibility adapter SHA
in the spec even though no2020 model or geometry is loaded. A one-time
metadata migration retains old qualification receipts/specs and PNG hardlinks.

CPU-only chaos tests cover four coordinator crash windows, inherited locks,
source mismatch rejection, capacity and the real frozen fleet's ACK/SSH-intent
window with a CPU receiver. Actual4K GT72/119 and a neural metadata roundtrip
were also checked; overlap and RGB code differences were zero. No GPU/model
experiment or training snapshot change is made by this wrapper.
