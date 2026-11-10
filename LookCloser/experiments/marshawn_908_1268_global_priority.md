# Global24fps priority and cooperative queues

## What was tested

The user requested that a worker finishing its own24fps tasks help other queues finish the complete72-frame, three-second sequence before starting dense remainder frames. The fixed source cadence is120fps; priority frames are908+5*k for k=0..71. All361 models remain in scope.

The executor now blocks only new dense jobs behind a global completion receipt. Existing jobs and their crash recovery continue. The separate persistent scheduler transfers unstarted sparse tasks to workers without sparse work waiting. It prefers a donor with a longer pending queue and a distant task, preserving the next staged task where possible. WS07 receives only frames≤1111; WS08 only frames≥1112. Other enrolled training hosts can receive either source half. A receiver uses its own qualified production command, runtime and GPU assignment.

Queue transactions record before/after states before writing, hold per-agent scheduling guards and roll forward after a crash. Agents acknowledge the guards through their normal five-second polling; states are revalidated after a six-second grace period before commit. No model is moved after it starts. Original masks, AABB, recipe, optimizer limits and numerical code remain fixed. Prefetch follows the receiving queue automatically.

## Results

Four automated checks passed: new-job gate versus recovery, sparse transfer preserving361 jobs and excluding started tasks, source-half admission, and recovery after a partial queue commit. A real executor canary also verified the scheduling guard, sparse-before-dense execution, gate release and preservation of the child process across an executor restart.

Deployment evidence is stored alongside this report after live verification. The global gate requires all72 agent jobs complete with completion receipts. Export integrity and visual inspection remain separate requirements and are not implied by the scheduling gate.

## Insights

The policy prevents an early-finishing GPU from immediately starting dense frames while sparse tasks wait elsewhere. Already-running nonpriority models finish normally. With fewer remaining unstarted tasks than GPUs, idle workers may have to wait for active models; the scheduler never duplicates or restarts those models. ETA improvements require measurement and are not guaranteed by the queue policy alone.

Live deployment completed with all12 active production parent identities and neural PIDs preserved. All five hosts retained identical hashes for `common.py`, `runtime.py`, `worker.py` and `worker_background.py`; recipe and original tuning timestamps were unchanged. The enabled `marshawn908-priority.service` is active on Silver. First status:2/72 sparse models complete, global dense gate closed. Evidence: [deployment receipt](marshawn_908_1268_diagnostics/priority_scheduler_deployment.json). Current queue files remain authoritative; avoid manual queue edits during a pending `priority_queue_transaction.json` transaction.
