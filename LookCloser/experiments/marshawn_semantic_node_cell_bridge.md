# Conservative semantic-node to native-cell support bridge

## What was tested

CPU-only preparation for the independent direct-LC semantic head. No fitted head values, model, native occupancy, masks or training state were changed. The existing head interpolates **XYZ logits**, then applies sigmoid; PyTorch coordinates are supplied Z/Y/X, `align_corners=True`, border padding, and an explicit inclusive actor-box gate. LC floor-frame transform is identity. The implementation also tests `align_corners=False` explicitly, rather than silently substituting it.

Semantic128 nodes include both box endpoints, giving pitch extent/127. Actual native128 cells use extent/128 and their saved cascade AABBs. On the actual source capture, native pitches are `.0077343751/.0049218750/.0054687499`; semantic pitches are `.0077952754/.0049606298/.0055118110`. These are scene-coordinate units, not certified metres. The slight float-bound differences are preserved.

The helper bounds a cell by **all node basis functions whose support can overlap it**. A trilinear logit is a convex combination, so its maximum cannot exceed the maximum of these coefficients. This conservative bound may retain extra cells. Testing only the eight native-cell corners is invalid when an interior semantic knot contains a higher value; an adversarial case demonstrates that failure. An exact tighter alternative would evaluate the tensor product of native boundaries and crossed knot planes.

For hard `p>=.1`/`p>=.5`, threshold equality is retained. For smoothstep(.1,.5), exact mathematical support is `p>.1`; equality can be zero only when no numerical slack is requested. Near-threshold nodes are retained under the explicit default logit slack `1e-5`. Coordinate slack `2e-7` is separate from arithmetic slack and any geometric margin. Finite soft sigmoid remains potentially nonzero everywhere inside: small class probability cannot bound selected density without a bound on native sigma. The occupancy EMA threshold `.01` is not a semantic class threshold and is not used here.

## Results

[Receipt](assets/marshawn_semantic_node_cell_bridge_20261107/receipt.json) binds actual schema2 cascade bounds, sealed clothing103+shoe6 queries and helper bytes. The following are **synthetic contract cases**, not learned semantics or source labels:

| Synthetic node field | Envelope cells | Envelope with fresh +1 | 109 observations retained |
|---|---:|---:|---:|
| All positive | 2,097,152 | 2,097,152 | 109 |
| All negative | 0 | 0 | 0 |
| One positive interior knot | 27 | 125 | 0 |
| Positive basis nodes deliberately surrounding witnesses | 1,641 | 4,688 | 109 |

The witness construction tests coordinates and indexing only; it provides no evidence that a fitted head will preserve these points. The 109 observations represent 41 unique positions and do not cover all hair, clothing contours or feet.

Fourteen CPU tests pass: independent Torch trilinear queries with both alignment conventions; hard/smooth threshold equality; FP32 nextafter points at node/boundary planes; interior-knot maxima missed by eight-corner tests; anisotropic XYZ axes; exact border/partial overlap; separate cascade bounds; finite/transform validation; monotonic selector support; and fresh noncumulative +1 dilation. CUDA remained uninitialized in these tests.

The +1 operation here dilates the computed semantic envelope in native-cell coordinates. It is not a cumulative update, a source-mask padding change, or an automatic change to the model's learned-core halo. It can preserve traversal near the envelope but **cannot restore density already zeroed by the semantic head**.

## Insights and pending integration gates

This is a conservative mathematical bound for the chosen selector under the declared arithmetic slack, **not yet a CUDA deployment certificate**. Actual GPU grid-sampling/threshold arithmetic must be checked against the exact fitted head and its physical selector before using rejected cells in traversal. Source-head values and exact source files must be hash-bound; all109 actual probabilities/support decisions, edge cases and fresh0/+1 grids must be reported. No such fitted-head evaluation has occurred here.

The guarantee applies to the semantic envelope alone. Intersecting it with an existing learned occupancy grid does not prove that the learned grid itself contains every nonzero field value. Likewise skipping zero-density cells can change native stepping phase and tail coverage; support containment alone does not prove rendered identity. Actual density/RGB/transmittance and full-render parity remain separate gates.

Neither a class-grid prediction nor this support bound establishes physical free space or converts occluded/unknown hair into a negative label. No integration or further fitting is authorized by this report.
