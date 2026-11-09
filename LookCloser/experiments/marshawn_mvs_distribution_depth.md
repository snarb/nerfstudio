# Expected samplewise MVS depth error: read-only preparation

The mean-depth trial fits its target while retaining opposite tails: dense median absolute normal mean error improves .003951→.000133, but 10–90%normal thickness p90 worsens .01331→.01608. Its floor mottling remains visually unacceptable. The new staged objective replaces ρ(E[normal error]) by E[ρ(normal error)] using the same graph-connected native packed weights, 128 camera-balanced TRAIN rays and private RNG. Here ρ(e)=sqrt(e²+.005²)−.005; the scale is an optimization robustness choice, not calibrated MVS confidence. All NN parameters/architecture, input RGB, original losses and true-LOS derivative remain unchanged.

Actual helper CPU check: symmetric normal errors−.01,+.01 with equal weights give old mean loss0 and new loss.00618034; gradients−.4472,+.4472 oppose separation. No new density or carving constraints are generated.

The local actual16k full-state GPU proof passed without optimizer updates. Parent checkpoint SHA5aa08c7cc725cb38e1de33141acc9a5dfd77041d9e808d7d9d97c3832a4c11f2. Initial64 rays/4451 samples and adapted15077 rays/1088826 samples plus128 extra rays/4506 samples were tested. Full NN/Adam/scheduler/scaler/RNG and occupancy state remained exact. Only field.mlp_base.model.params receives auxiliary gradients; color head receives none. Combined AMP gradients are finite at scaler262144. Native main-forward parity ≤6.56e−7 RGB and ≤3.10e−6 depth.

| λ | Adapted auxiliary/RGB geometry gradient norm |
|---:|---:|
| .001 | .00960 |
| .01 | .09589 |
| .1 | .95892 |

No scanned value lies strictly in .1–.5. The prepared .01 is a conservative tested9.6% choice; .1 is substantially stronger. Source-ray opacity minimum .999900. Before optimization, sampled midpoint mass within normal±.005 is majority on115/128 rays, median.82194; within±.01 on128/128, median.96416. Midpoint band tests are diagnostic and do not define MVS confidence.

Fresh stage `/fsx/tmp/marshawn_001660_hd/native_png_MVSdist16k_STAGED` has separate config, helper, runner, zeroopt proof, CPU math receipt and dry-run log. No old worker/source was patched. Optional future continuation starts16001, pauses before18001 after2000 new attempts, ends before18830 with2829 maximum new attempts. Prior lineage charged19170, total cap21999. Default stopping and final terminal evaluation are retained. **No new training launched; root decision pending independent AA results/review.**
