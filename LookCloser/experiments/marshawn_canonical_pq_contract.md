# Marshawn signed PQ contract audit

## What was tested

The take's signed radiometric contract was compared with the experimental gain4
SDR/HDR/PQ variants. Source: `/fsx/oregon/gta_marshawn_2_1_prep_20261028/DECISIONS.md`
lines 28–30 and `manifests/footage_range_signed.json`, SHA256
`7c475132bd6d48775780167161ef521a638c84981b3bcbc9b2d0b8ca585fee18`.
The take-specific p999-half-peak scale was signed on 10-16 and used for Marshawn
1300–2020. The source-location receipt proves global 1660 corresponds to
segment-local 408 (`1660-1252`), so the selected raw TIFF frame mapping is correct.

## Results

| Property | Signed vendor tier | Previous experimental HDR/PQ |
|---|---|---|
| Gain from original decoded relative linear RGB | 52.13360318135631 | 4×203=812, plus scalar in scalar derivatives |
| Floor | .005 nits, clamp | 0 in archival export; loss had separate conventions |
| Content peak | 4000 nits | 10000 nits |
| PQ normalization constant | 10000 nits | 10000 nits |
| RGB channels | AP1 from vendor matrix | Rec.709 master converted to Rec.2020 PQ |
| Negative clip order | Decode RWG → AP1 matrix → clip AP1 | Clip RWG → Rec.709 matrix → clip Rec.709 |

For a neutral value, the old gain4+203 chain is 15.57537× brighter (+3.96119 EV)
before content clipping. Relative midgrey .18 becomes 9.38405 nits / PQ .294710 in the
signed chain versus 146.16 nits / PQ .546635 previously. This is a substantive input
contract difference, not an EXR/PNG container change. Existing measured SDR
quality remains valid for its declared experimental RGB input; it does not
establish vendor-canonical PQ equivalence.

Vendor tier metadata binds params_hash `78d8d53c80f92b6b`, matrix_id
`rwg->ap1/bradford-d65+rot:0+und:5739b71cecce8754a7027b91b3ff21f9`.
However the actual vendor `pq.py` defines AP1 white xy (.321680,.337670), ACES D60,
and Bradford-adapts RWG D65 → AP1 D60; later exporter metadata says `ap1-d65`.
The new contract records actual matrix/white constants explicitly. Independent actual-tier proof resolves the numeric contract: all 24576 sampled
channel codes exactly match AP1 D60. The D65 alternative differs by up to 143
codes; the wrong RWG pre-clamp by up to 873. The historical `ap1-d65` label is
inconsistent with the delivered pixels; no Rec.2020 assumption is substituted.

New canonical 66-camera export:
`/fsx/tmp/marshawn_001660_hd/color_variants/canonical_signed_ap1`.
The physical EXR stores unbaked relative AP1 linear RGB. The archival PQ16 uses
only the signed gain/floor/content peak. No extra +2 EV, camera scalar fitting,
white-balance fitting, or altered camera geometry enters this master. HD linear
prefiltering/remap is explicitly a derivative of the native vendor tier; native
full-resolution byte identity is not claimed. QA and independent source/root visual review passed; sealed input gates bind
all 133 files per dataset. No training was launched by these exporters.

A separate completed direct-code control contains actual PQ16/65535 as float32 EXR,
explicitly nonlinear AP1 PQ codes, not a physical linear EXR. It must use a code
RGB objective, not encode PQ again. The physical HDR path separately needs a
canonical AP1 encoder/loss adapter: the existing extra black offset/10000-content
peak conventions are not the signed tier by flags alone.

[Numeric source audit](/fsx/tmp/marshawn_001660_hd/color_variants/canonical_signed_pq_audit/comparison.json).

## Insights and next steps

203 nits is a declared display/loss reference, not the take's signed camera-linear
scale. Compare both physical 203-reference previews and explicitly matched
old-display exposure separately. Converting the old PNG back to canonical RGB
cannot undo nonlinear tone mapping or negative clipping; regenerate from raw.
The accepted SDR package remains unchanged. Both new input gates passed; model
adapter correctness and any training launch remain separate reviewed steps.

The unbaked master is `canonical_signed_ap1/linear_ap1_exr`; its signed AP1 PQ16
archive is `canonical_signed_ap1/pq_ap1_png16`. Direct-code EXRs are under
`canonical_signed_ap1_pq_codes_exr`, with exact uint16 roundtrip for every image.
All 66 numeric camera records match the original geometry; EXR filename manifests
have SHA 18a1ef2ea0ce8a720586f5171a062d3ed9f0301d3ff53c716c78ca8036b86315.
The physical master and numerical code EXR are distinct datasets and contracts.

Train-only AP1-Y statistics use the vendor luminance row and all 122411520
valid-border pixels: relative q50=.0201105075, q999=30.89463468; signed nits
q50=1.04843322, q999=1610.64862483. No image clips at the 4000-nit content ceiling.
PQ16 quantization maximum error is .14173 nits, worst-camera mean absolute error
.00101634 nits. These are conversion checks, not trained reconstruction metrics.

- [Physical input gate](/fsx/tmp/marshawn_001660_hd/color_variants/canonical_signed_ap1/color_gate.json)
- [Direct-code input gate](/fsx/tmp/marshawn_001660_hd/color_variants/canonical_signed_ap1_pq_codes_exr/color_gate.json)
- [Physical QA/statistics](/fsx/tmp/marshawn_001660_hd/color_variants/canonical_signed_ap1/QA.json)
- [Direct-code exact roundtrip QA](/fsx/tmp/marshawn_001660_hd/color_variants/canonical_signed_ap1_pq_codes_exr/QA.json)
- [Root three-view/ROI visual gate](/fsx/tmp/marshawn_001660_hd/color_variants/canonical_signed_ap1/root_visual_gate.json)
- [Independent actual-tier proof](/fsx/tmp/marshawn_001660_hd/report/canonical_colour_independent/audit.json)
- [Whole-view physical/matched-display comparison](/fsx/tmp/marshawn_001660_hd/color_variants/canonical_signed_ap1/physical_and_matched_display_preview/comparison.jpg)
- [Fixed main-pair ROI comparison](/fsx/tmp/marshawn_001660_hd/color_variants/canonical_signed_ap1/physical_and_matched_display_preview/mainpair_rois/comparison.jpg)

Implementation: `scripts/prepare_marshawn_canonical_pq.py` exports the master,
archive and declared previews; `scripts/finalize_marshawn_canonical_pq.py` audits
all files, derives train-only statistics and exports the separate code control.
Exact helper/vendor bytes are frozen in the master `export_code` directory.
