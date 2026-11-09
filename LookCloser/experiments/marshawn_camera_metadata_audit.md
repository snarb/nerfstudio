# Marshawn acquisition and RED decode metadata audit

What was tested: read-only acquisition/look metadata for KA, DC, HC, MA, NA, NB, AA, AB and BB before considering any new colour calibration. Read TIFF IFDs at global1660/local408,1700/448,1740/488; copied only the first8MiB of each camera's original R3D segment003 and printed its metadata with the exact vendor REDline binary on DEV3. No full image decode, training or gain changes.

Results: all nine cameras have the same recorded acquisition and look settings.

| Recorded setting | All nine cameras |
|---|---|
| ISO | 800 |
| White balance | 5600K, tint0 |
| Shutter | 1/240s,180°; header rounded ms field4 |
| Aperture | F8 |
| Record/timecode rates | 120fps /24fps |
| Exposure / Exposure Adjust / FLUT | 0 /0 /0 |
| Red/Green/Blue gain | 1 /1 /1 |
| Saturation / contrast / brightness | 1 /0 /0 |
| Colour/gamma IDs | 25 /34: REDWideGamutRGB /Log3G10 |
| CDL /3D LUT enabled | false /false |
| Firmware | 2.2.3beta |

The original camera identity and R3D filename were checked in every printed header. The saved432segment probe table independently confirms matching ISO800/5600K/1/240/F8 in all54 rows for these nine cameras across six segments each. ND Stops is blank; no numeric ND setting is inferred.

The actual decode recipe uses `--useMeta --metaIgnoreFrameGuide --colorSciVersion 3 --colorSpace 25 --gammaCurve 34 --deb 0 --format 1 --res 1`. The saved six-second decoder preserves the home-of-record GPU flags: `--gpuPlatform 2 --enableGpuDecode --cudaDeviceIndexes 0 --decodeThreads 16`. No fixed ISO, Kelvin, tint or exposure override and no RMD/RSX loading option appears. No RMD/RSX/XML/JSON/TXT look sidecar was found in the nine original clip directories.

Pinned REDline help states that `--useMeta` takes R3D look metadata as defaults unless explicitly overridden. The bare CLI ISO default is320, but the actual recipe enables metadata and the actual headers say800. There is no evidence here of an accidental ISO320 decode. The11-05 GPU continuation independently matched cedar HC/DC frame1900 byte-for-byte. The10-28 CPU decoder has the same look flags but is not byte-identical to cedar GPU output; do not substitute it when preparing calibration-frame comparisons.

All27 TIFFs exist at212336804bytes and are native8192×4320 contiguous RGB uint16 with the same12 basic IFD tags. No ISO, exposure, white-balance, tint, ICC or description metadata is embedded. The source TIFFs remain lens-distorted REDWideGamutRGB/Log3G10, not PQ or Rec709. Source-frame mapping remains global−1252 for segment003:408/448/488. This verifies the selected calibration/holdout frames' native canvas; it does not independently prove every historical decoder execution.

Evidence:

- [Hash-bound audit and source manifest](/fsx/tmp/marshawn_001660_hd/camera_metadata_audit/audit.json), SHA `eaca6d8e2b98d6faf3fc3bb8b434b8861a16d01b02835447e27abf696b0549ad`.
- [Actual nine R3D metadata records and exact commands](/fsx/tmp/marshawn_001660_hd/camera_metadata_audit/r3d_segment3_metadata.json).
- [Header/source R3D bindings](/fsx/tmp/marshawn_001660_hd/camera_metadata_audit/header_bindings.json).
- [27 TIFF IFD probes](/fsx/tmp/marshawn_001660_hd/camera_metadata_audit/TIFF_headers_9cams_3frames.json).
- [Pinned REDline help](/fsx/tmp/marshawn_001660_hd/camera_metadata_audit/REDline_help_pinned_DEV3.txt).
- [Actual decoder/SDK binding](/fsx/tmp/marshawn_001660_hd/camera_metadata_audit/decoder_runtime_binding_DEV3.txt): binary MD5 `0cf0db325d4e3265f14a706114306f10`, same as the vendor recipe.

The first local metadata attempt failed to load its SDK; the pinned native DEV3 decoder succeeded on the same header copy. This is an execution-environment issue, not source corruption. Metadata mode returns1 despite printing valid records; records were validated by expected camera and original filename rather than interpreting that exit status as a decode failure.

Insight: no recorded per-camera ISO/WB/tint/exposure mismatch explains the KA colour gap. This does not establish matched sensor/lens spectral response, vignetting, lighting or material BRDF. These are segment-header values, not an independent extraction of per-frame408/448/488 metadata; dynamic per-frame overrides are not excluded. Truncated header `Total Frames=1` is an artifact of the header-only probe, not the actual take length. No new gains, matrices, calibration or model changes follow from this metadata audit alone.
