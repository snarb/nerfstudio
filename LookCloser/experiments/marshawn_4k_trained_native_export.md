# Native 4K export from the 4K-trained model

## What was tested

User-requested HC export at 3840×2024 from the selected 4K-trained checkpoint126316, SHA `cb6aa1bc3e54011df9995f744f9b4d4a1beb29ebbb261147baa00ecd28402136`. This is the separate `semantic_4k_production_v1` model, whose training images are3840×2024; it is distinct from the HD-trained raw-LPIPS leader340e794. The model's original1502-file source snapshot, checkpoint, config, FR.3, native2048-ray chunks, admission and occupancy are preserved.

## Results

The actual CUDA export completed successfully on silverGPU3 without optimizer updates or OOM. The original HC1920×1012 camera first reproduced its archived RGB: only3 channels differ, each by one8-bit code. The camera intrinsics/raster were then scaled by2 and the model evaluated on new3840×2024 camera rays. No output-image upsampling was used. Persistent pipeline and nonpersistent feature-weight cache remained unchanged; RNG was restored. Peak allocated CUDA memory was3,364,498,432bytes.

- [Native HC 4K RGB](/fsx/tmp/marshawn_001660_hd/tuning_20261106/user_4K_trained_native_render_v1/render/HC_4K.png)
- [Native HC 4K alpha16](/fsx/tmp/marshawn_001660_hd/tuning_20261106/user_4K_trained_native_render_v1/render/HC_4K_alpha16.png)
- [Source, parity and state receipt](/fsx/tmp/marshawn_001660_hd/tuning_20261106/user_4K_trained_native_render_v1/render/complete.json)

RGB SHA: `8efb21d1e9628af233f0717d7ce510260ef183f7b0ee043be38e221ff9912e2c`. The first local dispatch invocation used the wrong CLI action and exited before any GPU work; its log is retained. The corrected `launch --manifest` invocation completed normally.

## Insights

Root visually inspected the exported HC image. Both actors, hair and footwear remain visible, with substantial existing floor fragments and silhouette haze. A larger output raster does not repair these geometric artifacts or establish a new quality winner. The reusable renderer is [render_marshawn_checkpoint_camera.py](../scripts/render_marshawn_checkpoint_camera.py); its spec binds the exact sealed source and checkpoint.
