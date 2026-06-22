# VideoMamba3 Fixed-Budget Comparison

| model | budget | resize | n | top1_pct | top5_pct | mean_processed_frames | mean_model_input_frames | mean_inference_ms | mean_total_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| VideoMamba3-middle-complex-d4 | 2 | 112 | 2020 | 4.16 | 8.71 | 2.0 | 2.0 | 101.241 | 510.771 |
| VideoMamba | 4 | 224 | 2020 | 85.05 | 96.29 | 4.0 | 8.0 | 3.942 | 316.979 |
| VideoMamba3-middle-complex-d4 | 4 | 112 | 2020 | 4.16 | 8.71 | 4.0 | 2.0 | 100.572 | 436.927 |
| VideoMamba | 8 | 224 | 2020 | 88.37 | 97.77 | 8.0 | 8.0 | 3.42 | 335.458 |
| VideoMamba3-middle-complex-d4 | 8 | 112 | 2020 | 4.16 | 8.71 | 8.0 | 2.0 | 100.195 | 445.244 |
| VideoMamba | 16 | 224 | 2020 | 88.22 | 98.12 | 16.0 | 8.0 | 3.43 | 389.196 |
| VideoMamba | 32 | 224 | 2020 | 87.82 | 97.62 | 32.0 | 8.0 | 3.42 | 440.942 |

Note: the current VideoMamba3 checkpoint is a fast validation candidate trained at 112px and 2 input frames. The existing VideoMamba baseline was trained/evaluated at 224px and 8 model input frames, so this table is a sanity comparison, not yet a matched SOTA claim.
