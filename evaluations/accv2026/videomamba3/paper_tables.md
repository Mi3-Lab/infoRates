# VideoMamba3 Experiment Tables

## Training Summary

| checkpoint | variant | num_frames | input_size | depth | val_acc | epoch_seconds | samples_per_second |
| --- | --- | --- | --- | --- | --- | --- | --- |
| fine_tuned_models/accv2026_videomamba3_middle_complex_ucf101_112r_f2_d4_e3_fast_h200 | complex | 2 | 112 | 4 | 0.44140625 | 653.349 | 1.56730958 |
| fine_tuned_models/accv2026_videomamba3_middle_complex_ucf101_112r_f2_d4_e3_fast_h200_epoch1 | complex | 2 | 112 | 4 | 0.44140625 |  |  |
| fine_tuned_models/accv2026_videomamba3_middle_complex_ucf101_112r_f2_d4_e3_fast_h200_epoch2 | complex | 2 | 112 | 4 | 0.328125 |  |  |
| fine_tuned_models/accv2026_videomamba3_middle_complex_ucf101_112r_f2_d4_e3_fast_h200_epoch3 | complex | 2 | 112 | 4 | 0.43359375 |  |  |
| fine_tuned_models/accv2026_videomamba3_tiny_complex_ucf101_112r_f2_e1_h200 | complex | 2 | 112 | 24 | 0.5 | 2554.134 | 0.20045934 |
| fine_tuned_models/accv2026_videomamba3_tiny_complex_ucf101_112r_f2_e1_h200_epoch1 | complex | 2 | 112 | 24 | 0.5 |  |  |
| fine_tuned_models/accv2026_videomamba3_tiny_complex_ucf101_h200_112_smoke | complex | 2 | 112 | 24 | 1.0 |  |  |
| fine_tuned_models/accv2026_videomamba3_tiny_complex_ucf101_h200_112_smoke_epoch1 | complex | 2 | 112 | 24 | 1.0 |  |  |
| fine_tuned_models/accv2026_videomamba3_tiny_mimo_ucf101_112r_f2_e1_h200 | mimo | 2 | 112 | 24 | 0.296875 | 2993.271 | 0.17105034 |
| fine_tuned_models/accv2026_videomamba3_tiny_mimo_ucf101_112r_f2_e1_h200_epoch1 | mimo | 2 | 112 | 24 | 0.296875 |  |  |
| fine_tuned_models/accv2026_videomamba3_tiny_trapezoidal_ucf101_112r_f2_e1_h200 | trapezoidal | 2 | 112 | 24 | 0.3515625 | 1706.072 | 0.30010452 |
| fine_tuned_models/accv2026_videomamba3_tiny_trapezoidal_ucf101_112r_f2_e1_h200_epoch1 | trapezoidal | 2 | 112 | 24 | 0.3515625 |  |  |

## Latency Summary

| variant | seq_len | depth | mean_inference_s | videos_per_second | peak_memory_mb | params |
| --- | --- | --- | --- | --- | --- | --- |
| trapezoidal | 99 | 24 | 0.7354807270069917 | 2.719309869803002 | 57.48876953125 | 6004325 |
| complex | 99 | 24 | 1.2622485366688732 | 1.5844740095940884 | 57.48876953125 | 6004325 |
| mimo | 99 | 24 | 1.4862995283328928 | 1.3456237870460062 | 59.6142578125 | 6363749 |

## System Validation

| source | status | variant | depth | mode | batch_size | num_frames | input_size | seq_len | mean_s | videos_per_second | tokens_per_second | peak_allocated_mb | peak_reserved_mb | torch_compile | gpu_name |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| system_validation_cpu_smoke.csv | ok | complex | 1 | inference | 1 | 1 | 32 | 5 | 0.165454528061673 | 6.043956679307385 | 30.21978339653693 | 0.0 | 0.0 | 0 | cpu |
| system_validation_cpu_smoke.csv | ok | complex | 1 | train | 1 | 1 | 32 | 5 | 1.6042971829883754 | 0.6233259090670895 | 3.116629545335448 | 0.0 | 0.0 | 0 | cpu |
| system_validation_fast_ucf101_72207.csv | ok | complex | 4 | inference | 4 | 4 | 112 | 197 | 0.7746506703357833 | 5.163617812744089 | 1017.2327091105856 | 23.0283203125 | 34.0 | 0 | NVIDIA A100-PCIE-40GB |
| system_validation_fast_ucf101_72207.csv | ok | complex | 4 | train | 4 | 4 | 112 | 197 | 5.099777331342921 | 0.7843479705312316 | 154.5165501946526 | 331.9833984375 | 354.0 | 0 | NVIDIA A100-PCIE-40GB |
| system_validation_fast_ucf101_72207.csv | ok | complex | 8 | inference | 4 | 4 | 112 | 197 | 1.553001661998375 | 2.5756572564467644 | 507.40447952001256 | 36.7236328125 | 40.0 | 0 | NVIDIA A100-PCIE-40GB |
| system_validation_fast_ucf101_72207.csv | ok | complex | 8 | train | 4 | 4 | 112 | 197 | 9.917168869675756 | 0.4033409184178569 | 79.45816092831781 | 638.7158203125 | 662.0 | 0 | NVIDIA A100-PCIE-40GB |
| system_validation_fast_ucf101_72207.csv | ok | complex | 12 | inference | 4 | 4 | 112 | 197 | 2.326735992663695 | 1.7191464835770722 | 338.67185726468324 | 42.2939453125 | 46.0 | 0 | NVIDIA A100-PCIE-40GB |
| system_validation_fast_ucf101_72207.csv | ok | complex | 12 | train | 4 | 4 | 112 | 197 | 14.93528532331887 | 0.267822134857691 | 52.76096056696513 | 945.4482421875 | 968.0 | 0 | NVIDIA A100-PCIE-40GB |
| system_validation_fast_ucf101_72207.csv | ok | trapezoidal | 4 | inference | 4 | 4 | 112 | 197 | 0.5066657633481858 | 7.894750917383695 | 1555.265930724588 | 31.1533203125 | 34.0 | 0 | NVIDIA A100-PCIE-40GB |
| system_validation_fast_ucf101_72207.csv | ok | trapezoidal | 4 | train | 4 | 4 | 112 | 197 | 3.206039369668966 | 1.2476453152267473 | 245.7861270996692 | 324.5517578125 | 348.0 | 0 | NVIDIA A100-PCIE-40GB |
| system_validation_fast_ucf101_72207.csv | ok | trapezoidal | 8 | inference | 4 | 4 | 112 | 197 | 1.014772043020154 | 3.9417719748124327 | 776.5290790380493 | 36.7236328125 | 40.0 | 0 | NVIDIA A100-PCIE-40GB |
| system_validation_fast_ucf101_72207.csv | ok | trapezoidal | 8 | train | 4 | 4 | 112 | 197 | 6.3192044973257 | 0.6329910674188196 | 124.69924028150746 | 623.7802734375 | 646.0 | 0 | NVIDIA A100-PCIE-40GB |
| system_validation_fast_ucf101_72207.csv | ok | trapezoidal | 12 | inference | 4 | 4 | 112 | 197 | 1.5255002256599255 | 2.622090729792987 | 516.5518737692184 | 42.2939453125 | 46.0 | 0 | NVIDIA A100-PCIE-40GB |
| system_validation_fast_ucf101_72207.csv | ok | trapezoidal | 12 | train | 4 | 4 | 112 | 197 | 9.772999794998515 | 0.4092909120950828 | 80.63030968273131 | 923.0087890625 | 946.0 | 0 | NVIDIA A100-PCIE-40GB |
| system_validation_fast_ucf101_72210.csv | ok | complex | 4 | inference | 4 | 2 | 112 | 99 | 0.4043361980002373 | 9.89275761058042 | 979.3830034474616 | 69.15673828125 | 76.0 | 0 | NVIDIA A100-PCIE-40GB |
| system_validation_fast_ucf101_72210.csv | ok | complex | 4 | train | 4 | 2 | 112 | 99 | 2.4484879936402044 | 1.633661267847648 | 161.73246551691713 | 579.5224609375 | 654.0 | 0 | NVIDIA A100-PCIE-40GB |
| system_validation_fast_ucf101_72210.csv | ok | trapezoidal | 4 | inference | 4 | 2 | 112 | 99 | 0.2617054810010207 | 15.284356998179948 | 1513.1513428198148 | 77.28173828125 | 98.0 | 0 | NVIDIA A100-PCIE-40GB |
| system_validation_fast_ucf101_72210.csv | ok | trapezoidal | 4 | train | 4 | 2 | 112 | 99 | 1.6272415363540251 | 2.458147675459628 | 243.35661987050312 | 568.31884765625 | 644.0 | 0 | NVIDIA A100-PCIE-40GB |
