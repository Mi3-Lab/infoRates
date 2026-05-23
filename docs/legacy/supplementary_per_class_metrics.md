# Supplementary Table S1 — Per-class mean drops & variances

This table lists per-class mean 100→25 accuracy drops (%) and mean variance across coverage (%) aggregated across models, plus per-model breakdowns (drops and variances). Classes chosen for Figure 3 are the 3 sensitive and 3 robust per dataset shown below.

| Dataset | Class | Role | Mean Drop (%) | Mean Var (%) | Presence | Timesformer Drop | Videomae Drop | ViViT Drop | Timesformer Var | Videomae Var | ViViT Var |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ucf101 | YoYo | sensitive | 51.376 | 9.808 | 3 | 25.688 | 54.128 | 74.312 | 4.343 | 11.573 | 13.506 |
| ucf101 | JumpingJack | sensitive | 47.009 | 8.459 | 3 | 25.641 | 64.103 | 51.282 | 4.142 | 9.158 | 12.078 |
| ucf101 | SalsaSpin | sensitive | 43.487 | 6.799 | 3 | 35.058 | 40.23 | 55.172 | 7.857 | 6.071 | 6.468 |
| ucf101 | Billiards | robust | -0.718 | 0.002 | 3 | -0.431 | -1.293 | -0.431 | 0.001 | 0.004 | 0.001 |
| ucf101 | Bowling | robust | 0.379 | 0.004 | 3 |  |  | 1.136 | 0.003 | 0.004 | 0.006 |
| ucf101 | Typing | robust | -0.447 | 0.005 | 3 | -1.342 | 0.671 | -0.671 | 0.005 | 0.008 | 0.003 |
| kinetics400 | diving cliff | sensitive | 31.293 | 2.979 | 3 | 34.694 | 34.694 | 24.49 | 2.853 | 4.885 | 1.2 |
| kinetics400 | waiting in line | sensitive | 30.667 | 3.631 | 3 | 20.0 | 56.0 | 16.0 | 2.288 | 7.952 | 0.652 |
| kinetics400 | dunking basketball | sensitive | 29.167 | 2.558 | 3 | 33.333 | 18.75 | 35.417 | 3.832 | 0.985 | 2.856 |
| kinetics400 | shearing sheep | robust | 1.361 | 0.026 | 3 |  | 2.041 | 2.041 | 0.033 | 0.033 | 0.012 |
| kinetics400 | playing harp | robust | -0.667 | 0.027 | 3 | 2.0 |  | -4.0 | 0.028 | 0.032 | 0.02 |
| kinetics400 | bowling | robust | 1.361 | 0.031 | 3 |  | 2.041 | 2.041 | 0.029 | 0.029 | 0.033 |
