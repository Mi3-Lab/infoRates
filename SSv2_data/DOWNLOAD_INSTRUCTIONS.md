# Something-Something V2 Download Instructions

The Something-Something V2 dataset requires manual download due to licensing.

## Option 1: Official Source (Recommended)
1. Visit: https://developer.qualcomm.com/software/ai-datasets/something-something
2. Register for free (academic/research use)
3. Download validation split (~6-8 GB)
4. Extract to this directory (SSv2_data/)

## Option 2: Academic Request
Contact: datasets@twentybn.com
Request academic access to Something-Something V2

## Dataset Structure After Download
```
SSv2_data/
├── validation/
│   ├── 1.webm
│   ├── 2.webm
│   └── ...
├── something-something-v2-validation.json
└── something-something-v2-labels.json
```

## Alternative: Proceed Without SSv2
You already have 3 strong datasets:
- UCF-101: 101 classes, 13k clips
- HMDB-51: 51 classes, 7k clips  
- Kinetics-400: 400 classes, 19k clips

Total: 552 classes, 39k clips - MORE than sufficient for publication.
SSv2 is optional enhancement, not required.
