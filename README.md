# Confidence is not Timeless: Modeling Temporal Validity for Rule-based Temporal Knowledge Graph Forecasting

This repository contains the code for the paper [Confidence is not Timeless: Modeling Temporal Validity for Rule-based Temporal Knowledge Graph Forecasting](https://aclanthology.org/2024.acl-long.580/).

## Qucik Start
### Env
```
conda create -n TempValid python=3.8
conda actvate TempValid
pip install -r requirements.txt
cd src
```

### Rule Mining
```
python learn.py -d ICEWS14 -l 1 2 3 -n 200 -s 12 -p 16
```
### Feature Generating
```
python generate_feature.py -d ICEWS14 -r rules_dict.json -l 1 2 3 -w 0 -p 12 -s train
python generate_feature.py -d ICEWS14 -r rules_dict.json -l 1 2 3 -w 0 -p 12 -s valid
python generate_feature.py -d ICEWS14 -r rules_dict.json -l 1 2 3 -w 0 -p 12 -s test
```

### Model Training
```
python TempValid.py --cuda -d ICEWS14 --ta
```

### Acknowledgments
Our code are developed based on https://github.com/liu-yushan/TLogic and https://github.com/nec-research/TKG-Forecasting-Evaluation .

### Citation
```
@inproceedings{huang-etal-2024-confidence,
    title = "Confidence is not Timeless: Modeling Temporal Validity for Rule-based Temporal Knowledge Graph Forecasting",
    author = "Huang, Rikui  and
      Wei, Wei  and
      Qu, Xiaoye  and
      Zhang, Shengzhe  and
      Chen, Dangyang  and
      Cheng, Yu",
    editor = "Ku, Lun-Wei  and
      Martins, Andre  and
      Srikumar, Vivek",
    booktitle = "Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = aug,
    year = "2024",
    address = "Bangkok, Thailand",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.acl-long.580",
    pages = "10783--10794",
}
```
