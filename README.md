# Data
Location of A-OKVQA train / val data split:
- `VQA/data/aokvqa/exp_data/train_data`
- `VQA/data/aokvqa/exp_data/val_data`

# Image
- Image folder (put all your `.JPEG`/`.jpg` files here): `VQA/data/aokvqa/exp_data/images/images`
- Image feature:
  - **FVQA**
    - `fvqa-resnet-14x14.h5` pretrained: [GoogleDrive](https://drive.google.com/file/d/1p-7AUrveEPWYQKub77UN4Nr8munFtYsV/view?usp=sharing)
    - `fvqa36_imgid2idx.pkl`: [url](https://drive.google.com/file/d/1aGzZyuSgM0OFi281ppby3Lw-NAAWBIIP/view?usp=sharing) and `fvqa_36.hdf5`: [url](https://drive.google.com/file/d/1qWJrHYR_LvRPuUsMGm-qqZfR8G8wkcOs/view?usp=drive_link)
  - **A-OKVQA**
    - `aokvqa-resnet-14x14.h5` pretrained: [GoogleDrive](https://drive.google.com/file/d/1Wls6bZ80PboxrYod1IZWDTrvkSb-F5oi/view?usp=sharing)
    - `aokvqa36_imgid2idx.pkl`: [url](https://drive.google.com/file/d/1eKIayndwgIqWDkixIZQqaGYmqRZlpymA/view?usp=sharing) and `aokvqa_36.hdf5`: [url](https://drive.google.com/file/d/1eFTI8sv24T046kMjsQ08biV5druBMaA2/view?usp=drive_link)
- Original images are available at [FVQA](https://github.com/wangpengnorman/FVQA) and [AOKVQA](https://github.com/allenai/aokvqa).
  - The generated `.h` file should be placed in:
    - `VQA/data/aokvqa/exp_data/common_data/`
    - `VQA/data/fvqa/exp_data/common_data/`

# Answer / Question vocab
The generated files `answer.vocab.aokvqa.json` & `question.vocab.aokvqa.json` are now available at:
- `VQA/data/aokvqa/exp_data/common_data/`
- `VQA/data/fvqa/exp_data/common_data/`

# Pretrained Model
[Download Pretrained Model](https://www.dropbox.com/scl/fo/07aierwz3tlsg4nzyr8bv/ADUZAuYtPQKmN4qHoQN8hWw?rlkey=qtpf14v1pak3lv2a5i6zh0png&st=2hsdhmcz&dl=1)
> **Note:** Download it and overwrite the `VQA/model_save` directory.

# Parameter
```bash
[--KGE {TransE,ComplEx,TransR,DistMult}] [--KGE_init KGE_INIT] [--GAE_init GAE_INIT] 
[--ZSL ZSL] [--entity_num {all,4302}] [--data_choice {0,1,2,3,4}] [--name NAME] 
[--no-tensorboard] --exp_name EXP_NAME [--dump_path DUMP_PATH] [--exp_id EXP_ID] 
[--random_seed RANDOM_SEED] [--freeze_w2v {0,1}] [--ans_net_lay {0,1,2}] 
[--fact_map {0,1}] [--relation_map {0,1}] [--now_test {0,1}] [--save_model {0,1}] 
[--joint_test_way {0,1}] [--top_rel TOP_REL] [--top_fact TOP_FACT] 
[--soft_score SOFT_SCORE] [--mrr MRR]
```

# Running 
- cd VQA/code

Data check:
- python deal_data.py --exp_name data_check

General VQA:

- train: bash run_FVQA_train.sh
- test: bash run_FVQA.sh   

General AOKVQA: 

- train: bash run_AOKVQA_train.sh
- test: bash run_AOKVQA.sh 
