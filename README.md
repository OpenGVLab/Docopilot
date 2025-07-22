# Docopilot: Improving Multimodal Models for Document-Level Understanding

The official implementation of the paper "[Docopilot: Improving Multimodal Models for Document-Level Understanding](https://arxiv.org/abs/2507.14675)". 

<!-- <div align="center">
    <img src="assets/fig1_hf_00.png" alt="drawing" width="600"/>
</div> -->

<div align="center">

[\[📜 Paper\]](https://arxiv.org/abs/2507.14675)  [ \[🤗 HF Models 2B |](https://huggingface.co/OpenGVLab/Docopilot-2B)[ 8B \]](https://huggingface.co/OpenGVLab/Docopilot-8B)  [\[📖 HF Datasets\]](https://huggingface.co/datasets/OpenGVLab/Doc-750K)

</div>



## 📕 Overview
- We construct `Doc-750K`, the first large-scale, high-quality dataset for **document-level multimodal understanding**, with 758K QA pairs covering 9 task types.

- We propose `Docopilot`, a native **document-level VLM** that outperforms existing methods and Gemini-1.5-Pro on MMLongBench-Doc, making it the closest open-source model to GPT-4o.

- `Docopilot` achieves much lower inference latency than RAG-based methods, and when combined with RAG, its performance further improves, showing that RAG effectively enhances its retrieval and reasoning.


## 🗓️ Schedule

- [ ] Release Evaluation Code
- [x] Release Training Code
- [x] Release `Doc-750K`
- [x] Release `Docopilot` Checkpoints

## ⚙️ Data Preparation
### Download [`Doc-750K`](https://huggingface.co/datasets/OpenGVLab/Doc-750K) (need about 1.5TB space)
```sh
mkdir data
cd data
# pip install -U huggingface_hub
huggingface-cli download --resume-download --local-dir-use-symlinks False OpenGVLab/Doc-750K --local-dir Doc-750K --repo-type dataset

# unzip each images folder
cd Doc-750K/openreview
unzip images.zip
cd ../generated
unzip images.zip
cd ../arxivqa
unzip images.zip
cd ../scihub
unzip images.zip
```
### Custom your own training data (Optional)
Follow [this link](https://internvl.readthedocs.io/en/latest/internvl2.0/finetune.html#prepare-customized-data) to prepare your own training data.

Notice: Put the meta in a single json, similar to `playground/Doc-750K.json`.

## 🔥 Supervised Finetuning
### Pretrained Model Preparation
Our models are finetuned from `InternVL2-2B` and `InternVL2-8B`.
Please download the above model weights and place them in the `pretrained/` folder.


| model name              | type | download                                                               |  size  |
| ----------------------- |------| ---------------------------------------------------------------------- |:------:|
| InternVL2-2B    | VLM  | 🤗 [HF link](https://huggingface.co/OpenGVLab/InternVL2-2B) | 4.4 GB |
| InternVL2-8B    | VLM  | 🤗 [HF link](https://huggingface.co/OpenGVLab/InternVL2-8B) | 16 GB |


```sh
mkdir pretrained
cd pretrained/
# pip install -U huggingface_hub
huggingface-cli download --resume-download --local-dir-use-symlinks False OpenGVLab/InternVL2-2B --local-dir InternVL2-2B
huggingface-cli download --resume-download --local-dir-use-symlinks False OpenGVLab/InternVL2-8B --local-dir InternVL2-8B
```


### Training
```sh
sh shell/slurm_train_example.sh
```

## 📦 Model Zoo


| model name              | type | download                                                               |  size  |
| ----------------------- |------| ---------------------------------------------------------------------- |:------:|
| Docopilot-2B    | VLM  | 🤗 [HF link](https://huggingface.co/OpenGVLab/Docopilot-2B) | 4.4 GB |
| Docopilot-8B    | VLM  | 🤗 [HF link](https://huggingface.co/OpenGVLab/Docopilot-8B) | 16 GB |


## 🖊️ Citation

If you find this work helpful in your research, please consider citing:

```bibtex
@inproceedings{duan2025docopilot,
  title={Docopilot: Improving Multimodal Models for Document-Level Understanding},
  author={Duan, Yuchen and Chen, Zhe and Hu, Yusong and Wang, Weiyun and Ye, Shenglong and Shi, Botian and Lu, Lewei and Hou, Qibin and Lu, Tong and Li, Hongsheng and others},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
  pages={4026--4037},
  year={2025}
}
```
