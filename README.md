# ORL


## 环境配置：

- conda env  create -n T5 -f T5.yml
- conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
- pip install ipykernel
- python -m ipykernel install --name T5
- pip install nltk
- pip install pandas
- pip install transformers
- pip install datasets
- pip install matplotlib

## 项目结构
- test_AL_copy.ipynb: 训练代码
- data_analysis.ipynb: 数据可视化代码
- RANDOM_result: 随机采样结果
- BEAM_result: least confident策略结果
- MARGIN_result: margin sampling策略结果
- FULL_result: 全量训练结果