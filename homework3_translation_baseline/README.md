# 任务说明

实现从英语到中文的翻译任务，输入是一段英语文本，输出是与输入所对应的中文文本。

## 数据集

在dataset文件夹中，data_train.json为训练集，data_valid.json为验证集，src是需要翻译的语句，tgt是目标语句。

## 模型

选用Hugging Face上的opus-mt-en-zh(https://huggingface.co/Helsinki-NLP/opus-mt-en-zh)作为本次任务的基准模型。

## 评价指标

采用bleu分数作为翻译任务的评价指标，重点关注bleu-4的得分，分数越高代表模型效果越好。

## 特别说明

跑baseline.py文件需安装numpy、GPUtil、datasets、evaluate、transformers等包。模型训练完成后，会在当前目录下新建一个result文件夹，在result文件夹中保存两个文件夹记录训练过程中的不同步数的模型参数(例如checkpoint-3000就是第3000步模型的参数)，同时会保存一个train.log文件，里面记录了训练时间和显存占用情况。大家可以根据两个checkpoint保存的模型参数，跑一下在验证集集上的bleu分数，将得分高的模型留下，方法如下(baseline_example.py给了相应示例)：

1.将15行的model_checkpoint改为result文件夹下的check_point文件，例如`model_checkpoint = "./result/check_point3000"`

2.将`trainer.train()`与计算时间和GPU的代码注释，只跑`print(trainer.predict(tokenized_valid_dataset).metrics)`



训练的步数不是像3000这样的整数，故如果只保存一个模型参数，训练结束后则文件里一个模型参数都没有，故保存两个模型参数。

