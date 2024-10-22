# UseBert

We all want to use BERT quickly to build a baseline. This repo gives you the code necessary to run this baseline. Also this repo has tensorboard functionality to look at the loss while training.

I have used DataParallel to speed up the process of training. One should ideally use DistributedDataParallel. I'll work on changing it in the code.

## To train the model on SNLI, run the following command:
`python -m model.train_nli --train-file ../../Data/snli_1.0/snli_1.0_train.jsonl --test-file ../../Data/snli_1.0/snli_1.0_test.jsonl  --batch-size 32`

# Pretrain Bert

## Prepare the Data

In order to demonstrate how to fine tune the LM, I have used the Wikitext-2 datatset (https://blog.einstein.ai/the-wikitext-long-term-dependency-language-modeling-dataset/). Download the daataset and unzip the folder. Pass the path of the unzipped folder to the preprocess code. Preprocess the dataset by running the following command:

`python -m preprocess.wikitext --wiki-input-path ../../Data/wikitext-2/ --wiki-output-path ../../Data/wiki-preprocessed/`

## To pretrain bert on a specific domain, run the following command:
`python -m model.lm_train --train_file ../../Data/wiki-preprocessed/wiki.train.tokens --test_file ../../Data/wiki-preprocessed/wiki.test.tokens`

## Train Loss on running batches
<img src="./Train_LOSS.svg" height="50%" width="50%">

## Eval loss on the eval file
<img src="./Eval_LOSS.svg" height="50%" width="50%">
