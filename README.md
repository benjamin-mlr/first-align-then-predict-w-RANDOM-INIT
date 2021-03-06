# First Align, then Predict: Understanding the Cross-Lingual Ability of Multilingual BERT 

This repository includes pointers and scripts to reproduce the experiments presented in our paper [First Align, then Predict: Understanding the Cross-Lingual Ability of Multilingual BERT](https://arxiv.org/abs/2101.11109) accepted at [EACL 2021](https://2021.eacl.org/)


## Setting up 

`cd ./first-align-then-predict`

`bash install.sh` 

`conda activate align-then-predict` 

## Computing Cross-Lingual Similarity  

We measure mBERT's hidden representation similarity between source and target languages with the [Central Kernel Alignment metric (CKA)](https://arxiv.org/abs/1905.00414) 

### Downloading parallel data

In our paper, we use the parrallel sentences provided by the PUD UD treebanks.  

The PUD treebanks are available here: 

`curl --remote-name-all https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-3424{/ud-treebanks-v2.7.tgz,/ud-documentation-v2.7.tgz,/ud-tools-v2.7.tgz}`

After uncompressing them, we can use the `./{lang_src}-ud-test.conllu` and `./{lang_trg}-ud-test.conllu` as parrallel sentences between `lang_src` and `lang_trg`. 


### Computing Cross-Lingual Similarity with the CKA metric 


```
python ./measure_similarity.py \
--data_dir ./data/ \                # location of parrallel data in the source and target languages
--source_lang_dataset en_pud \      # prefix of the dataset name of the source language
--target_lang_list fr_pud de_pud \  # list of prefix of the dataset name of the source language
--dataset_suffix ' -ud-test.conllu'\# suffix of the dataset names (should be the same for all source and target languages e.g. en_pud-ud-test.conllu )
--line_filter '# text =' \          # if we work with conllu files, we filter-in only the raw sentences starting with '# text =' 
--report_dir ./  \                  # directory where a json file will be stored with the similarity metric
--n_sent_total 100 \                # how many parrallel sentences picked from each file (it will sample the n_sent_total top sentences)
```

This script will printout and write in report_dir the CKA score between the source language and each target language for each layer hidden layer of mBERT. 


NB: 
- We assume that each dataset will follow the template: args.data_dir/{dataset_name}{dataset_suffix}
- To measure the similarity , the dataset between the source and the target languages should be aligned at the sentence level (for instance `en_pud-ud-test.conllu` and `de_pud-ud-test.conllu` are aligned). 


## RANDOM-INIT

Random-init now supports a more recent version of transformers

`pip install transformers==4.10.2`  

You can use random-init by importing BertModel from random_init.modeling_bert_random_init instead of the original transformers library. 
  
Then, to use random-init, simply add the `random_init_layers` argument to the `from_pretrained()` method: 

 
random_init_layers: List(str)

To apply random-init to a specific layer in a model, add a string or a regex to random_init_layers that matches a layer name in the model.    
For instance, to match the self-attention of the first layer in BERT, add `bert.encoder.layer.0.attention.*` to random_init_layers. This will randomly-initialize the first attention layer of mBERT.


Example: Applying random-init to mBERT layer 0 and 1
```
                                                          					 
>>> from random_init.modeling_bert_random_init import BertForTokenClassification
>>> mbert_w_random_init = BertForTokenClassification.from_pretrained("bert-base-multilingual-cased", random_init_layers=['bert.encoder.layer.[0-1]{1}.attention.*', 'bert.encoder.layer.[0-1]{1}.output.*', 'bert.encoder.layer.[0-1]{1}.intermediate.*'])
(stdout) RANDOM-INIT was applied to the following layers ['bert.encoder.layer.0.attention.', 'bert.encoder.layer.0.intermediate.', 'bert.encoder.layer.0.output.', 'bert.encoder.layer.1.attention.', 'bert.encoder.layer.1.intermediate.', 'bert.encoder.layer.1.output.'] based on argument ['bert.encoder.layer.[0-1]{1}.attention.*', 'bert.encoder.layer.[0-1]{1}.output.*', 'bert.encoder.layer.[0-1]{1}.intermediate.*']			  
```

Applying random-init to mBERT layer 10 and 11
```
                                                          					 
>>> from random_init.modeling_bert_random_init import BertForTokenClassification
>>> mbert_w_random_init = BertForTokenClassification.from_pretrained("bert-base-multilingual-cased", random_init_layers=['bert.encoder.layer.[10]{2}.attention.*', 'bert.encoder.layer.[10]{2}.output.*', 'bert.encoder.layer.[10]{2}.intermediate.*'])
(stdout) RANDOM-INIT was applied to the following layers ['bert.encoder.layer.10.attention.', 'bert.encoder.layer.10.intermediate.', 'bert.encoder.layer.10.output.', 'bert.encoder.layer.11.attention.', 'bert.encoder.layer.11.intermediate.', 'bert.encoder.layer.11.output.'] based on argument ['bert.encoder.layer.[10]{2}.attention.*', 'bert.encoder.layer.[10]{2}.output.*', 'bert.encoder.layer.[10]{2}.intermediate.*']
```





# How to cite 

If you extend or use this work, please cite:

```
@misc{muller2021align,
      title={First Align, then Predict: Understanding the Cross-Lingual Ability of Multilingual BERT}, 
      author={Benjamin Muller and Yanai Elazar and Beno??t Sagot and Djam?? Seddah},
      year={2021},
      eprint={2101.11109},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```