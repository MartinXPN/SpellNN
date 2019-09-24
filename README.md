# SpellNN
Multi-language Neural Spell-checking


[![Build Status](https://travis-ci.org/MartinXPN/SpellNN.svg?branch=master)](https://travis-ci.org/MartinXPN/SpellNN)
[![license](https://img.shields.io/github/license/mashape/apistatus.svg?maxAge=2592000)](https://github.com/MartinXPN/SpellNN/blob/master/LICENSE)


### Code organization
TODO

### Datasets
TODO
```bash
wiki2text.py --locale en --path datasets/en-articles.bz2 \                      # download latest pages-articles in XML format
             -o datasets/data/wiki/ --processes 4 datasets/en-articles.bz2 \    # parse the XML with WikiExtractor
             --wiki_dir datasets/data/wiki --output_path datasets/en_wiki.txt   # convert to .txt

# Cleanup
rm -rf datasets/data datasets/*articles.bz2
```

### Model Architecture
TODO

### Training
TODO

### Evaluation & Results
TODO

### Installation
```bash
git clone git@github.com:MartinXPN/SpellNN.git && cd SpellNN
pip install .[tf-gpu]  # Or `pip install .[tf]` if no gpu support is needed
```
