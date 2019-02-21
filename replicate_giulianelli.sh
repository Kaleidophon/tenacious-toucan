mkdir -p ./data/corpora/labeled/
python3 convert_corpora.py
wget -O ./data/models/gulordava/model.pt https://dl.fbaipublicfiles.com/colorless-green-rnns/best-models/English/hidden650_batch128_dropout0.2_lr20.0.pt
mkdir -p ./data/activations/giulianelli/
mkdir -p ./data/activations/gulordava/
mkdir -p ./data/classifiers/giulianelli/preds/
mkdir -p ./data/classifiers/giulianelli/models/
python3 replication/extract.py -c configs/extract_gulordava.json
python3 replication/extract.py -c configs/extract_giulianelli.json
python3 replication/classify.py -c configs/classify.json
python3 replication/replicate.py -c configs/replicate.json
