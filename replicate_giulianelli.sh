python3 convert_corpora.py
wget -0 ./data/models/gulordava/model.pt https://dl.fbaipublicfiles.com/colorless-green-rnns/best-models/English/hidden650_batch128_dropout0.2_lr20.0.pt
python3 extract.py -c config/extract_gulordava.json
python3 extract.py -c config/extract_giulianelli.json
python3 classify.py -c config/classify.json
python3 replication.py -c config/replication.json
