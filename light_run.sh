mkdir $1
mkdir $1/artefact
mkdir $1/result

python preprocess.py light_config.yaml
python light_train.py light_config.yaml
python light_inference.py light_config.yaml
python light_evaluation.py light_config.yaml