ROOT=savecode/
export PYTHONPATH=$PYTHONPATH:$ROOT
mkdir snapshot
CUDA_VISIBLE_DEVICES=0  python -u $ROOT/main.py --log loguvg.txt --testuvg --pretrain snapshot/iter363456.model --config config.json
