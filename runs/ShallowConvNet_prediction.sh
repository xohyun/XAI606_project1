# LOAD_PATH="./result/train/5"  # example

# python main.py --all_subject --mode=test --get_prediction --load_path=${LOAD_PATH} --device=0

LOAD_PATH="./result/train/521"  # example
criterion="CEE"
python main.py --all_subject --mode=test --get_prediction --load_path=${LOAD_PATH} --criterion=${criterion} --device=0
