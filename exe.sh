# python3 main.py no_vgg
python3 main.py "multihead_70-30_ColorFilter_DataAugX1_KSplits8" --preproc 1 --traintest 0.30 --epochs 40 --ksplits 8 --dataaug 1
python3 main.py "multihead_70-30_ColorCircleCrop_DataAugX1_KSplits8" --preproc 2 --traintest 0.30 --epochs 40 --ksplits 8 --dataaug 1
python3 main.py "multihead_70-30_ColorFilter_DataAugX2_KSplits8" --preproc 1 --traintest 0.30 --epochs 40 --ksplits 8 --dataaug 2
python3 main.py "multihead_70-30_ColorCircleCrop_DataAugX2_KSplits8" --preproc 2 --traintest 0.30 --epochs 40 --ksplits 8 --dataaug 2
