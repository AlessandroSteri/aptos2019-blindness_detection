# python3 main.py no_vgg
# python3 main.py "multihead_70-30_ColorFilter_DataAugX1_KSplits8" --preproc 1 --traintest 0.30 --epochs 40 --ksplits 8 --dataaug 1
# python3 main.py "multihead_70-30_ColorCircleCrop_DataAugX1_KSplits8" --preproc 2 --traintest 0.30 --epochs 40 --ksplits 8 --dataaug 1
# python3 main.py "multihead_70-30_ColorFilter_DataAugX2_KSplits8" --preproc 1 --traintest 0.30 --epochs 40 --ksplits 8 --dataaug 2
# python3 main.py "SimpleResNet_85-15_ColorFilter_DataAugX1_KSplits10" --preproc 1 --traintest 0.15 --epochs 30 --ksplits 10 --dataaug 1
# python3 main.py "SimpleResNet_70-30_ColorFilter_DataAugX1_KSplits10" --preproc 1 --traintest 0.30 --epochs 30 --ksplits 10 --dataaug 1
# python3 main.py "BestModel_40-60_ColorFilter_DataAugX0_KSplits10" --preproc 1 --traintest 0.60 --epochs 30 --ksplits 10 --dataaug 0
# python3 main.py "BestModel_50-50_ColorFilter_DataAugX0_KSplits10" --preproc 1 --traintest 0.50 --epochs 30 --ksplits 10 --dataaug 0
# python3 main.py "BestModel_60-40_ColorFilter_DataAugX0_KSplits10" --preproc 1 --traintest 0.40 --epochs 30 --ksplits 10 --dataaug 0
# python3 main.py "BestModel_70-30_ColorFilter_DataAugX0_KSplits10" --preproc 1 --traintest 0.30 --epochs 30 --ksplits 10 --dataaug 0
# python3 main.py "BestModel_85-15_ColorFilter_DataAugX0_KSplits10" --preproc 1 --traintest 0.15 --epochs 30 --ksplits 10 --dataaug 0

models="MLP Attention ResNet Multihead"
preproc="1 2"
splits="0.30 0.15"
aug="2"

for a in $aug
do
    for s in $splits
    do
        for p in $preproc
        do
            for m in $models
            do
        python3 main.py "04Sett2019_${m}_Split-${s}_Preproc${p}_DataAugX${a}_KSplits10" --model ${m} --preproc ${p} --ksplits 10 --epochs 30 --traintest ${s} --dataaug ${a}
            done
        done
    done
done


models="MLP Attention ResNet Multihead"
preproc="1 2"
splits="0.50 0.40 0.30 0.15"
aug="3"
for a in $aug
do
    for s in $splits
    do
        for p in $preproc
        do
            for m in $models
            do
        python3 main.py "04Sett2019_${m}_Split-${s}_Preproc${p}_DataAugX${a}_KSplits10" --model ${m} --preproc ${p} --ksplits 10 --epochs 30 --traintest ${s} --dataaug ${a}
            done
        done
    done
done
