# python3 main.py no_vgg
# python3 main.py "multihead_70-30_ColorFilter_DataAugX1_KSplits8" --preproc 1 --traintest 0.30 --epochs 40 --ksplits 8 --dataaug 1

# models="Attention"
# preproc="1 2"
# splits="0.50 0.40 0.30 0.15"
# aug="1"
#
# for a in $aug
# do
#     for s in $splits
#     do
#         for p in $preproc
#         do
#             for m in $models
#             do
#         python3 main.py "06Sett2019_${m}_Split-${s}_Preproc${p}_DataAugX${a}_KSplits10" --model ${m} --preproc ${p} --ksplits 10 --epochs 30 --traintest ${s} --dataaug ${a}
#             done
#         done
#     done
# done

models="Attention"
preproc="1 2"
splits="0.30"
aug="3"

for a in $aug
do
    for s in $splits
    do
        for p in $preproc
        do
            for m in $models
            do
        python3 main.py "06Sett2019_${m}_Split-${s}_Preproc${p}_DataAugX${a}_KSplits10" --model ${m} --preproc ${p} --ksplits 10 --epochs 30 --traintest ${s} --dataaug ${a}
            done
        done
    done
done
# models="MLP ResNet Multihead"
# preproc="1 2"
# splits="0.15"
# aug="1"
# for a in $aug
# do
#     for s in $splits
#     do
#         for p in $preproc
#         do
#             for m in $models
#             do
#         python3 main.py "05Sett2019_${m}_Split-${s}_Preproc${p}_DataAugX${a}_KSplits10" --model ${m} --preproc ${p} --ksplits 10 --epochs 30 --traintest ${s} --dataaug ${a}
#             done
#         done
#     done
# done
