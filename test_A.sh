for dir in ./data/test_A/*
do
    if test -d $dir # check if it is derectory
    then
        echo processing $dir saving to results/${dir##*/}
        python test_fastdvdnet.py --device_id 0 --model_file ./results/net.pth --test_path $dir --save_path results/${dir##*/}
    fi
done

python seq2mp4.py

