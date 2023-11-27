export gpu_num=$1
export id_limit=40
export id=0
export n_train=$[50/$gpu_num/2+1]

while [ $gpu_num -ge 0 ];
do
    if [ $id -ge $id_limit ]; then
        break
    fi
    for ((gpu=0; gpu < gpu_num ; gpu++)) 
    do
        if [ $id -ge $id_limit ]; then
            break
        fi
        export offset=$[$gpu*$n_train*2]
        export offset_1=$[$gpu*$n_train*2+$n_train]
        bash llama2_chat_7b.sh $id $gpu $n_train $offset &
        bash llama2_chat_7b.sh $id $gpu $n_train $offset_1 &
    done
    export id=$[$id+1]
    wait
done
