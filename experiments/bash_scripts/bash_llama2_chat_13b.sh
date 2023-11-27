export gpu_num=$1
export id_limit=50
export id=0
export n_train=$[50/$gpu_num+1]

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
        export offset=$[$gpu*$n_train]
        bash llama2_chat_13b.sh $id $gpu $n_train $offset &
    done
    export id=$[$id+1]
    wait
done
