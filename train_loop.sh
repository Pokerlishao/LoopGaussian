LIB_PATH=$(pwd)
export LD_LIBRARY_PATH=$LIB_PATH/CinemaGaussian/lib:$LD_LIBRARY_PATH

python train_loop.py --task_name $1


