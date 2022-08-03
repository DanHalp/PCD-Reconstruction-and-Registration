PROJ_HOME=`pwd`

sudo chmod -R ugo+rwx $PROJ_HOME/SpareNet

pip install --upgrade https://github.com/unlimblue/KNN_CUDA/releases/download/0.2/KNN_CUDA-0.2-py3-none-any.whl

cd $PROJ_HOME/SpareNet/cuda/emd/
rm -rf build/*
python setup.py install

cd $PROJ_HOME/SpareNet/cuda/MDS/
rm -rf build/*
python setup.py install

cd $PROJ_HOME/SpareNet/cuda/expansion_penalty/
rm -rf build/*
python setup.py install