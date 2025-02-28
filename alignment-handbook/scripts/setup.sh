apt-get install git-lfs
python -m pip install .
#export PATH=/usr/local/cuda/bin:$PATH
#export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
python -m pip install flash-attn --no-build-isolation
pip install git+https://github.com/huggingface/trl.git
