# har-pytorch
Deep Learning for HAR: models and tools for Human Activity Recognition from IMU sensor data

virtualenv -p python3.6 venv
start

pip install transformers
pip install tensorflow tensorflow_datasets
pip install torch torchvision
pip install pandas sklearn matplotlib
pip install ipykernel cloudpickle

cd ..
# git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --no-cache-dir ./
cd ../hugface

## https://gist.github.com/W4ngatang/60c2bdb54d156a41194446737ce03e2e
# python download_glue_data.py --data_dir /home/david/data/hugface/glue --tasks all