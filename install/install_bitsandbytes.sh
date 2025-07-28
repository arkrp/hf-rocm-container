# Clone the github repo
git clone --recurse https://github.com/ROCm/bitsandbytes.git
cd bitsandbytes
git checkout rocm_enabled_multi_backend

# Install dependencies
pip install -r requirements-dev.txt

# Use -DBNB_ROCM_ARCH to specify target GPU arch
cmake -DBNB_ROCM_ARCH="gfx1100" -DCOMPUTE_BACKEND=hip -S .

# Compile the project
make

# Install
python setup.py install
