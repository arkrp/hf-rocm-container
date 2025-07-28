application_name="arkrp-gemma-api"
project_directory=$(realpath $(dirname $BASH_SOURCE))
cd $project_directory
sudo docker build -t $application_name $project_directory && \
sudo docker run -it --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
--device=/dev/kfd --device=/dev/dri --group-add video \
--ipc=host --shm-size 8G \
-v $HF_HOME:/mnt/HF_HOME -e HF_HOME=/mnt/HF_HOME \
$application_name
