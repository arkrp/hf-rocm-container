# hf-rocm-container

Hello there! Hannah here!
Basically I want to run some LLMs right? But I have AMD!!! So this container is for hosting a OpenAI api using huggingface LLMS inside a rocm standard docker container! Welcome to the wonderful world of working around the fact that everything is built for NIVIDIA!

Yeah, but basically. This is a rocm based container for a Open Ai API interfaced Huggingface chat model instance. Have fun!

Supported models:
Gemma 3 (WIP)

# Requirements for host
ROCM >= 6.4.2
User has to go into install/install_bitsandbytes.sh and manually specify the amd gpu architecture that they are using!
