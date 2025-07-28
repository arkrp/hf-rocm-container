# simple container for LLMChat application
# requires all the rocm/pytorch requirements
# requires a TRANSFORMERS_CACHE at mounted at /mnt/TRANSFORMERS_CACHE
FROM rocm/pytorch
RUN apt-get install -y libhiredis1.1.0
RUN pip install transformers accelerate
COPY ./install/install_bitsandbytes.sh /install/install_bitsandbytes.sh
RUN bash /install/install_bitsandbytes.sh
WORKDIR /app
COPY ./app /app
CMD ["python", "/app/main.py"]

