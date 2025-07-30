# simple container for LLMChat application
# requires all the rocm/pytorch requirements
FROM rocm/pytorch:rocm6.4.1_ubuntu24.04_py3.12_pytorch_release_2.6.0
RUN pip install transformers==4.54.1
RUN pip install accelerate==1.9.0
RUN pip install -v gptqmodel==2.2.0 --no-build-isolation
RUN pip install optimum
RUN pip install datasets
RUN pip install logbar
RUN pip install tokenicer
RUN pip install device_smi
RUN pip install random_word
RUN pip install fastapi
Run pip install uvicorn
WORKDIR /app
COPY ./app /app
CMD ["python", "/app/main.py"]
