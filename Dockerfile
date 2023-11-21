FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ENV PYTHONUNBUFFERED=1

RUN apt update && apt install -y python3 wget python-is-python3
RUN wget https://bootstrap.pypa.io/get-pip.py && python get-pip.py

RUN apt update && apt install --no-install-recommends -y git

RUN \
    pip install git+https://github.com/kopyl/oneflow-diffusers.git && \
    pip install \
        torch==2.0.1+cu118 \
        --extra-index-url https://download.pytorch.org/whl/cu118 && \
    pip install xformers==0.0.20 && \
    pip install transformers==4.27.1 && \
    pip install diffusers[torch]==0.19.3 && \
    pip install --pre oneflow -f https://oneflow-pro.oss-cn-beijing.aliyuncs.com/branch/master_open_source/cu118 && \
    pip cache purge

COPY examples/text_to_image.py /text_to_image_example.py
COPY examples/text_to_image_no_args.py /text_to_image_no_args_example.py

CMD ["sleep", "infinity"]

# depot project name: `kopyl-oneflow-diffusers`
# depot build command: `sudo depot build --project f3fwh7xht9 . -t kopyl/oneflow-diffusers --push`