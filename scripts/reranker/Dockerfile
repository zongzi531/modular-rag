FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

COPY ./model ./model

COPY requirements.txt .

RUN python3 -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

COPY app.py Dockerfile .

ENTRYPOINT python3 app.py
