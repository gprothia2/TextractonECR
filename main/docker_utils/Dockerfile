FROM python:3.8
WORKDIR /ee

COPY ./EulerEye-master3/requirements.txt .
RUN pip3 install -r requirements.txt

COPY ./EulerEye-master3/requirements2.txt .
RUN pip3 install -r requirements2.txt

COPY ./EulerEye-master3 .

ENTRYPOINT ["sh","./runscript"]
