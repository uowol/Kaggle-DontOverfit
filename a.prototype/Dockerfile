FROM amd64/python:3.9-slim

WORKDIR /usr/app

RUN pip install -U pip &&\
    pip install streamlit pandas requests 

COPY app.py app.py 

CMD ["streamlit", "run", "app.py"]