FROM python

COPY . /app
WORKDIR /app

RUN pip install -r requirements.txt

EXPOSE 8501 

CMD ["streamlit", "run", "app.py", "--server.port", "8501"]
