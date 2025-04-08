FROM python:3.10-slim

WORKDIR /APP

COPY flask_app /APP/flask_app
COPY model/vectorizer.pkl /APP/model/vectorizer.pkl
COPY flask_app/.project-root /APP/.project-root
COPY src /APP/src

RUN pip install -r /APP/flask_app/requirements.txt

RUN python -m nltk.downloader stopwords wordnet

# Set PYTHONPATH so Python can find 'flask_app'
ENV PYTHONPATH=/APP

EXPOSE 5000

# Local run
CMD [ "python", "flask_app/app.py" ]

# #cicd
# CMD ["gunincorn " , "--blind"   "0.0.0.0:5000","--timeout" , "120" ,"app:app"]