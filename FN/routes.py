from flask import Flask, session, escape, render_template, url_for, flash, redirect, request
from flask_dropzone import Dropzone
from flask_uploads import UploadSet, configure_uploads, IMAGES, patch_request_class
import os, sys
import secrets
#from FN import app
from PIL import Image as img
from sqlalchemy.orm import Session
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from math import log, sqrt
import pandas as pd
import numpy as np
import re
from sklearn.externals import joblib
import requests
from sqlalchemy import or_ , and_
from keras.preprocessing.image import ImageDataGenerator
from sklearn.externals import joblib
from flask import Flask, render_template, request, redirect,  session, url_for
from flask_sqlalchemy import SQLAlchemy
from flask_uploads import UploadSet, configure_uploads, IMAGES, patch_request_class
from flask_dropzone import Dropzone
import os
from werkzeug.utils import secure_filename
app = Flask(__name__)
dropzone = Dropzone(app)

# Dropzone settings
app.config['DROPZONE_UPLOAD_MULTIPLE'] = True
app.config['DROPZONE_ALLOWED_FILE_CUSTOM'] = True
app.config['DROPZONE_ALLOWED_FILE_TYPE'] = 'image/*'
app.config['DROPZONE_REDIRECT_VIEW'] = 'results'

app.config['UPLOADED_PHOTOS_DEST'] = 'models/text'


photos = UploadSet('photos',extensions=('txt','doc', 'docx','jpg', 'jpe', 'jpeg', 'png', 'mp4' ), default_dest=None) 
app.config['UPLOADED_PHOTOS_DEST'] = 'models/text'
#app.config['UPLOADED_PHOTOS_DEST'] = 'models/imaage/df'
configure_uploads(app, photos)
patch_request_class(app)  # set maximum file size, default is 16MB

app.config['SECRET_KEY'] = '5791628bb0b13ce0c676dfde280ba245'

#from FN import routes
#from FN.classifiers import *
#from FN.pipeline import *


@app.route("/")
@app.route("/home", methods=['GET','POST'])
def home():
    return render_template('index.html')


photos = UploadSet('photos',extensions=('txt','doc', 'docx','jpg', 'jpe', 'jpeg', 'png', 'mp4' ), default_dest=None) 
app.config['UPLOADED_PHOTOS_DEST'] = 'models/text'
configure_uploads(app, photos)
#from classifiers import *
#from pipeline import *
@app.route('/upload-text', methods=['GET', 'POST'])
def upload_text():
    if request.method == 'POST' and 'text' in request.files:
        filename = photos.save(request.files['text'])
        f = open(os.path.join(app.config['UPLOADED_PHOTOS_DEST'], filename))
        text = f.read()
        f.close()
        os.remove(os.path.join(app.config['UPLOADED_PHOTOS_DEST'], filename))
        return redirect
    return render_template('news.html')

@app.route('/upload-text-simple', methods=['GET', 'POST'])
def upload_text_simple():
    if request.method == 'POST':
        textfeed = request.form['textfeed']
        model = joblib.load('./FN/newsmodel.pkl') 
        tfidf_vect = joblib.load('./FN/vectorizer.pickle')
        mylist = []
        mylist.append(textfeed)
        mylist = list(mylist)
        df2 = pd.DataFrame(mylist, columns = ['textinput'])
        myX = df2.textinput
        mytest = tfidf_vect.transform(myX)
        #return mytest
        prediction = str(model.predict(mytest)[0])
        print('pred: ', prediction)
        if prediction == '1':
            prediction = "Real"
        else:
            prediction = "Fake"
        return render_template('news.html', prediction=prediction)
    return render_template('news.html')



UPLOAD_FOLDER = 'models/images'
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
@app.route('/upload-image', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        return filename
    return render_template('upload.html')

def upload_image():
    if request.method == 'POST' and 'photo' in request.files:
        filename = photos.save(request.files['photo'])
        os.mkdir(os.path.join(app.config['UPLOADED_PHOTOS_DEST'], filename))
        classifier = Meso4()
        classifier.load('weights/Meso4_DF')
        dataGenerator = ImageDataGenerator(rescale=1./255)
        generator = dataGenerator.flow_from_directory('test_images',target_size=(256, 256),batch_size=1,class_mode='binary',subset='training')
        X, y = generator.next()
        image_label=classifier.predict(X)
        if(image_label>0.8):
            image_label=1
        else:
            image_label=0
        print('function executed')
        os.remove(os.path.join(app.config['UPLOADED_PHOTOS_DEST'], filename))
        #return redirect(url_for('images'), image_label=image_label)
        #return render_template('images.html', image_label=image_label)
        return filename
    return render_template('upload.html')
'''
@app.route('/upload-video', methods=['GET', 'POST'])
def upload_video():
    if request.method == 'POST' and 'photo' in request.files:
        filename = photos.save(request.files['photo'])
        
        classifier.load('weights/Meso4_F2F')

        predictions = compute_accuracy(classifier, 'test_videos')
        print(predictions)
        os.remove(os.path.join(app.config['UPLOADED_PHOTOS_DEST'], filename))
        return filename
    return render_template('video.html')
'''
 

'''
@app.route('/dropzone', methods=['GET', 'POST'])
def index():
    
    # set session for image results
    if "file_urls" not in session:
        session['file_urls'] = []
    # list to hold our uploaded image urls
    file_urls = session['file_urls']
    # handle image upload from Dropzone
    if request.method == 'POST':
        file_obj = request.files
        for f in file_obj:
            file = request.files.get(f)
            
            # save the file with to our photos folder
            filename = photos.save(
                file,
                name=file.filename    
            )
            # append image urls
            file_urls.append(photos.url(filename))
            
        session['file_urls'] = file_urls
        return "uploading..."
    # return dropzone template on GET request    
    return render_template('index.html')

@app.route('/results')
def results():
    
    # redirect to home if no images to display
    if "file_urls" not in session or session['file_urls'] == []:
        return redirect(url_for('index'))
        
    # set the file_urls and remove the session variable
    file_urls = session['file_urls']
    session.pop('file_urls', None)
    
    return render_template('results.html', file_urls=file_urls)
        return filename
    return render_template('news.html')'''

if __name__=='__main__':
    app.run(debug=True)