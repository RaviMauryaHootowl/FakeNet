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
import numpy as np
from classifiers import *
from pipeline import *
from keras.preprocessing.image import ImageDataGenerator
 
from classifiers import *
from pipeline import *
from keras.preprocessing.image import ImageDataGenerator
import keras.backend.tensorflow_backend as tb
tb._SYMBOLIC_SCOPE.value = True
app = Flask(__name__)

dropzone = Dropzone(app)

# Dropzone settings
app.config['DROPZONE_UPLOAD_MULTIPLE'] = True
app.config['DROPZONE_ALLOWED_FILE_CUSTOM'] = True
app.config['DROPZONE_ALLOWED_FILE_TYPE'] = 'image/*'
app.config['DROPZONE_REDIRECT_VIEW'] = 'results'
app.config['UPLOADED_PHOTOS_DEST'] = 'test_images/df'

patch_request_class(app)  # set maximum file size, default is 16MB
app.config['SECRET_KEY'] = '5791628bb0b13ce0c676dfde280ba245'

photos = UploadSet('photos',extensions=('txt','doc', 'docx','jpg', 'jpe', 'jpeg', 'png', 'mp4' ), default_dest=None) 
app.config['UPLOADED_PHOTOS_DEST'] = 'test_images/df'
configure_uploads(app, photos)


classifier = Meso4()
classifier.load('weights/Meso4_DF')

# 2 - Minimial image generator
# We did use it to read and compute the prediction by batchs on test videos
# but do as you please, the models were trained on 256x256 images in [0,1]^(n*n)

dataGenerator = ImageDataGenerator(rescale=1./255)
generator = dataGenerator.flow_from_directory(
        'test_images',
        target_size=(256, 256),
        batch_size=1,
        class_mode='binary',
        subset='training')

classifier.load('weights/Meso4_F2F')
predictions = compute_accuracy(classifier, 'test_videos')
for video_name in predictions:
    answer=str(predictions[video_name][0])
    file1 = open("myvideo.txt","w") 
    #L = [answer]  
    file1.write(answer)
    file1.close() 
    print('`{}` video class prediction :'.format(video_name), predictions[video_name][0])

# 3 - Predict
X, y = generator.next()
print('Predicted :', classifier.predict(X), '\nReal class :', y)
answer=str(classifier.predict(X))
file1 = open("myimage.txt","w") 
#L = [answer]  
file1.write(answer)
file1.close() 

@app.route("/")
@app.route("/home", methods=['GET','POST'])
def home():
    return render_template('index.html')

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
        model = joblib.load('./newsmodel.pkl') 
        tfidf_vect = joblib.load('./vectorizer.pickle')
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

@app.route('/upload-image', methods=['GET', 'POST'])
def upload_image():
    tb._SYMBOLIC_SCOPE.value = True
    if request.method == 'POST' and 'photo' in request.files:
        filename = photos.save(request.files['photo'])
        classifier = Meso4()
        classifier.load('weights/Meso4_DF')

        # 2 - Minimial image generator
        # We did use it to read and compute the prediction by batchs on test videos
        # but do as you please, the models were trained on 256x256 images in [0,1]^(n*n)

        dataGenerator = ImageDataGenerator(rescale=1./255)
        generator = dataGenerator.flow_from_directory('test_images',target_size=(256, 256), batch_size=1,class_mode='binary',subset='training')

        # 3 - Predict
        X, y = generator.next()
        answer=classifier.predict(X)
        print('Predicted :', answer)
        answer=str(answer)
        file1 = open("myfile.txt","w") 
        L = ["This is Delhi \n","This is Paris \n","This is London \n"]  
        file1.close() 
        print("write file")
        # if(image_label>0.8):
        #     image_label=1
        
        # else:
        #     image_label=0
        print('function executed')
        os.remove(os.path.join(app.config['UPLOADED_PHOTOS_DEST'], filename))
        return filename
    return render_template('upload.html')
'''
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
    return render_template('upload_image.html')

@app.route('/results')
def results():
    
    # redirect to home if no images to display
    if "file_urls" not in session or session['file_urls'] == []:
        return redirect(url_for('index'))
        
    # set the file_urls and remove the session variable
    file_urls = session['file_urls']
    session.pop('file_urls', None)
    
    return render_template('results.html', file_urls=file_urls)

'''

@app.route('/upload-video', methods=['GET', 'POST'])
def upload_video():
    if request.method == 'POST' and 'photo' in request.files:
        filename = photos.save(request.files['photo'])
        classifier.load('weights/Meso4_F2F')

        predictions = compute_accuracy(classifier, 'test_videos')

        for video_name in predictions:
            answer=str(predictions[video_name][0])
            file1 = open("myvideo.txt","w") 
            #L = [answer]  
            file1.write(answer)
            file1.close() 
            print('`{}` video class prediction :'.format(video_name), predictions[video_name][0])
        return filename
    return render_template('upload_video.html')


if __name__ == '__main__':
	app.run(debug=True)