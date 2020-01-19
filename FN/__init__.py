from flask import Flask, render_template, request, redirect,  session, url_for
from flask_sqlalchemy import SQLAlchemy
from flask_uploads import UploadSet, configure_uploads, IMAGES, patch_request_class
from flask_dropzone import Dropzone
import os
app = Flask(__name__)
dropzone = Dropzone(app)

# Dropzone settings
app.config['DROPZONE_UPLOAD_MULTIPLE'] = True
app.config['DROPZONE_ALLOWED_FILE_CUSTOM'] = True
app.config['DROPZONE_ALLOWED_FILE_TYPE'] = 'image/*'
app.config['DROPZONE_REDIRECT_VIEW'] = 'results'

app.config['UPLOADED_PHOTOS_DEST'] = '/savehere'


photos = UploadSet('photos',extensions=('txt','doc', 'docx','jpg', 'jpe', 'jpeg', 'png', 'mp4' ), default_dest=None) 

#app.config['UPLOADED_PHOTOS_DEST'] = 'models/imaage/df'
configure_uploads(app, photos)
patch_request_class(app)  # set maximum file size, default is 16MB

app.config['SECRET_KEY'] = '5791628bb0b13ce0c676dfde280ba245'

from FN import routes
from FN.classifiers import *
from FN.pipeline import *
