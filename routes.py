import sys
import os
import glob
import re
import numpy as np
import tensorflow
from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image


from flask import Flask,render_template,request,flash
import cv2
import time
import gdown
import numpy
import pandas
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from pathlib import Path
from keras import backend as K
from livelossplot import PlotLossesKeras
from sklearn.model_selection import KFold
from tensorflow.keras.utils import img_to_array
import keras
from keras import layers
from keras.models import Sequential
from keras.utils import load_img
input_shape = (224, 224, 3)

from predictbmi.forms import RegistrationForm,LoginForm
from predictbmi.models import User
from flask_login import login_user, current_user, logout_user, login_required


from predictbmi import app,bcrypt,db,login_manager



def coeff_determination(y_true, y_pred):
    SS_res =  K.sum(K.square( y_true-y_pred )) 
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) ) 
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )




@app.route('/',methods=['GET'])
def welcomePage():
    return render_template("welcomepage.html")

@app.route('/predictbmi',methods=['GET'])
def predictBMI():
    return render_template("predictpage.html")


@app.route('/login', methods=['GET','POST'])
def loginPage():
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(email=form.email.data).first()
        if user and bcrypt.check_password_hash(user.password, form.password.data):
            login_user(user, remember=form.remember.data)
            next_page = request.args.get('next')
            return redirect(next_page) if next_page else redirect(url_for('welcomePage'))
            print("THE CURRENT USER ADMIN CHECK........",current_user.isAdmin)
        else:
            flash('Login Unsuccessful. Please check email and password', 'danger')
    return render_template('loginpage.html', form=form)


@app.route('/register',methods=['GET','POST'])
def registerPage():
    form = RegistrationForm()
    if form.validate_on_submit():
        hashed_password = bcrypt.generate_password_hash(form.password.data).decode('utf-8')
        user = User(username=form.username.data, email=form.email.data, age=form.age.data ,password=hashed_password)
        db.session.add(user)
        db.session.commit()
        flash('Your account has been created! You are now able to log in', 'success')
        return redirect(url_for('loginPage'))
    return render_template('registerpage.html', form=form)

@app.route("/logout")
def logout():
    logout_user()
    return redirect(url_for('welcomePage'))

@app.route('/overweight',methods=['GET'])
@login_required
def overweightPage():
    return  render_template('overweight.html')

@app.route('/underweight',methods=['GET'])
@login_required
def underweightPage():
    return  render_template('underweight.html')

@app.route('/predict', methods=['GET', 'POST'])
@login_required
def upload():
    bmi_result="Normal"
    bmi_pred=0
    bmi_status=1
    if request.method == 'POST':
        f = request.files['file']
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        raw_input_image = cv2.imread(file_path)
        raw_input_image = cv2.cvtColor(raw_input_image, cv2.COLOR_BGR2RGB)
        raw_input_image = cv2.resize(raw_input_image, (input_shape[0], input_shape[1]))
        preprocessed_input_image = load_img(file_path, target_size=input_shape)
        preprocessed_input_image = img_to_array(preprocessed_input_image)

        preprocessed_input_image[preprocessed_input_image[:,:,0] > 0] = 1
        preprocessed_input_image[preprocessed_input_image[:,:,1] > 0] = 1
        preprocessed_input_image[preprocessed_input_image[:,:,2] > 0] = 1

        final_input_image = raw_input_image * preprocessed_input_image        

        test_datagen = ImageDataGenerator(
        samplewise_center=True,
        )

        base_model = keras.models.load_model('predictbmi/lib/model/base_model.h5')

        generator = test_datagen.flow(
            numpy.expand_dims(final_input_image, axis=0),
            batch_size=1
        )
        features_batch = base_model.predict(generator)

        dependencies = {
            'coeff_determination': coeff_determination
        }

        model = keras.models.load_model('predictbmi/lib/model/3.935_model.h5', custom_objects=dependencies)
        preds = model.predict(features_batch)
        bmi_pred = preds[0][0]
        print(f"BMI: {bmi_pred}")
        bmi_status=1
        bmi_result="Normal"
        if bmi_pred < 15:
            bmi_result="Very severely underweight"
            bmi_status=0
        elif 15 <= bmi_pred < 16:
            bmi_result="Severely underweight"
            bmi_status=0
        elif 16 <= bmi_pred < 18.5:
            bmi_result="Underweight"
            bmi_status=0
        elif 18.5 <= bmi_pred < 25:
            bmi_result="Normal"
            bmi_status=1
        elif 25 <= bmi_pred < 30:
            bmi_result="Overweight"
            bmi_status=2
        elif 30 <= bmi_pred < 35:
            bmi_result="Moderately obese"
            bmi_status=2
        elif 35 <= bmi_pred < 40:
            bmi_result="Severely obese"
            bmi_status=2
        elif bmi_pred >= 40:
            bmi_result="Very severely obese"
            bmi_status=2
    return render_template("resultPage.html",bmi_result=bmi_result,bmi_pred=bmi_pred,bmi_status=bmi_status)