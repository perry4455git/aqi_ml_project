from django.shortcuts import render,redirect
from django.contrib.auth.models import User
from django.contrib import messages
from django.contrib import auth
from django.contrib.auth import authenticate
from .models import *
from django.contrib.sessions.models import Session
import pandas as pd
import numpy as np
from .WeatherPrediction import *
import csv
# Create your views here.
def login(request):
    if request.POST:
        email = request.POST['email']
        password = request.POST['password']
        count = User.objects.filter(email=email,password=password).count()
        if count >0:
            #request.session['is_logged'] = True
            request.session['user_id'] = User.objects.values('id').filter(email=email,password=password)[0]['id']
            return redirect('index')
        else:
            messages.error(request, 'Invalid email or password')
            return redirect('login')
    return render(request, 'login.html')

def signup(request):
    if request.POST:
        username = request.POST['username']
        email = request.POST['email']
        password = request.POST['password']
        obj = User(username=username,email=email,password=password)
        obj.save()
        messages.success(request, 'You are registered successfully')
        return redirect('login')
    return render(request, 'signup.html')

def index(request):
    predict_accuracy = ''
    svm_predict = ''
    neigh_predict = ''
    tree = ''
    count=''
    svm_score=''
    neigh_score=''
    aqi_level = ''
    predicted_aqi_level = ''
    if request.POST:

        post_request_allowed = ['PM2.5','PM10','NO','NO2','NOx','NH3','CO','SO2','O3','Benzene','Toluene','AQI']
        prediction_list = []
        for param in post_request_allowed:
            #prediction_list.append(1)#harcode value uncomment below and comment this line
            prediction_list.append(float(request.POST[param]))
        print('prediction_list --- ', prediction_list)

        post_aqi_list = ['PM2.5', 'PM10', 'NO2', 'NH3', 'SO2', 'CO', 'O3']
        predict_aqi_input = []
        for param in post_aqi_list:
            predict_aqi_input.append(float(request.POST[param]))
        #predict_aqi_input = [123, 45, 67, 34, 5, 0, 23]
        print('predict_aqi_input --- ', predict_aqi_input)

        predict_accuracy , svm_score,neigh_score,svm_predict,neigh_predict = predict_weather(prediction_list)
        predicted_aqi_level = predict_aqi(predict_aqi_input)

        if 0 <= predicted_aqi_level <= 50:
            aqi_level = 'Good'
        elif 51 <= predicted_aqi_level <= 100:
            aqi_level = 'Satisfactory'
        elif 101 <= predicted_aqi_level <= 200:
            aqi_level = 'Moderate'
        elif 201 <= predicted_aqi_level <= 300:
            aqi_level = 'Poor'
        elif 301 <= predicted_aqi_level <= 400:
            aqi_level = 'Very Poor'
        else:
            aqi_level = 'Severe'

        if (svm_predict==0)and (neigh_predict==0):
            bad="BAD"

            with open(r'C:/Users/Prasheel/Downloads/AIR_QUALITY_PREDICTION Final_v1/AIR_QUALITY_PREDICTION Final1/AIR_QUALITY_PREDICTION/air/air_app/tree.csv', mode='r') as csv_file:
                csv_reader = csv.DictReader(csv_file)
                line_count = 0
                name = aqi_level
                print(name)
                for row in csv_reader:
                    if row["result"] == name:
                        tree = row["name"]
                        count=row["count"]
                        print(tree)
                        print(count)

                        print("Pollution is  "+aqi_level,tree,count)

          #  messages='Pollution is Bad .Plant more Trees'
        else:
            good="GOOD"

            with open(r'C:/Users/Admin/Downloads/AIR_QUALITY_PREDICTION Final1/AIR_QUALITY_PREDICTION Final1/AIR_QUALITY_PREDICTION/air/air_app/tree.csv', mode='r') as csv_file:
                csv_reader = csv.DictReader(csv_file)
                line_count = 0
                name = aqi_level
                #print(name)
                for row in csv_reader:
                    if row["result"] == name:
                        tree= row["name"]
                        count = row["count"]
                        print(tree)
                        print(count)
                        print("Solution  "+aqi_level,tree,count)

       # messages='Pollution is Low .Plant Trees For Better Living'


    return render(request, 'index.html', {'predict_accuracy': predict_accuracy,'svm_predict':svm_score,'neigh_predict':neigh_score,'tree':tree,'count':count, 'predicted_aqi_level': predicted_aqi_level, 'aqi_level': aqi_level})

def info(request):
    return render(request, 'info.html')

def about(request):
    return render(request, 'about.html')

def contact(request):
    if request.POST:
        first_name = request.POST['first_name']
        last_name = request.POST['last_name']
        email = request.POST['email']
        comment = request.POST['comment']
        obj = Contact(first_name=first_name, last_name=last_name, email=email, comment=comment)
        obj.save()
        return redirect('index')
    return render(request, 'contact.html')

def logout(request):
    auth.logout(request)
    return redirect('login')