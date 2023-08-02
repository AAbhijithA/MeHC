'''
Importing the libraries needed
'''
from flask import Flask, render_template, request, url_for, session, send_file, redirect, render_template_string
from flask_sqlalchemy import SQLAlchemy
from dotenv import load_dotenv
import os
import random
import requests
from datetime import date
from bs4 import BeautifulSoup
import pickle
import json
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow import keras
from keras.models import load_model
from textblob import TextBlob
import time
import openai
import folium

'''
Setting up the database and apis for the backend
'''
load_dotenv()
openai.api_key = os.getenv("OAI_API_KEY")
gapikey = os.getenv("GA_API_KEY")
app = Flask(__name__,template_folder='templates')
app.secret_key = os.getenv("MS_KEY")
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('SDU')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

'''
Setting global variables for later use
'''
FILTER = ['disease','medical','accident','health','illness']
FILTER = set(FILTER)
gdate = None
links_fh = []
lemmatizer = WordNetLemmatizer()
intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl','rb'))
classes = pickle.load(open('classes.pkl','rb'))
model = load_model('mhcb.h5')
nltk.download('punkt')
nltk.download('wordnet')

'''
Construction of the SQLite Database
-Users for the basic usercredentials
-prevConversations for storing older conversations from the LLM model
-userSentiment for tracking sentiment data from MeHC Chatbot conversations
'''
class Users(db.Model):
    sno = db.Column(db.Integer, primary_key = True)
    username = db.Column(db.String(100), nullable = False)
    password = db.Column(db.String(100), nullable = False)
    def __repr__(self) -> str:
        return f"Users(username={self.username}, password={self.password})"

class prevConversation(db.Model):
    sno = db.Column(db.Integer, primary_key = True)
    username = db.Column(db.String(100), nullable = False)
    question = db.Column(db.String(1000), nullable = False)
    conversation = db.Column(db.String(1000), nullable = False)
    def __repr__(self) -> str:
        return f"prevConversation(username={self.username}, question={self.question}, conversation={self.conversation})"

class userSentiment(db.Model):
    sno = db.Column(db.Integer, primary_key = True)
    username = db.Column(db.String(100), nullable = False)
    sentiment = db.Column(db.Float, nullable = False)
    def __repr__(self) -> str:
        return f"userSentiment(username={self.username}, sentiment={self.sentiment})"

with app.app_context():
    db.create_all()

'''
Functionalities used by the backend code
'''
#Strip unwanted nextlines and spaces from text
def stripNLWS(txt):
    txt = txt.lstrip('\n')
    txt = txt.rstrip('\n')
    txt = txt.lstrip()
    txt = txt.rstrip()
    return txt

#construct a News Link from the given text and link
def construct_NL(txt,link):
    txt = stripNLWS(txt)
    tl = [txt,link]
    return tl

#Model pre-processing for predictions
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence,words):
    sentence_words = clean_up_sentence(sentence)
    bag = [0]*len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence,words,classes):
    bow = bag_of_words(sentence,words)
    res = model.predict(np.array([bow]))[0]
    results = [[i,r] for i, r in enumerate(res) if r > 0.22]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent':classes[r[0]],'probability':str(r[1])})
    return return_list

def get_response(intents_list,intents_json):
    if(len(intents_list) == 0):
        return "I am sorry I can't understand or help you on that, I still have a lot to improve on..."
    tag = intents_list[0]['intent']
    loi = intents_json['intents']
    result = None
    random.seed(time.time())
    for i in loi:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result

#Linear regression from given variables
def reg(n,x,y,xx,yy,xy):
    return ((y*xx) - (x*xy))/((n*xx) - (x*x)), y/n

#Response for motivating the user based on pre-calculated regression values
def response_generator(fs,ss,ts,fm,sm):
    predicted = None
    if fs >= ss and ss >= ts and fm >= sm:
        if fm > 0.2:
            predicted='You are in a stable condition and becoming relatively happy :)'
        else:
            predicted='Slowly on the path to stability and betterment, break out of the stigma!'
    elif fs >= ss and ss <= ts:
        if fm > 0.2:
            predicted='You are improving and keep going at it! You may have downfalls but it is alright!'
        else:
            predicted='You might be sad now but you can improve! You may have downfalls but it is alright!'
    elif fs <= ss and ss <= ts:
        predicted='Take some rest and think better, you will surely make it!'
    else:
        predicted='Be grateful for your chances and good in life and relax as well'
    return predicted

#Chatbot query verifier
def cquery_verifier(query):
    query = query.lstrip()
    query = query.rstrip()
    if (len(query) == 0) or (query == "Enter Message"):
        return False
    else:
        return True

#GPT query verifier 
def mquery_verifier(query):
    url = "https://www.google.com/search?q=" + query
    response = requests.get(url)
    global FILTER
    if response.status_code == 200:
        data = response.text
        data = str(data)
        if any(word in data for word in FILTER):
            return True
        else:
            return False
    else:
        return False

#Geoapify-link request construction
def ch_url(lat,lon):
    global gapikey
    return "https://api.geoapify.com/v2/places?categories=healthcare.hospital&filter=circle:"+lon+","+lat+",5000&bias=proximity:"+lon+","+lat+"&lang=en&limit=20&apiKey="+gapikey

'''
Dealing with the templates for the web-application
Has the following listed services
'''
#LOGIN-PAGE
@app.route('/',methods=['GET','POST'])
def login():
    if request.method == 'POST':
        usrnm = request.form['username']
        pswd = request.form['password']
        with app.app_context():
            usrnm = str(usrnm)
            pswd = str(pswd)
            try:
                qry = Users.query.filter_by(username=usrnm,password=pswd)
                res = qry.all()
                if len(res) == 0:
                    return render_template('login.html',error='Invalid credentials, Try Again')
                else:
                    session['username'] = usrnm
                    return redirect(url_for('home'))
            except:
                return render_template('login.html',error='Server error, try again later')
    return render_template('login.html',error = None)

#REGISTRATION-PAGE
@app.route('/register',methods=['GET','POST'])
def register():
    if request.method == 'POST':
        usrnm = request.form['username']
        pswd = request.form['password']
        with app.app_context():
            try:
                usrnm = str(usrnm)
                pswd = str(pswd)
                qry = Users.query.filter_by(username=usrnm)
                res = qry.all()
                if len(res) == 0:
                    try:
                        newuser = Users(username=usrnm,password=pswd)
                        db.session.add(newuser)
                        db.session.commit()
                        return render_template('register.html',error='Successfully registered!')
                    except:
                        return render_template('register.html',error='Server error, try again later')
                else:
                    return render_template('register.html',error='Username already exists')
            except:
                return render_template('register.html',error='Server error, try again later')
    return render_template('register.html',error = None)

#HOME-PAGE
@app.route('/home',methods=['GET','POST'])
def home():
    if 'username' in session:
        global gdate
        global links_fh
        if gdate == date.today():
            return render_template('home.html',news = links_fh,username = session['username'])
        gdate = date.today()
        links_fh = []
        page = requests.get("https://www.health.harvard.edu/blog")
        soup = BeautifulSoup(page.content, 'html.parser')
        res = soup.find_all('a',class_="hover:text-red transition-colors duration-200")
        random.seed(time.time())
        try:
            i = random.randint(0,len(res))
            j = i
            k = i
            while j == i:
                j = random.randint(0,len(res))
            while ((k == i) or (k == j)):
                k = random.randint(0,len(res))
            links_fh.append(construct_NL(res[i].text,res[i]['href']))
            links_fh.append(construct_NL(res[j].text,res[j]['href']))
            links_fh.append(construct_NL(res[k].text,res[k]['href']))
        except:
            pass
        try:
            page = requests.get("https://www.thehealthyhomeeconomist.com/")
            soup = BeautifulSoup(page.content, 'html.parser')
            res = soup.find_all('a',class_ = 'entry-title-link')
            i = random.randint(0,len(res))
            j = i
            while j == i:
                j = random.randint(0,len(res))
            while ((k == i) or (k == j)):
                k = random.randint(0,len(res))
            links_fh.append(construct_NL(res[i].text,res[i]['href']))
            links_fh.append(construct_NL(res[j].text,res[j]['href']))
            links_fh.append(construct_NL(res[k].text,res[k]['href']))
        except:
            pass
        return render_template('home.html',news = links_fh,username = session['username'])
    else:
        return redirect(url_for('login'))

#MEHC-CHATBOT-PAGE
@app.route('/Chatbot',methods=['GET','POST'])
def Chatbot():
    if 'username' in session:
        if request.method == 'POST':
            prompt = request.form['prompt']
            prompt = str(prompt)
            prompt = prompt + " "
            if cquery_verifier(prompt) == False:
                render_template('Chatbot.html',response = "I can't help you with that... I am a health assistant")
            else:
                global words
                global classes
                global intents
                ints = predict_class(prompt,words,classes)
                res = get_response(ints,intents)
                usrnm = session['username']
                pol = TextBlob(prompt).sentiment.polarity
                nuS = userSentiment(username=usrnm,sentiment=pol)
                with app.app_context():
                    db.session.add(nuS)
                    db.session.commit()
                return render_template('Chatbot.html',response = res)
        return render_template('Chatbot.html',response = None)
    else:
        return redirect(url_for('login'))

#USER-STATUS-PAGE
@app.route('/status',methods=['GET','POST'])
def status():
    if 'username' in session:
        usrnm = session['username']
        qry = userSentiment.query.filter_by(username = usrnm)
        results = qry.all()
        x = [i for i in range(1,len(results)+1)]
        y = [results[i].sentiment for i in range(0,len(results))]
        itr = len(x) - 1
        fp = 2*len(x)/3
        sp = len(x)/3
        xys = 0
        xxs = 0
        yys = 0
        xs = 0
        ys = 0
        fs = None
        fm = None
        ss = None
        sm = None
        ts = None
        tm = None
        while(itr >= 0):
            if((fs == None) and (itr < fp)):
                fs, fm = reg(len(x)-1-itr,ys,xs,xxs,yys,xys)
            elif((ss == None) and (itr < sp)):
                ss, sm = reg(len(x)-1-itr,xs,ys,xxs,yys,xys)
            xs += x[itr]
            ys += y[itr]
            yys += y[itr]*y[itr]
            xxs += x[itr]*x[itr]
            xys += x[itr]*y[itr]
            itr -= 1
        ts, tm = reg(len(x)-1-itr,xs,ys,xxs,yys,xys)
        return render_template('status.html',username=usrnm,pred=response_generator(fs,ss,ts,fm,sm),x_data=x,y_data=y)
    else:
        return redirect(url_for('login'))

#MedGPT/MEDICAL-QUERY-PAGE
@app.route('/mquery',methods=['GET','POST'])
def mquery():
    if 'username' in session:
        if request.method == 'POST':
            prompt = request.form['prompt']
            prompt = str(prompt)
            prompt = prompt + " "
            print(prompt)
            if mquery_verifier(prompt) == True:
                cgpt_response = None
                try:
                    cgpt_response = openai.ChatCompletion.create(
                        model = "gpt-3.5-turbo",
                        messages = [{"role": "user", "content": prompt}]
                    )
                except:
                    return render_template('mquery.html',response="We are working on a issue with the backend, we will continue where we left off shortly")
                res = cgpt_response.choices[0].message.content
                usrnm = session['username']
                pol = TextBlob(prompt).sentiment.polarity
                with app.app_context():
                    nuS = userSentiment(username=usrnm,sentiment=pol)
                    nprevconv = prevConversation(username=usrnm,question=prompt,conversation=res)
                    db.session.add(nuS)
                    db.session.commit()
                    db.session.add(nprevconv)
                    db.session.commit()
                return render_template('mquery.html',response=res)
            return render_template('mquery.html',response="I can't help you with that as I am a Medical Assistant.")
        return render_template('mquery.html',response=None)
    else:
        return redirect(url_for('login'))

#PAST-CONVERSATIONS-PAGE
@app.route('/pconv',methods=['GET','POST'])
def pconv():
    if 'username' in session:
        usrnm = session['username']
        qry = prevConversation.query.filter_by(username = usrnm)
        CBconv = qry.all()
        CBconv = CBconv[::-1]
        return render_template('pconv.html',conversations=CBconv)
    else:
        return redirect(url_for('login'))

#HOSPITALS-NEARBY-PAGE
@app.route('/hmap',methods=['GET','POST'])
def hmap():
    if 'username' in session:
        if request.method == 'POST':
            lat = request.form['lat']
            lon = request.form['lon']
            lat = str(lat)
            lon = str(lon)
            print(lat)
            print(lon)
            requrl = ch_url(lat,lon)
            data = requests.get(requrl)
            data = data.json()
            lat = float(lat)
            lon = float(lon)
            map = folium.Map(
                location=[lat,lon],
                zoom_start=13,width=800,height=600
            )
            folium.Marker(
                [float(lat),float(lon)],
                icon = folium.Icon(icon="home",prefix="fa",color="green"),
                popup="<i>Your Location<i>"
                ).add_to(map)
            for i in range(0,len(data['features'])):
                folium.Marker(
                    location=[data['features'][i]['properties']['lat'],data['features'][i]['properties']['lon']],
                    icon = folium.Icon(icon="plus",prefix="fa",color="red"),
                    popup="<i>"+data['features'][i]['properties']['formatted']+"<i>"
                ).add_to(map)
            map.get_root().render()
            header_map = map.get_root().header.render()
            body_map = map.get_root().html.render()
            script_map = map.get_root().script.render()
            return render_template_string("""
            <!DOCTYPE html>
            <html>
                <head>
                    <meta charset="utf-8">
                    <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no">
                    <link rel="shortcut icon" href="{{ url_for('static', filename='favicon.ico') }}">
                    <title>MeHC</title>
                    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@4.0.0/dist/css/bootstrap.min.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">
                    {{ header_map | safe}}
                    <style>
                        .options{
                            color:aliceblue;
                        }
                        .options:hover{
                            color:black;
                        }
                        body{
                            background-color: #5ba9c6;
                        }
                    </style>
                </head>
                <body>
                    <nav class="navbar navbar-expand-lg navbar-light" style="background-color: rgb(22, 187, 121);">
                        <a class="navbar-brand" href="#"><p style="color: aliceblue;"><b>MeHC</b></p></a>
                        <button class="navbar-toggler" type="button" data-toggle="collapse" data-target="#navbarNav" aria-controls="navbarNav" aria-expanded="false" aria-label="Toggle navigation">
                            <span class="navbar-toggler-icon"></span>
                        </button>
                        <div class="collapse navbar-collapse" id="navbarNav">
                        <ul class="navbar-nav">
                            <li class="nav-item">
                                <a class="nav-link" href="{{ url_for('home') }}"><p class="options">Home</p></a>
                            </li>
                            <li class="nav-item active">
                                <a class="nav-link" href="#">Hospitals Near You</a>
                            </li>
                            <li class="nav-item">
                                <a class="nav-link" href="{{ url_for('mquery') }}"><p class="options">Medical Queries</p></a>
                            </li>
                            <li class="nav-item">
                                <a class="nav-link" href="{{ url_for('pconv') }}"><p class="options">Previous Queries</p></a>
                            </li>
                            <li class="nav-item">
                                <a class="nav-link" href="{{ url_for('Chatbot') }}"><p class="options">Mental Health Chatbot</p></a>
                            </li>
                            <li class="nav-item">
                                <a class="nav-link" href="{{ url_for('status') }}"><p class="options">Your Status</p></a>
                            </li>
                            <li class="nav-item">
                                <a class="nav-link" href="{{ url_for('logout')}}"><p class="options">Logout</p></a>
                            </li>
                        </ul>
                        </div>
                    </nav>
                    <div align="center" style="padding-left:5%;padding-right:5%;">
                        <h2 style="color = rgb(203, 241, 146);"><u>Hospitals nearby via Map</u></h2>
                        {{ body_map | safe}}
                        <br>
                        <a class="btn btn-primary" href="{{ url_for('hmap') }}" role="button" style="background-color: rgb(203, 241, 146);">Go Back</a>
                        <br>
                        <br>
                    </div>
                    <script>
                        {{ script_map | safe }}
                    </script>
                    <script src="https://code.jquery.com/jquery-3.2.1.slim.min.js" integrity="sha384-KJ3o2DKtIkvYIK3UENzmM7KCkRr/rE9/Qpg6aAZGJwFDMVNA/GpGFF93hXpG5KkN" crossorigin="anonymous"></script>
                    <script src="https://cdn.jsdelivr.net/npm/popper.js@1.12.9/dist/umd/popper.min.js" integrity="sha384-ApNbgh9B+Y1QKtv3Rn7W3mgPxhU9K/ScQsAP7hUibX39j7fakFPskvXusvfa0b4Q" crossorigin="anonymous"></script>
                    <script src="https://cdn.jsdelivr.net/npm/bootstrap@4.0.0/dist/js/bootstrap.min.js" integrity="sha384-JZR6Spejh4U02d8jOt6vLEHfe/JQGiRRSQQxSfFWpi1MquVdAyjUar5+76PVCmYl" crossorigin="anonymous"></script>
                </body>
            </html>                   
            """,header_map=header_map,body_map=body_map,script_map=script_map)
        return render_template('hmap.html')
    else:
        return redirect(url_for('login'))

#LOGOUT-FUNCTIONALITY
@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('login'))

#STATIC-FILE-RETURNER
@app.route('/static/<filename>')
def static_file(filename):
    return send_file(f'static/{filename}')

'''
Running the MEHC-Application
'''
if __name__=="__main__":
    app.run(debug=True)