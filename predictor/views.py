from django.shortcuts import render
import numpy as np
import joblib
from .report_ai import analyze_report
import os
from .explain import explain_prediction
from .llm import ask_llm
from .agent import run_agent
from django.conf import settings


# Load model
model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'ml', "heart_disease_model.pkl")

try:
    model = joblib.load(model_path)
except:
    model = None

def index(request):

    prediction = None

    if request.method == 'POST':

        age = float(request.POST['age'])
        sex = float(request.POST['sex'])
        cp = float(request.POST['cp'])
        trestbps = float(request.POST['trestbps'])
        chol = float(request.POST['chol'])
        fbs = float(request.POST['fbs'])
        restecg = float(request.POST['restecg'])
        thalach = float(request.POST['thalach'])
        exang = float(request.POST['exang'])
        oldpeak = float(request.POST['oldpeak'])
        slope = float(request.POST['slope'])
        ca = float(request.POST['ca'])
        thal = float(request.POST['thal'])

        data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                          thalach, exang, oldpeak, slope, ca, thal]])

        result = model.predict(data)

        if result[0] == 1:
            prediction = "Heart Disease Risk Detected"
        else:
            prediction = "No Heart Disease Risk"

    return render(request, 'predict.html', {'prediction': prediction})

def home(request):
    return render(request, 'home.html')

def about(request):
    return render(request, "about.html")

def summarize(request):

    summary = None

    if request.method == "POST":

        report = request.POST["report"]

        # temporary summarization logic
        summary = report[:200] + "..."

    return render(request, "summarize.html", {"summary": summary})

def chat(request):

    answer = None
    question = None

    if request.method == 'POST':

        question = request.POST.get('question')

        if question:
            answer = run_agent(question)

    return render(request, 'chat.html',
    {
        'answer': answer,
        'question': question
    })


def predict(request):

    if request.method == "POST":

        age = int(request.POST["age",0])
        sex = int(request.POST["sex"])
        cp = int(request.POST["cp"])
        trestbps = int(request.POST["trestbps"])
        chol = int(request.POST["chol"])
        fbs = int(request.POST["fbs"])
        restecg = int(request.POST["restecg"])
        thalach = int(request.POST["thalach"])
        exang = int(request.POST["exang"])
        oldpeak = float(request.POST["oldpeak"])
        slope = int(request.POST["slope"])
        ca = int(request.POST["ca"])
        thal = int(request.POST["thal"])

        input_data = [[
            age,sex,cp,trestbps,chol,fbs,restecg,
            thalach,exang,oldpeak,slope,ca,thal
        ]]

        prediction = model.predict(input_data)

        explanation = explain_prediction(input_data)

        return render(request,"result.html",
        {
            "prediction":prediction[0],
            "explanation":explanation
        })
    

def report_upload(request):

    result = None

    if request.method == "POST":

        pdf = request.FILES["report"]

        file_path = os.path.join(settings.MEDIA_ROOT, pdf.name)

        with open(file_path, "wb+") as f:
            for chunk in pdf.chunks():
                f.write(chunk)

        result = analyze_report(file_path)

    return render(request, "report.html", {"result": result})