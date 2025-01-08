import spacy as sp
import random
from spacy.tokens import DocBin
from pymongo import MongoClient
import numpy as np
import datetime
import pandas
import re
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates
from matplotlib.ticker import FormatStrFormatter
from fpdf import FPDF # type: ignore
from flask import Flask, request, render_template, send_file
from io import BytesIO
from PIL import Image
import tempfile
import json
import base64

uri = "mongodb+srv://wizbot21:0NtTiVkNptbch4KN@intern.hd30j.mongodb.net/?retryWrites=true&w=majority&appName=intern"

client = MongoClient(uri) #connect to my mongo atlas cloud database

mydb = client["Asta"]
mycol = mydb["sensor"]

nlp = sp.load("en_core_web_sm") 

nlp = sp.load("../output/model-best")

app = Flask(__name__)
plt.ioff()
matplotlib.use("Agg")

indgph = 0

def graph(d, tim, val, att, inde, ad, gl):
    keys = []
    for each in d.keys():
        keys.append(each)
    #plt.figure(figsize=(14,7))
    #keys = [time, on, off] / [time, max, min, avg]
    for i in range(1,len(keys)):
        plt.plot(d["Time"],d[keys[i]],label=keys[i],marker="o")
    plt.title(val + " based on " + tim + " " + f"({att})")
    plt.xlabel(tim)
    if tim == "monthly":
        plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))
        plt.gca().xaxis.set_major_formatter(FormatStrFormatter('% 1.0f'))
    elif tim == "hourly":
        plt.xticks([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23])
    elif tim == "daily":
        plt.xticks(rotation=39)
    elif tim == "weekly":
        plt.xticks([0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46,48,50,52,54])
    plt.ylabel(val)
    plt.legend(loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    #plt.show()
    img_stream = BytesIO()
    plt.savefig(img_stream, format="png")
    plt.close()
    ad.append(d)
    gl.append(img_stream)

    img_stream.seek(0)

    global indgph
    indgph += 1

def mmaDF(df):
    df['Max'] = df['Max'].apply(lambda x: float("{:.2f}".format(x)))
    df['Min'] = df['Min'].apply(lambda x: float("{:.2f}".format(x)))
    df['Avg'] = df['Avg'].apply(lambda x: float("{:.2f}".format(x)))
    df = df.set_index("Time").transpose()

    print("---------------------Report---------------------")
    print(df)
    print("----------------------End-----------------------")

def ofDF(df):
    df['1s'] = df['1s'].apply(lambda x: float("{:.2f}".format(x)))
    df['0s'] = df['0s'].apply(lambda x: float("{:.2f}".format(x)))
    df = df.set_index("Time").transpose()

    print("---------------------Report---------------------")
    print(df)
    print("----------------------End-----------------------")

def printDF(df):
    df['Probability of 0'] = df['Probability of 0'].apply(lambda x: float("{:.2f}".format(x)))
    df['EF of 0'] = df['EF of 0'].apply(lambda x: float("{:.2f}".format(x)))
    df['True Value of 0'] = df['True Value of 0'].apply(lambda x: float("{:.2f}".format(x)))
    df['Probability of 1'] = df['Probability of 1'].apply(lambda x: float("{:.2f}".format(x)))
    df['EF of 1'] = df['EF of 1'].apply(lambda x: float("{:.2f}".format(x)))
    df['True Value of 1'] = df['True Value of 1'].apply(lambda x: float("{:.2f}".format(x)))
    df = df.set_index("Time").transpose()

    print("---------------------Report---------------------")
    print(df)
    print("----------------------End-----------------------")

def plotGraph(pipe, col, tim, val, att, inde, ad, gl):
    result = col.aggregate(pipe)

    on_values = []
    off_values = []

    minh = []
    maxh = []
    avgh = []

    time = []
    timeh = []

    if val == "binary":
        for x in result:
            on_values.append(x["On"])
            off_values.append(x["Off"])
            time.append(x["_id"])

        data = {
            "Time": time,
            "1s": on_values,
            "0s": off_values
        }
        df = pandas.DataFrame(data)
        ofDF(df)
        graph(df, tim, "Count of 0s and 1s", att, inde, ad, gl)
        return df

    elif val == "decimal":
        for x in result:
            minh.append(x["min"])
            maxh.append(x["max"])
            avgh.append(x["avg"])
            timeh.append(x["_id"])

        data = {
            "Time": timeh,
            "Max": maxh,
            "Min": minh,
            "Avg": avgh
        }
        df = pandas.DataFrame(data)
        mmaDF(df)
        graph(df, tim, "Max, Min, Avg", att, inde, ad, gl)
        return df
    else:
        print("Error. Unable to create DF object!")

def plotE(df, timing, ind, ad, gl):
    #plt.figure(figsize=(14,7))
    if timing == "24h":
        day = df["Time"].astype(int)
    else:
        day = df["Time"]
    ef0 = df["EF of 0"]
    ef1 = df["EF of 1"]
    ac0 = df["True Value of 0"]
    ac1 = df["True Value of 1"]
    plt.scatter(day, ef0, color="blue", marker="X", label="Expected frequency for 0", s=65)
    plt.scatter(day, ac0, color="blue", marker="o", label="Actual frequency for 0")
    plt.scatter(day, ef1, color="red", marker="X", label="Expected frequency for 1", s=65)
    plt.scatter(day, ac1, color="red", marker="o", label="Actual frequency for 1")
    plt.legend(loc='upper left')
    plt.title("Expected Frequency" + " based on " + timing + " for 0s and 1s")
    if timing == "24h":
        plt.xticks([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23])
    elif timing == "7d":
        plt.xticks(rotation=30)

    if type(day) == datetime.datetime:
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M:%S'))
    
    plt.xlabel(timing)
    plt.ylabel("Count")
    
    plt.grid(True)
    plt.tight_layout()
    img_stream = BytesIO()
    plt.savefig(img_stream, format="png")
    plt.close()
    ad.append(df)
    gl.append(img_stream)

    img_stream.seek(0)

    global indgph
    indgph += 1
    return df

def calculation(alist, blist, day):
    trail = len(alist)
    total = len(blist)
    if total != 0 and trail !=0:
        num0 = blist.count(0)
        num1 = blist.count(1)
        prob0 = num0 / total
        prob1 = num1 / total

        ef0 = prob0 * trail
        ef1 = prob1 * trail

        ac0 = alist.count(0)
        ac1 = alist.count(1)

        tryData = {
                "Time": str(day),
                "Probability of 0": prob0,
                "Probability of 1": prob1,
                "EF of 0": ef0,
                "EF of 1": ef1,
                "True Value of 0": ac0,
                "True Value of 1": ac1
            }

        return tryData

    else:
        print("Math error! No data to calculate the probability")

def ef24hour(dae,convd, id, ad, gl):
    try:
        rdate = datetime.datetime.strptime(dae, '%Y-%m-%d').date()
        prevdate = rdate-datetime.timedelta(7)
        prevdate = prevdate.strftime("%Y-%m-%d")
    except:
        print("Invalid input!")

    hours = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]
    nlist = []
    for h in hours:
        timing = datetime.datetime.strptime(dae, "%Y-%m-%d").replace(hour=h)
        endtiming = datetime.datetime.strptime(dae, "%Y-%m-%d").replace(hour=h,minute=59,second=59)
        pti = datetime.datetime.strptime(prevdate, "%Y-%m-%d").replace(hour=h)
        peti = datetime.datetime.strptime(prevdate, "%Y-%m-%d").replace(hour=h, minute=59, second=59)
        trailData = []
        probData = []
        for e in convd:
            x = e["time"]
            y = datetime.datetime.strptime(x, '%Y-%m-%dT%H:%M:%S')
            if timing <= y <= endtiming:
                trailData.append(int(list(e["params"].values())[0]))
            elif pti <= y <= peti:
                probData.append(int(list(e["params"].values())[0]))
        nst = calculation(trailData, probData, h)
        if nst != None:
            nlist.append(nst)
    if len(nlist) != 0:
        df = pandas.DataFrame(nlist)
        printDF(df)
        return plotE(df, "24h", id, ad, gl)

def ef7day(da,cod, id, ad, gl):
    wkData = []
    oldwkData = []
    timeData = []
    oldtimeData = []
    storeList = []

    try:
        sweek = datetime.datetime.strptime(da, '%Y-%m-%d').date()
        weekno = sweek.strftime("%V")
    except:
        print("Invalid input!")
        return False

    for each in cod:
        x = each["time"]
        y = datetime.datetime.strptime(x, '%Y-%m-%dT%H:%M:%S').date()
        daw = y.strftime("%V")
        if daw == weekno:
            wkData.append(each)
        elif int(daw) == int(weekno)-1:
            oldwkData.append(each)

    for d in wkData:
        ctime = datetime.datetime.strptime(d["time"], '%Y-%m-%dT%H:%M:%S').date()
        if ctime not in timeData:
            timeData.append(ctime)
    for d in oldwkData:
        ctime = datetime.datetime.strptime(d["time"], '%Y-%m-%dT%H:%M:%S').date()
        if ctime not in oldtimeData:
            oldtimeData.append(ctime)

    if len(timeData) == 7 and len(oldtimeData) == 7:
        for f in range(7):
            edData = []
            pbData = []
            for g in wkData:
                convg = datetime.datetime.strptime(g["time"], '%Y-%m-%dT%H:%M:%S').date()
                if convg == timeData[f]: 
                    edData.append(int(list(g["params"].values())[0]))
            for h in oldwkData:
                convg = datetime.datetime.strptime(h["time"], '%Y-%m-%dT%H:%M:%S').date()
                if convg == oldtimeData[f]:
                    pbData.append(int(list(h["params"].values())[0]))
            edict = calculation(edData, pbData, timeData[f])
            storeList.append(edict)
        df = pandas.DataFrame(storeList)
        printDF(df)
        return plotE(df, "7d", id, ad, gl)
    else:
        print("Not enough data. Both weeks must have 7 days of data")

"""def exportPDF(ad, gl):
    pdf = FPDF("P","mm","A4")
    pdf.add_page()

    pdf.set_font("Arial", "B", 16)
    pdf.cell(200,10, txt="Analysis Report", ln=True, align="C")

    pdf.ln(10)

    for i in range(len(ad)):
        #print(i)
        pdf.set_font("Arial", "B", 12)
        pdf.cell(200,10, txt="Table Report", ln=True, align="C")

        pdf.ln(5)
        df = ad[i]

        pdf.set_font("Arial", '', 9)
        x = pandas.DataFrame.to_dict(df, 'list')
        for col in df.columns:
            pdf.cell(25, 10, col, border=1, align="C")
        pdf.ln()
        y = json.dumps(x)
        print(y)

        for n in range(len(df)):
            for col in df.columns:
                pdf.cell(25,10, str(df.iloc[n][col]), border=1, align="C")
            pdf.ln()
        
        pdf.ln(5)

        
        img = Image.open(gl[i])

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
            img.save(tmp_file.name)
            pdf.image(tmp_file.name, w=180, h=150)
            img.close()
        
        if i != len(ad)-1:
            pdf.add_page()
            pdf.ln(10)

    output = BytesIO()
    pdf_content = pdf.output(dest='S').encode('latin1')
    output.write(pdf_content)
    output.seek(0)

    return output
"""

def jsc(ad):
    nl = []
    for i in range(len(ad)):
        df = ad[i]
        x = pandas.DataFrame.to_dict(df, 'list')
        print(x)
        y = json.dumps(x)
        print(y)
        nl.append(y)
    return nl

def dataFHTML(ad):
    ol = []    
    for n in range(len(ad)):
        d = []
        for col in ad.columns:
            d.append(str(ad.iloc[n][col]))
        ol.append(d)
    return ol

def colFHTML(ad):
    c = []
    for col in ad.columns:
        c.append(col)
    return c

def bytesHTML(gl):
    b64ls = []
    for i in gl:
        b64ls.append(base64.b64encode(i.getvalue()).decode("utf-8"))
    return b64ls
        

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/', methods=['POST'])
def formpost():
    attributes = []
    timing = []
    date = ""
    convData = []
    allDF = []
    efDF = []
    graphls = []
    htlist = []

    global indgph
    indgph = 0

    text = request.form.get("textarea")
    print(text)
    doc = nlp(text)
    for ent in doc.ents: #Printing of labels and values retrieved from the text
        print(ent.text, ent.label_)
        if ent.label_ == "attr":
            attributes.append(ent.text)
        elif ent.label_ == "time":
            timing.append(ent.text)
        elif ent.label_ == "date":
            date = ent.text
        else:
            print("Error. Unknown label")
    
    for elem in attributes: #perform the following code on all the attributes the user input
        ef = 0
        if elem == "door" or elem == "motion":
            ef = 1
        
        for t in timing:
            newValue = {} #attribute, #header if exists, data if exists, ef headers if exists, data if exists
            newValue["attr"] = elem
            if ef == 1: #binary values
                #Check if date is empty *** for hourly and weekly (EF)
                if t == "hourly" and date == "":
                    print("Provide an date")
                    continue
                bpipe = [
                    {"$match": {f"params.{elem}": {"$exists": "True"}}}, 
                    {"$addFields": {"isoDate": {"$dateFromString": {"dateString": "$time"}}}},


                    {"$addFields": {"date" : {"$dateToParts": {"date": "$isoDate", "timezone": "Asia/Singapore"}}}},
                    {"$addFields": {"newDate": {"$dateFromParts": {"year": "$date.year", "month": "$date.month", "day": "$date.day", 
                                                                "hour": "$date.hour", "minute": "$date.minute", "second": "$date.second"}}}},
                    {"$addFields": {"time": {"$dateToString": {"format": "%Y-%m-%dT%H:%M:%S", "date": "$newDate"}}, 
                                    "tdate": {"$dateToString": {"format": "%Y-%m-%d", "date": "$newDate"}}}},

                    {"$project": {"time":1, "on": {"$cond": [{"$eq":[f"$params.{elem}","1"]},1,0]}, "off": {"$cond":[{"$eq":[f"$params.{elem}","0"]},1,0]}, 
                                    "daily": "$tdate", "hourly": "$date.hour", "weekly": {"$isoWeek": "$newDate"}, "monthly": "$date.month", "yearly": "$date.yearly"}},
                    
                    *([{"$match": {"daily": f"{date}"}}] if t == "hourly" else []),
                    
                    {"$group": {"_id": f"${t}", "On": {"$sum": "$on"}, "Off": {"$sum": "$off"}}},
                    {"$sort": {"_id": 1}}
                ]
                da = plotGraph(bpipe, mycol, t, "binary", elem, indgph, allDF, graphls)
                h = colFHTML(da)
                d = dataFHTML(da)
                newValue["head"] = h
                newValue["data"] = d
                
                pipeline = [ #Retrieve the data to use in calculating the EF
                    {"$match": {f"params.{elem}": {"$exists": "True"}}},
                    {"$addFields": {"isoDate": {"$dateFromString": {"dateString": "$time"}}}},

                    {"$addFields": {"date" : {"$dateToParts": {"date": "$isoDate", "timezone": "Asia/Singapore"}}}},
                    {"$addFields": {"newDate": {"$dateFromParts": {"year": "$date.year", "month": "$date.month", "day": "$date.day", 
                                                                "hour": "$date.hour", "minute": "$date.minute", "second": "$date.second"}}}},
                    {"$addFields": {"time": {"$dateToString": {"format": "%Y-%m-%dT%H:%M:%S", "date": "$newDate"}}}},

                    {"$project": {"_id":0, "date":0, "newDate": 0, "isoDate": 0}}
                ]
                if len(convData) != 0:
                    convData.clear()                   

                res = mycol.aggregate(pipeline)
                for i in res:
                    if list(i["params"].values())[0] != "unavailable":
                        convData.append(i)
                    else:
                        print("removed")
                if t == "hourly":
                    efd = ef24hour(date, convData, indgph, allDF, graphls)
                    if not efd.empty: 
                        efh = colFHTML(efd)
                        efd = dataFHTML(efd)
                        newValue["efhe"] = efh
                        newValue["efda"] = efd
                elif t == "weekly" and date != "":
                    efd = ef7day(date, convData, indgph, allDF, graphls)
                    if not efd.empty: 
                        efh = colFHTML(efd)
                        efd = dataFHTML(efd)
                        newValue["efhe"] = efh
                        newValue["efda"] = efd
                    
            else: #dec values
            #Check if date is empty *** hourly only
                bpipe = [
                        {"$match": {f"params.{elem}": {"$exists": "True"}}}, 

                        {"$addFields": {"decValue": {"$toDouble": f"$params.{elem}"}, "isoDate": {"$dateFromString": {"dateString": "$time"}}}},

                        {"$addFields": {"date" : {"$dateToParts": {"date": "$isoDate", "timezone": "Asia/Singapore"}}}},
                        {"$addFields": {"newDate": {"$dateFromParts": {"year": "$date.year", "month": "$date.month", "day": "$date.day", 
                                                                    "hour": "$date.hour", "minute": "$date.minute", "second": "$date.second"}}}},
                        {"$addFields": {"time": {"$dateToString": {"format": "%Y-%m-%dT%H:%M:%S", "date": "$newDate"}}, 
                                        "tdate": {"$dateToString": {"format": "%Y-%m-%d", "date": "$newDate"}}}},

                        {"$project": {"time":1, "decValue":1, 
                                    "daily": "$tdate", "hourly": "$date.hour", "weekly": {"$isoWeek": "$newDate"}, 
                                    "monthly": "$date.month", "yearly": "$date.year"}},

                        *([{"$match": {"daily": f"{date}"}}] if t == "hourly" else []),

                        {"$group": {"_id": f"${t}", "min": {"$min": "$decValue"}, "max": {"$max": "$decValue"}, "avg": {"$avg": "$decValue"}}},
                        {"$sort": {"_id": 1}}
                    ]
                da = plotGraph(bpipe, mycol, t, "decimal", elem, indgph, allDF, graphls)
                h = colFHTML(da)
                d = dataFHTML(da)
                newValue["head"] = h
                newValue["data"] = d

            #print(newValue)
            htlist.append(newValue)

    print(len(graphls))
    print(indgph)
    #print(htlist)
    #pdfRes = exportPDF(allDF, graphls)
    jsoncal = jsc(allDF)
    #dataa = dataFHTML(allDF)
    #header = colFHTML(allDF)
    #efd = dataFHTML(efDF)
    #efhead = colFHTML(efDF)
    imgls = bytesHTML(graphls)
    #return send_file(pdfRes, mimetype="application/pdf" ,as_attachment=True, download_name="analysis_report.pdf")
    return render_template("index.html", dl = htlist, alljson=jsoncal, allimg=imgls)

if __name__ == "__main__":
    app.run()
