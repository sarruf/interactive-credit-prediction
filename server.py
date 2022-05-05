from flask import Flask, jsonify
import time
from sklearn.metrics import accuracy_score, confusion_matrix
from pymongo import MongoClient, DESCENDING
from datetime import datetime

from creditcase import train_model, predict

app = Flask(__name__)
tech = "svm"
meses = 0
atraso = 1
model, X_train, X_test, y_train, y_test, y_predict = train_model(tech, meses, atraso)

cliente = MongoClient("localhost", 27017)
banco = cliente["projeto"]
colecao = banco["tested_models"]
'''dados = {"nome": "Jan K. S.", "idade": 32}
colecao.insert(dados)
colecao.insert([dados1, dados2])
busca = {"chave1": valor1, "chave2": {"$gt": valor2}}
ordenacao = [ ["idade", DESCENDING] ]
documento = colecao.find_one(busca, sort=ordenacao)
documentos = list( colecao.find(busca, sort=ordenacao) )'''

@app.route('/teste')
def pagina_principal():
    return jsonify({'time': time.time()})

@app.route('/retrain/<int:tech>/<int:meses>/<int:atraso>')
def treinar_modelo(tech, meses, atraso):
    global model, X_train, X_test, y_train, y_test, y_predict
    model, X_train, X_test, y_train, y_test, y_predict = train_model(tech, meses, atraso)
    tn = ""
    if tech == 0:
        tn = "Regressão Logística"
    elif tech == 1:
        tn = "Árvore de Decisão"
    elif tech == 2:
        tn = "Random Forest"
    elif tech == 3:
        tn = "SVM"
    elif tech == 4:
        tn = "LGBM"
    elif tech == 5:
        tn = "XGBoost"
    else:
        tn = "Cat Boost"
    dados = {"data": datetime.now(), "tech": tn, "meses": meses, "atraso": atraso, 
             "accuracy": accuracy_score(y_test, y_predict)}
    colecao.insert_one(dados)
    return jsonify({'model': tech * meses * atraso})

@app.route('/update')
def atualiza_modelos_treinados():
    busca = {}
    ordenacao = [ ["accuracy", DESCENDING] ]
    documentos = list( colecao.find(busca, sort=ordenacao) )
    ret = []
    for documento in documentos:
        print(documento["accuracy"])
        ret.append({"data": documento["data"], "tech": documento["tech"], "meses": documento["meses"], 
                    "atraso": documento["atraso"], "accuracy": documento["accuracy"]})
    return jsonify(ret)
    
@app.route('/reset')
def reseta_modelos_treinados():
    colecao.drop()
    return ""

@app.route('/simular/<int:nf>/<int:rn>/<string:dn>/<string:da>/<int:ni>/<int:re>/<int:es>/<int:ec>/<int:tm>/<int:to>/<int:gn>/<string:car>/<string:rp>/<string:ce>/<string:tp>/<string:tf>/<string:em>')
def simula_credito(nf, rn, dn, da, ni, re, es, ec, tm, to, gn, car, rp, ce, tp, tf, em):
    prediction = predict(model, nf, rn, dn, da, ni, re, es, ec, tm, to, gn, car, rp, ce, tp, tf, em)
    if prediction == [0]:
        return jsonify({'simul': 'Previsão: receber crédito'})
    return jsonify({'simul': 'Previsão: não receber crédito'})

app.run(port=5000)