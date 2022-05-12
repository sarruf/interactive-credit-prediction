from flask import Flask, jsonify
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from pymongo import MongoClient, DESCENDING
from datetime import datetime

from creditcase import train_model, predict

app = Flask(__name__)
tech = "svm"
n_meses = 1
x = 1
model, X_train, X_test, y_train, y_test, y_predict = train_model(tech, n_meses, x)

cliente = MongoClient("localhost", 27017)
banco = cliente["projeto"]
colecao = banco["tested_models"]

@app.route('/retrain/<int:tech>/<int:n_meses>/<int:x>')
def treinar_modelo(tech, n_meses, x):
    global model, X_train, X_test, y_train, y_test, y_predict
    
    # índice comeca em 0, mas n_meses e x comecam em 1
    n_meses += 1
    x += 1
    
    model, X_train, X_test, y_train, y_test, y_predict = train_model(tech, n_meses, x)
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
    dados = {"data": datetime.now(), "tech": tn, "meses": n_meses, "atraso": x, 
             "accuracy": accuracy_score(y_test, y_predict),
             "precision": precision_score(y_test, y_predict),
             "recall": recall_score(y_test, y_predict),
             "f1": f1_score(y_test, y_predict)}
    colecao.insert_one(dados)
    return jsonify({'model': 'Modelo retreinado e métricas salvas na tabela'})

@app.route('/update')
def atualiza_modelos_treinados():
    busca = {}
    ordenacao = [ ["accuracy", DESCENDING] ]
    documentos = list( colecao.find(busca, sort=ordenacao) )
    ret = []
    for documento in documentos:
        print(documento["accuracy"])
        ret.append({"data": documento["data"], "tech": documento["tech"], "meses": documento["meses"], 
                    "atraso": documento["atraso"], "accuracy": documento["accuracy"], 
                    "precision": documento["precision"], "recall": documento["recall"], "f1": documento["f1"],})
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