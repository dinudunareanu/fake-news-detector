from flask import Flask, request, render_template
import predict

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        article = request.form['article']
        label, confidence = predict.predict_news(article)
        return render_template('index.html', 
                            result=label, 
                            confidence=confidence)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)