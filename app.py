from flask import Flask, render_template, request
from model import compare_answers

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        teacher_answer = request.form['teacher_answer']
        student_answer = request.form['student_answer']
        result, similarity_percentage = compare_answers(teacher_answer, student_answer)
        return render_template('result.html', result=result, similarity_percentage=similarity_percentage)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
