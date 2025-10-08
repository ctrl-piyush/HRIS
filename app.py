# app.py - main Flask application (enhanced + fixed)
from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
from models import (
    bcrypt, seed_sample_data, find_user_by_email, insert_user,
    list_employees, list_payroll, list_attendance, get_employee_by_id,
    train_and_save_model, load_model, db
)
import os
from bson import ObjectId
import warnings
from sklearn.exceptions import DataConversionWarning

# Suppress sklearn warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DataConversionWarning)

app = Flask(__name__)
app.secret_key = os.urandom(24)

# Seed DB and train model once
seed_sample_data()
model = train_and_save_model()

# -------------------- Helper --------------------
def clean_mongo_docs(docs):
    """Convert MongoDB ObjectIds to strings for JSON/template safety."""
    cleaned = []
    for d in docs:
        d = dict(d)
        if "_id" in d:
            d["_id"] = str(d["_id"])
        cleaned.append(d)
    return cleaned

# -------------------- Routes --------------------

@app.route('/')
def index():
    return render_template('index.html')


# HR login
@app.route('/login/hr', methods=['GET', 'POST'])
def login_hr():
    if request.method == 'POST':
        email = request.form['email']
        pwd = request.form['password']
        user = find_user_by_email(email)
        if user and user.get('role') == 'hr' and bcrypt.check_password_hash(user['password'], pwd):
            session['user'] = {'email': user['email'], 'role': 'hr', 'name': user.get('name')}
            return redirect(url_for('hr_dashboard'))
        flash('Invalid HR credentials', 'danger')
    return render_template('login_hr.html')


# Employee login
@app.route('/login/employee', methods=['GET', 'POST'])
def login_employee():
    if request.method == 'POST':
        email = request.form['email']
        pwd = request.form['password']
        user = find_user_by_email(email)
        if user and user.get('role') == 'employee' and bcrypt.check_password_hash(user['password'], pwd):
            session['user'] = {
                'email': user['email'],
                'role': 'employee',
                'name': user.get('name'),
                'employee_id': user.get('employee_id')
            }
            return redirect(url_for('employee_dashboard'))
        flash('Invalid employee credentials', 'danger')
    return render_template('login_employee.html')


# HR Dashboard
@app.route('/hr/dashboard')
def hr_dashboard():
    if 'user' not in session or session['user'].get('role') != 'hr':
        return redirect(url_for('login_hr'))

    employees = clean_mongo_docs(list_employees())
    payroll = clean_mongo_docs(list_payroll())
    attendance = clean_mongo_docs(list_attendance())

    total_employees = len(employees)
    pending_payroll = sum(1 for p in payroll if p.get('status') == 'pending')
    avg_salary = sum(e.get('salary', 0) for e in employees) / max(1, total_employees)

    preds = []
    for e in employees:
        X = [[
            e.get('tenure_years', 0),
            e.get('salary', 30000),
            {"Excellent": 3, "Good": 2, "Average": 1, "Below Average": 0}.get(e.get('performance', 'Average'), 1),
            e.get('absence_count', 0)
        ]]
        p = int(model.predict(X)[0])
        preds.append({
            "employee_id": e.get('employee_id'),
            "name": e.get('name'),
            "risk": bool(p)
        })
    risk_count = sum(1 for r in preds if r['risk'])

    return render_template(
        'hr_dashboard.html',
        employees=employees,
        payroll=payroll,
        attendance=attendance,
        total_employees=total_employees,
        pending_payroll=pending_payroll,
        avg_salary=int(avg_salary),
        preds=preds,
        risk_count=risk_count
    )


# Employee dashboard
@app.route('/employee/dashboard')
def employee_dashboard():
    if 'user' not in session or session['user'].get('role') != 'employee':
        return redirect(url_for('login_employee'))
    user_email = session['user']['email']
    user_doc = db.users.find_one({"email": user_email})
    emp = None
    if user_doc:
        emp = db.employees.find_one({"employee_id": user_doc.get('employee_id')})
    if emp and "_id" in emp:
        emp["_id"] = str(emp["_id"])
    return render_template('employee_dashboard.html', user=session['user'], employee=emp)


# Employee view (HR side)
@app.route('/employee/<eid>')
def employee_view(eid):
    if 'user' not in session or session['user'].get('role') != 'hr':
        return redirect(url_for('login_hr'))
    emp = get_employee_by_id(eid)
    payrolls = clean_mongo_docs(list(db.payroll.find({"employee_id": eid})))
    attendance = clean_mongo_docs(list(db.attendance.find({"employee_id": eid}).sort("date", 1)))
    return render_template('employee_view.html', emp=emp, payrolls=payrolls, attendance=attendance)


# Prediction API
@app.route('/api/predict', methods=['POST'])
def api_predict():
    data = request.get_json() or {}
    X = [[
        int(data.get('tenure', 0)),
        int(data.get('salary', 30000)),
        int(data.get('perf_score', 1)),
        int(data.get('absence', 0))
    ]]
    pred = int(model.predict(X)[0])
    prob = float(model.predict_proba(X)[0][1]) if hasattr(model, "predict_proba") else (0.0 if pred == 0 else 1.0)
    return jsonify({"risk": bool(pred), "probability": prob})


# Logout
@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))


# Register new user
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']
        role = request.form.get('role', 'employee')
        hashed = bcrypt.generate_password_hash(password).decode('utf-8')
        insert_user({'name': name, 'email': email, 'password': hashed, 'role': role})
        flash('User registered. You can login now.', 'success')
        return redirect(url_for('index'))
    return render_template('register.html')


if __name__ == '__main__':
    app.run(debug=True)
