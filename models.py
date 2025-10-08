# models.py - PyMongo helpers + ML training/loading
from pymongo import MongoClient
from flask_bcrypt import Bcrypt
from config import MONGO_URI, DB_NAME, MODEL_PATH
from bson.objectid import ObjectId
import os, joblib, random
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

client = MongoClient(MONGO_URI)
db = client[DB_NAME]
bcrypt = Bcrypt()

def seed_sample_data():
    # create users: 3 HR and 20 employees
    if db.users.count_documents({}) == 0:
        users = []
        # 3 HR users
        users += [
            {"name":"HR One","email":"hr1@company.com","password":bcrypt.generate_password_hash("hrpass1").decode('utf-8'), "role":"hr"},
            {"name":"HR Two","email":"hr2@company.com","password":bcrypt.generate_password_hash("hrpass2").decode('utf-8'), "role":"hr"},
            {"name":"HR Three","email":"hr3@company.com","password":bcrypt.generate_password_hash("hrpass3").decode('utf-8'), "role":"hr"},
        ]
        # 20 employees
        for i in range(1,21):
            eid = f"E{100+i:03d}"
            users.append({
                "name": f"Employee {i}",
                "email": f"emp{i}@company.com",
                "password": bcrypt.generate_password_hash(f"emppass{i}").decode('utf-8'),
                "role": "employee",
                "employee_id": eid
            })
        db.users.insert_many(users)
    if db.employees.count_documents({}) == 0:
        employees = []
        for i in range(1,21):
            eid = f"E{100+i:03d}"
            dept = random.choice(["Sales","HR","Dev","Support","Finance","Marketing"])
            salary = random.choice([30000,35000,40000,45000,50000,55000,60000])
            tenure = random.randint(0,10)
            perf = random.choice(["Excellent","Good","Average","Below Average"])
            absence = random.randint(0,8)
            pending = random.choice([False,False,False,True])
            employees.append({
                "employee_id": eid,
                "name": f"Employee {i}",
                "department": dept,
                "salary": salary,
                "tenure_years": tenure,
                "performance": perf,
                "absence_count": absence,
                "salary_pending": pending
            })
        db.employees.insert_many(employees)
    if db.payroll.count_documents({}) == 0:
        payrolls = []
        for e in db.employees.find():
            eid = e.get("employee_id")
            amount = e.get("salary",30000)
            status = random.choice(["processed","pending"])
            payrolls.append({"employee_id":eid,"month":"2025-09","amount":amount,"status":status})
        if payrolls:
            db.payroll.insert_many(payrolls)
    if db.attendance.count_documents({}) == 0:
        attendance = []
        for e in db.employees.find():
            eid = e.get("employee_id")
            for d in range(1,11):
                status = random.choice(["present"]*8 + ["absent"]*2)
                attendance.append({"employee_id":eid, "date": f"2025-09-{d:02d}", "status": status})
        if attendance:
            db.attendance.insert_many(attendance)

def find_user_by_email(email):
    return db.users.find_one({"email": email})

def insert_user(user_doc):
    return db.users.insert_one(user_doc)

def get_employee_by_id(eid):
    return db.employees.find_one({"employee_id": eid})

def list_employees():
    return list(db.employees.find())

def list_payroll():
    return list(db.payroll.find())

def list_attendance():
    return list(db.attendance.find())

# ML: train a simple model to predict payroll risk (synthetic)
def train_and_save_model(force=False):
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    if os.path.exists(MODEL_PATH) and not force:
        return joblib.load(MODEL_PATH)
    emps = list_employees()
    rows = []
    for e in emps:
        perf_score = {"Excellent":3,"Good":2,"Average":1,"Below Average":0}.get(e.get("performance","Average"),1)
        absence = int(e.get("absence_count",0))
        salary_pending = int(e.get("salary_pending", False))
        tenure = int(e.get("tenure_years",0))
        salary = int(e.get("salary",30000))
        label = 1 if (salary_pending or absence>=5 or perf_score<=0) else 0
        rows.append([tenure, salary, perf_score, absence, label])
    df = pd.DataFrame(rows, columns=["tenure","salary","perf_score","absence","label"])
    if len(df) < 5:
        df = pd.DataFrame({
            "tenure":[1,2,3,4,5,6,2,3],
            "salary":[30000,40000,50000,45000,35000,60000,32000,41000],
            "perf_score":[2,3,1,2,3,1,0,2],
            "absence":[0,1,6,2,0,7,4,1],
            "label":[0,0,1,0,0,1,1,0]
        })
    X = df[["tenure","salary","perf_score","absence"]]
    y = df["label"]
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X, y)
    joblib.dump(model, MODEL_PATH)
    return model

def load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return train_and_save_model()
