HRIS Advanced - Flask + MongoDB + ML (Full Stack)
------------------------------------------------
This enhanced project includes:
- Colorful CSS with animated buttons
- Two login pages (HR and Employee)
- HR dashboard with analytics and ML-backed payroll/attrition prediction
- Employee details page (attendance, salary status)
- Seeded dataset with 20 employees + 3 HR users
- Simple ML model trained on synthetic data at first run (scikit-learn) and saved to disk

How to run:
1. Edit config.py to point to your MongoDB (default: mongodb://localhost:27017/hris_db)
2. Create virtualenv and install: pip install -r requirements.txt
3. Run: python app.py
4. Open: http://127.0.0.1:5000

Default HR accounts (seeded):
- hr1@company.com / hrpass1
- hr2@company.com / hrpass2
- hr3@company.com / hrpass3

Default Employee sample:
- emp1@company.com / emppass1  (and 19 more)

Notes:
- ML model (a RandomForest classifier) is trained on synthetic features (tenure, salary, performance score, absence_count)
  to predict a simple "payroll_at_risk" label. This is illustrative: replace/train with your own production data.
