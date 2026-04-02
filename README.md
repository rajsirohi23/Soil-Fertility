# 🌱 Soil Fertility Prediction Web App

A Machine Learning-based web application that predicts soil fertility using environmental and soil parameters. This project uses a **Random Forest model** trained on soil data and is deployed using **Flask** with a simple and interactive web interface.

---


## 🚀 Live Demo

🌐 Frontend: https://soil-fertility-rkv4.onrender.com

---

## 🚀 Project Overview

Soil fertility plays a crucial role in agricultural productivity. This application helps users (farmers, researchers, students) determine soil fertility levels based on input parameters.

The system:

* Takes soil-related inputs from users
* Uses a trained ML model (`.pkl` file)
* Predicts soil fertility
* Displays results instantly on a web interface

---

## 🧠 Machine Learning Model

* Algorithm: **Random Forest Classifier**
* Library: **Scikit-learn**
* Model File: `soil_model.pkl`
* Training done on soil dataset with relevant features

---

## 🛠️ Tech Stack

| Technology     | Usage             |
| -------------- | ----------------- |
| Python         | Backend logic     |
| Flask          | Web framework     |
| HTML           | Structure         |
| CSS            | Styling           |
| Scikit-learn   | ML model          |
| Pandas / NumPy | Data processing   |
| Gunicorn       | Production server |

---

## 📁 Project Structure

```
SOILFER/
│
├── models/
│   └── soil_model.pkl
│
├── static/
│   └── css/
│       └── styles.css
│
├── templates/
│   └── index.html
│
├── app.py
├── requirements.txt
├── Procfile
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository

```
git clone https://github.com/your-username/soil-fertility-app.git
cd soil-fertility-app
```

---

### 2️⃣ Create Virtual Environment

```
python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows
```

---

### 3️⃣ Install Dependencies

```
pip install -r requirements.txt
```

---

### 4️⃣ Run the Application

```
python app.py
```

Open browser and go to:

```
http://127.0.0.1:5000/
```

---

## 🌐 Deployment

This app can be deployed on platforms like:

* Render (Recommended)
* Vercel (Advanced)
* Heroku (Alternative)

### Render Deployment Steps:

1. Push project to GitHub
2. Connect repository to Render
3. Set:

   * Build Command: `pip install -r requirements.txt`
   * Start Command: `gunicorn app:app`
4. Deploy 🚀

---

## 📊 Features

✔️ User-friendly interface
✔️ Fast predictions
✔️ Machine learning integration
✔️ Lightweight and deployable
✔️ Scalable for future improvements

---

## 🔮 Future Improvements

* Add more advanced ML models (XGBoost, Neural Networks)
* Improve UI/UX design
* Add data visualization dashboard
* Deploy mobile-friendly version
* Integrate real-time soil data APIs

---

## 🤝 Contributing

Contributions are welcome! Feel free to fork the repo and submit a pull request.

---

## 📜 License

This project is licensed under the MIT License.

---

## 👨‍💻 Author

**Raj Sirohi**

* GitHub: https://github.com/rajsirohi23
* LinkedIn: https://www.linkedin.com/in/raj-sirohi-64b330289/

---

## ⭐ Acknowledgements

* Scikit-learn documentation
* Flask documentation
* Open-source community

---

> 🌾 *Empowering agriculture with Machine Learning*
