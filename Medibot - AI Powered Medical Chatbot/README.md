# 🩺 Medibot – AI Powered Medical Chatbot

Medibot is an **AI-powered medical chatbot** developed as a **B.Tech Mini Project (Project-I)**.  
It provides **preliminary healthcare guidance** by analyzing user-reported symptoms using **Natural Language Processing (NLP)** and **Machine Learning**, assigning **severity levels (Low / Medium / High)**, and activating **safe fallback mechanisms** when confidence is low.

> ⚠️ **Disclaimer:** Medibot is not a replacement for professional medical diagnosis or treatment.  
> It is intended only for first-level guidance and awareness.

---

## 📌 Project Motivation

Access to timely and reliable healthcare remains a challenge due to long waiting times, limited availability of medical professionals, and geographical barriers.  
Medibot addresses this problem by offering **instant, confidence-aware medical guidance** while ensuring **user safety** and responsible AI usage.

---

## 🎯 Objectives

- Build an intelligent conversational interface for healthcare queries  
- Analyze free-text symptom descriptions using NLP  
- Classify symptoms into **severity levels** (low, medium, high)  
- Provide **confidence-aware fallback responses** when predictions are uncertain  
- Clearly flag **critical or emergency conditions**  
- Ensure scalability, reliability, and secure data handling  

---

## 🚀 Key Features

- 💬 Chat-based medical interaction (text & voice input)
- 🧠 NLP-based symptom extraction
- 📊 ML-driven severity scoring with confidence levels
- 🚨 Emergency alerts for high-risk symptoms
- 🛡️ Safe fallback mechanism for low-confidence predictions
- 📈 Symptom history and trend tracking
- ⚡ Real-time UI updates with smooth animations

---

## 🧰 Tech Stack

### Frontend
- **React (Vite)** – Interactive chat interface
- **Tailwind CSS** – Responsive UI design
- **Framer Motion** – Smooth animations and transitions
- **Browser Speech Recognition API** – Voice-based symptom input

### Backend
- **Node.js & Express.js** – API handling and orchestration
- **MongoDB (Mongoose)** – Chat history and severity trend storage
- **REST APIs** – Frontend–backend communication
- **Asynchronous ML invocation** – Non-blocking execution

### Machine Learning & NLP
- **Python**
- **TF-IDF Vectorization** – Text feature extraction
- **Logistic Regression** – Severity classification
- **Joblib** – Model persistence

### Data Sources
- Curated **CSV & JSON medical datasets** containing symptoms, causes, prevention steps, and emergency indicators

---

## 🏗️ System Architecture

Medibot follows a **three-layer client–server architecture**:

1. **Frontend (React)** – User interaction, severity visualization, alerts  
2. **Backend (Express)** – Request handling, rule-based logic, ML coordination  
3. **ML Engine (Python)** – Symptom analysis and severity prediction  

The backend maintains session context and applies fallback logic when the ML model returns low-confidence results.

---

## 🔄 Workflow

1. User enters symptoms via text or voice  
2. Backend classifies query intent (symptom / general information)  
3. ML model predicts severity level and confidence score  
4. Backend applies safety rules and fallback logic  
5. Structured response returned with:
   - Severity level
   - Confidence score
   - Possible causes
   - Health recommendations
   - Emergency guidance (if applicable)

---

## 🧪 Testing

- **Unit Testing:** ML model, backend APIs, frontend components  
- **Integration Testing:** Frontend ↔ Backend ↔ ML model  
- **Functional Testing:** Symptom analysis and emergency detection  
- **Performance Testing:** Real-time responsiveness under concurrent usage  
- **Security Testing:** Input validation and fallback reliability  

---

## 📊 Results

- Accurate symptom severity classification  
- Real-time confidence-aware healthcare guidance  
- Clear emergency alerts for critical symptoms  
- Smooth and responsive user interface  
- Reliable fallback behavior during uncertainty  
