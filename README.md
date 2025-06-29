# 🐶🐱 Cat vs Dog Classifier API (FastAPI + TensorFlow + Docker + AWS)

This project is an end-to-end machine learning pipeline for image classification — specifically, identifying whether an image contains a **cat or a dog**.  
It uses **Transfer Learning** with TensorFlow, exposes a prediction API with **FastAPI**, tracks training with **MLflow**, containerizes the app using **Docker**, and deploys it on **AWS EC2**.

---

## 🧰 Tech Stack

| Component          | Tool/Library         |
|--------------------|----------------------|
| Model Training     | TensorFlow, MobileNetV2 |
| Tracking           | MLflow               |
| Web API            | FastAPI, Uvicorn     |
| Image Processing   | Pillow, NumPy        |
| Containerization   | Docker               |
| Deployment         | AWS EC2              |

---

## 📁 Project Structure
    cat_dog_classifier/
        ├── app/                     # FastAPI code
        │   ├── main.py              # API logic
        │   └── utils.py             # Image preprocessing
        ├── data/                    # Dataset folder (local only, not in Docker)
        │   ├── train/               # Training images
        │   │   ├── cat/
        │   │   └── dog/
        │   └── test/                # Testing/validation images
        │       ├── cat/
        │       └── dog/
        ├── models/                  # Trained model file
        │   └── cat_dog_model.h5
        ├── training/                # Training pipeline
        │   └── train.py             # Training + MLflow logging
        ├── requirements.txt         # Dependencies
        ├── Dockerfile               # For containerization
        ├── .dockerignore            # Ignore data & logs
        ├── mlruns/ (local)          # MLflow experiment tracking logs
    README.md


---

# ⚙️ Installations


## 1️⃣ Clone the repository
    git clone https://github.com/DataWithSandeep/mlops-catdog-classification.git
    cd cat-dog-classifier

## 2️⃣ Create and activate virtual environment
    python -m venv venv
    source venv/bin/activate       # For Linux/Mac
    # venv\Scripts\activate        # For Windows

## 3️⃣ Install project dependencies
    pip install -r requirements.txt

## 4️⃣ (Optional) Launch MLflow UI for experiment tracking
    mlflow ui                      # Open http://localhost:5000 in your browser

## 5️⃣ Train the model using MobileNetV2
    cd training
    python train.py                # Model saved to ../models/cat_dog_model.h5
    cd ..

## 6️⃣ Run the FastAPI server locally
    uvicorn app.main:app --reload  # Open http://localhost:8000/docs

## 7️⃣ (Optional) Build and run Docker container
    docker build -t cat-dog-api .
    docker run -p 8000:8000 cat-dog-api

## 8️⃣ (Optional) Push to Docker Hub (requires login)
    docker login
    docker tag cat-dog-api <your-dockerhub-username>/cat-dog-api
    docker push <your-dockerhub-username>/cat-dog-api

## 9️⃣ (Optional) On AWS EC2 (after Docker install)
    docker pull <your-dockerhub-username>/cat-dog-api
    docker run -d -p 80:8000 <your-dockerhub-username>/cat-dog-api

---

# 🚀 **AWS EC2 Deployment: Full Setup Script**

## 1️⃣ Connect to your EC2 instance
    ssh -i "your-key.pem" ubuntu@<your-ec2-public-ip>

## 2️⃣ Update packages and install Docker
    sudo apt update
    sudo apt install docker.io -y
    sudo systemctl start docker
    sudo systemctl enable docker
    sudo usermod -aG docker ubuntu
    exit  # 👈 logout once to apply Docker group permissions

## 3️⃣ Reconnect to EC2
    ssh -i "your-key.pem" ubuntu@<your-ec2-public-ip>

## 4️⃣ Pull Docker image from Docker Hub (replace with your image)
    docker pull <your-dockerhub-username>/cat-dog-api

## 5️⃣ Run the container and expose it on port 80
    docker run -d -p 80:8000 <your-dockerhub-username>/cat-dog-api

## 6️⃣ (Optional) Check if container is running
    docker ps

## 7️⃣ (Important) Open EC2 Security Group and allow HTTP (port 80)

## 8️⃣ Test API in your browser or Postman
### 👉 http://<your-ec2-public-ip>/docs


