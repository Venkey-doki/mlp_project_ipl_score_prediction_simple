# IPL Score Prediction

A machine learning application that predicts the final score of Indian Premier League (IPL) cricket matches based on current match statistics.

![IPL Score Prediction](https://raw.githubusercontent.com/Venkey-doki/mlp_project_ipl_score_prediction_simple/main/frontend/src/assets/ipl_logo.png)

## Features

- Predicts final score based on current match statistics
- User-friendly web interface
- Real-time predictions with machine learning model
- Responsive design for various devices

## Technology Stack

### Frontend
- React.js
- CSS for styling
- Vite build tool

### Backend
- Flask (Python)
- TensorFlow/Keras for ML model
- NumPy for numerical operations
- Scikit-learn for data preprocessing

## Project Structure

```
.
├── backend/                # Flask server and ML model
│   ├── app.py              # Flask application
│   ├── model/              # Trained model files
│   │   ├── ipl_mlp_model.h5
│   │   ├── label_encoders.pkl
│   │   └── scaler.pkl
│   └── train_model.py      # Model training script
│
├── frontend/               # React application
│   ├── public/
│   ├── src/
│   │   ├── App.jsx         # Main React component
│   │   ├── components/     # React components
│   │   ├── assets/         # Images and static files
│   │   └── styles/         # CSS files
│   ├── index.html
│   ├── package.json
│   └── vite.config.js
│
├── ipl_data.csv            # Dataset for model training
└── README.md               # Project documentation
```

## Getting Started

### Prerequisites

- Python 3.8+ with pip
- Node.js 16+ with npm

### Installation

1. Clone this repository:
   ```
   git clone https://github.com/Venkey-doki/mlp_project_ipl_score_prediction_simple.git
   cd mlp_project_ipl_score_prediction_simple
   ```

2. Set up the backend:
   ```
   cd backend
   pip install flask flask-cors tensorflow numpy scikit-learn
   ```

3. Set up the frontend:
   ```
   cd ../frontend
   npm install
   ```

### Running the Application

1. Start the Flask backend server:
   ```
   cd backend
   python app.py
   ```

2. Start the React frontend development server:
   ```
   cd frontend
   npm run dev
   ```

3. Open your browser and navigate to `http://localhost:5173` to use the application.

## Usage

1. Select the batting team and bowling team
2. Enter the current match details:
   - Overs completed
   - Current runs
   - Wickets fallen
3. Click "Predict Score" to see the predicted final score

## Model Information

The prediction model is a Multilayer Perceptron (MLP) neural network built with TensorFlow/Keras. The model takes the following inputs:
- Batting team (encoded)
- Bowling team (encoded)
- Current overs completed
- Current runs scored
- Current wickets fallen

The model predicts the final total score of the match.

## Model Training

The model was trained on historical IPL data, using the following steps:
1. Data preprocessing and feature engineering
2. Label encoding for team names
3. Feature scaling using MinMaxScaler
4. Training a Sequential MLP model with 2 hidden layers
5. Evaluation on test data

To retrain the model:
```
cd backend
python train_model.py
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is open source and available under the MIT License.

## Acknowledgments

- IPL dataset contributors
- TensorFlow and Keras development teams
- Flask and React.js communities