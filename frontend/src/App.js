// frontend/src/App.js
import React, { useState } from 'react';
import axios from 'axios';
import './App.css'; // You can create this file for basic styling

function App() {
  const [formData, setFormData] = useState({
    ax_g: 0.45,
    ay_g: 0.6,
    az_g: 0.9,
    vibration: 300,
    bending: 100,
  });
  const [predictionResult, setPredictionResult] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData((prevData) => ({
      ...prevData,
      [name]: parseFloat(value), // Ensure numbers are parsed correctly
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError('');
    setPredictionResult('');

    try {
      // The API endpoint for prediction from your Flask app
      // When deployed, this should be a relative path if Flask serves React,
      // or the full backend URL if deployed separately (e.g., Render functions)
      const response = await axios.post('/predict', formData);
      setPredictionResult(`Prediction: ${response.data.status}`);
    } catch (err) {
      console.error('Prediction error:', err);
      setError('Failed to get prediction. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>Structural Health Prediction</h1>
      </header>
      <div className="container">
        <form onSubmit={handleSubmit}>
          <div className="form-group">
            <label htmlFor="ax_g">Ax_g:</label>
            <input type="number" id="ax_g" name="ax_g" step="0.01" value={formData.ax_g} onChange={handleChange} required />
          </div>
          <div className="form-group">
            <label htmlFor="ay_g">Ay_g:</label>
            <input type="number" id="ay_g" name="ay_g" step="0.01" value={formData.ay_g} onChange={handleChange} required />
          </div>
          <div className="form-group">
            <label htmlFor="az_g">Az_g:</label>
            <input type="number" id="az_g" name="az_g" step="0.01" value={formData.az_g} onChange={handleChange} required />
          </div>
          <div className="form-group">
            <label htmlFor="vibration">Vibration:</label>
            <input type="number" id="vibration" name="vibration" value={formData.vibration} onChange={handleChange} required />
          </div>
          <div className="form-group">
            <label htmlFor="bending">Bending:</label>
            <input type="number" id="bending" name="bending" value={formData.bending} onChange={handleChange} required />
          </div>
          <button type="submit" disabled={loading}>
            {loading ? 'Predicting...' : 'Predict'}
          </button>
        </form>

        {predictionResult && <div className="result">{predictionResult}</div>}
        {error && <div className="error">{error}</div>}
      </div>
    </div>
  );
}

export default App;