import { useState } from "react";
import axios from "axios";
import "./App.css";

function App() {
  const [form, setForm] = useState({
    batting_team: "Deccan Chargers",
    bowling_team: "Chennai Super Kings",
    overs: 10.0,
    runs: 80,
    wickets: 2
  });

  const [prediction, setPrediction] = useState(null);

  const handleChange = (e) => {
    const { name, value } = e.target;
    setForm({
      ...form,
      [name]:
        name === "overs" || name === "runs" || name === "wickets"
          ? Number(value)
          : value,
    });
  };
  

  const handleSubmit = async () => {
    try {
      const res = await axios.post("http://localhost:5000/predict", form, {
        headers: {
          "Content-Type": "application/json"
        }
      });
      setPrediction(res.data.predicted_total);
    } catch (err) {
      alert("Error: " + err.message + " (Status " + err.response?.status + ")");
    }
  };
  

  return (
    <div className="container">
      <h1>🏏 IPL Score Predictor</h1>

      <label>Batting Team</label>
      <select name="batting_team" onChange={handleChange} value={form.batting_team}>
        <option>Deccan Chargers</option><option>Chennai Super Kings</option><option>RCB</option><option>KKR</option>
      </select>

      <label>Bowling Team</label>
      <select name="bowling_team" onChange={handleChange} value={form.bowling_team}>
        <option>Chennai Super Kings</option><option>Deccan Chargers</option><option>RCB</option><option>KKR</option>
      </select>

      <label>Overs</label>
      <input type="number" name="overs" step="0.1" value={form.overs} onChange={handleChange} />

      <label>Runs</label>
      <input type="number" name="runs" value={form.runs} onChange={handleChange} />

      <label>Wickets</label>
      <input type="number" name="wickets" value={form.wickets} onChange={handleChange} />

      <button onClick={handleSubmit}>Predict</button>

      {prediction && <h2>Predicted Total: {Math.round(prediction)}</h2>}
    </div>
  );
}

export default App;
