
import { useEffect, useState } from "react";

const API_URL = "http://127.0.0.1:8000";

export default function ResultsDashboard({ jobId, demo = false }) {
  const [results, setResults] = useState(null);

  useEffect(() => {
    const resultsUrl = demo ? "/demo/results.json" : `${API_URL}/comparisons/${jobId}/results`;
    fetch(resultsUrl)
      .then((response) => response.json())
      .then((data) => setResults(data.algorithms))
      .catch(console.error);
  }, [jobId, demo]);

  if (!results) return <p>Loading results...</p>;

  return (
    <section>
      <h2>Comparison Results</h2>

      <table width="100%" cellPadding="12">
        <thead>
          <tr>
            <th>Algorithm</th>
            <th>Final Accuracy</th>
            <th>Final Loss</th>
            <th>Communication</th>
          </tr>
        </thead>

        <tbody>
          {Object.entries(results).map(([name, data]) => (
            <tr key={name}>
              <td>{name}</td>
              <td>{data.final_accuracy.toFixed(2)}%</td>
              <td>{data.final_loss.toFixed(4)}</td>
              <td>{data.total_communication_mb.toFixed(3)} MB</td>
            </tr>
          ))}
        </tbody>
      </table>

      <h3>Accuracy vs Communication Rounds</h3>
      <img
        src={demo ? "/demo/accuracy.png" : `${API_URL}/comparisons/${jobId}/plots/accuracy`}
        alt="Accuracy comparison"
        style={{ width: "100%" }}
        onError={(e) => {
          console.error("Accuracy image failed:", e.currentTarget.src);
        }}
      />

      <h3>Loss vs Communication Rounds</h3>
      <img
        src={demo ? "/demo/loss.png" : `${API_URL}/comparisons/${jobId}/plots/loss`}
        alt="Loss comparison"
        style={{ width: "100%" }}
        onError={(e) => {
          console.error("Loss image failed:", e.currentTarget.src);
        }}
      />
    </section>
  );
}
