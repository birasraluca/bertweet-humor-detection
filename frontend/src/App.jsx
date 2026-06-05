import { useState } from "react";
import "./App.css";

const API_URL = "http://127.0.0.1:5000/predict";

const exampleTexts = [
  "I used to play piano by ear, but now I use my hands.",
  "I told my computer I needed a break, and now it won't stop sending me vacation ads.",
  "The database backup completed successfully.",
  "The meeting will start tomorrow at 10 AM."
];

function App() {
  const [text, setText] = useState("");
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  const predictHumor = async () => {
    const trimmedText = text.trim();

    if (!trimmedText) {
      setError("Please enter a sentence first.");
      setResult(null);
      return;
    }

    setLoading(true);
    setError("");
    setResult(null);

    try {
      const response = await fetch(API_URL, {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({ text: trimmedText })
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || "Something went wrong.");
      }

      setResult(data);
    } catch (err) {
      setError(
        err.message ||
          "Could not connect to the backend. Make sure the Flask API is running on port 5000."
      );
    } finally {
      setLoading(false);
    }
  };

  const useExample = (example) => {
    setText(example);
    setResult(null);
    setError("");
  };

  const clearText = () => {
    setText("");
    setResult(null);
    setError("");
  };

  const formatPercent = (value) => {
    return `${(value * 100).toFixed(2)}%`;
  };

  return (
    <main className="page">
      <section className="hero-card">
        <div className="badge">BERTweet Humor Detection</div>

        <h1>Laugh Detector</h1>

        <p className="subtitle">
          Enter a short English text and the model will predict humor, humor intensity,
          offensiveness, and whether the humor may be controversial.
        </p>

        <div className="input-area">
          <textarea
            value={text}
            onChange={(event) => setText(event.target.value)}
            placeholder="Example: I used to play piano by ear, but now I use my hands."
            rows="6"
          />

          <div className="actions">
            <button onClick={predictHumor} disabled={loading}>
              {loading ? "Predicting..." : "Detect Humor"}
            </button>

            <button className="secondary-button" onClick={clearText}>
              Clear
            </button>
          </div>
        </div>

        <div className="examples">
          <p>Try an example:</p>

          <div className="example-buttons">
            {exampleTexts.map((example, index) => (
              <button
                key={index}
                className="example-button"
                onClick={() => useExample(example)}
              >
                Example {index + 1}
              </button>
            ))}
          </div>
        </div>

        {error && <div className="error-box">{error}</div>}

        {result && (
          <section className="result-card">
            <div className="result-header">
              <span
                className={
                  result.label === "Humorous"
                    ? "result-label humorous"
                    : "result-label not-humorous"
                }
              >
                {result.label}
              </span>

              <span className="confidence">
                Confidence: {formatPercent(result.confidence)}
              </span>
            </div>

            <div className="probabilities">
              <div className="probability-row">
                <span>Humorous</span>
                <strong>{formatPercent(result.humorous_probability)}</strong>
              </div>

              <div className="bar">
                <div
                  className="bar-fill humorous-fill"
                  style={{
                    width: formatPercent(result.humorous_probability)
                  }}
                />
              </div>

              <div className="probability-row">
                <span>Not humorous</span>
                <strong>
                  {formatPercent(result.not_humorous_probability)}
                </strong>
              </div>

              <div className="bar">
                <div
                  className="bar-fill not-humorous-fill"
                  style={{
                    width: formatPercent(result.not_humorous_probability)
                  }}
                />
              </div>

              <div className="extra-metrics">
                <div className="metric-card">
                  <span>Humor rating</span>
                  <strong>
                    {result.label === "Humorous"
                      ? `${result.humor_rating_0_to_5} / 5`
                      : "N/A"}
                  </strong>
                </div>

                <div className="metric-card">
                  <span>Offense rating</span>
                  <strong>{result.offense_rating_0_to_5} / 5</strong>
                </div>

                <div className="metric-card">
                  <span>Controversial</span>
                  <strong>{result.controversial ? "Yes" : "No"}</strong>
                </div>

                <div className="metric-card">
                  <span>Controversy probability</span>
                  <strong>{formatPercent(result.controversy_probability)}</strong>
                </div>
              </div>

              
            </div>

            <p className="note">
              The confidence score is the model's predicted probability, not an
              objective measurement of how funny the sentence is.
            </p>
          </section>
        )}
      </section>
    </main>
  );
}

export default App;