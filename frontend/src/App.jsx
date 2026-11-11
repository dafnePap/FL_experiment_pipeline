
import { useEffect, useState } from "react";
import "./App.css";
import ResultsDashboard from "./ResultsDashboard";

const API_URL = "http://127.0.0.1:8000";
const IS_DEMO = import.meta.env.VITE_DEMO_MODE === "true";

const algorithmOptions = {
  fedavg: ["local_lr"],
  fedprox: ["local_lr", "mu"],
  scaffold: ["local_lr", "server_lr"],
  fedadam: ["local_lr", "eta", "tau"],
  fdaopt: ["local_lr", "server_lr"],
};

const names = {
  fedavg: "FedAvg",
  fedprox: "FedProx",
  scaffold: "SCAFFOLD",
  fedadam: "FedAdam",
  fdaopt: "FDA-OPT",
};

const defaults = {
  local_lr: 0.1,
  mu: 0.01,
  eta: 0.01,
  tau: 0.001,
  server_lr: 1.0,
};
const modelOptions = {
  mnist: [
    ["LG", "Logistic Regression"],
    ["lenet", "LeNet-5"],
    ["CNN", "CNN"],
    ["VGG16", "VGG16"],
  ],
  emnist: [
    ["LG", "Logistic Regression"],
    ["lenet", "LeNet-5"],
    ["CNN", "CNN"],
    ["VGG16", "VGG16"],
  ],
  cifar10: [
    ["RSN", "ResNet18"],
  ],
};
function App() {
  const [shared, setShared] = useState({
    dataset: "mnist",
    model: "LG",
    global_epochs: 10,
    client_num_in_total: 100,
    client_num_per_round: 2,
    local_epochs: 1,
    batch_size: 64,
    s_iid: 0.2,
    seed: "",
    gpu: 0,
  });

  const [algorithms, setAlgorithms] = useState({});
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const [jobId, setJobId] = useState(null);
  const [jobStatus, setJobStatus] = useState(null);

  const [demoReady, setDemoReady] = useState(!IS_DEMO);

  useEffect(() => {
    if (!IS_DEMO) return;

    fetch("/demo/config.json")
      .then((response) => {
        if (!response.ok) {
          throw new Error("Demo configuration not found");
        }
        return response.json();
      })
      .then((config) => {
        setShared(config.shared);

        setAlgorithms(
          Object.fromEntries(
            config.algorithms.map(({ name, ...params }) => [
              name,
              params,
            ])
          )
        );

        setDemoReady(true);
      })
      .catch((error) => {
        setResult({
          success: false,
          messages: [error.message],
        });
      });
  }, []);
function updateShared(key, value) {
  setShared((previous) => {
    if (key === "dataset") {
      const availableModels = modelOptions[value];

      const modelIsSupported = availableModels.some(
        ([model]) => model === previous.model
      );

      return {
        ...previous,
        dataset: value,
        model: modelIsSupported
          ? previous.model
          : availableModels[0][0],
      };
    }

    return {
      ...previous,
      [key]: value,
    };
  });
}

  function toggleAlgorithm(name) {
    setAlgorithms((previous) => {
      const updated = { ...previous };

      if (name in updated) {
        delete updated[name];
      } else {
        updated[name] = Object.fromEntries(
          algorithmOptions[name].map((key) => [key, defaults[key]])
        );
      }

      return updated;
    });
  }

  function updateAlgorithm(name, key, value) {
    setAlgorithms((previous) => ({
      ...previous,
      [name]: {
        ...previous[name],
        [key]: value,
      },
    }));
  }

  
  // Send the experiment setup to FastAPI
  async function submitComparison(event) {
    event.preventDefault();
    if (IS_DEMO) {
      setJobId("demo-mnist");
      setJobStatus("completed");

      setResult({
        success: true,
        messages: [
          "Showing results from a previously completed experiment. No new training was started.",
        ],
      });

      return;
    }
    const config = {
      shared: {
        ...shared,
        seed: Number(shared.seed),
      },
      algorithms: Object.entries(algorithms).map(([name, params]) => ({
        name,
        ...params,
      })),
    };

    setLoading(true);
    setResult(null);
    setJobId(null);
    setJobStatus(null);

    try {
      const response = await fetch(`${API_URL}/comparisons`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(config),
      });

      const data = await response.json();

      if (!response.ok) {
        const errors = Array.isArray(data.detail)
          ? data.detail.map(
              (error) => `${error.loc.join(" → ")}: ${error.msg}`
            )
          : [data.detail || "Request failed"];

        setResult({
          success: false,
          messages: errors,
        });
        return;
      }

      setJobId(data.job_id);
      setJobStatus(data.status);

      setResult({
        success: true,
        messages: [`Comparison submitted! Job ID: ${data.job_id}`],
      });
    } catch {
      setResult({
        success: false,
        messages: ["Could not connect to FastAPI. Is the backend running?"],
      });
    } finally {
      setLoading(false);
    }
  }

  // Automatically check the experiment status every 2 seconds
  useEffect(() => {
    if ( IS_DEMO ||!jobId || ["completed", "failed"].includes(jobStatus)) {
      return;
    }

    const interval = setInterval(async () => {
      try {
        const response = await fetch(
          `${API_URL}/comparisons/${jobId}`
        );

        if (!response.ok) {
          throw new Error("Could not retrieve job status");
        }

        const data = await response.json();
        setJobStatus(data.status);
      } catch (error) {
        console.error("Status check failed:", error);
      }
    }, 2000);

    return () => clearInterval(interval);
  }, [jobId, jobStatus]);

  return (
    <main className="container">
      <header>
        <div className="eyebrow">FEDERATED LEARNING EXPERIMENTS</div>
        <h2>Create a comparison between FL algorithms</h2>
        <p>
          Select your algorithms and their experimental settings.
        </p>
      </header>

      <form onSubmit={submitComparison}>
        <section>
          <h2>Shared settings</h2>
          <p>These conditions apply to all selected algorithms.</p>

          <div className="fields">
            <label>
              Dataset
              <select
                disabled={IS_DEMO}
                value={shared.dataset}
                onChange={(e) =>
                  updateShared("dataset", e.target.value)
                }
              >
                <option value="mnist">MNIST</option>
                <option value="emnist">EMNIST ByClass</option>
                <option value="cifar10">CIFAR-10</option>
              </select>
            </label>

            <label>
              Model
              <select
                disabled={IS_DEMO}
                value={shared.model}
                onChange={(e) =>
                  updateShared("model", e.target.value)
                }
              >
              {modelOptions[shared.dataset].map(([value, label]) => (
                <option key={value} value={value}>
                  {label}
                </option>
              ))}
              </select>
            </label>

            <label>
              Total clients
              <input
                disabled={IS_DEMO}
                type="number"
                min="1"
                required
                value={shared.client_num_in_total}
                onChange={(e) =>
                  updateShared(
                    "client_num_in_total",
                    e.target.value === "" ? "" : Number(e.target.value)
                  )
                }
              />
            </label>

            <label>
              IID fraction (s_iid)
              <input
                disabled={IS_DEMO}
                type="number"
                min="0"
                max="1"
                step="0.1"
                required
                value={shared.s_iid}
                onChange={(e) =>
                  updateShared(
                    "s_iid",
                    e.target.value === "" ? "" : Number(e.target.value)
                  )
                }
              />
            </label>
            {[
              ["global_epochs", "Global rounds"],
              ["client_num_per_round", "Clients per round"],
              ["local_epochs", "Local epochs"],
              ["batch_size", "Batch size"],
              ["seed", "Random seed"],
            ].map(([key, label]) => (
              <label key={key}>
                {label}
                <input
                  disabled={IS_DEMO}
                  type="number"
                  required
                  step="1"
                  min={key === "seed" ? "0" : "1"}
                  placeholder={
                    key === "seed" ? "Enter a seed" : ""
                  }
                  value={shared[key]}
                  onChange={(e) =>
                    updateShared(
                      key,
                      e.target.value === ""
                        ? ""
                        : Number(e.target.value)
                    )
                  }
                />
              </label>
            ))}
          </div>
        </section>

        <section>
          <h2> Select algorithms</h2>
          <p>
            Choose which algorithms you want to compare.  
          </p>

          {Object.entries(algorithmOptions).map(
            ([name, parameters]) => (
              <div className="algorithm" key={name}>
                <label className="algorithm-choice">
                  <input
                    disabled={IS_DEMO}
                    type="checkbox"
                    checked={name in algorithms}
                    onChange={() => toggleAlgorithm(name)}
                  />
                  <strong>{names[name]}</strong>
                </label>

                {name in algorithms && (
                  <div className="fields">
                    {parameters.map((key) => (
                      <label key={key}>
                        {key}
                        <input
                          disabled={IS_DEMO}
                          type="number"
                          required
                          step="any"
                          min={
                            key === "mu"
                              ? "0"
                              : "0.000000001"
                          }
                          value={algorithms[name][key]}
                          onChange={(e) =>
                            updateAlgorithm(
                              name,
                              key,
                              e.target.value === ""
                                ? ""
                                : Number(e.target.value)
                            )
                          }
                        />
                      </label>
                    ))}
                  </div>
                )}
              </div>
            )
          )}
        </section>

        <button
          type="submit"
          disabled={
            !demoReady ||
            loading ||
            ["pending", "running"].includes(jobStatus) ||
            Object.keys(algorithms).length === 0
          }
        >
          {IS_DEMO
            ? "View saved results"
            : loading
              ? "Submitting..."
              : "Run experiment"}
        </button>
      </form>

      {result && (
        <section className={result.success ? "success" : "error"}>
          <h2>
            {result.success
              ? "Comparison submitted"
              : "Validation error"}
          </h2>

          {result.messages.map((message, index) => (
            <p key={index}>{message}</p>
          ))}
        </section>
      )}

      {jobId && (
        <section>
          <h2>Experiment status</h2>

          <p>
            Job ID: <strong>{jobId}</strong>
          </p>

          <p>
            Status: <strong>{jobStatus}</strong>
          </p>

          {["pending", "running"].includes(jobStatus) && (
            <p>Training is in progress...</p>
          )}

          {jobStatus === "completed" && (
            <p>✅ Training completed successfully!</p>
          )}

          {jobStatus === "failed" && (
            <p>❌ Training failed. Check the experiment logs.</p>
          )}
        </section>
      )}
      {jobStatus === "completed" && jobId && (
        <ResultsDashboard jobId={jobId} demo={IS_DEMO} />
      )}
    </main>
  );
}

export default App;
