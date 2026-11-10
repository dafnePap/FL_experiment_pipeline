
import subprocess
import threading
import uuid
from pathlib import Path
from typing import Literal
import pickle
import yaml
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator, model_validator
from fastapi.responses import FileResponse
from plot import generate_all_plots
# --------------------------------------------------
# FastAPI application
# --------------------------------------------------

app = FastAPI(title="FL Experiment API")

app.add_middleware(
    CORSMiddleware,  # Adds CORS middleware to FastAPI
    allow_origins=[
        "http://localhost:5173",  # Allows requests from the React frontend
        "http://127.0.0.1:5173",  # Also allows requests using this local address
    ],
    allow_methods=["GET", "POST"],  
    allow_headers=["Content-Type"], 
)


# --------------------------------------------------
# Validation
# --------------------------------------------------

def must_be_positive(value):
    if value <= 0:
        raise ValueError("Must be greater than 0")
    return value


class SharedConfig(BaseModel):
    dataset: Literal["mnist", "emnist", "cifar10"]
    model: str
    global_epochs: int
    client_num_per_round: int
    local_epochs: int
    batch_size: int
    seed: int
    gpu: Literal[0] = 0

    @field_validator(
        "global_epochs",
        "client_num_per_round",
        "local_epochs",
        "batch_size",
    )
    @classmethod
    def validate_positive(cls, value):
        return must_be_positive(value)


class AlgorithmConfig(BaseModel):
    name: Literal[
        "fedavg", "fedprox", "scaffold", "fedadam", "fdaopt"
    ]
    local_lr: float
    mu: float | None = None
    eta: float | None = None
    tau: float | None = None
    server_lr: float | None = None

    @field_validator("local_lr", "eta", "tau", "server_lr")
    @classmethod
    def validate_positive(cls, value):
        if value is not None:
            return must_be_positive(value)
        return value

    @field_validator("mu")
    @classmethod
    def validate_mu(cls, value):
        if value is not None and value < 0:
            raise ValueError("Must be greater than or equal to 0")
        return value


class ComparisonConfig(BaseModel):
    shared: SharedConfig
    algorithms: list[AlgorithmConfig] = Field(min_length=1)

    @model_validator(mode="after")
    def check_duplicates(self):
        names = [algo.name for algo in self.algorithms]

        if len(names) != len(set(names)):
            raise ValueError("Each algorithm can only be selected once")

        return self


# --------------------------------------------------
# Job management
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

jobs = {}
job_lock = threading.Lock() # Lock that prevents multiple threads from modifying shared job

# Function responsible for executing an experiment
def run_job(job_id):
    with job_lock:
        jobs[job_id]["status"] = "running"

    try:
        command = [ # Build the Docker Compose command as a list of argument
            "docker",
            "compose",
            "run",
            "--rm",
            "--no-deps",
            "-e",
            f"FL_COMPARISON_ID=comparison_{job_id}",
            "experiment1",
            "python",
            "experiment_runner.py",
            f"experiments_config/jobs/{job_id}.yaml",
        ]

        log_path = PROJECT_ROOT / "results" / f"{job_id}.log"

        with open(log_path, "w") as log:
            result = subprocess.run( # Executes the command, and captures the output in a log file
                command,
                cwd=PROJECT_ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )

        with job_lock:
            jobs[job_id]["status"] = (
                "completed" if result.returncode == 0 else "failed"
            )

    except Exception as error:
        with job_lock:
            jobs[job_id]["status"] = "failed"
            jobs[job_id]["error"] = str(error)


# --------------------------------------------------
# API endpoints
# --------------------------------------------------

@app.get("/")
def home():
    return {"message": "FL Experiment API is running"}

# Define a POST endpoint for creating a new comparison
@app.post("/comparisons", status_code=202)
def create_comparison(config: ComparisonConfig):

    # Allow only one comparison at a time
    with job_lock:
        if any(
            job["status"] in ("pending", "running")
            for job in jobs.values()
        ):
            raise HTTPException( # Reject the request if another comparison is active
                status_code=409,
                detail="Another comparison is already running",
            )

        # Generate a unique ID for this comparison
        job_id = uuid.uuid4().hex[:12]
        # Register the new job in the jobs dictionary
        jobs[job_id] = {
            "status": "pending",
            "results_dir": f"results/comparison_{job_id}",
        }

    # Save the submitted configuration as YAML
    config_dir = PROJECT_ROOT / "experiments_config" / "jobs"
    config_path = config_dir / f"{job_id}.yaml"

    try:
        config_dir.mkdir(parents=True, exist_ok=True)
        (PROJECT_ROOT / "results").mkdir(exist_ok=True)

        with open(config_path, "w") as file:
            yaml.safe_dump(
                config.model_dump(exclude_none=True),
                file,
                sort_keys=False,
            )

        # Start the job without blocking the HTTP response
        threading.Thread(
            target=run_job,
            args=(job_id,),
            daemon=True,
        ).start()

    except Exception:
        with job_lock:
            del jobs[job_id]
        raise HTTPException(
            status_code=500,
            detail="Could not create comparison job",
        )

    return {
        "job_id": job_id,
        "status": "pending",
    }


@app.get("/comparisons/{job_id}")
def get_comparison(job_id: str):
    with job_lock:
        if job_id not in jobs:
            raise HTTPException(
                status_code=404,
                detail="Job not found",
            )

        return jobs[job_id].copy()


@app.get("/comparisons/{job_id}/results")
def get_results(job_id: str):

    with job_lock:
        job = jobs.get(job_id)

        if job is None:
            raise HTTPException(404, "Job not found")

        if job["status"] != "completed":
            raise HTTPException(409, "Comparison is not completed")

        results_dir = PROJECT_ROOT / job["results_dir"]

    results = {}

    for file in results_dir.glob("*.pkl"):
        with open(file, "rb") as f:
            data = pickle.load(f)

        algorithm = file.stem.split("_")[0]

        total_bytes = data["total_down"] + data["total_up"]

        results[algorithm] = {
            "rounds": data["rounds"],
            "accuracy": data["accuracy"],
            "loss": data["loss"],
            "final_accuracy": data["accuracy"][-1],
            "final_loss": data["loss"][-1],
            "down_per_round": data["down_per_round"],
            "up_per_round": data["up_per_round"],
            "total_communication_mb": total_bytes / 1_000_000,
        }

    return {
        "job_id": job_id,
        "algorithms": results,
    }

@app.get("/comparisons/{job_id}/plots/{metric}")
def get_plot(job_id: str, metric: Literal["accuracy", "loss"]):
    with job_lock:
        job = jobs.get(job_id)

        if job is None:
            raise HTTPException(404, "Job not found")

        if job["status"] != "completed":
            raise HTTPException(409, "Comparison not completed")

        results_dir = PROJECT_ROOT / job["results_dir"]

    plot_path = results_dir / "plots" / f"{metric}.png"

    if not plot_path.is_file():
        generate_all_plots(results_dir)

    return FileResponse(plot_path, media_type="image/png")
