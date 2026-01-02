function collectTrainingConfig() {
  const batchSize = parseInt(
    document.getElementById("train-batch-size").value,
    10
  );

  const shuffle = document.getElementById("train-shuffle").checked;

  const mode = document.getElementById("training-mode").value;

  const config = {
    batch_size: batchSize,
    shuffle: shuffle,
    mode: mode
  };

  if (mode === "epochs") {
    config.epochs = parseFloat(
      document.getElementById("train-epochs").value
    );
  }

  if (mode === "batches") {
    config.total_batches = parseInt(
      document.getElementById("train-batches").value,
      10
    );
  }

  return config;
}

function initializeTraining() {
  const trainingConfig = collectTrainingConfig();
  const projectId = document.getElementById("project-id").value;

  console.log("🚀 Training config:", trainingConfig);

  // Временно сохраняем глобально
  window.trainingConfig = window.trainingConfig || {};
  window.trainingConfig.training = trainingConfig;

  console.log("📦 Full training config:", window.trainingConfig);

  fetch("/initialize_training/" + projectId, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ projectId, ...window.trainingConfig })
  })
    .then((response) => response.json()).then((data) => {
      if (data.status === "ok") {
        alert("Training initialized successfully.");
        } else {
        alert(`Failed to initialize training: ${data.detail}`);
        }
    }).catch((error) => {
        console.error("Error initializing training:", error);
        alert("An error occurred while initializing training.");
    }
    );
}

function startTraining() {
  const projectId = document.getElementById("project-id").value;
    const trainingConfig = window.trainingConfig || {};
    const payload = { projectId, ...trainingConfig };

    fetch("/start_training/" + projectId, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
    }).then(response => response.json())
      .then(data => {
          if (data.status === "ok") {
                alert("Training started successfully.");
            } else {
                alert(`Failed to start training: ${data.detail}`);
            }
        }).catch(error => {
            console.error("Error starting training:", error);
            alert("An error occurred while starting training.");
        });
}