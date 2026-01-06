function prepareDataset() {
  const datasetId = document.getElementById('dataset-for-project-select').value;
  const projectId = document.getElementById('project-id').value;
  fetch(`/prepare_dataset/${datasetId}/for_project/${projectId}`)
    .then(response => {
      if (!response.ok) {
        throw new Error('Network response was not ok');
      }
      return response.json();
    })
    .then(data => {
      alert('Dataset prepared successfully for the project.');
    })
    .catch(error => {
      console.error('There was a problem with the fetch operation:', error);
      alert('Failed to prepare dataset for the project.');
    });
}

async function initializeTraining() {
  const projectId = document.getElementById('project-id').value;
  try {
    const response = await fetch(`/initialize_training/${projectId}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' }
    });
    const data = await response.json();
    if (response.ok) {
      alert('Training initialized successfully.');
    } else {
      // alert(`Failed to initialize training: ${data.detail}`);
    }
  } catch (error) {
    console.error('Error initializing training:', error);
    alert('An error occurred while initializing training.');
  }
}

async function resetTraining(){
  const projectId = document.getElementById('project-id').value;
  try {
    const response = await fetch(`/projects/${projectId}/reset`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' }
    });
    const data = await response.json();
    if (response.ok) {
      alert('Training was reset successfully.');
    }
    else {
      alert(`Failed to reset training: ${data.detail}`);
    }
  } catch (error) {
    console.error('Error resetting training:', error);
    alert('An error occurred while resetting training.');
  }
}