function prepareDataset() {
  const datasetId = document.getElementById('dataset-select').value;
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