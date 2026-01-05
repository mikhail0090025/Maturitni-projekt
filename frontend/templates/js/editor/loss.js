const projectId = document.getElementById('project-id').value;

const lossSelect = document.getElementById('loss-select');
const statusField = document.getElementById('loss-status');

// 1️⃣ загрузка доступных лоссов
async function loadLossTypes() {
  try {
    const res = await fetch('/losstypes');
    const losses = await res.json();

    lossSelect.innerHTML = '';

    losses.forEach(loss => {
      const option = document.createElement('option');
      option.value = loss;
      option.textContent = loss;
      lossSelect.appendChild(option);
    });

  } catch (e) {
    lossSelect.innerHTML = '<option>Error loading losses</option>';
    statusField.textContent = 'Failed to load loss types';
    statusField.className = 'status error';
  }
}

// 2️⃣ обновление проекта при выборе
lossSelect.addEventListener('change', async () => {
  const selectedLoss = lossSelect.value;

  statusField.textContent = 'Saving…';
  statusField.className = 'status';

  try {
    const res = await fetch(`/loss`, {
      method: 'PUT',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        loss_function: selectedLoss,
        project_id: projectId
      })
    });

    if (!res.ok) {
      throw new Error('Request failed');
    }

    statusField.textContent = 'Saved';
    statusField.className = 'status ok';

  } catch (e) {
    statusField.textContent = 'Error saving loss';
    statusField.className = 'status error';
  }
});

// init
loadLossTypes();