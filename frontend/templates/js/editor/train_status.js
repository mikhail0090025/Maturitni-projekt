 const statusText = document.getElementById('status-text');
 const progressText = document.getElementById('progress-text');
 const lossText = document.getElementById('loss-text');
 const progressFill = document.getElementById('progress-fill');
 function updateStatus() {
   const projectId = document.getElementById('project-id').value;
   if (!projectId) return;
   fetch(`/get_train_status/${projectId}`)
     .then(res => res.json())
     .then(data => {
       statusText.textContent = data.status;
       if (data.status === "running") {
         const percent = Math.round((data.current / data.total) * 100);
         progressText.textContent = `${data.current} / ${data.total} (${percent}%)`;
         lossText.textContent = data.loss.toFixed(6);
         progressFill.style.width = percent + "%";
       } else {
         progressText.textContent = "—";
         lossText.textContent = "—";
         progressFill.style.width = "0%";
       }
     })
     .catch(err => {
       statusText.textContent = "error";
       console.error(err);
     });
 }
 // опрос каждые 500 мс
 setInterval(updateStatus, 500);