document.addEventListener("DOMContentLoaded", () => {
    const maxWait = 5000; // максимум ждать 5 секунд
    const intervalTime = 100; // проверять каждые 100 мс
    let waited = 0;

    const intervalId = setInterval(() => {
        const buttons = document.querySelectorAll(".delete-project-btn");
        if (buttons.length > 0 || waited >= maxWait) {
            clearInterval(intervalId); // останавливаем проверку
        }

        if (buttons.length > 0) {
            buttons.forEach(btn => {
                console.log("Attaching listener to", btn);
                // чтобы не навесить несколько раз
                if (!btn.dataset.listener) {
                    btn.dataset.listener = "true";
                    btn.addEventListener("click", () => {
                        const projectId = btn.getAttribute("data-id");
                        fetch(`/delete_project/${projectId}`, {
                            method: "POST",
                            credentials: "include"
                        }).then(res => {
                            if (res.ok) location.reload();
                            else alert("Failed to delete project.");
                        });
                    });
                }
            });
        }

        waited += intervalTime;
    }, intervalTime);
});
