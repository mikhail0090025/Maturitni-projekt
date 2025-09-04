document.getElementById("delete-account-button").addEventListener("click", async () => {
    try {
        const response = await fetch("/delete_me", {
            method: "DELETE",
            credentials: "include"
        });
        if (response.ok) {
            window.location.href = "/login_page";
        } else {
            console.error("Delete failed");
        }
    } catch (err) {
        console.error("Service unavailable: " + err.message);
    }
});
