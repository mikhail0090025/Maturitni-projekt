document.getElementById("logout_button").addEventListener("click", async () => {
    try {
        const response = await fetch("/logout", {
            method: "POST",
            credentials: "include"
        });
        if (response.ok) {
            window.location.href = "/login_page";
        } else {
            console.error("Logout failed");
        }
    } catch (err) {
        console.error("Service unavailable: " + err.message);
    }
});

document.getElementById("settings_button").addEventListener("click", async () => {
    window.location.href = "/settings_page";
});

/*
fetch("/my_projects", {
    method: "GET",
    credentials: "include"
}).then(response => response.json()).then(data => {
    console.log(data);
}).catch(error => console.error(error));*/