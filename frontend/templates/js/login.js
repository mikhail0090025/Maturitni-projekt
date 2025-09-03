document.addEventListener("DOMContentLoaded", () => {
    const form = document.querySelector("form");
    const resultDiv = document.querySelector("div");

    form.addEventListener("submit", async (event) => {
        event.preventDefault();

        const data = {
            username: document.getElementById("username").value,
            password: document.getElementById("password").value
        };
        console.log(data);

        try {
            const response = await fetch("/login", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify(data),
                credentials: "include"
            });

            const json = await response.json();
            if (response.ok) {
                resultDiv.textContent = json.message || "Login successful!";
                window.location.href = "/profile_page";
            } else {
                console.log(json);
                resultDiv.textContent = json.error || "Invalid username or password!";
            }
        } catch (err) {
            resultDiv.textContent = "Service unavailable: " + err.message;
        }
    });
});