document.addEventListener("DOMContentLoaded", () => {
    const form = document.querySelector("form");
    const resultDiv = document.querySelector("div");

    form.addEventListener("submit", async (event) => {
        event.preventDefault();

        const data = {
            name: document.getElementById("name").value,
            surname: document.getElementById("surname").value,
            username: document.getElementById("username").value,
            password: document.getElementById("password").value,
            born_date: document.getElementById("birth_date").value
        };
        console.log(data);

        try {
            const response = await fetch("/register", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify(data)
            });

            const json = await response.json();
            if (response.ok) {
                resultDiv.textContent = json.message || "User created!";
            } else {
                resultDiv.textContent = json.error || "Something went wrong!";
            }
        } catch (err) {
            resultDiv.textContent = "Service unavailable: " + err.message;
        }
    });
});