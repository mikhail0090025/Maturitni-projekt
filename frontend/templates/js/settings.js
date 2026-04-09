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

document.getElementById("update-account-button").addEventListener("click", async () => {
    try {
        content = {
                'username': document.getElementById('initial_username').value,
                'new_username': document.getElementById('new_username').value,
                'new_name': document.getElementById('new_name').value,
                'new_surname': document.getElementById('new_surname').value,
                'new_born_date': document.getElementById('new_born_date').value,
                'new_bio': document.getElementById('new_bio').value
            }
        console.log(content);
        const response = await fetch("/edit_user", {
            method: "POST",
            credentials: "include",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify(content)
        });
        if (response.ok) {
            location.reload();
        } else {
            console.error("Edit failed");
        }
    } catch (err) {
        console.error("Service unavailable: " + err.message);
    }
});