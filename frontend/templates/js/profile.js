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

document.addEventListener("DOMContentLoaded", () => {

    const container = document.querySelector(".profile-container");
    const projects = document.getElementById("projects");

    // Плавное появление профиля
    setTimeout(() => {
        container.classList.add("show");
    }, 150);

    // Отдельное появление проектов (чтобы было ощущение динамики)
    setTimeout(() => {
        projects.classList.add("show-projects");
    }, 500);

});

const buttons = document.querySelectorAll(".btn");

buttons.forEach((btn, index) => {
    btn.style.opacity = "0";
    btn.style.transform = "translateY(15px)";
    btn.style.transition = "all 0.4s ease";

    setTimeout(() => {
        btn.style.opacity = "1";
        btn.style.transform = "translateY(0)";
    }, 200 + index * 100);
});