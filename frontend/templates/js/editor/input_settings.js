document.querySelectorAll(".tab-button").forEach(btn => {
    btn.addEventListener("click", () => {
        const tab = btn.getAttribute("data-tab");

        // Кнопки
        document.querySelectorAll(".tab-button")
            .forEach(b => b.classList.remove("active"));
        btn.classList.add("active");

        // Контент
        document.querySelectorAll(".tab-content")
            .forEach(c => c.classList.remove("active"));
        document.getElementById("tab-" + tab).classList.add("active");
    });
});