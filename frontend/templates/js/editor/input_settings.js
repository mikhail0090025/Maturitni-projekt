
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

/*
document.addEventListener("DOMContentLoaded", () => {
    // Все вкладки изначально скрыты
    document.querySelectorAll(".tab-content").forEach(c => c.classList.remove("active"));
    document.querySelectorAll(".tab-button").forEach(b => b.classList.remove("active"));

    document.querySelectorAll(".tab-button").forEach(btn => {
        btn.addEventListener("click", () => {
            const tab = btn.getAttribute("data-tab");
            const content = document.getElementById("tab-" + tab);
            const isActive = btn.classList.contains("active");

            if (isActive) {
                // Вкладка уже открыта → свернуть
                btn.classList.remove("active");
                content.classList.remove("active");
                return;
            }

            // Иначе — открыть новую вкладку
            document.querySelectorAll(".tab-button")
                .forEach(b => b.classList.remove("active"));
            document.querySelectorAll(".tab-content")
                .forEach(c => c.classList.remove("active"));

            btn.classList.add("active");
            content.classList.add("active");
        });
    });
});
*/