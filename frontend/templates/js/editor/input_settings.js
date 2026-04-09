document.addEventListener("DOMContentLoaded", () => {
    const sidebar = document.querySelector(".editor-sidebar");
    if (!sidebar) return;

    const buttons = sidebar.querySelectorAll(".tabs button");
    const content = sidebar.querySelector(".tab-content");

    // Изначально скрываем содержимое
    content.style.display = "none";
    buttons.forEach(btn => btn.classList.remove("active"));

    buttons.forEach(btn => {
        btn.addEventListener("click", () => {
            const isActive = btn.classList.contains("active");

            // Скрываем содержимое
            content.style.display = "none";
            buttons.forEach(b => b.classList.remove("active"));

            if (!isActive) {
                // Показываем содержимое, если вкладка была неактивна
                btn.classList.add("active");
                content.style.display = "block";
            }
        });
    });
});

document.addEventListener("DOMContentLoaded", () => {
    // Находим все блоки с вкладками
    const tabBlocks = document.querySelectorAll(".input-settings-block");

    tabBlocks.forEach(block => {
        const buttons = block.querySelectorAll(".tabs .tab-button");
        const contents = block.querySelectorAll(".tab-content");

        // Изначально скрываем все контенты кроме активного
        contents.forEach(content => {
            if (!content.classList.contains("active")) {
                content.style.display = "none";
            } else {
                content.style.display = "block";
            }
        });

        buttons.forEach(btn => {
            btn.addEventListener("click", () => {
                const targetId = btn.getAttribute("data-tab");
                const targetContent = block.querySelector("#tab-" + targetId);

                const isActive = btn.classList.contains("active");

                // Сначала скрываем все
                buttons.forEach(b => b.classList.remove("active"));
                contents.forEach(c => c.style.display = "none");

                // Если вкладка была неактивна, показываем её
                if (!isActive) {
                    btn.classList.add("active");
                    if (targetContent) {
                        targetContent.style.display = "block";
                    }
                }
            });
        });
    });
});
