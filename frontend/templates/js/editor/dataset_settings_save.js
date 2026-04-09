/* ===============================
   Helpers
================================ */

function el(id) {
    return document.getElementById(id);
}

function val(id) {
    const e = el(id);
    return e ? e.value : null;
}

function num(id) {
    const v = val(id);
    return v !== null ? Number(v) : null;
}

function checked(id) {
    const e = el(id);
    return e ? e.checked : false;
}

/* ===============================
   INPUT SETTINGS
================================ */

function collectInputSettings(inputType) {
    const input = {
        type: inputType,
        basic: {},
        augmentations: {}
    };

    /* ----- IMAGE INPUT ----- */
    if (inputType === "image") {
        // BASIC
        input.basic = {
            color_mode: val("select-input-image-type"),
            width: num("width-image-input"),
            height: num("height-image-input")
        };

        // AUGMENTATIONS
        input.augmentations = {
            horizontal_flip_prob: num("horizontal-flip"),
            vertical_flip_prob: num("vertical-flip"),

            color_jitter: {
                brightness: num("input-image-brightness"),
                contrast: num("input-image-contrast"),
                saturation: num("input-image-saturation")
            },

            affine: {
                shift_h: num("input-image-shift-h"),
                shift_v: num("input-image-shift-v"),
                scale_h: num("input-image-scale-h"),
                scale_v: num("input-image-scale-v"),
                rotate_deg: num("input-image-rotate")
            }
        };
    }

    /* ----- NOISE INPUT ----- */
    if (inputType === "noise") {
        const is2d = checked("2d-noise-input");

        input.basic = {
            mean: num("noise-input-mean"),
            std: num("noise-input-std"),
            mode: is2d ? "2d" : "1d"
        };

        if (is2d) {
            input.basic.width = num("width-noise-input");
            input.basic.height = num("height-noise-input");
        } else {
            input.basic.size = num("size-noise-input");
        }
    }

    /* ----- VECTOR / OTHER (future-safe) ----- */
    if (Object.keys(input.basic).length === 0) {
        input.basic = null;
        input.augmentations = null;
    }

    return input;
}

/* ===============================
   OUTPUT SETTINGS
================================ */

function collectOutputSettings(outputType) {
    const output = {
        type: outputType,
        basic: {}
    };

    if (outputType === "image") {
        output.basic = {
            color_mode: val("output-image-type"),
            width: num("output-width"),
            height: num("output-height")
        };
    }

    if (outputType === "vector") {
        output.basic = {
            size: num("output-vector-size")
        };
    }

    if (outputType === "binary") {
        output.basic = {
            size: num("output-binary-size")
        };
    }

    if (outputType === "noise") {
        output.basic = null;
    }

    return output;
}

/* ===============================
   FINAL COLLECTOR
================================ */

function collectPreprocessingConfig(inputType, outputType) {
    return {
        input: collectInputSettings(inputType),
        output: collectOutputSettings(outputType)
    };
}

/* ===============================
   Example usage
================================ */

// inputType / outputType должны прийти из backend или data-атрибутов
// например: "image", "vector", "binary", "noise"

function buildAndSendConfig(inputType, outputType) {
    const config = collectPreprocessingConfig(inputType, outputType);

    return config;
}

function buildConfigFromPage() {
    const inputTypeEl = document.getElementById("input-type");
    const outputTypeEl = document.getElementById("output-type");

    if (!inputTypeEl || !outputTypeEl) {
        console.error("Input/output type not found on page");
        return null;
    }

    const inputType = inputTypeEl.value;
    const outputType = outputTypeEl.value;

    const config = buildAndSendConfig(inputType, outputType);

    return config;
}

function getCurrentDatasetId() {
    const select = document.getElementById("dataset-for-project-select");

    if (!select) {
        console.error("Dataset select not found");
        return null;
    }

    return select.value || null;
}

function getProjectId() {
    const projectIdEl = document.getElementById("project-id");
    if (!projectIdEl) {
        console.error("Project ID element not found");
        return null;
    }
    return projectIdEl.value || null;
}

async function sendDatasetSettingsToBackend() {
    const datasetId = getCurrentDatasetId();
    const config = buildConfigFromPage();
    const projectId = getProjectId();

    if (!datasetId || !config || !projectId) {
        console.error("Cannot send dataset settings: missing data");
        return;
    }

    const payload = {
        dataset_id: datasetId,
        preprocessing_config: JSON.stringify(config),
        project_id: projectId
    };

    try {
        const response = await fetch("http://localhost:8001/save_dataset_settings", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify(payload)
        });

        if (!response.ok) {
            console.error("Backend error:", response.status);
            return;
        }

        const result = await response.json();
    } catch (err) {
        console.error("Request failed:", err);
    }
}

async function loadDatasetSettingsFromBackend() {
    const projectId = getProjectId();

    if (!projectId) {
        console.error("[loadDatasetSettings] projectId not found");
        return;
    }

    try {
        const response = await fetch(
            `http://localhost:8001/load_dataset_settings/${projectId}`,
            {
                method: "GET",
                credentials: "include" // важно для cookies
            }
        );

        if (!response.ok) {
            console.error("[loadDatasetSettings] backend error:", response.status);
            return;
        }

        const data = await response.json();

        const { dataset_id, dataset_preprocess_json } = data;

        // 1️⃣ применяем датасет
        applyDatasetSelection(dataset_id);

        // 2️⃣ применяем preprocessing
        if (dataset_preprocess_json) {
            let parsedConfig;
            try {
                parsedConfig = JSON.parse(dataset_preprocess_json);
            } catch (e) {
                console.error("Failed to parse preprocessing JSON:", e);
                return;
            }

            applyPreprocessingConfigToUI(parsedConfig);
        }

    } catch (err) {
        console.error("[loadDatasetSettings] request failed:", err);
    }
}

function applyDatasetSelection(datasetId) {
    const select = document.getElementById("dataset-for-project-select");

    if (!select) {
        console.error("Dataset select not found");
        return;
    }

    if (!datasetId) {
        console.warn("No dataset_id to apply");
        return;
    }

    select.value = String(datasetId);
}

function applyPreprocessingConfigToUI(config) {
    if (!config) {
        console.warn("No preprocessing config to apply");
        return;
    }
    const { input, output } = config;

    // Apply input settings
    if (input) {
        if (input.type === "image") {
            el("select-input-image-type").value = input.basic.color_mode || "rgb";
            el("width-image-input").value = input.basic.width || 64;
            el("height-image-input").value = input.basic.height || 64;
            el("horizontal-flip").value = input.augmentations.horizontal_flip_prob || 0;
            el("vertical-flip").value = input.augmentations.vertical_flip_prob || 0;
            el("input-image-brightness").value = input.augmentations.color_jitter.brightness || 0;
            el("input-image-contrast").value = input.augmentations.color_jitter.contrast || 0;
            el("input-image-saturation").value = input.augmentations.color_jitter.saturation || 0;
            el("input-image-shift-h").value = input.augmentations.affine.shift_h || 0;
            el("input-image-shift-v").value = input.augmentations.affine.shift_v || 0;
            el("input-image-scale-h").value = input.augmentations.affine.scale_h || 0;
            el("input-image-scale-v").value = input.augmentations.affine.scale_v || 0;
            el("input-image-rotate").value = input.augmentations.affine.rotate_deg || 0;
        }
        if (input.type === "noise") {
            const is2d = input.basic.mode === "2d";
            el("2d-noise-input").checked = is2d;
            el("noise-input-mean").value = input.basic.mean || 0;
            el("noise-input-std").value = input.basic.std || 1;
            if (is2d) {
                el("width-noise-input").value = input.basic.width || 28;
                el("height-noise-input").value = input.basic.height || 28;
            } else {
                el("size-noise-input").value = input.basic.size || 100;
            }
        }
    }
    // Apply output settings
    if (output) {
        if (output.type === "image") {
            el("output-image-type").value = output.basic.color_mode || "rgb";
            el("output-width").value = output.basic.width || 64;
            el("output-height").value = output.basic.height || 64;
        }
        if (output.type === "vector") {
            el("output-vector-size").value = output.basic.size || 10;
        }
        if (output.type === "binary") {
            el("output-binary-size").value = output.basic.size || 1;
        }
    }
}

let dataLoaded = false;
document.addEventListener("DOMContentLoaded", () => {
    loadDatasetSettingsFromBackend();
    dataLoaded = true;
    setInterval(() => {
        sendDatasetSettingsToBackend();
    }, 10000); // every 10 seconds
});
