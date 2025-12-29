function imageTypeToChannels(type) {
    switch (type) {
        case "rgb": return 3;
        case "grayscale": return 1;
        case "cmyk": return 4;
        default: return null;
    }
}

function getInputTensorShape() {
    const inputType = document.getElementById("input-type")?.value;

    if (!inputType) return null;

    if (inputType === "image") {
        const imageType = document.getElementById("select-input-image-type")?.value;
        const width = parseInt(document.getElementById("width-image-input")?.value);
        const height = parseInt(document.getElementById("height-image-input")?.value);

        const channels = imageTypeToChannels(imageType);

        if (!channels || !width || !height) return null;

        return {
            kind: "image",
            channels,
            height,
            width
        };
    }

    if (inputType === "vector") {
        const size = parseInt(document.getElementById("input-vector-size")?.value);
        if (!size) return null;

        return {
            kind: "vector",
            length: size
        };
    }

    return null;
}

function getOutputTensorShape() {
    const outputType = document.getElementById("output-type")?.value;

    if (!outputType) return null;

    if (outputType === "image") {
        const imageType = document.getElementById("output-image-type")?.value;
        const width = parseInt(document.getElementById("output-width")?.value);
        const height = parseInt(document.getElementById("output-height")?.value);

        const channels = imageTypeToChannels(imageType);

        if (!channels || !width || !height) return null;

        return {
            kind: "image",
            channels,
            height,
            width
        };
    }

    if (outputType === "vector") {
        const size = parseInt(document.getElementById("output-vector-size")?.value);
        if (!size) return null;

        return {
            kind: "vector",
            length: size
        };
    }

    return null;
}

function formatShape(shape) {
    if (!shape) return "—";

    if (shape.kind === "image") {
        return `(${shape.channels}, ${shape.height}, ${shape.width})`;
    }

    if (shape.kind === "vector") {
        return `(${shape.length})`;
    }

    return "—";
}

function updateTensorShapeInfo() {
    const inputShape = getInputTensorShape();
    const outputShape = getOutputTensorShape();

    const inputEl = document.getElementById("input-shape-value");
    const outputEl = document.getElementById("output-shape-value");

    if (inputEl) inputEl.textContent = formatShape(inputShape);
    if (outputEl) outputEl.textContent = "Expected: " + formatShape(outputShape);

    console.log("[tensor-shape]", { inputShape, outputShape });
}

document.addEventListener("DOMContentLoaded", () => {
    while (!dataLoaded) {}
    console.log(dataLoaded);
    setTimeout(() => {
        updateTensorShapeInfo();
    }, 1000);

    const container = document.getElementById("dataset-settings");
    if (!container) return;

    container.addEventListener("input", updateTensorShapeInfo);
    container.addEventListener("change", updateTensorShapeInfo);
});