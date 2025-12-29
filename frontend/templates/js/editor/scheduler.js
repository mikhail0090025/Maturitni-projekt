const schedulerType = document.getElementById("scheduler-type");
const cosineBlock = document.getElementById("scheduler-cosine");
const plateauBlock = document.getElementById("scheduler-plateau");

const warmupCheckbox = document.getElementById("cosine-warmup-enabled");
const warmupBlock = document.getElementById("cosine-warmup-block");

schedulerType.addEventListener("change", () => {
  cosineBlock.classList.add("hidden");
  plateauBlock.classList.add("hidden");

  if (schedulerType.value === "cosine") {
    cosineBlock.classList.remove("hidden");
  }

  if (schedulerType.value === "plateau") {
    plateauBlock.classList.remove("hidden");
  }
});

warmupCheckbox.addEventListener("change", () => {
  warmupBlock.classList.toggle("hidden", !warmupCheckbox.checked);
});

function getOptimizerConfig() {
  return {
    type: "AdamW",
    lr: Number(document.getElementById("opt-lr").value),
    weight_decay: Number(document.getElementById("opt-wd").value),
    betas: [
      Number(document.getElementById("opt-beta1").value),
      Number(document.getElementById("opt-beta2").value)
    ]
  };
}

function getSchedulerConfig() {
  const type = schedulerType.value;

  if (type === "none") return null;

  if (type === "cosine") {
    return {
      type: "cosine",
      total_steps: Number(document.getElementById("cosine-total-steps").value),
      min_lr: Number(document.getElementById("cosine-min-lr").value),
      warmup_steps: warmupCheckbox.checked
        ? Number(document.getElementById("cosine-warmup-steps").value)
        : 0
    };
  }

  if (type === "plateau") {
    return {
      type: "plateau",
      mode: document.getElementById("plateau-mode").value,
      factor: Number(document.getElementById("plateau-factor").value),
      patience: Number(document.getElementById("plateau-patience").value),
      min_lr: Number(document.getElementById("plateau-min-lr").value)
    };
  }
}