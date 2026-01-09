function readJSONFromHidden(id, fallback = null) {
  const el = document.getElementById(id);
  if (!el || !el.value) return fallback;
  try {
    return JSON.parse(el.value);
  } catch (e) {
    console.warn(`Invalid JSON in #${id}`, e);
    return fallback;
  }
}

function setValueIfExists(id, value) {
  const el = document.getElementById(id);
  if (!el) return;
  if (el.tagName === "SELECT") {
    el.value = value;
    // try to trigger change for any listeners
    el.dispatchEvent(new Event("change", { bubbles: true }));
  } else {
    el.value = value;
    el.dispatchEvent(new Event("input", { bubbles: true }));
  }
}

function showElement(id) {
  const el = document.getElementById(id);
  if (el) el.classList.remove("hidden");
}
function hideElement(id) {
  const el = document.getElementById(id);
  if (el) el.classList.add("hidden");
}

function applyOptimizerConfig(cfg) {
  if (!cfg) return;
  // lr, weight_decay, betas (array)
  setValueIfExists("opt-lr", typeof cfg.lr === "number" ? cfg.lr : (cfg.learning_rate ?? 0.0003));
  setValueIfExists("opt-wd", typeof cfg.weight_decay === "number" ? cfg.weight_decay : (cfg.weight_decay ?? 0.0));

  const betas = Array.isArray(cfg.betas) ? cfg.betas : (Array.isArray(cfg.beta) ? cfg.beta : null);
  if (betas) {
    setValueIfExists("opt-beta1", betas[0]);
    setValueIfExists("opt-beta2", betas[1]);
  } else {
    // fallbacks
    setValueIfExists("opt-beta1", cfg.beta1 ?? 0.9);
    setValueIfExists("opt-beta2", cfg.beta2 ?? 0.999);
  }
}

function applySchedulerConfig(cfg) {
  // default - hide all subblocks
  if (!cfg) {
    setValueIfExists("scheduler-type", "none");
    hideElement("scheduler-cosine");
    hideElement("scheduler-plateau");
    hideElement("cosine-warmup-block");
    return;
  }

  const type = (cfg.type || cfg.scheduler_type || "").toString().toLowerCase();

  if (type === "cosine" || type === "cosineannealing" || type === "cosine_annealing") {
    setValueIfExists("scheduler-type", "cosine");
    showElement("scheduler-cosine");
    hideElement("scheduler-plateau");

    setValueIfExists("cosine-total-steps", cfg.total_steps ?? cfg.totalSteps ?? cfg.max_steps ?? 20000);
    setValueIfExists("cosine-min-lr", cfg.min_lr ?? cfg.min_lr ?? cfg.minLr ?? 1e-6);

    const warmupEnabled = Boolean(cfg.warmup_steps && Number(cfg.warmup_steps) > 0);
    const warmupCheckbox = document.getElementById("cosine-warmup-enabled");
    if (warmupCheckbox) {
      warmupCheckbox.checked = warmupEnabled;
      warmupCheckbox.dispatchEvent(new Event("change", { bubbles: true }));
    }
    if (warmupEnabled) {
      setValueIfExists("cosine-warmup-steps", cfg.warmup_steps ?? cfg.warmupSteps ?? 0);
      showElement("cosine-warmup-block");
    } else {
      hideElement("cosine-warmup-block");
    }

  } else if (type === "plateau" || type === "reduceonplateau" || type === "reduce_on_plateau") {
    setValueIfExists("scheduler-type", "plateau");
    showElement("scheduler-plateau");
    hideElement("scheduler-cosine");

    setValueIfExists("plateau-mode", cfg.mode ?? "min");
    setValueIfExists("plateau-factor", cfg.factor ?? 0.1);
    setValueIfExists("plateau-patience", cfg.patience ?? 5);
    setValueIfExists("plateau-min-lr", cfg.min_lr ?? cfg.minLr ?? 1e-6);
    hideElement("cosine-warmup-block");
  } else {
    setValueIfExists("scheduler-type", "none");
    hideElement("scheduler-cosine");
    hideElement("scheduler-plateau");
    hideElement("cosine-warmup-block");
  }
}

document.addEventListener("DOMContentLoaded", () => {
  const optimizerConfig = readJSONFromHidden("optimizer-json", null);
  const schedulerConfig = readJSONFromHidden("scheduler-json", null);
  const lossFunction = document.getElementById("loss-function")?.value ?? null;

  console.log("✅ Loaded optimizer config:", optimizerConfig);
  console.log("✅ Loaded scheduler config:", schedulerConfig);
  console.log("✅ Loaded loss:", lossFunction);

  // apply to UI
  applyOptimizerConfig(optimizerConfig);
  applySchedulerConfig(schedulerConfig);

  // keep global store in sync
  window.trainingConfig = {
    optimizer: optimizerConfig,
    scheduler: schedulerConfig,
    loss: lossFunction
  };

  // Ensure scheduler select toggles blocks if user changes it later
  const schedulerTypeEl = document.getElementById("scheduler-type");
  const warmupCheckboxEl = document.getElementById("cosine-warmup-enabled");

  if (schedulerTypeEl) {
    schedulerTypeEl.addEventListener("change", () => {
      const v = schedulerTypeEl.value;
      if (v === "cosine") {
        showElement("scheduler-cosine");
        hideElement("scheduler-plateau");
      } else if (v === "plateau") {
        hideElement("scheduler-cosine");
        showElement("scheduler-plateau");
      } else {
        hideElement("scheduler-cosine");
        hideElement("scheduler-plateau");
      }
    });
  }

  if (warmupCheckboxEl) {
    warmupCheckboxEl.addEventListener("change", () => {
      if (warmupCheckboxEl.checked) showElement("cosine-warmup-block");
      else hideElement("cosine-warmup-block");
    });
  }
});
