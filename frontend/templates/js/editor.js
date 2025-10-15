// editor.js — React-based editor that listens to the existing buttons in the page
const { useState, useEffect } = React;

const LAYER_TEMPLATES = {
  Linear: { in_features: 1, out_features: 1 },
  Conv2D: { in_channels: 1, out_channels: 8, kernel_size: 3, stride: 1, padding: 0 },
  ReLU: {},
  LeakyReLU: { alpha: 0.01 },
  PReLU: { /* params handled differently in backend if needed */ },
  Sigmoid: {},
  Tanh: {},
  Softmax: { dim: 1 },
  BatchNorm1d: { num_features: 1 },
  BatchNorm2d: { num_features: 1 },
  LayerNorm: { normalized_shape: 1 },
};

function LayerCard({ layer, index, onUpdate, onDelete, moveUp, moveDown, total }) {
  const handleChange = (key, val) => {
    onUpdate(index, { ...layer, params: { ...layer.params, [key]: val } });
  };

  const renderFields = () => {
    const p = layer.params || {};
    switch (layer.type) {
      case "Linear":
        return (
          <>
            <label>In features</label>
            <input type="number" value={p.in_features} onChange={e => handleChange("in_features", Number(e.target.value))} />
            <label>Out features</label>
            <input type="number" value={p.out_features} onChange={e => handleChange("out_features", Number(e.target.value))} />
          </>
        );
      case "Conv2D":
        return (
          <div className="layer-params-row">
            <label>In channels</label>
            <input type="number" value={p.in_channels} onChange={e => handleChange("in_channels", Number(e.target.value))} />
            <label>Out channels</label>
            <input type="number" value={p.out_channels} onChange={e => handleChange("out_channels", Number(e.target.value))} />
            <label>Kernel size</label>
            <input type="number" value={p.kernel_size} onChange={e => handleChange("kernel_size", Number(e.target.value))} />
            <label>Stride</label>
            <input type="number" value={p.stride} onChange={e => handleChange("stride", Number(e.target.value))} />
            <label>Padding</label>
            <input type="number" value={p.padding} onChange={e => handleChange("padding", Number(e.target.value))} />
          </div>
        );
      case "LeakyReLU":
        return (
          <>
            <label>Alpha</label>
            <input type="number" step="0.01" value={p.alpha} onChange={e => handleChange("alpha", Number(e.target.value))} />
          </>
        );
      case "Softmax":
        return (
          <>
            <label>Dim</label>
            <input type="number" value={p.dim ?? 1} onChange={e => handleChange("dim", Number(e.target.value))} />
          </>
        );
      case "BatchNorm1d":
      case "BatchNorm2d":
        return (
          <>
            <label>Num features</label>
            <input type="number" value={p.num_features} onChange={e => handleChange("num_features", Number(e.target.value))} />
          </>
        );
      case "LayerNorm":
        return (
          <>
            <label>Normalized shape</label>
            <input type="number" value={p.normalized_shape} onChange={e => handleChange("normalized_shape", Number(e.target.value))} />
          </>
        );
      default:
        return <em className="no-params">No editable params</em>;
    }
  };

  return (
    <div className="layer-card">
      <div className="layer-header">
        <div><strong>{index + 1}. {layer.type}</strong></div>
        <div className="layer-controls">
          <button className="btn btn-sm" onClick={() => moveUp(index)} disabled={index === 0}>▲</button>
          <button className="btn btn-sm" onClick={() => moveDown(index)} disabled={index === total - 1}>▼</button>
          <button className="btn btn-danger btn-sm" onClick={() => onDelete(index)}>Delete</button>
        </div>
      </div>

      <div className="layer-body">
        {renderFields()}
      </div>
    </div>
  );
}

function Editor() {
  const [layers, setLayers] = useState([]);

  // Adding layer by type (used from internal calls or from external buttons)
  const pushLayer = (type) => {
    const template = LAYER_TEMPLATES[type] ? JSON.parse(JSON.stringify(LAYER_TEMPLATES[type])) : {};
    setLayers(prev => [...prev, { type, params: template }]);
  };

  // Update / delete / reorder
  const updateLayer = (idx, newLayer) => setLayers(prev => prev.map((l, i) => (i === idx ? newLayer : l)));
  const deleteLayer = (idx) => setLayers(prev => prev.filter((_, i) => i !== idx));
  const moveUp = (idx) => setLayers(prev => {
    if (idx <= 0) return prev;
    const arr = [...prev];
    [arr[idx-1], arr[idx]] = [arr[idx], arr[idx-1]];
    return arr;
  });
  const moveDown = (idx) => setLayers(prev => {
    if (idx >= prev.length - 1) return prev;
    const arr = [...prev];
    [arr[idx], arr[idx+1]] = [arr[idx+1], arr[idx]];
    return arr;
  });

  // Hook: attach DOM click listeners to existing buttons on page (only once)
  useEffect(() => {
    const bind = (id, type) => {
      const el = document.getElementById(id);
      if (!el) return null;
      const handler = () => pushLayer(type);
      el.addEventListener("click", handler);
      return () => el.removeEventListener("click", handler);
    };

    const cleaners = [
      bind("add-linear-layer-button", "Linear"),
      bind("add-conv-layer-button", "Conv2D"),
      bind("add-relu-button", "ReLU"),
      bind("add-leaky-relu-button", "LeakyReLU"),
      bind("add-prelu-button", "PReLU"),
      bind("add-sigmoid-button", "Sigmoid"),
      bind("add-tanh-button", "Tanh"),
      bind("add-softmax-button", "Softmax"),
      bind("add-batchnorm1d-button", "BatchNorm1d"),
      bind("add-batchnorm2d-button", "BatchNorm2d"),
      bind("add-layernorm-button", "LayerNorm"),
    ].filter(Boolean);

    return () => cleaners.forEach(c => c && c());
  }, []); // empty deps → attach once on mount

  // Export JSON — can later send to backend
  const exportJSON = () => {
    const json = JSON.stringify(layers, null, 2);
    console.log("EXPORT MODEL JSON:", json);
    // Example: POST to backend
    // fetch('/api/projects/123/layers', { method:'POST', headers: {'Content-Type':'application/json'}, body: json })
    //   .then(r=>r.json()).then(console.log)
    alert("Model JSON printed to console (and ready to POST).");
  };

  return (
    <div className="editor-root">
      <div className="editor-top">
        <p className="hint">Click buttons above to add layers. Edit parameters below.</p>
        <div className="action-row">
          <button className="btn btn-success" onClick={exportJSON}>Export JSON</button>
          <span className="small-muted"> (will print JSON to console)</span>
        </div>
      </div>

      <div id="layers-list">
        {layers.length === 0 ? <div className="empty">No layers yet — add some.</div> : null}
        {layers.map((layer, idx) => (
          <LayerCard
            key={idx}
            layer={layer}
            index={idx}
            total={layers.length}
            onUpdate={updateLayer}
            onDelete={deleteLayer}
            moveUp={moveUp}
            moveDown={moveDown}
          />
        ))}
      </div>
    </div>
  );
}

// mount React into the existing #layers-block element — robust version
function mountEditor() {
  const rootEl = document.getElementById("layers-block");
  if (!rootEl) {
    console.warn("No #layers-block found for React editor.");
    return;
  }
  // if already mounted, avoid double-mounts
  if (rootEl._react_root_attached) return;
  ReactDOM.createRoot(rootEl).render(<Editor />);
  rootEl._react_root_attached = true;
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", mountEditor);
} else {
  mountEditor();
}
