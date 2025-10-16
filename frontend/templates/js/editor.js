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
  const [activeTab, setActiveTab] = useState("layers"); // 'layers' | 'activations' | 'normalizations'

  const pushLayer = (type) => {
    const template = LAYER_TEMPLATES[type] ? JSON.parse(JSON.stringify(LAYER_TEMPLATES[type])) : {};
    setLayers(prev => [...prev, { type, params: template }]);
  };

  const updateLayer = (idx, newLayer) => setLayers(prev => prev.map((l, i) => (i === idx ? newLayer : l)));
  const deleteLayer = (idx) => setLayers(prev => prev.filter((_, i) => i !== idx));

  // Вкладки слева
  const tabContent = () => {
    switch(activeTab) {
      case "layers":
        return (
          <>
            <button className="btn btn-primary" onClick={() => pushLayer("Linear")}>Add Linear</button>
            <button className="btn btn-primary" onClick={() => pushLayer("Conv2D")}>Add Conv2D</button>
          </>
        );
      case "activations":
        return (
          <>
            <button className="btn btn-primary" onClick={() => pushLayer("ReLU")}>Add ReLU</button>
            <button className="btn btn-primary" onClick={() => pushLayer("LeakyReLU")}>Add LeakyReLU</button>
            <button className="btn btn-primary" onClick={() => pushLayer("PReLU")}>Add PReLU</button>
            <button className="btn btn-primary" onClick={() => pushLayer("Sigmoid")}>Add Sigmoid</button>
            <button className="btn btn-primary" onClick={() => pushLayer("Tanh")}>Add Tanh</button>
            <button className="btn btn-primary" onClick={() => pushLayer("Softmax")}>Add Softmax</button>
          </>
        );
      case "normalizations":
        return (
          <>
            <button className="btn btn-primary" onClick={() => pushLayer("BatchNorm1d")}>Add BatchNorm1d</button>
            <button className="btn btn-primary" onClick={() => pushLayer("BatchNorm2d")}>Add BatchNorm2d</button>
            <button className="btn btn-primary" onClick={() => pushLayer("LayerNorm")}>Add LayerNorm</button>
          </>
        );
      default: return null;
    }
  };

  return (
    <div className="editor-container">
      {/* Левая панель вкладок */}
      <div className="editor-sidebar">
        <div className="tabs">
          <button className={activeTab === "layers" ? "active" : ""} onClick={() => setActiveTab("layers")}>Layers</button>
          <button className={activeTab === "activations" ? "active" : ""} onClick={() => setActiveTab("activations")}>Activations</button>
          <button className={activeTab === "normalizations" ? "active" : ""} onClick={() => setActiveTab("normalizations")}>Normalizations</button>
        </div>
        <div className="tab-content">
          {tabContent()}
        </div>
      </div>

      {/* Центр — список слоёв */}
      <div className="editor-main">
        {layers.length === 0 ? <div className="empty">No layers yet — add some.</div> : null}
        {layers.map((layer, idx) => (
          <LayerCard
            key={idx}
            layer={layer}
            index={idx}
            total={layers.length}
            onUpdate={updateLayer}
            onDelete={deleteLayer}
            moveUp={(i)=> {/*...*/}}
            moveDown={(i)=> {/*...*/}}
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
