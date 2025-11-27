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
          <>
            <label>In C</label>
            <input type="number" value={p.in_channels} onChange={e => handleChange("in_channels", Number(e.target.value))} />
            <label>Out C</label>
            <input type="number" value={p.out_channels} onChange={e => handleChange("out_channels", Number(e.target.value))} />
            <label>Kernel</label>
            <input type="number" value={p.kernel_size} onChange={e => handleChange("kernel_size", Number(e.target.value))} />
            <label>Stride</label>
            <input type="number" value={p.stride} onChange={e => handleChange("stride", Number(e.target.value))} />
            <label>Pad</label>
            <input type="number" value={p.padding} onChange={e => handleChange("padding", Number(e.target.value))} />
          </>
        );
      case "DSConv2D":
        return (
          <>
            <label>In C</label>
            <input type="number" value={p.in_channels} onChange={e => handleChange("in_channels", Number(e.target.value))} />
            <label>Out C</label>
            <input type="number" value={p.out_channels} onChange={e => handleChange("out_channels", Number(e.target.value))} />
            <label>Kernel</label>
            <input type="number" value={p.kernel_size} onChange={e => handleChange("kernel_size", Number(e.target.value))} />
            <label>Stride</label>
            <input type="number" value={p.stride} onChange={e => handleChange("stride", Number(e.target.value))} />
            <label>Pad</label>
            <input type="number" value={p.padding} onChange={e => handleChange("padding", Number(e.target.value))} />
          </>
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
      case "LayerNorm":
        return <em className="no-params">No editable params</em>;

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

  /*
  const pushLayer = (type) => {
    const template = LAYER_TEMPLATES[type] ? JSON.parse(JSON.stringify(LAYER_TEMPLATES[type])) : {};
    setLayers(prev => [...prev, { type, params: template }]);
    };
  
  const updateLayer = (idx, newLayer) => setLayers(prev => prev.map((l, i) => (i === idx ? newLayer : l)));
  */
 /*
  const pushLayer = (type) => {
    const template = LAYER_TEMPLATES[type] ? JSON.parse(JSON.stringify(LAYER_TEMPLATES[type])) : {};
    
    // Автоматическая коррекция входов
    setLayers(prev => {
      const newLayer = { type, params: template };

      if (prev.length > 0) {
        const lastLayer = prev[prev.length - 1];

        // Если оба Linear, синхронизируем in_features
        if (type === "Linear" && lastLayer.type === "Linear") {
          newLayer.params.in_features = lastLayer.params.out_features;
        }

        // Если оба Conv2D, синхронизируем in_channels
        if (type === "Conv2D" && lastLayer.type === "Conv2D") {
          newLayer.params.in_channels = lastLayer.params.out_channels;
        }
      }

      return [...prev, newLayer];
    });
  };

  const updateLayer = (idx, newLayer) => {
    setLayers(prev => {
      const updated = prev.map((l, i) => (i === idx ? newLayer : l));

      // Авто-связка с предыдущим слоем
      if (idx > 0) {
        const prevLayer = updated[idx - 1];

        if (newLayer.type === "Linear" && prevLayer.type === "Linear") {
          newLayer.params.in_features = prevLayer.params.out_features;
        }

        if (newLayer.type === "Conv2D" && prevLayer.type === "Conv2D") {
          newLayer.params.in_channels = prevLayer.params.out_channels;
        }
      }

      // Авто-связка с последующим слоем (если он Linear/Conv2D)
      if (idx < updated.length - 1) {
        const nextLayer = updated[idx + 1];
        if (nextLayer.type === "Linear" && newLayer.type === "Linear") {
          nextLayer.params.in_features = newLayer.params.out_features;
        }
        if (nextLayer.type === "Conv2D" && newLayer.type === "Conv2D") {
          nextLayer.params.in_channels = newLayer.params.out_channels;
        }
      }

      return updated;
    });
  };
*/
  const isDataLayer = (type) => ["Linear", "Conv2D"].includes(type);

  const pushLayer = (type) => {
    const template = LAYER_TEMPLATES[type] ? JSON.parse(JSON.stringify(LAYER_TEMPLATES[type])) : {};
    setLayers(prev => {
      const newLayer = { type, params: template };

      // Авто-связка с последним Data Layer
      for (let i = prev.length - 1; i >= 0; i--) {
        if (isDataLayer(prev[i].type)) {
          if (type === "Linear" && prev[i].type === "Linear") {
            newLayer.params.in_features = prev[i].params.out_features;
          }
          if ((type === "Conv2D" || type === "DSConv2D") && (prev[i].type === "Conv2D" || prev[i].type === "DSConv2D")) {
            newLayer.params.in_channels = prev[i].params.out_channels;
          }
          break;
        }
      }

      return [...prev, newLayer];
    });
  };

  const updateLayer = (idx, newLayer) => {
    setLayers(prev => {
      const updated = prev.map((l, i) => (i === idx ? newLayer : l));

      // Авто-связка: пройтись вперёд и обновить все следующие Data Layer
      let lastOutLinear = null;
      let lastOutConv = null;
      for (let i = 0; i < updated.length; i++) {
        const l = updated[i];

        if (l.type === "Linear") {
          if (lastOutLinear !== null) l.params.in_features = lastOutLinear;
          lastOutLinear = l.params.out_features;
        }

        if (l.type === "Conv2D" || l.type === "DSConv2D") {
          if (lastOutConv !== null) l.params.in_channels = lastOutConv;
          lastOutConv = l.params.out_channels;
        }
      }

      return updated;
    });
  };

  const deleteLayer = (idx) => setLayers(prev => prev.filter((_, i) => i !== idx));

  const exportJSON = () => {
    const json = JSON.stringify(layers, null, 2);
    console.log("Model JSON:", json);
    alert("Model JSON printed to console.");
    return json;
  };

  const importJSON = (json) => {
    try {
      const parsed = JSON.parse(json);
      setLayers(parsed);
    } catch (e) {
      alert("Invalid JSON");
    }
  };

  useEffect(() => {
    // On mount, try to load existing project JSON
    const existingJson = document.getElementById('project-json').value;
    if (existingJson) {
      importJSON(existingJson);
    }
  }, []);

  const SaveJSON = () => {
    const json = JSON.stringify(layers, null, 2);
    console.log("Model JSON:", json);
    alert("Model JSON printed to console.");
  };
  const SaveProject = () => {
    const json = JSON.stringify(layers, null, 2);
    fetch('/projects/save/', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        project_json: json,
        project_id: document.getElementById('project-id').value
      }),
    })
  };

  // Вкладки слева
  const tabContent = () => {
    switch(activeTab) {
      case "layers":
        return (
          <>
            <button className="btn btn-primary" onClick={() => pushLayer("Linear")}>Add Linear</button>
            <button className="btn btn-primary" onClick={() => pushLayer("Conv2D")}>Add Conv2D</button>
            <button className="btn btn-primary" onClick={() => pushLayer("DSConv2D")}>Add Depthwise Separable Conv2D</button>
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
    <button onClick={() => exportJSON()} className="btn btn-primary button-save">Get JSON</button>
    <button onClick={() => SaveProject()} className="btn btn-primary button-save">Save project</button>
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
