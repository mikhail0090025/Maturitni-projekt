// editor.js — React-based editor that listens to the existing buttons in the page
const { useState, useEffect } = React;

const LAYER_TEMPLATES = {
  Linear: { in_features: 1, out_features: 1 },
  Conv2D: { in_features: 1, out_features: 8, kernel_size: 3, stride: 1, padding: 1 },
  DSConv2D: { in_features: 1, out_features: 8, kernel_size: 3, stride: 1, padding: 1 },
  ReLU: { in_features: 1, out_features: 8 },
  LeakyReLU: { in_features: 1, out_features: 8, alpha: 0.01 },
  PReLU: { in_features: 1, out_features: 8 },
  Sigmoid: { in_features: 1, out_features: 8 },
  Tanh: { in_features: 1, out_features: 8 },
  Softmax: { in_features: 1, out_features: 8 },
  BatchNorm1d: { in_features: 1, out_features: 8 },
  BatchNorm2d: { in_features: 1, out_features: 8 },
  LayerNorm: { in_features: 1, out_features: 8 },

  Conv2DTranspose: { in_features: 1, out_features: 8, kernel_size: 3, stride: 1, padding: 1, scale_factor: 2 },
  Upsample: { in_features: 1, out_features: 8, scale_factor: 2, mode: "nearest" },
  MaxPool2D: { in_features: 1, out_features: 8, kernel_size: 2, stride: 2 },
  AvgPool2D: { in_features: 1, out_features: 8, kernel_size: 2, stride: 2 },
  ConvolutionalAttention: { in_features: 1, out_features: 8, num_heads: 4},
  SEBlock: { in_features: 1, out_features: 8, reduction: 16 },
  ResidualBlock: { in_features: 1, out_features: 8, kernel_size: 3, stride: 1, padding: 1, depth: 2 },
  InstanceNorm1D: { in_features: 1, out_features: 8},
  InstanceNorm2D: { in_features: 1, out_features: 8},
  GroupNorm: { in_features: 1, out_features: 8, num_groups: 1 },
  Dropout: { in_features: 1, out_features: 8, p: 0.5 },
  PixelShuffle: { in_features: 8, out_features: 8, upscale_factor: 1 },
};

function countParams(layer) {
  const p = layer.params || {};

  switch (layer.type) {
    case "Linear":
      return p.in_features * p.out_features + p.out_features;

    case "Conv2D":
      return (
        p.out_features *
        p.in_features *
        p.kernel_size *
        p.kernel_size +
        p.out_features
      );

    case "BatchNorm1d":
    case "BatchNorm2d":
      return 2 * p.num_features;

    case "LayerNorm":
      return 2 * p.normalized_shape;

    case "PReLU":
      return 1;
    
    case "DSConv2D":
      return (
        p.in_features * p.kernel_size * p.kernel_size + // depthwise
        p.in_features * p.out_features + // pointwise
        p.out_features // biases
      );
    
    case "ConvolutionalAttention":
      const head_dim = p.in_features / p.num_heads;
      return 3 * (p.in_features * head_dim) + (p.num_heads * head_dim * p.in_features); // Q,K,V and output

    case "SEBlock":
      return (p.in_features * (p.in_features / p.reduction)) + ((p.in_features / p.reduction) * p.in_features);
    
    case "ResidualBlock":
      const first_conv_params = (p.out_features * p.in_features * p.kernel_size * p.kernel_size) + p.out_features;
      const other_convs_params = (p.depth - 1) * ((p.out_features * p.out_features * p.kernel_size * p.kernel_size) + p.out_features);
      return first_conv_params + other_convs_params;
    
    case "Conv2DTranspose":
      return (
        p.in_features * p.out_features * p.kernel_size * p.kernel_size +
        p.out_features
      );
    
    case "InstanceNorm1D":
    case "InstanceNorm2D":
      return 2 * p.num_features;
    case "GroupNorm":
      return 2 * p.num_channels;
    case "Dropout":
      return 0;
    case "PixelShuffle":
      return 0;

    default:
      return 0;
  }
}

function validUpscaleFactors(inChannels) {
  const factors = [];
  for (let r = 1; r <= Math.sqrt(inChannels); r++) {
    if (inChannels % (r * r) === 0) {
      factors.push(r);
    }
  }
  return factors;
}

function LayerCard({ layer, index, onUpdate, onDelete, moveUp, moveDown, total }) {
  const p = layer.params || {};

  const handleChange = (key, val) => {
    console.log("Changing", key, "to", val);
    onUpdate(index, { ...layer, params: { ...p, [key]: val } });
  };

  const paramCount = countParams(layer);

  const renderFields = () => {
    switch (layer.type) {
      case "Linear":
        return (
          <>
            <label>In features</label>
            <input min="0" type="number" value={p.in_features}
              onChange={e => handleChange("in_features", Number(e.target.value))}/>

            <label>Out features</label>
            <input min="0" type="number" value={p.out_features}
              onChange={e => handleChange("out_features", Number(e.target.value))} />
          </>
        );

      case "Conv2D":
        return (
          <>
            <label>In C</label>
            <input min="0" type="number" value={p.in_features}
              onChange={e => handleChange("in_features", Number(e.target.value))} />

            <label>Out C</label>
            <input min="0" type="number" value={p.out_features}
              onChange={e => handleChange("out_features", Number(e.target.value))} />

            <label>Kernel</label>
            <input min="1" type="number" value={p.kernel_size}
              onChange={e => handleChange("kernel_size", Number(e.target.value))} />

            <label>Stride</label>
            <input min="1" type="number" value={p.stride}
              onChange={e => handleChange("stride", Number(e.target.value))} />

            <label>Pad</label>
            <input min="0" type="number" value={p.padding}
              onChange={e => handleChange("padding", Number(e.target.value))} />
          </>
        );
      case "DSConv2D":
        return (
          <>
            <label>In C</label>
            <input min="0" type="number" value={p.in_features} onChange={e => handleChange("in_features", Number(e.target.value))} />
            <label>Out C</label>
            <input min="0" type="number" value={p.out_features} onChange={e => handleChange("out_features", Number(e.target.value))} />
            <label>Kernel</label>
            <input min="1" type="number" value={p.kernel_size} onChange={e => handleChange("kernel_size", Number(e.target.value))} />
            <label>Stride</label>
            <input min="1" type="number" value={p.stride} onChange={e => handleChange("stride", Number(e.target.value))} />
            <label>Pad</label>
            <input min="0" type="number" value={p.padding} onChange={e => handleChange("padding", Number(e.target.value))} />
          </>
        );
      case "LeakyReLU":
        return (
          <>
            <label>Alpha</label>
            <input type="number" step="0.01" value={p.alpha}
              onChange={e => handleChange("alpha", Number(e.target.value))} />
          </>
        );

      case "Softmax":
        return (
          <>
            <label>Dim</label>
            <input type="number" value={p.dim ?? 1}
              onChange={e => handleChange("dim", Number(e.target.value))} />
          </>
        );
      
      case "Conv2DTranspose":
        return (
          <>
            <label>In C</label>
            <input min="0" type="number" value={p.in_features}
              onChange={e => handleChange("in_features", Number(e.target.value))} />
            <label>Out C</label>
            <input min="0" type="number" value={p.out_features}
              onChange={e => handleChange("out_features", Number(e.target.value))} />
            <label>Kernel</label>
            <input min="1" type="number" value={p.kernel_size}
              onChange={e => handleChange("kernel_size", Number(e.target.value))} />
            <label>Stride</label>
            <input min="1" type="number" value={p.stride}
              onChange={e => handleChange("stride", Number(e.target.value))} />
            <label>Pad</label>
            <input min="0" type="number" value={p.padding}
              onChange={e => handleChange("padding", Number(e.target.value))} />
            <label>Scale Factor</label>
            <input min="1" type="number" value={p.scale_factor}
              onChange={e => handleChange("scale_factor", Number(e.target.value))} />
          </>
        );
      
      case "Upsample":
        return (
          <>
            <label>Scale Factor</label>
            <input min="1" type="number" value={p.scale_factor}
              onChange={e => handleChange("scale_factor", Number(e.target.value))} />
            <label>Mode</label>
            <input type="text" value={p.mode}
              onChange={e => handleChange("mode", e.target.value)} />
          </>
        );

      case "MaxPool2D":
        return (
          <>
            <label>Kernel Size</label>
            <input min="1" type="number" value={p.kernel_size}
              onChange={e => handleChange("kernel_size", Number(e.target.value))} />
            <label>Stride</label>
            <input min="1" type="number" value={p.stride}
              onChange={e => handleChange("stride", Number(e.target.value))} />
          </>
        );
      case "AvgPool2D":
        return (
          <>
            <label>Kernel Size</label>
            <input min="1" type="number" value={p.kernel_size}
              onChange={e => handleChange("kernel_size", Number(e.target.value))} />
            <label>Stride</label>
            <input min="1" type="number" value={p.stride}
              onChange={e => handleChange("stride", Number(e.target.value))} />
          </>
        );
      
      case "ConvolutionalAttention":
        return (
          <>
            <label>In Channels</label>
            <input min="0" type="number" value={p.in_features}
              onChange={e => handleChange("in_features", Number(e.target.value))} />
            <label>Num Heads</label>
            <input min="1" type="number" value={p.num_heads}
              onChange={e => handleChange("num_heads", Number(e.target.value))} />
          </>
        );
      
      case "SEBlock":
        return (
          <>
            <label>In Channels</label>
            <input min="0" type="number" value={p.in_features}
              onChange={e => handleChange("in_features", Number(e.target.value))} />
            <label>Reduction</label>
            <input min="1" type="number" value={p.reduction}
              onChange={e => handleChange("reduction", Number(e.target.value))} />
          </>
        );
      case "ResidualBlock":
        return (
          <>
            <label>In Channels</label>
            <input min="0" type="number" value={p.in_features}
              onChange={e => handleChange("in_features", Number(e.target.value))} />
            <label>Out Channels</label>
            <input min="0" type="number" value={p.out_features}
              onChange={e => handleChange("out_features", Number(e.target.value))} />
            <label>Kernel Size</label>
            <input min="1" type="number" value={p.kernel_size}
              onChange={e => handleChange("kernel_size", Number(e.target.value))} />
            <label>Stride</label>
            <input min="1" type="number" value={p.stride}
              onChange={e => handleChange("stride", Number(e.target.value))} />
            <label>Padding</label>
            <input min="0" type="number" value={p.padding}
              onChange={e => handleChange("padding", Number(e.target.value))} />
            <label>Depth</label>
            <input min="1" type="number" value={p.depth}
              onChange={e => handleChange("depth", Number(e.target.value))} />
          </>
        );
      
      case "InstanceNorm1D":
      case "InstanceNorm2D":
        return (
          <>
            <label>Num Features</label>
            <input type="number" value={p.num_features}
              onChange={e => handleChange("num_features", Number(e.target.value))} />
          </>
        );
      case "GroupNorm":
        return (
          <>
            <label>Num Groups</label>
            <input min="1" type="number" value={p.num_groups}
              onChange={e => handleChange("num_groups", Number(e.target.value))} />
            <label>Num Channels</label>
            <input min="1" type="number" value={p.num_channels}
              onChange={e => handleChange("num_channels", Number(e.target.value))} />
          </>
        );
      case "Dropout":
        return (
          <>
            <label>Probability (p)</label>
            <input min="0" type="number" step="0.01" value={p.p}
              onChange={e => handleChange("p", Number(e.target.value))} />
          </>
        );
      case "PixelShuffle": {
        // const valid = validUpscaleFactors(p.in_features).filter(r => r !== 1);
        const valid = validUpscaleFactors(p.in_features);

        return (
          <>
            <label>Upscale Factor</label>
            <select
              value={p.upscale_factor}
              onChange={e => {
                const r = Number(e.target.value);

                const newParams = {
                  ...p,
                  upscale_factor: r,
                  out_features: p.in_features / (r * r),
                };

                onUpdate(index, { ...layer, params: newParams });
              }}
            >
              {valid.map(r => (
                <option key={r} value={r}>{r}</option>
              ))}
            </select>
          </>
        );
      }

      default:
        return <em className="no-params">No editable params</em>;
    }
  };

  return (
    <div className="layer-card">
      <div className="layer-header">
        <div>
          <strong>{index + 1}. {layer.type}</strong>
          <div className="param-count">
            🧮 Params: <strong>{paramCount.toLocaleString()}</strong>
          </div>
        </div>

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
    const template = LAYER_TEMPLATES[type] ? JSON.parse(JSON.stringify(LAYER_TEMPLATES[type])) : {
      in_features: 1, out_features: 1
    };
    setLayers(prev => {
      const newLayer = { type, params: template };
      return propagateChannels([...prev, newLayer]);
    });
  };

  const LayersThatDontChangeChannels = (layer) => {
    const types = ["ReLU", "LeakyReLU", "PReLU", "Sigmoid", "Tanh", "Softmax",
      "BatchNorm1d", "BatchNorm2d", "LayerNorm", "InstanceNorm1D", "InstanceNorm2D",
      "GroupNorm", "Dropout"];
    return types.includes(layer.type);
  }

  const ActivationLayers = (layer) => {
    const types = ["ReLU", "LeakyReLU", "PReLU", "Sigmoid", "Tanh", "Softmax"];
    return types.includes(layer.type);
  }

  const LastLayerIsActivation = (layers) => {
    if (layers.length === 0) return false;
    return ActivationLayers(layers[layers.length - 1]);
  }

  function propagateChannels(layers) {
    console.log("Prev:");
    console.log(layers);
    
    let result = layers.map((layer, i) => {
      if (i === 0) return layer;

      const prev = layers[i - 1].params.out_features;

      return {
        ...layer,
        params: {
          ...layer.params,
          in_features: prev,
          out_features: LayersThatDontChangeChannels(layer)
            ? prev
            : layer.params.out_features
        }
      };
    });

    for (let iterator = 0; iterator < 10; iterator++) {
      for (let iterator = 0; iterator < result.length; iterator++) {
        for (let i = 1; i < result.length; i++) {
          result[i] = {
            ...result[i],
            params: {
              ...result[i].params,
              in_features: result[i - 1].params.out_features
            }
          };
        }
      }
    }

    console.log("Result:");
    console.log(result);

    return result
  }

  const updateLayer = (idx, newLayer) => {
    setLayers(prev => {
      const updated = prev.map((l, i) => i === idx ? newLayer : l);
      return propagateChannels(updated);
    });
  };

  const deleteLayer = (idx) => {
    setLayers(prev => {
      const updated = prev.filter((_, i) => i !== idx);
      return propagateChannels(updated);
    });
  };

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
            <button className="btn btn-primary" onClick={() => pushLayer("Conv2DTranspose")}>Add Conv2DTranspose</button>
            <button className="btn btn-primary" onClick={() => pushLayer("Upsample")}>Add Upsample</button>
            <button className="btn btn-primary" onClick={() => pushLayer("MaxPool2D")}>Add MaxPool2D</button>
            <button className="btn btn-primary" onClick={() => pushLayer("AvgPool2D")}>Add AvgPool2D</button>
            <button className="btn btn-primary" onClick={() => pushLayer("ConvolutionalAttention")}>Add Convolutional Attention</button>
            <button className="btn btn-primary" onClick={() => pushLayer("SEBlock")}>Add SE Block</button>
            <button className="btn btn-primary" onClick={() => pushLayer("ResidualBlock")}>Add Residual Block</button>
            <button className="btn btn-primary" onClick={() => pushLayer("PixelShuffle")}>Add PixelShuffle</button>
          </>
        );
      case "activations":
        return (
          <>
            <button className="btn btn-primary" disabled={LastLayerIsActivation(layers)} onClick={() => pushLayer("ReLU")}>Add ReLU</button>
            <button className="btn btn-primary" disabled={LastLayerIsActivation(layers)} onClick={() => pushLayer("LeakyReLU")}>Add LeakyReLU</button>
            <button className="btn btn-primary" disabled={LastLayerIsActivation(layers)} onClick={() => pushLayer("PReLU")}>Add PReLU</button>
            <button className="btn btn-primary" disabled={LastLayerIsActivation(layers)} onClick={() => pushLayer("Sigmoid")}>Add Sigmoid</button>
            <button className="btn btn-primary" disabled={LastLayerIsActivation(layers)} onClick={() => pushLayer("Tanh")}>Add Tanh</button>
            <button className="btn btn-primary" disabled={LastLayerIsActivation(layers)} onClick={() => pushLayer("Softmax")}>Add Softmax</button>
          </>
        );
      case "normalizations":
        return (
          <>
            <button className="btn btn-primary" onClick={() => pushLayer("BatchNorm1d")}>Add BatchNorm1d</button>
            <button className="btn btn-primary" onClick={() => pushLayer("BatchNorm2d")}>Add BatchNorm2d</button>
            <button className="btn btn-primary" onClick={() => pushLayer("LayerNorm")}>Add LayerNorm</button>
            <button className="btn btn-primary" onClick={() => pushLayer("InstanceNorm1D")}>Add InstanceNorm1D</button>
            <button className="btn btn-primary" onClick={() => pushLayer("InstanceNorm2D")}>Add InstanceNorm2D</button>
            <button className="btn btn-primary" onClick={() => pushLayer("GroupNorm")}>Add GroupNorm</button>
            <button className="btn btn-primary" onClick={() => pushLayer("Dropout")}>Add Dropout</button>
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
