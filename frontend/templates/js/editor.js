// editor.js
const { useState } = React;

function LayerCard({ layer, index, removeLayer }) {
  return (
    <div className="layer-card">
      <strong>{index + 1}. {layer.type}</strong>
      <pre>{JSON.stringify(layer.params, null, 2)}</pre>
      <button className="btn btn-danger btn-sm" onClick={() => removeLayer(index)}>Delete</button>
    </div>
  );
}

function Editor() {
  const [layers, setLayers] = useState([]);

  const addLayer = (type) => {
    let params = {};
    switch (type) {
      case "Linear":
        params = { in_features: 0, out_features: 0 };
        break;
      case "Conv2D":
        params = { in_channels: 0, out_channels: 0, kernel_size: 3, stride: 1, padding: 0 };
        break;
      case "ReLU":
      case "LeakyReLU":
      case "PReLU":
      case "Sigmoid":
      case "Tanh":
      case "Softmax":
        params = {};
        break;
      case "BatchNorm1d":
      case "BatchNorm2d":
      case "LayerNorm":
        params = {};
        break;
      default:
        params = {};
    }
    setLayers([...layers, { type, params }]);
  };

  const removeLayer = (index) => {
    setLayers(layers.filter((_, i) => i !== index));
  };

  return (
    <div>
      <div id="add-layer-buttons">
        <button onClick={() => addLayer("Linear")} className="btn btn-primary">Add Linear</button>
        <button onClick={() => addLayer("Conv2D")} className="btn btn-primary">Add Conv2D</button>
        <button onClick={() => addLayer("ReLU")} className="btn btn-primary">Add ReLU</button>
        <button onClick={() => addLayer("LeakyReLU")} className="btn btn-primary">Add LeakyReLU</button>
        <button onClick={() => addLayer("PReLU")} className="btn btn-primary">Add PReLU</button>
        <button onClick={() => addLayer("Sigmoid")} className="btn btn-primary">Add Sigmoid</button>
        <button onClick={() => addLayer("Tanh")} className="btn btn-primary">Add Tanh</button>
        <button onClick={() => addLayer("Softmax")} className="btn btn-primary">Add Softmax</button>
        <button onClick={() => addLayer("BatchNorm1d")} className="btn btn-primary">Add BatchNorm1d</button>
        <button onClick={() => addLayer("BatchNorm2d")} className="btn btn-primary">Add BatchNorm2d</button>
        <button onClick={() => addLayer("LayerNorm")} className="btn btn-primary">Add LayerNorm</button>
      </div>
      <div id="layers-block">
        {layers.map((layer, index) => (
          <LayerCard key={index} layer={layer} index={index} removeLayer={removeLayer} />
        ))}
      </div>
    </div>
  );
}

document.addEventListener("DOMContentLoaded", () => {
  ReactDOM.render(<Editor />, document.getElementById("editor-block"));
});
