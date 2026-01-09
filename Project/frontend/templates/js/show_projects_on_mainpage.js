const { useState, useEffect } = React;

const MyProjects = () => {
    const [projects, setProjects] = useState([]);

    useEffect(() => {
        fetch("/my_projects", { credentials: "include" })
            .then(res => res.ok ? res.json() : [])
            .then(data => setProjects(data))
            .catch(() => setProjects([]));
    }, []);

    console.log(projects);

    if (projects.length === 0) {
        return <div style={{ fontSize: "0.9rem", color: "#666" }}>No projects yet</div>;
    }

    return (
        <div style={{
            display: "flex",
            flexDirection: "column",
            gap: "6px",
            padding: "8px",
            border: "1px solid #ddd",
            borderRadius: "8px",
            backgroundColor: "#fafafa",
            maxHeight: "120px",
            overflowY: "auto"
        }}>
            {projects.map((p, i) => (
                <div key={i}>
                <a
                    href={`/projects/${p.id}`}
                    style={{
                        padding: "6px 10px",
                        borderRadius: "6px",
                        backgroundColor: "#fff",
                        boxShadow: "0 1px 3px rgba(0,0,0,0.1)",
                        textDecoration: "none",
                        color: "#2563eb",
                        fontWeight: "500",
                        fontSize: "0.9rem",
                        transition: "background 0.2s"
                    }}
                    onMouseOver={(e) => e.currentTarget.style.backgroundColor = "#f0f4ff"}
                    onMouseOut={(e) => e.currentTarget.style.backgroundColor = "#fff"}
                >
                    {p.name}
                </a>
                <button className="btn btn-danger btn-sm delete-project-btn" data-id={p.id}>Delete</button>
                </div>
            ))}
        </div>
    );
};

ReactDOM.createRoot(document.getElementById("projects")).render(<MyProjects />);