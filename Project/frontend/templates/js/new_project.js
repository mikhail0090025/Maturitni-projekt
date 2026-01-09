document.getElementById("create_project_button").addEventListener("click", async () => {
    try {
        const inputType = document.getElementById("input_type").value;
        const outputType = document.getElementById("output_type").value;
        console.log(inputType, outputType);

        const inputResp = await fetch("/data_type_to_index/" + inputType, {
            method: "GET",
            credentials: "include"
        });
        const outputResp = await fetch("/data_type_to_index/" + outputType, {
            method: "GET",
            credentials: "include"
        });
        const inputData = await inputResp.json();
        const outputData = await outputResp.json();

        console.log(inputData);
        console.log(outputData);

        var query_body = JSON.stringify({
            name: document.getElementById("project_name").value,
            description: document.getElementById("project_description").value,
            /*
            input_type: parseInt(inputData.index),
            output_type: parseInt(outputData.index),
            */
            input_type: inputType,
            output_type: outputType,
            owner_username: document.getElementById("owner_username").value
        });
        console.log(query_body);
        const response = await fetch("/new_project", {
            method: "POST",
            body: query_body,
            credentials: "include"
        });
        if (response.ok) {
            window.location.href = "/profile_page";
        } else {
            console.error("Project creation failed: " + response.statusText);
        }
    } catch (err) {
        console.error("Service unavailable: " + err.message);
    }
});