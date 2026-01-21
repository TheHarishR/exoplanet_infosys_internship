document.getElementById("predictForm").addEventListener("submit", async function (e) {
    e.preventDefault();

    const data = {
        planet_temp_z: document.getElementById("planet_temp_z").value,
        planet_size_z: document.getElementById("planet_size_z").value,
        star_temp_z: document.getElementById("star_temp_z").value,
        star_energy_z: document.getElementById("star_energy_z").value
    };

    const response = await fetch("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(data)
    });

    const result = await response.json();

    document.getElementById("score").innerText = result.habitability_score;
    document.getElementById("prediction").innerText = result.prediction;

    document.getElementById("result").classList.remove("hidden");
});
