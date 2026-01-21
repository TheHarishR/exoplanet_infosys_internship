const sliders = [
  ["planet_temp_z","v1"],
  ["planet_size_z","v2"],
  ["star_temp_z","v3"],
  ["star_energy_z","v4"]
];

sliders.forEach(([s,v]) => {
  document.getElementById(s).oninput = () =>
    document.getElementById(v).innerText =
    document.getElementById(s).value;
});

async function predict() {
  const data = {
    planet_temp_z: planet_temp_z.value,
    planet_size_z: planet_size_z.value,
    star_temp_z: star_temp_z.value,
    star_energy_z: star_energy_z.value
  };

  const res = await fetch("/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data)
  });

  const r = await res.json();

  document.getElementById("result").innerHTML =
    `<h3>Result</h3>
     <p><b>Habitability Score:</b> ${r.habitability_score}</p>
     <p><b>Status:</b> ${r.prediction}</p>`;
}
