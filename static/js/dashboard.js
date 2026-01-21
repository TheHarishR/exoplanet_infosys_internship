// LINE CHART – CLEAN & SMOOTH
new Chart(document.getElementById("trendChart"), {
  type: "line",
  data: {
    labels: ["Run 1", "Run 2", "Run 3", "Run 4", "Run 5"],
    datasets: [{
      label: "Habitability Score",
      data: predictionData,
      borderColor: "#7b8cff",
      backgroundColor: "rgba(123,140,255,0.25)",
      fill: true,
      tension: 0.45,
      pointRadius: 4
    }]
  },
  options: {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: { labels: { color: "#cfd8ff" } }
    },
    scales: {
      x: { ticks: { color: "#cfd8ff" } },
      y: {
        min: 0,
        max: 1,
        ticks: { color: "#cfd8ff" }
      }
    }
  }
});

// DOUGHNUT – PROPER SIZE
new Chart(document.getElementById("confidenceChart"), {
  type: "doughnut",
  data: {
    labels: ["Habitable", "Non-Habitable"],
    datasets: [{
      data: [lastScore * 100, 100 - lastScore * 100],
      backgroundColor: ["#6efcff", "#2a2f9c"],
      borderWidth: 2
    }]
  },
 options: {
  responsive: true,
  maintainAspectRatio: false,
  cutout: "70%",
  plugins: {
    legend: {
      position: "bottom",
      labels: {
        color: "#cfd8ff",
        padding: 20
      }
    }
  }
}
});
