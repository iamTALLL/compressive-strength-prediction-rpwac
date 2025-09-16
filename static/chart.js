document.addEventListener('DOMContentLoaded', function() {
    const canvases = document.querySelectorAll('canvas[id^="chart_"]');

    canvases.forEach((canvas) => {
        const logFile = canvas.getAttribute('data-log-file');
        const ctx = canvas.getContext('2d');

        fetch(`/chart_data/${logFile}`)
            .then(res => res.json())
            .then(data => {
                if (data.error) {
                    ctx.font = "16px Arial";
                    ctx.fillStyle = "red";
                    ctx.fillText(data.error, 10, 50);
                    return;
                }

                const predictedPoints = [];
                let actualPoints = [];

                data.forEach((row, idx) => {
                    predictedPoints.push({
                        x: idx + 1,
                        y: row.predicted,
                        info: row
                    });
                    // chưa push actualPoints lúc này
                });

                const chart = new Chart(ctx, {
                    type: 'scatter',
                    data: {
                        datasets: [
                            {
                                label: 'Predicted',
                                data: predictedPoints,
                                backgroundColor: 'green',
                                pointRadius: 6
                            }
                        ]
                    },
                    options: {
                        responsive: true,
                        plugins: {
                            legend: { position: 'top' },
                            tooltip: {
                                callbacks: {
                                    label: function(context) {
                                        const info = context.raw.info;
                                        let label = `Predicted: ${context.raw.y}`;
                                        // show 1 số input features
                                        label += ` | Cement: ${info.cement_dosage}, Water: ${info.water}`;
                                        return label;
                                    }
                                }
                            }
                        },
                        scales: {
                            x: {
                                title: { display: true, text: 'Sample' },
                                ticks: { stepSize: 1 }
                            },
                            y: {
                                title: { display: true, text: 'Concrete Compressive Strength (MPa)' }
                            }
                        }
                    }
                });

                // Nếu y_true đã có, vẽ thêm actualPoints + đường x=y
                if (data[0].y_true !== undefined) {
                    actualPoints = data.map((row, idx) => ({
                        x: idx + 1,
                        y: row.y_true,
                        info: row
                    }));

                    // thêm dataset Actual
                    chart.data.datasets.push({
                        label: 'Actual',
                        data: actualPoints,
                        backgroundColor: 'red',
                        pointRadius: 6
                    });

                    // thêm line x=y
                    const maxVal = Math.max(
                        ...predictedPoints.map(p => p.y),
                        ...actualPoints.map(p => p.y)
                    );
                    chart.data.datasets.push({
                        label: 'y = x',
                        data: [
                            {x: 0, y: 0},
                            {x: maxVal, y: maxVal}
                        ],
                        type: 'line',
                        borderColor: 'gray',
                        borderWidth: 1,
                        fill: false,
                        pointRadius: 0,
                        borderDash: [5,5]
                    });

                    chart.update();
                }

            })
            .catch(err => console.error('Error loading chart data:', err));
    });
});
