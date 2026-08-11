// The API (FastAPI) returns errors as `{"detail": "..."}` for a plain
// string message (e.g. unknown team) or `{"detail": [{"msg": "...", ...}]}`
// for Pydantic validation errors (e.g. same team twice).
function extractErrorMessage(data) {
    if (typeof data.detail === 'string') {
        return data.detail;
    }
    if (Array.isArray(data.detail) && data.detail.length > 0) {
        return data.detail.map(e => e.msg || JSON.stringify(e)).join('; ');
    }
    return 'Prediction failed';
}

document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('prediction-form');
    const predictBtn = document.getElementById('predict-btn');
    const btnText = predictBtn.querySelector('.btn-text');
    const spinner = predictBtn.querySelector('.spinner');
    const errorMsg = document.getElementById('error-message');
    const resultsSection = document.getElementById('results-section');
    
    const homeTeamSelect = document.getElementById('home-team');
    const awayTeamSelect = document.getElementById('away-team');
    
    let shapChartInstance = null;

    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        const homeTeam = homeTeamSelect.value;
        const awayTeam = awayTeamSelect.value;

        // Reset state
        errorMsg.classList.add('hidden');
        resultsSection.classList.add('hidden');
        
        if (homeTeam === awayTeam) {
            errorMsg.textContent = "Please select two different teams.";
            errorMsg.classList.remove('hidden');
            return;
        }

        // Loading state
        predictBtn.disabled = true;
        btnText.textContent = "Running prediction...";
        spinner.classList.remove('hidden');

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ home_team: homeTeam, away_team: awayTeam })
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(extractErrorMessage(data));
            }

            renderResults(data);
            
        } catch (error) {
            errorMsg.textContent = error.message;
            errorMsg.classList.remove('hidden');
        } finally {
            // End loading state
            predictBtn.disabled = false;
            btnText.textContent = "Predict Outcome";
            spinner.classList.add('hidden');
        }
    });

    function renderResults(data) {
        // Winner Banner
        const banner = document.getElementById('winner-banner');
        const isHomeWinner = data.predicted_winner === data.home_team;
        const winnerText = `🏆 Predicted Winner: ${data.predicted_winner} (${isHomeWinner ? 'Home' : 'Away'})`;
        banner.innerHTML = `<h2>${winnerText}</h2>`;

        // Metrics
        document.getElementById('home-prob-title').textContent = `${data.home_team} Win Prob`;
        document.getElementById('away-prob-title').textContent = `${data.away_team} Win Prob`;
        
        document.getElementById('home-prob-value').textContent = `${(data.home_win_prob * 100).toFixed(1)}%`;
        document.getElementById('home-prob-value').style.color = 'var(--accent-home)';
        
        document.getElementById('away-prob-value').textContent = `${(data.away_win_prob * 100).toFixed(1)}%`;
        document.getElementById('away-prob-value').style.color = 'var(--accent-away)';
        
        document.getElementById('confidence-value').textContent = `${(data.confidence * 100).toFixed(1)}%`;

        // Chart — shap_explanation is already sorted by |impact| descending
        renderChart(
            data.shap_explanation.map(f => f.shap_value),
            data.shap_explanation.map(f => f.feature),
        );

        // Show results
        resultsSection.classList.remove('hidden');
        
        // Scroll to results smoothly
        setTimeout(() => {
            resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }, 100);
    }

    function renderChart(shapValues, featureNames) {
        // Pair and sort by absolute SHAP value (top 10)
        let pairs = shapValues.map((val, idx) => ({
            val: val,
            absVal: Math.abs(val),
            name: featureNames[idx]
        }));
        
        pairs.sort((a, b) => b.absVal - a.absVal);
        pairs = pairs.slice(0, 10);
        
        // Prepare chart data
        const labels = pairs.map(p => p.name);
        const data = pairs.map(p => p.val);
        const backgroundColors = data.map(v => v > 0 ? 'rgba(247, 148, 29, 0.8)' : 'rgba(29, 110, 247, 0.8)');
        const borderColors = data.map(v => v > 0 ? 'rgba(247, 148, 29, 1)' : 'rgba(29, 110, 247, 1)');

        const ctx = document.getElementById('shapChart').getContext('2d');
        
        if (shapChartInstance) {
            shapChartInstance.destroy();
        }

        Chart.defaults.color = '#94a3b8';
        Chart.defaults.font.family = "'Outfit', sans-serif";

        shapChartInstance = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [{
                    label: 'SHAP Value',
                    data: data,
                    backgroundColor: backgroundColors,
                    borderColor: borderColors,
                    borderWidth: 1,
                    borderRadius: 4,
                }]
            },
            options: {
                indexAxis: 'y',
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        backgroundColor: 'rgba(11, 15, 25, 0.9)',
                        titleColor: '#fff',
                        bodyColor: '#cbd5e1',
                        borderColor: 'rgba(255,255,255,0.1)',
                        borderWidth: 1,
                        padding: 12,
                        callbacks: {
                            label: function(context) {
                                let label = context.dataset.label || '';
                                if (label) { label += ': '; }
                                if (context.parsed.x !== null) {
                                    let v = context.parsed.x;
                                    label += (v > 0 ? '+' : '') + v.toFixed(4) + (v > 0 ? ' (Favors Home)' : ' (Favors Away)');
                                }
                                return label;
                            }
                        }
                    }
                },
                scales: {
                    x: {
                        grid: { color: 'rgba(255, 255, 255, 0.05)' },
                        title: { display: true, text: 'SHAP Value (impact on home win probability)', color: '#94a3b8' }
                    },
                    y: {
                        grid: { display: false }
                    }
                }
            }
        });
    }
});
