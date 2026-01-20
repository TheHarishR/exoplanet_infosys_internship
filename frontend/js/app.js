// EXOPLANET HABITABILITY EXPLORER - COMPLETE ANALYTICS EDITION
// Version 3.1 - Fixed Global Analytics
// IMPORTANT: This should match your Flask server port
const API_URL = "https://exoplanet-habitability-lqlj.onrender.com";

let currentPredictionData = null;
let batchPredictionData = null;
let currentRankingsData = null;

console.log('%c🌌 EXOPLANET EXPLORER - ANALYTICS EDITION', 'font-size:20px; color:#00d4ff; font-weight:bold;');

document.addEventListener('DOMContentLoaded', function() {
    console.log('✅ Initializing...');
    initializeNavigation();
    initializeTooltips();
    initializeRangeSliders();
    initializeForms();
    initializeModeSelector();
    initializeCSVUpload();
    initializeExportButtons();
    initializeAnalytics();
    console.log('✅ Ready!');
});

// ========== ANALYTICS ===========
function initializeAnalytics() {
    const btn = document.getElementById('loadGlobalAnalytics');
    if (btn) btn.addEventListener('click', loadGlobalAnalytics);
}

async function loadGlobalAnalytics() {
    console.log('📊 Loading global analytics...');
    
    // Show loading state
    const btn = document.getElementById('loadGlobalAnalytics');
    const originalText = btn.querySelector('span').textContent;
    btn.querySelector('span').textContent = '⏳ Loading...';
    btn.disabled = true;
    
    try {
        // First check if backend is running
        const healthCheck = await fetch(`${API_BASE_URL}/health`).catch(() => null);
        
        if (!healthCheck || !healthCheck.ok) {
            throw new Error('Backend server not running. Please start Flask with: python app.py');
        }
        
        const healthData = await healthCheck.json();
        console.log('Health check:', healthData);
        
        // Check if there are planets in the database
        if (healthData.total_planets === 0) {
            throw new Error('No planets in database yet. Please add planets using "Add Planet" section or upload CSV in "Predict" section first.');
        }
        
        // Load analytics data
        const [featureImportanceRes, distributionRes, correlationsRes] = await Promise.all([
            fetch(`${API_BASE_URL}/analytics/feature_importance`),
            fetch(`${API_BASE_URL}/analytics/distribution`),
            fetch(`${API_BASE_URL}/analytics/correlations`)
        ]);
        
        // Check responses
        if (!featureImportanceRes.ok || !distributionRes.ok || !correlationsRes.ok) {
            throw new Error('Failed to load analytics data. Check Flask console for errors.');
        }
        
        // Parse responses
        const featureImportance = await featureImportanceRes.json();
        const distribution = await distributionRes.json();
        const correlations = await correlationsRes.json();
        
        console.log('Feature Importance:', featureImportance);
        console.log('Distribution:', distribution);
        console.log('Correlations:', correlations);
        
        // Display charts
        if (featureImportance.status === 'success' && featureImportance.data) {
            displayFeatureImportance(featureImportance.data);
        } else {
            console.error('Feature importance data missing');
        }
        
        if (distribution.status === 'success' && distribution.data) {
            displayDistribution(distribution.data);
        } else {
            console.error('Distribution data missing');
        }
        
        if (correlations.status === 'success' && correlations.data) {
            displayCorrelations(correlations.data);
        } else {
            console.error('Correlations data missing');
        }
        
        // Show analytics content
        document.getElementById('analyticsContent').style.display = 'block';
        
        // Reset button
        btn.querySelector('span').textContent = originalText;
        btn.disabled = false;
        
        console.log('✅ Analytics loaded successfully!');
        
    } catch (error) {
        console.error('❌ Analytics error:', error);
        
        let errorMessage = error.message;
        
        // Detect specific errors
        if (error instanceof TypeError && error.message.includes('Failed to fetch')) {
            errorMessage = 'Backend server not running. Start with: python app.py (in the terminal)';
        } else if (error instanceof SyntaxError && error.message.includes('JSON')) {
            errorMessage = 'Backend returning invalid response. Check Flask terminal for errors.';
        }
        
        // Show user-friendly error using correct element ID
        showError('predictErrorMessage', errorMessage);
        
        // Also display in analytics section
        const analyticsContent = document.getElementById('analyticsContent');
        if (analyticsContent) {
            analyticsContent.innerHTML = `
                <div class="glass-card" style="text-align: center; padding: 3rem;">
                    <div style="font-size: 3rem; margin-bottom: 1rem;">⚠️</div>
                    <h3 style="color: #ff3b5c; margin-bottom: 1rem;">Unable to Load Analytics</h3>
                    <p style="color: #b4b4ff; font-size: 1.1rem; margin-bottom: 2rem;">${errorMessage}</p>
                    <div style="background: rgba(0, 212, 255, 0.1); border-left: 4px solid #00d4ff; padding: 1.5rem; border-radius: 8px; text-align: left;">
                        <div style="font-weight: 700; color: #00d4ff; margin-bottom: 0.5rem;">💡 Troubleshooting Steps:</div>
                        <ol style="color: #b4b4ff; line-height: 1.8;">
                            <li>Make sure Flask backend is running: <code style="background: rgba(0,0,0,0.3); padding: 0.25rem 0.5rem; border-radius: 4px;">python app.py</code></li>
                            <li>Check that the model file exists: <code style="background: rgba(0,0,0,0.3); padding: 0.25rem 0.5rem; border-radius: 4px;">habitability_model.pkl</code></li>
                            <li>Add planets to the database first (use "Add Planet" or upload CSV)</li>
                            <li>Verify Flask is running on port 5000 (check terminal output)</li>
                            <li>Check browser console (F12) for detailed error messages</li>
                        </ol>
                    </div>
                </div>
            `;
            analyticsContent.style.display = 'block';
        }
        
        // Reset button
        btn.querySelector('span').textContent = originalText;
        btn.disabled = false;
    }
}

function displayFeatureImportance(data) {
    if (!data || data.length === 0) {
        document.getElementById('featureImportanceChart').innerHTML = 
            '<div style="text-align:center;padding:3rem;color:#b4b4ff;">No feature importance data available</div>';
        return;
    }
    
    const trace = {
        x: data.map(d => d.importance),
        y: data.map(d => d.feature_label),
        type: 'bar',
        orientation: 'h',
        marker: {
            color: data.map(d => d.importance),
            colorscale: [[0, '#8b5cf6'], [0.5, '#00d4ff'], [1, '#00ff88']],
            line: { color: '#00d4ff', width: 2 }
        },
        text: data.map(i => (i.importance * 100).toFixed(1) + '%'),
        textposition: 'outside'
    };
    
    const layout = {
        title: { text: 'Feature Importance', font: { family: 'Orbitron', size: 20, color: '#00d4ff' }},
        xaxis: { title: 'Importance', gridcolor: 'rgba(0, 212, 255, 0.1)', color: '#b4b4ff' },
        yaxis: { gridcolor: 'rgba(0, 212, 255, 0.1)', color: '#b4b4ff' },
        plot_bgcolor: 'rgba(0, 0, 0, 0.3)',
        paper_bgcolor: 'transparent',
        font: { family: 'Rajdhani', color: '#b4b4ff' },
        margin: { l: 200, r: 50, t: 80, b: 80 },
        height: 500
    };
    
    Plotly.newPlot('featureImportanceChart', [trace], layout, { responsive: true, displayModeBar: false });
    
    const top = data[0];
    document.getElementById('featureImportanceInsight').innerHTML = `
        <div class="insight-title">🔍 Key Insight</div>
        <div class="insight-text"><strong>${top.feature_label}</strong> is most important (${(top.importance * 100).toFixed(1)}%)</div>
    `;
}

function displayDistribution(data) {
    if (!data.scores || data.scores.length === 0) {
        document.getElementById('distributionChart').innerHTML = 
            '<div style="text-align:center;padding:3rem;color:#b4b4ff;">No distribution data available. Add planets to the database first.</div>';
        document.getElementById('distributionInsight').innerHTML = `
            <div class="insight-title">📊 No Data</div>
            <div class="insight-text">Add planets using "Add Planet" or upload CSV to see distribution.</div>
        `;
        return;
    }
    
    const trace = {
        x: data.scores,
        type: 'histogram',
        nbinsx: 20,
        marker: { color: '#00d4ff', line: { color: '#00ff88', width: 2 }}
    };
    
    const layout = {
        title: { text: 'Score Distribution', font: { family: 'Orbitron', size: 20, color: '#00d4ff' }},
        xaxis: { title: 'Habitability Score', gridcolor: 'rgba(0, 212, 255, 0.1)', color: '#b4b4ff', range: [0, 1] },
        yaxis: { title: 'Count', gridcolor: 'rgba(0, 212, 255, 0.1)', color: '#b4b4ff' },
        plot_bgcolor: 'rgba(0, 0, 0, 0.3)',
        paper_bgcolor: 'transparent',
        font: { family: 'Rajdhani', color: '#b4b4ff' },
        margin: { l: 80, r: 50, t: 80, b: 80 },
        height: 500
    };
    
    Plotly.newPlot('distributionChart', [trace], layout, { responsive: true, displayModeBar: false });
    
    const hab = data.scores.filter(s => s > 0.5).length;
    const pct = (hab / data.total_planets * 100).toFixed(1);
    document.getElementById('distributionInsight').innerHTML = `
        <div class="insight-title">📈 Statistics</div>
        <div class="insight-text">${hab} of ${data.total_planets} (${pct}%) are habitable. Avg: ${(data.mean_score * 100).toFixed(1)}%</div>
    `;
}

function displayCorrelations(data) {
    if (!data || data.length === 0) {
        document.getElementById('correlationChart').innerHTML = 
            '<div style="text-align:center;padding:3rem;color:#b4b4ff;">No correlation data available. Add more planets to see correlations.</div>';
        document.getElementById('correlationInsight').innerHTML = `
            <div class="insight-title">📊 No Data</div>
            <div class="insight-text">Add at least 2 planets to calculate correlations.</div>
        `;
        return;
    }
    
    const colors = data.map(c => c.correlation > 0.3 ? '#00ff88' : c.correlation < -0.3 ? '#ff3b5c' : '#00d4ff');
    
    const trace = {
        x: data.map(d => d.correlation),
        y: data.map(d => d.feature_label),
        type: 'bar',
        orientation: 'h',
        marker: { color: colors, line: { color: '#ffffff', width: 1 }},
        text: data.map(c => c.correlation.toFixed(3)),
        textposition: 'outside'
    };
    
    const layout = {
        title: { text: 'Correlations', font: { family: 'Orbitron', size: 20, color: '#00d4ff' }},
        xaxis: { title: 'Correlation', gridcolor: 'rgba(0, 212, 255, 0.1)', color: '#b4b4ff', range: [-1, 1], zeroline: true, zerolinecolor: '#ffffff', zerolinewidth: 2 },
        yaxis: { gridcolor: 'rgba(0, 212, 255, 0.1)', color: '#b4b4ff' },
        plot_bgcolor: 'rgba(0, 0, 0, 0.3)',
        paper_bgcolor: 'transparent',
        font: { family: 'Rajdhani', color: '#b4b4ff' },
        margin: { l: 200, r: 50, t: 80, b: 80 },
        height: 500
    };
    
    Plotly.newPlot('correlationChart', [trace], layout, { responsive: true, displayModeBar: false });
    
    const top = data[0];
    const type = top.correlation > 0 ? 'positive' : 'negative';
    document.getElementById('correlationInsight').innerHTML = `
        <div class="insight-title">🔗 Insights</div>
        <div class="insight-text"><strong>${top.feature_label}</strong> has strongest ${type} correlation (${top.correlation.toFixed(3)})</div>
    `;
}

async function displayIndividualAnalytics(planetData) {
    const features = ['Stellar Temp', 'Stellar Radius', 'Stellar Mass', 'Metallicity', 'Luminosity', 'Orbital Period', 'Eccentricity', 'Insolation'];
    const values = [
        planetData.st_teff / 15000,
        planetData.st_rad / 10,
        planetData.st_mass / 20,
        (planetData.st_met + 2) / 3,
        Math.min(planetData.st_luminosity / 10, 1),
        planetData.pl_orbper / 1000,
        planetData.pl_orbeccen,
        Math.min(planetData.pl_insol / 10, 1)
    ];
    
    const trace = {
        type: 'scatterpolar',
        r: values,
        theta: features,
        fill: 'toself',
        fillcolor: 'rgba(0, 212, 255, 0.3)',
        line: { color: '#00d4ff', width: 3 },
        marker: { color: '#00ff88', size: 8 }
    };
    
    const layout = {
        polar: {
            bgcolor: 'rgba(0, 0, 0, 0.3)',
            radialaxis: { visible: true, range: [0, 1], gridcolor: 'rgba(0, 212, 255, 0.2)', color: '#b4b4ff' },
            angularaxis: { gridcolor: 'rgba(0, 212, 255, 0.2)', color: '#00d4ff' }
        },
        paper_bgcolor: 'transparent',
        font: { family: 'Rajdhani', color: '#b4b4ff', size: 12 },
        margin: { l: 80, r: 80, t: 80, b: 80 },
        height: 500,
        showlegend: false
    };
    
    Plotly.newPlot('individualRadarChart', [trace], layout, { responsive: true, displayModeBar: false });
    
    displayParameterBreakdown(planetData);
    document.getElementById('individualAnalytics').style.display = 'block';
}

function displayParameterBreakdown(planetData) {
    const params = [
        { name: 'Stellar Temperature', value: planetData.st_teff, unit: 'K', optimal: 5500 },
        { name: 'Stellar Radius', value: planetData.st_rad, unit: 'R☉', optimal: 1 },
        { name: 'Stellar Mass', value: planetData.st_mass, unit: 'M☉', optimal: 1 },
        { name: 'Metallicity', value: planetData.st_met, unit: 'dex', optimal: 0 },
        { name: 'Luminosity', value: planetData.st_luminosity, unit: 'L☉', optimal: 1 },
        { name: 'Orbital Period', value: planetData.pl_orbper, unit: 'days', optimal: 365 },
        { name: 'Eccentricity', value: planetData.pl_orbeccen, unit: '', optimal: 0.017 },
        { name: 'Insolation', value: planetData.pl_insol, unit: 'Earth=1', optimal: 1 }
    ];
    
    let html = '<div class="parameter-list">';
    params.forEach(p => {
        const dev = Math.abs(p.value - p.optimal) / p.optimal;
        let status = 'good', text = 'Optimal', color = '#00ff88';
        if (dev > 0.5) { status = 'poor'; text = 'Sub-optimal'; color = '#ff3b5c'; }
        else if (dev > 0.2) { status = 'moderate'; text = 'Moderate'; color = '#ffd700'; }
        
        html += `<div class="parameter-item" style="background:rgba(0,212,255,0.05);border-left:4px solid ${color};padding:1rem;margin-bottom:0.5rem;border-radius:8px;display:flex;justify-content:space-between">
            <div>
                <div style="font-weight:700;color:#00d4ff">${p.name}</div>
                <div style="color:#b4b4ff;font-size:0.9rem">${p.value.toFixed(3)} ${p.unit} <span style="color:#7a7a9f">(Optimal: ${p.optimal} ${p.unit})</span></div>
            </div>
            <div style="background:rgba(${color==='#00ff88'?'0,255,136':color==='#ffd700'?'255,215,0':'255,59,92'},0.2);color:${color};padding:0.5rem 1rem;border-radius:20px;font-weight:700;font-size:0.85rem;border:1px solid ${color}">${text}</div>
        </div>`;
    });
    html += '</div>';
    document.getElementById('parameterBreakdown').innerHTML = html;
}

async function displayBatchAnalytics(predictions) {
    const scores = predictions.map(p => p.score);
    
    const trace = {
        x: scores,
        type: 'histogram',
        nbinsx: 15,
        marker: { color: '#00d4ff', line: { color: '#00ff88', width: 2 }}
    };
    
    const layout = {
        title: { text: 'Batch Score Distribution', font: { family: 'Orbitron', size: 18, color: '#00d4ff' }},
        xaxis: { title: 'Score', gridcolor: 'rgba(0, 212, 255, 0.1)', color: '#b4b4ff' },
        yaxis: { title: 'Count', gridcolor: 'rgba(0, 212, 255, 0.1)', color: '#b4b4ff' },
        plot_bgcolor: 'rgba(0, 0, 0, 0.3)',
        paper_bgcolor: 'transparent',
        font: { family: 'Rajdhani', color: '#b4b4ff' },
        margin: { l: 60, r: 40, t: 80, b: 60 },
        height: 400
    };
    
    Plotly.newPlot('batchDistributionChart', [trace], layout, { responsive: true, displayModeBar: false });
    document.getElementById('batchAnalytics').style.display = 'block';
}

// ========== NAVIGATION ==========
function initializeNavigation() {
    document.querySelectorAll('.nav-link, [data-navigate]').forEach(el => {
        el.addEventListener('click', function(e) {
            e.preventDefault();
            const section = this.getAttribute('data-navigate') || this.getAttribute('href').substring(1);
            navigateToSection(section);
        });
    });
}

function navigateToSection(id) {
    document.querySelectorAll('.content-section').forEach(s => s.classList.remove('active'));
    const target = document.getElementById(id);
    if (target) {
        target.classList.add('active');
        if (id === 'predict') setTimeout(() => initializeRangeSlidersForSection('predict'), 150);
        if (id === 'rank') loadRankings();
    }
    document.querySelectorAll('.nav-link').forEach(l => {
        l.classList.remove('active');
        if (l.getAttribute('href') === '#' + id) l.classList.add('active');
    });
    window.scrollTo({ top: 0, behavior: 'smooth' });
}

// ========== FORMS ==========
function initializeForms() {
    const addForm = document.getElementById('addPlanetForm');
    const predForm = document.getElementById('predictForm');
    if (addForm) addForm.addEventListener('submit', handleAddPlanet);
    if (predForm) predForm.addEventListener('submit', handlePredict);
    const refresh = document.getElementById('refreshRankings');
    const topSel = document.getElementById('topRankSelect');
    if (refresh) refresh.addEventListener('click', loadRankings);
    if (topSel) topSel.addEventListener('change', loadRankings);
}

async function handleAddPlanet(e) {
    e.preventDefault();
    const form = e.target;
    const data = {
        planet_name: form.planet_name.value,
        st_teff: parseFloat(form.st_teff.value),
        st_rad: parseFloat(form.st_rad.value),
        st_mass: parseFloat(form.st_mass.value),
        st_met: parseFloat(form.st_met.value),
        st_luminosity: parseFloat(form.st_luminosity.value),
        pl_orbper: parseFloat(form.pl_orbper.value),
        pl_orbeccen: parseFloat(form.pl_orbeccen.value),
        pl_insol: parseFloat(form.pl_insol.value)
    };
    
    try {
        const res = await fetch(`${API_BASE_URL}/add_planet`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data)
        });
        const result = await res.json();
        if (res.ok) {
            document.getElementById('addSuccessMessage').style.display = 'flex';
            form.reset();
            resetFormRanges(form);
            setTimeout(() => document.getElementById('addSuccessMessage').style.display = 'none', 5000);
        } else throw new Error(result.message);
    } catch (error) {
        showError('addErrorMessage', error.message);
    }
}

async function handlePredict(e) {
    e.preventDefault();
    const form = e.target;
    const data = {
        planet_name: form.planet_name.value,
        st_teff: parseFloat(form.st_teff.value),
        st_rad: parseFloat(form.st_rad.value),
        st_mass: parseFloat(form.st_mass.value),
        st_met: parseFloat(form.st_met.value),
        st_luminosity: parseFloat(form.st_luminosity.value),
        pl_orbper: parseFloat(form.pl_orbper.value),
        pl_orbeccen: parseFloat(form.pl_orbeccen.value),
        pl_insol: parseFloat(form.pl_insol.value)
    };
    
    try {
        const res = await fetch(`${API_BASE_URL}/predict`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data)
        });
        const result = await res.json();
        if (res.ok) {
            currentPredictionData = { ...data, ...result.data };
            displayPredictionResults(result.data);
            displayIndividualAnalytics(currentPredictionData);
        } else throw new Error(result.message);
    } catch (error) {
        showError('predictErrorMessage', error.message);
    }
}

function displayPredictionResults(data) {
    document.getElementById('resultPlanetName').textContent = data.planet_name;
    document.getElementById('resultScore').textContent = (data.score * 100).toFixed(1) + '%';
    document.getElementById('resultConfidence').textContent = (data.confidence * 100).toFixed(1) + '%';
    document.getElementById('scoreFill').style.width = (data.score * 100) + '%';
    const isHab = data.habitability === 1;
    const classEl = document.getElementById('resultClass');
    classEl.textContent = isHab ? 'Habitable' : 'Non-Habitable';
    classEl.className = isHab ? 'class-badge class-habitable' : 'class-badge class-non-habitable';
    document.getElementById('predictionResults').style.display = 'block';
    document.getElementById('predictionResults').scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// ========== CSV & BATCH ==========
function initializeModeSelector() {
    document.querySelectorAll('.mode-btn').forEach(btn => {
        btn.addEventListener('click', function() {
            const mode = this.getAttribute('data-mode');
            document.querySelectorAll('.mode-btn').forEach(b => b.classList.remove('active'));
            this.classList.add('active');
            if (mode === 'manual') {
                document.getElementById('manualMode').style.display = 'block';
                document.getElementById('batchMode').style.display = 'none';
            } else {
                document.getElementById('manualMode').style.display = 'none';
                document.getElementById('batchMode').style.display = 'block';
            }
        });
    });
}

function initializeCSVUpload() {
    const uploadArea = document.getElementById('uploadArea');
    const fileInput = document.getElementById('csvFileInput');
    const browseBtn = document.getElementById('browseBtn');
    const removeBtn = document.getElementById('removeFile');
    const processBtn = document.getElementById('processCsvBtn');
    
    browseBtn.addEventListener('click', () => fileInput.click());
    fileInput.addEventListener('change', e => e.target.files.length && handleFileSelect(e.target.files[0]));
    
    uploadArea.addEventListener('dragover', e => { e.preventDefault(); uploadArea.classList.add('dragover'); });
    uploadArea.addEventListener('dragleave', e => { e.preventDefault(); uploadArea.classList.remove('dragover'); });
    uploadArea.addEventListener('drop', e => {
        e.preventDefault();
        uploadArea.classList.remove('dragover');
        if (e.dataTransfer.files.length) handleFileSelect(e.dataTransfer.files[0]);
    });
    
    removeBtn.addEventListener('click', () => {
        fileInput.value = '';
        document.getElementById('fileInfo').style.display = 'none';
        uploadArea.style.display = 'block';
        document.getElementById('batchResults').style.display = 'none';
    });
    
    processBtn.addEventListener('click', processCsvFile);
}

function handleFileSelect(file) {
    if (!file.name.endsWith('.csv')) return showError('predictErrorMessage', 'Please select CSV');
    document.getElementById('fileName').textContent = file.name;
    document.getElementById('fileStats').textContent = `Size: ${(file.size / 1024).toFixed(2)} KB`;
    document.getElementById('uploadArea').style.display = 'none';
    document.getElementById('fileInfo').style.display = 'block';
}

async function processCsvFile() {
    const file = document.getElementById('csvFileInput').files[0];
    if (!file) return showError('predictErrorMessage', 'No file');
    
    const btn = document.getElementById('processCsvBtn');
    const orig = btn.querySelector('span').textContent;
    btn.querySelector('span').textContent = 'Processing...';
    btn.disabled = true;
    
    Papa.parse(file, {
        header: true,
        skipEmptyLines: true,
        complete: async function(results) {
            try {
                const predictions = [];
                for (const row of results.data) {
                    const planetData = {
                        planet_name: row.planet_name || 'Unknown',
                        st_teff: parseFloat(row.st_teff),
                        st_rad: parseFloat(row.st_rad),
                        st_mass: parseFloat(row.st_mass),
                        st_met: parseFloat(row.st_met),
                        st_luminosity: parseFloat(row.st_luminosity),
                        pl_orbper: parseFloat(row.pl_orbper),
                        pl_orbeccen: parseFloat(row.pl_orbeccen),
                        pl_insol: parseFloat(row.pl_insol)
                    };
                    
                    const isValid = Object.values(planetData).every(v => v !== null && !isNaN(v) || typeof v === 'string');
                    if (!isValid) continue;
                    
                    const res = await fetch(`${API_BASE_URL}/predict`, {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify(planetData)
                    });
                    const result = await res.json();
                    if (res.ok) predictions.push({ ...planetData, ...result.data });
                }
                
                batchPredictionData = predictions;
                displayBatchResults(predictions);
                displayBatchAnalytics(predictions);
                btn.querySelector('span').textContent = orig;
                btn.disabled = false;
            } catch (error) {
                showError('predictErrorMessage', error.message);
                btn.querySelector('span').textContent = orig;
                btn.disabled = false;
            }
        }
    });
}

function displayBatchResults(predictions) {
    const habitable = predictions.filter(p => p.habitability === 1).length;
    const avgScore = predictions.reduce((s, p) => s + p.score, 0) / predictions.length;
    
    document.getElementById('totalProcessed').textContent = predictions.length;
    document.getElementById('totalHabitable').textContent = habitable;
    document.getElementById('avgScore').textContent = (avgScore * 100).toFixed(1) + '%';
    
    const tbody = document.getElementById('batchResultsTable');
    tbody.innerHTML = '';
    predictions.forEach((p, i) => {
        const row = document.createElement('tr');
        const isHab = p.habitability === 1;
        row.innerHTML = `
            <td>${i + 1}</td>
            <td>${p.planet_name}</td>
            <td>${(p.score * 100).toFixed(2)}%</td>
            <td><span class="class-badge ${isHab ? 'class-habitable' : 'class-non-habitable'}">${isHab ? 'Habitable' : 'Non-Habitable'}</span></td>
            <td>${(p.confidence * 100).toFixed(1)}%</td>
        `;
        tbody.appendChild(row);
    });
    
    document.getElementById('batchResults').style.display = 'block';
}

// ========== EXPORTS ==========
function initializeExportButtons() {
    const single = document.getElementById('downloadSingleResult');
    const batch = document.getElementById('downloadBatchResults');
    const pdf = document.getElementById('exportPdfBtn');
    const excel = document.getElementById('exportExcelBtn');
    
    if (single) single.addEventListener('click', downloadSingleResult);
    if (batch) batch.addEventListener('click', downloadBatchResults);
    if (pdf) pdf.addEventListener('click', exportRankingsPDF);
    if (excel) excel.addEventListener('click', exportRankingsExcel);
}

function downloadSingleResult() {
    if (!currentPredictionData) return showError('predictErrorMessage', 'No data');
    const d = currentPredictionData;
    const csv = [
        ['Planet Name', 'Stellar Temp (K)', 'Stellar Radius', 'Stellar Mass', 'Metallicity', 'Luminosity', 'Orbital Period', 'Eccentricity', 'Insolation', 'Score', 'Class', 'Confidence'],
        [d.planet_name, d.st_teff, d.st_rad, d.st_mass, d.st_met, d.st_luminosity, d.pl_orbper, d.pl_orbeccen, d.pl_insol, (d.score*100).toFixed(2)+'%', d.habitability===1?'Habitable':'Non-Habitable', (d.confidence*100).toFixed(2)+'%']
    ].map(r => r.join(',')).join('\n');
    downloadCSV(csv, `${d.planet_name}_prediction.csv`);
}

function downloadBatchResults() {
    if (!batchPredictionData?.length) return showError('predictErrorMessage', 'No data');
    const headers = ['Planet Name', 'Stellar Temp (K)', 'Stellar Radius', 'Stellar Mass', 'Metallicity', 'Luminosity', 'Orbital Period', 'Eccentricity', 'Insolation', 'Score', 'Class', 'Confidence'];
    const rows = batchPredictionData.map(p => [p.planet_name, p.st_teff, p.st_rad, p.st_mass, p.st_met, p.st_luminosity, p.pl_orbper, p.pl_orbeccen, p.pl_insol, (p.score*100).toFixed(2)+'%', p.habitability===1?'Habitable':'Non-Habitable', (p.confidence*100).toFixed(2)+'%']);
    const csv = [headers, ...rows].map(r => r.join(',')).join('\n');
    downloadCSV(csv, `batch_predictions_${new Date().toISOString().split('T')[0]}.csv`);
}

function downloadCSV(content, filename) {
    const blob = new Blob([content], { type: 'text/csv' });
    const link = document.createElement('a');
    link.href = URL.createObjectURL(blob);
    link.download = filename;
    link.click();
}

function exportRankingsPDF() {
    if (!currentRankingsData?.length) return showError('rankErrorMessage', 'No data');
    const { jsPDF } = window.jspdf;
    const doc = new jsPDF();
    doc.setFontSize(20);
    doc.setTextColor(0, 212, 255);
    doc.text('Exoplanet Habitability Rankings', 14, 20);
    doc.setFontSize(12);
    doc.setTextColor(100, 100, 100);
    doc.text(`Generated: ${new Date().toLocaleString()}`, 14, 28);
    const tableData = currentRankingsData.map(p => [p.rank, p.planet_name, (p.habitability_score*100).toFixed(2)+'%', getStatusText(p.habitability_score)]);
    doc.autoTable({
        head: [['Rank', 'Planet Name', 'Score', 'Status']],
        body: tableData,
        startY: 35,
        theme: 'grid',
        headStyles: { fillColor: [0, 212, 255] }
    });
    doc.save(`exoplanet_rankings_${new Date().toISOString().split('T')[0]}.pdf`);
}

function exportRankingsExcel() {
    if (!currentRankingsData?.length) return showError('rankErrorMessage', 'No data');
    const ws_data = [['Rank', 'Planet Name', 'Score', 'Status']];
    currentRankingsData.forEach(p => ws_data.push([p.rank, p.planet_name, (p.habitability_score*100).toFixed(2)+'%', getStatusText(p.habitability_score)]));
    const wb = XLSX.utils.book_new();
    const ws = XLSX.utils.aoa_to_sheet(ws_data);
    ws['!cols'] = [{ wch: 8 }, { wch: 30 }, { wch: 18 }, { wch: 15 }];
    XLSX.utils.book_append_sheet(wb, ws, 'Rankings');
    XLSX.writeFile(wb, `exoplanet_rankings_${new Date().toISOString().split('T')[0]}.xlsx`);
}

function getStatusText(score) {
    if (score >= 0.8) return 'Excellent';
    if (score >= 0.6) return 'Good';
    if (score >= 0.4) return 'Moderate';
    return 'Low';
}

// ========== RANKINGS ==========
async function loadRankings() {
    const tbody = document.getElementById('rankingsTableBody');
    const loading = document.getElementById('rankingsLoading');
    const empty = document.getElementById('rankingsEmpty');
    const table = document.getElementById('rankingsTableContainer');
    
    table.style.display = 'none';
    empty.style.display = 'none';
    loading.style.display = 'block';
    
    const topN = document.getElementById('topRankSelect')?.value || 20;
    
    try {
        const res = await fetch(`${API_BASE_URL}/rank?top=${topN}`);
        const data = await res.json();
        if (res.ok) {
            currentRankingsData = data.data;
            if (!data.data.length) {
                loading.style.display = 'none';
                empty.style.display = 'block';
            } else {
                tbody.innerHTML = '';
                data.data.forEach((p, i) => tbody.appendChild(createRankingRow(p, i)));
                loading.style.display = 'none';
                table.style.display = 'block';
            }
        } else throw new Error(data.message);
    } catch (error) {
        loading.style.display = 'none';
        showError('rankErrorMessage', error.message);
    }
}

function createRankingRow(planet, index) {
    const row = document.createElement('tr');
    const rank = planet.rank;
    const score = planet.habitability_score;
    let rankClass = '', medal = '';
    if (rank === 1) { rankClass = 'rank-top-1'; medal = '🥇'; }
    else if (rank === 2) { rankClass = 'rank-top-2'; medal = '🥈'; }
    else if (rank === 3) { rankClass = 'rank-top-3'; medal = '🥉'; }
    else if (rank <= 5) rankClass = 'rank-top-5';
    else if (rank <= 10) rankClass = 'rank-top-10';
    
    let statusClass = 'status-low', statusText = 'Low';
    if (score >= 0.8) { statusClass = 'status-excellent'; statusText = 'Excellent'; }
    else if (score >= 0.6) { statusClass = 'status-good'; statusText = 'Good'; }
    else if (score >= 0.4) { statusClass = 'status-moderate'; statusText = 'Moderate'; }
    
    row.innerHTML = `
        <td class="${rankClass}"><span class="rank-medal">${medal}</span>${rank}</td>
        <td>${planet.planet_name}</td>
        <td>${(score * 100).toFixed(2)}%</td>
        <td><span class="status-badge ${statusClass}">${statusText}</span></td>
    `;
    return row;
}

// ========== UTILITIES ==========
function initializeTooltips() {
    try {
        [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]')).map(el => new bootstrap.Tooltip(el));
    } catch (e) {}
}

function initializeRangeSliders() {
    document.querySelectorAll('.range-cosmic').forEach(range => {
        const name = range.getAttribute('name');
        const num = document.querySelector(`input[name="${name}_val"]`);
        if (num) {
            range.addEventListener('input', () => num.value = range.value);
            num.addEventListener('input', () => range.value = num.value);
        }
    });
}

function initializeRangeSlidersForSection(sectionId) {
    const section = document.getElementById(sectionId);
    if (!section) return;
    section.querySelectorAll('.range-cosmic').forEach(range => {
        const name = range.getAttribute('name');
        const num = section.querySelector(`input[name="${name}_val"]`);
        if (num) {
            range.addEventListener('input', () => num.value = range.value);
            num.addEventListener('input', () => range.value = num.value);
        }
    });
}

function resetFormRanges(form) {
    const defaults = { st_teff: 5500, st_rad: 1, st_mass: 1, st_met: 0, st_luminosity: 1, pl_orbper: 365, pl_orbeccen: 0.05, pl_insol: 1 };
    for (const [name, value] of Object.entries(defaults)) {
        const range = form.querySelector(`input[name="${name}"]`);
        const num = form.querySelector(`input[name="${name}_val"]`);
        if (range) range.value = value;
        if (num) num.value = value;
    }
}

function showError(elementId, message) {
    const el = document.getElementById(elementId);
    if (el) {
        const textEl = el.querySelector('.alert-text');
        if (textEl) {
            textEl.textContent = message;
        }
        el.style.display = 'flex';
        setTimeout(() => el.style.display = 'none', 8000);
    }
}

console.log('%c✅ All Systems Ready!', 'color: #00ff88; font-size: 14px; font-weight: bold');