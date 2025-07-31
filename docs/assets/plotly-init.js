// Custom Plotly initialization for mkdocs
(function() {
    // Wait for DOM to be ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', initializePlotly);
    } else {
        initializePlotly();
    }
    
    function initializePlotly() {
        // Check if Plotly is available
        if (typeof window.Plotly === 'undefined') {
            console.log('Plotly not loaded, waiting...');
            setTimeout(initializePlotly, 100);
            return;
        }
        
        // Find all Plotly divs and reinitialize them
        const plotlyDivs = document.querySelectorAll('div[data-plotly]');
        plotlyDivs.forEach(function(div) {
            try {
                const data = JSON.parse(div.getAttribute('data-plotly'));
                window.Plotly.newPlot(div, data.data, data.layout, data.config);
                console.log('Initialized Plotly plot');
            } catch (e) {
                console.error('Failed to initialize Plotly plot:', e);
            }
        });
        
        // Also check for jupyter notebook plotly outputs
        const jupyterPlotly = document.querySelectorAll('.jp-OutputArea-output[data-mime-type="application/vnd.plotly.v1+json"]');
        jupyterPlotly.forEach(function(element) {
            try {
                const plotlyJson = JSON.parse(element.textContent);
                const plotDiv = document.createElement('div');
                plotDiv.style.width = '100%';
                plotDiv.style.height = '400px';
                element.appendChild(plotDiv);
                window.Plotly.newPlot(plotDiv, plotlyJson.data, plotlyJson.layout, plotlyJson.config);
                console.log('Initialized Jupyter Plotly plot');
            } catch (e) {
                console.error('Failed to initialize Jupyter Plotly plot:', e);
            }
        });
    }
})();