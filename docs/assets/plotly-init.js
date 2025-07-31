// Ensure Plotly.js loads before any notebook content tries to use it
(function() {
    'use strict';
    
    // Store the original DOMContentLoaded to delay it if needed
    var originalDOMReady = false;
    var delayedCallbacks = [];
    
    // Function to wait for Plotly to be available
    function waitForPlotly(callback, maxAttempts = 50) {
        var attempts = 0;
        
        function checkPlotly() {
            attempts++;
            
            if (typeof window.Plotly !== 'undefined') {
                console.log('✅ Plotly.js loaded and ready');
                callback();
            } else if (attempts < maxAttempts) {
                setTimeout(checkPlotly, 100); // Wait 100ms and try again
            } else {
                console.warn('⚠️ Plotly.js failed to load after 5 seconds');
                callback(); // Proceed anyway
            }
        }
        
        checkPlotly();
    }
    
    // Override any immediate Plotly calls to wait for the library
    if (typeof window.Plotly === 'undefined') {
        window.Plotly = {
            newPlot: function() {
                var args = arguments;
                var element = args[0];
                
                waitForPlotly(function() {
                    if (window.Plotly && typeof window.Plotly.newPlot === 'function') {
                        window.Plotly.newPlot.apply(window.Plotly, args);
                    }
                });
            },
            
            purge: function() {
                var args = arguments;
                
                waitForPlotly(function() {
                    if (window.Plotly && typeof window.Plotly.purge === 'function') {
                        window.Plotly.purge.apply(window.Plotly, args);
                    }
                });
            }
        };
    }
    
    // When the page loads, make sure Plotly is ready
    document.addEventListener('DOMContentLoaded', function() {
        waitForPlotly(function() {
            // Plotly is ready, now handle any existing plots
            var plotlyDivs = document.querySelectorAll('div[id*="plotly"], .plotly-graph-div');
            
            plotlyDivs.forEach(function(div) {
                if (div.data && div.layout && typeof window.Plotly.redraw === 'function') {
                    try {
                        window.Plotly.redraw(div);
                    } catch(e) {
                        console.log('Plotly redraw attempted for:', div.id);
                    }
                }
            });
        });
    });
})();