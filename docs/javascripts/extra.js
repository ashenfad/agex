// Custom JavaScript for agex documentation

document.addEventListener("DOMContentLoaded", function() {
    // Add any custom JavaScript functionality here
    
    // Example: Track demo button clicks (could be useful for analytics)
    const demoButtons = document.querySelectorAll('a[href*="lite/lab"]');
    demoButtons.forEach(button => {
        button.addEventListener('click', function() {
            console.log('Demo launched');
            // Could add analytics tracking here
        });
    });
}); 