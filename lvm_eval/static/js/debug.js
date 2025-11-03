// Simple debug script to verify JavaScript is working
console.log('✅ Debug.js loaded successfully');
console.log('Testing basic functionality...');

$(document).ready(function() {
    console.log('✅ jQuery is working');
    console.log('✅ Document ready fired');
    
    // Test API
    $.ajax({
        url: '/api/models',
        method: 'GET',
        success: function(data) {
            console.log('✅ API /api/models works - found', data.models.length, 'models');
        },
        error: function(error) {
            console.error('❌ API /api/models failed:', error);
        }
    });
});
