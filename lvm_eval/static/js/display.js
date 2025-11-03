// Display evaluation results
function displayResults(results) {
    console.log('displayResults called with:', results);
    const resultsContainer = $('#evaluationResults');
    const noResultsDiv = $('#noResults');
    
    // Hide the "no results" message
    noResultsDiv.hide();
    
    // Clear and show the results container
    resultsContainer.empty().show();
    
    if (!results || results.length === 0) {
        resultsContainer.html('<div class="alert alert-warning">No results to display</div>');
        return;
    }
    
    console.log('Processing', results.length, 'model results');
    
    let html = '<div class="card mb-4">' +
        '<div class="card-header bg-primary text-white d-flex justify-content-between align-items-center">' +
            '<h5 class="mb-0">Evaluation Results</h5>' +
            '<button class="btn btn-sm btn-light" onclick="$(\'.test-cases\').slideToggle()">Toggle All Details</button>' +
        '</div>' +
        '<div class="card-body">';
    
    // Sort results by average cosine similarity (descending)
    results.sort((a, b) => (b.avg_cosine_similarity || 0) - (a.avg_cosine_similarity || 0));
    
    results.forEach((modelResult, index) => {
        const modelName = modelResult.model_name || 'Model ' + (index + 1);
        const avgCosine = modelResult.avg_cosine_similarity !== undefined ? 
            (modelResult.avg_cosine_similarity * 100).toFixed(2) : 'N/A';
        const avgLatency = modelResult.avg_latency !== undefined ? 
            modelResult.avg_latency.toFixed(3) : 'N/A';
        const memoryUsage = modelResult.memory_usage_mb !== undefined ? 
            modelResult.memory_usage_mb.toFixed(1) : 'N/A';
        
        // Build enhanced model display name (relative to project root)
        let enhancedModelName = modelName;
        if (modelResult.model_metadata && modelResult.model_metadata.full_path) {
            const metadata = modelResult.model_metadata;
            const fullPath = metadata.full_path || '';
            const projectMarker = '/lnsp-phase-4/';
            const projIdx = fullPath.indexOf(projectMarker);
            if (projIdx !== -1) {
                enhancedModelName = fullPath.substring(projIdx + projectMarker.length - 1); // keep leading /
            } else {
                const artifactsMarker = '/artifacts/';
                const artIdx = fullPath.indexOf(artifactsMarker);
                if (artIdx !== -1) {
                    enhancedModelName = fullPath.substring(artIdx); // /artifacts/...
                } else {
                    // Fallback: show last 2 segments
                    try {
                        enhancedModelName = '/' + fullPath.split('/').slice(-2).join('/');
                    } catch (_) {
                        enhancedModelName = fullPath;
                    }
                }
            }
        }
        
        // Calculate average ROUGE scores if available
        let avgRouge1 = 0, avgRouge2 = 0, avgRougeL = 0;
        const testCases = modelResult.test_cases || [];
        
        if (testCases.length > 0) {
            const rougeScores = testCases
                .filter(tc => tc.rouge_scores)
                .map(tc => ({
                    rouge1: tc.rouge_scores.rouge1?.f1 || 0,
                    rouge2: tc.rouge_scores.rouge2?.f1 || 0,
                    rougeL: tc.rouge_scores.rougeL?.f1 || 0
                }));
            
            if (rougeScores.length > 0) {
                avgRouge1 = (rougeScores.reduce((sum, s) => sum + s.rouge1, 0) / rougeScores.length * 100).toFixed(2);
                avgRouge2 = (rougeScores.reduce((sum, s) => sum + s.rouge2, 0) / rougeScores.length * 100).toFixed(2);
                avgRougeL = (rougeScores.reduce((sum, s) => sum + s.rougeL, 0) / rougeScores.length * 100).toFixed(2);
            }
        }
        
        html += '<div class="card mb-3">' +
            '<div class="card-header">' +
                '<h5 class="mb-0">' + escapeHtml(enhancedModelName) + '</h5>' +
            '</div>' +
            '<div class="card-body">' +
                '<div class="row">' +
                    '<div class="col-md-2">' +
                        '<div class="card h-100">' +
                            '<div class="card-body text-center">' +
                                '<h6 class="card-subtitle mb-2 text-muted">Avg Cosine</h6>' +
                                '<h3 class="text-primary">' + avgCosine + '%</h3>' +
                                '<div class="progress">' +
                                    '<div class="progress-bar" role="progressbar" ' +
                                        'style="width: ' + avgCosine + '%" ' +
                                        'aria-valuenow="' + avgCosine + '" ' +
                                        'aria-valuemin="0" aria-valuemax="100"></div>' +
                                '</div>' +
                            '</div>' +
                        '</div>' +
                    '</div>' +
                    '<div class="col-md-2">' +
                        '<div class="card h-100">' +
                            '<div class="card-body text-center">' +
                                '<h6 class="card-subtitle mb-2 text-muted">ROUGE-1</h6>' +
                                '<h3 class="text-success">' + avgRouge1 + '%</h3>' +
                                '<div class="progress">' +
                                    '<div class="progress-bar bg-success" role="progressbar" ' +
                                        'style="width: ' + avgRouge1 + '%" ' +
                                        'aria-valuenow="' + avgRouge1 + '" ' +
                                        'aria-valuemin="0" aria-valuemax="100"></div>' +
                                '</div>' +
                            '</div>' +
                        '</div>' +
                    '</div>' +
                    '<div class="col-md-2">' +
                        '<div class="card h-100">' +
                            '<div class="card-body text-center">' +
                                '<h6 class="card-subtitle mb-2 text-muted">ROUGE-L</h6>' +
                                '<h3 class="text-info">' + avgRougeL + '%</h3>' +
                                '<div class="progress">' +
                                    '<div class="progress-bar bg-info" role="progressbar" ' +
                                        'style="width: ' + avgRougeL + '%" ' +
                                        'aria-valuenow="' + avgRougeL + '" ' +
                                        'aria-valuemin="0" aria-valuemax="100"></div>' +
                                '</div>' +
                            '</div>' +
                        '</div>' +
                    '</div>' +
                    '<div class="col-md-2">' +
                        '<div class="card h-100">' +
                            '<div class="card-body text-center">' +
                                '<h6 class="card-subtitle mb-2 text-muted">Latency</h6>' +
                                '<h3 class="text-warning">' + avgLatency + 's</h3>' +
                                '<div class="text-muted small">avg per test</div>' +
                            '</div>' +
                        '</div>' +
                    '</div>' +
                    '<div class="col-md-2">' +
                        '<div class="card h-100">' +
                            '<div class="card-body text-center">' +
                                '<h6 class="card-subtitle mb-2 text-muted">Memory</h6>' +
                                '<h3 class="text-danger">' + memoryUsage + 'MB</h3>' +
                                '<div class="text-muted small">usage</div>' +
                            '</div>' +
                        '</div>' +
                    '</div>' +
                    '<div class="col-md-2">' +
                        '<div class="card h-100">' +
                            '<div class="card-body text-center">' +
                                '<h6 class="card-subtitle mb-2 text-muted">Test Cases</h6>' +
                                '<h3>' + testCases.length + '</h3>' +
                                '<div class="text-muted small">' +
                                    testCases.filter(tc => tc.cosine_similarity > 0.8).length + ' passed' +
                                '</div>' +
                            '</div>' +
                        '</div>' +
                    '</div>' +
                '</div>' +
                
                '<div class="test-cases" style="display: none;">' +
                    '<h6>Detailed Results</h6>';
        
        if (testCases.length > 0) {
            testCases.forEach((testCase, i) => {
                const rouge1 = testCase.rouge_scores?.rouge1?.f1 * 100 || 0;
                const rougeL = testCase.rouge_scores?.rougeL?.f1 * 100 || 0;
                const cosine = testCase.cosine_similarity !== undefined ? 
                    (testCase.cosine_similarity * 100).toFixed(2) : 'N/A';
                const latency = testCase.latency !== undefined ? 
                    testCase.latency.toFixed(3) : 'N/A';
                
                // Determine badge color based on cosine similarity
                let badgeClass = 'bg-danger';
                if (testCase.cosine_similarity > 0.8) badgeClass = 'bg-success';
                else if (testCase.cosine_similarity > 0.5) badgeClass = 'bg-warning';
                
                html += '<div class="card mb-3">' +
                    '<div class="card-header d-flex justify-content-between align-items-center" ' +
                         'data-bs-toggle="collapse" href="#testCase' + index + i + '">' +
                        '<div>' +
                            '<span class="badge ' + badgeClass + ' me-2">' + cosine + '% Similarity</span>' +
                            '<span class="badge bg-secondary me-2">' + latency + 's</span>' +
                            'Test Case ' + (i + 1) +
                        '</div>' +
                        '<i class="bi bi-chevron-down"></i>' +
                    '</div>' +
                    '<div class="collapse" id="testCase' + index + i + '">' +
                        '<div class="card-body">' +
                            '<div class="row">' +
                                '<div class="col-md-6">' +
                                    '<h6>Input Text</h6>' +
                                    '<div class="concept-preview" style="white-space: pre-wrap; font-family: monospace; font-size: 0.9rem; background: #f8f9fa; padding: 10px; border-radius: 4px; max-height: 300px; overflow-y: auto;">' + escapeHtml(testCase.input || 'N/A') + '</div>' +
                                '</div>' +
                                '<div class="col-md-6">' +
                                    '<h6>Expected Output</h6>' +
                                    '<div class="concept-preview" style="white-space: pre-wrap; font-family: monospace; font-size: 0.9rem; background: #fff3cd; padding: 10px; border-radius: 4px; max-height: 300px; overflow-y: auto;">' + escapeHtml(testCase.expected || 'N/A') + '</div>' +
                                '</div>' +
                            '</div>' +
                            '<div class="row mt-3">' +
                                '<div class="col-md-12">' +
                                    '<h6>Actual Output</h6>' +
                                    '<div class="concept-preview" style="white-space: pre-wrap; font-family: monospace; font-size: 0.9rem; background: #d1ecf1; padding: 10px; border-radius: 4px; max-height: 300px; overflow-y: auto;">' + escapeHtml(testCase.output || 'N/A') + '</div>' +
                                '</div>' +
                            '</div>' +
                            '<div class="row mt-3">' +
                                '<div class="col-md-3">' +
                                    '<h6>Cosine Similarity</h6>' +
                                    '<div class="d-flex align-items-center">' +
                                        '<span>' + cosine + '%</span>' +
                                        '<div class="progress flex-grow-1 ms-2">' +
                                            '<div class="progress-bar" role="progressbar" ' +
                                                'style="width: ' + cosine + '%" ' +
                                                'aria-valuenow="' + cosine + '" ' +
                                                'aria-valuemin="0" aria-valuemax="100"></div>' +
                                        '</div>' +
                                    '</div>' +
                                '</div>' +
                                '<div class="col-md-3">' +
                                    '<h6>ROUGE-1 F1</h6>' +
                                    '<div class="d-flex align-items-center">' +
                                        '<span>' + rouge1.toFixed(2) + '%</span>' +
                                        '<div class="progress flex-grow-1 ms-2">' +
                                            '<div class="progress-bar bg-success" role="progressbar" ' +
                                                'style="width: ' + rouge1 + '%" ' +
                                                'aria-valuenow="' + rouge1 + '" ' +
                                                'aria-valuemin="0" aria-valuemax="100"></div>' +
                                        '</div>' +
                                    '</div>' +
                                '</div>' +
                                '<div class="col-md-3">' +
                                    '<h6>ROUGE-L F1</h6>' +
                                    '<div class="d-flex align-items-center">' +
                                        '<span>' + rougeL.toFixed(2) + '%</span>' +
                                        '<div class="progress flex-grow-1 ms-2">' +
                                            '<div class="progress-bar bg-info" role="progressbar" ' +
                                                'style="width: ' + rougeL + '%" ' +
                                                'aria-valuenow="' + rougeL + '" ' +
                                                'aria-valuemin="0" aria-valuemax="100"></div>' +
                                        '</div>' +
                                    '</div>' +
                                '</div>' +
                                '<div class="col-md-3">' +
                                    '<h6>Latency</h6>' +
                                    '<div class="d-flex align-items-center">' +
                                        '<span>' + latency + 's</span>' +
                                        '<div class="progress flex-grow-1 ms-2">' +
                                            '<div class="progress-bar bg-warning" role="progressbar" ' +
                                                'style="width: ' + Math.min(latency * 100, 100) + '%" ' +
                                                'aria-valuenow="' + latency + '" ' +
                                                'aria-valuemin="0" aria-valuemax="100"></div>' +
                                        '</div>' +
                                    '</div>' +
                                '</div>' +
                            '</div>' +
                        '</div>' +
                    '</div>' +
                '</div>';
            });
        }
        
        html += '</div>' + // end test-cases
            '</div>' + // end card-body
        '</div>'; // end card
    });
    
    html += '</div></div>'; // end card-body and card
    
    resultsContainer.html(html);
}

// Helper function to escape HTML
function escapeHtml(unsafe) {
    if (typeof unsafe !== 'string') return '';
    return unsafe
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
}
