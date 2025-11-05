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

// Expose displayResults globally for template callbacks
window.displayResults = displayResults;

function buildSummaryTable(results) {
    const tbody = document.getElementById('summaryTableBody');
    if (!tbody) return;
    const rows = results.map(r => {
        const cosine = r.avg_cosine_similarity || 0;
        // derive ROUGE-L from test_cases
        let rougeL = 0; let n = 0;
        (r.test_cases || []).forEach(tc => {
            if (tc.rouge_scores && tc.rouge_scores.rougeL && typeof tc.rouge_scores.rougeL.f1 === 'number') {
                rougeL += tc.rouge_scores.rougeL.f1; n += 1;
            }
        });
        rougeL = n ? (rougeL / n) : 0;
        const latency = r.avg_latency || 0.0001;
        const bertF1 = (r.avg_bert && r.avg_bert.f1) ? r.avg_bert.f1 : 0;
        const latNorm = Math.min(1, 1 / (1 + latency));
        const composite = 0.6 * cosine + 0.2 * rougeL + 0.2 * latNorm;
        return {
            name: r.model_name,
            path: (r.model_metadata && r.model_metadata.full_path) ? r.model_metadata.full_path : r.model_path,
            composite, cosine, rougeL, bertF1, latency
        };
    }).sort((a,b)=>b.composite-a.composite);
    tbody.innerHTML = rows.map((x,i)=>
        `<tr><td>${i+1}</td><td class="text-truncate" title="${escapeHtml(x.path)}">${escapeHtml(x.name||'')}</td>`+
        `<td>${x.composite.toFixed(4)}</td><td>${(x.cosine*100).toFixed(2)}%</td>`+
        `<td>${(x.rougeL*100).toFixed(2)}%</td><td>${(x.bertF1*100).toFixed(2)}%</td>`+
        `<td>${x.latency.toFixed(3)}s</td></tr>`
    ).join('');
}

function setupExportTopK(results) {
    const btnCsv = document.getElementById('exportTopKCsv');
    const btnJson = document.getElementById('exportTopKJson');
    const kInput = document.getElementById('exportTopKCount');
    if (!btnCsv || !btnJson || !kInput) return;
    function ranked() {
        const arr = [];
        results.forEach(r => {
            const cosine = r.avg_cosine_similarity || 0;
            let rougeL = 0; let n = 0;
            (r.test_cases || []).forEach(tc => {
                if (tc.rouge_scores && tc.rouge_scores.rougeL && typeof tc.rouge_scores.rougeL.f1 === 'number') { rougeL += tc.rouge_scores.rougeL.f1; n++; }
            });
            rougeL = n ? (rougeL / n) : 0;
            const latency = r.avg_latency || 0.0001;
            const bertF1 = (r.avg_bert && r.avg_bert.f1) ? r.avg_bert.f1 : 0;
            const latNorm = Math.min(1, 1 / (1 + latency));
            const composite = 0.6 * cosine + 0.2 * rougeL + 0.2 * latNorm;
            arr.push({
                model: r.model_name,
                path: (r.model_metadata && r.model_metadata.full_path) ? r.model_metadata.full_path : r.model_path,
                composite, cosine, rougeL, bertF1, latency
            });
        });
        arr.sort((a,b)=>b.composite-a.composite);
        const k = Math.max(1, Math.min(arr.length, parseInt(kInput.value)||5));
        return arr.slice(0,k);
    }
    btnCsv.onclick = () => {
        const top = ranked();
        const header = ['rank','model','path','composite','cosine','rougeL','bertF1','latency'];
        const lines = [header.join(',')].concat(top.map((x,i)=>[
            i+1,x.model,x.path,x.composite.toFixed(6),(x.cosine*100).toFixed(2),(x.rougeL*100).toFixed(2),(x.bertF1*100).toFixed(2),x.latency.toFixed(3)
        ].join(',')));
        const blob = new Blob([lines.join('\n')], {type:'text/csv'});
        const url = URL.createObjectURL(blob); const a = document.createElement('a');
        a.href=url; a.download='top_k_models.csv'; a.click(); URL.revokeObjectURL(url);
    };
    btnJson.onclick = () => {
        const top = ranked();
        const blob = new Blob([JSON.stringify(top,null,2)], {type:'application/json'});
        const url = URL.createObjectURL(blob); const a = document.createElement('a');
        a.href=url; a.download='top_k_models.json'; a.click(); URL.revokeObjectURL(url);
    };
}

function setupABCompare(results) {
    const selA = document.getElementById('abModelA');
    const selB = document.getElementById('abModelB');
    const btn = document.getElementById('abCompareBtn');
    const body = document.getElementById('abCompareBody');
    if (!selA || !selB || !btn || !body) return;
    selA.innerHTML=''; selB.innerHTML='';
    results.forEach((r,i)=>{
        const o1 = document.createElement('option'); o1.value=i; o1.textContent=r.model_name; selA.appendChild(o1);
        const o2 = document.createElement('option'); o2.value=i; o2.textContent=r.model_name; selB.appendChild(o2);
    });
    btn.onclick = () => {
        const a = results[parseInt(selA.value,10)], b = results[parseInt(selB.value,10)];
        if (!a || !b) return;
        function m(r){
            // derive rougeL avg
            let rougeL=0,n=0; (r.test_cases||[]).forEach(tc=>{ if(tc.rouge_scores&&tc.rouge_scores.rougeL&&typeof tc.rouge_scores.rougeL.f1==='number'){ rougeL+=tc.rouge_scores.rougeL.f1; n++; }}); rougeL=n?(rougeL/n):0;
            return {cos:r.avg_cosine_similarity||0, rl:rougeL, bf:(r.avg_bert&&r.avg_bert.f1)?r.avg_bert.f1:0, lat:r.avg_latency||0, cr:r.avg_compression_ratio||0};
        }
        const A=m(a), B=m(b);
        const rows=[
            ['Cosine (%)',(A.cos*100).toFixed(2),(B.cos*100).toFixed(2),((A.cos-B.cos)*100).toFixed(2)],
            ['ROUGE-L (%)',(A.rl*100).toFixed(2),(B.rl*100).toFixed(2),((A.rl-B.rl)*100).toFixed(2)],
            ['BERT F1 (%)',(A.bf*100).toFixed(2),(B.bf*100).toFixed(2),((A.bf-B.bf)*100).toFixed(2)],
            ['Latency (s)',A.lat.toFixed(3),B.lat.toFixed(3),(A.lat-B.lat).toFixed(3)],
            ['Compression',A.cr.toFixed(2),B.cr.toFixed(2),(A.cr-B.cr).toFixed(2)]
        ];
        body.innerHTML = '<table class="table table-sm">' + rows.map(r=>`<tr><th>${r[0]}</th><td>${r[1]}</td><td>${r[2]}</td><td>${r[3]}</td></tr>`).join('') + '</table>';
        const modal = new bootstrap.Modal(document.getElementById('abCompareModal'));
        modal.show();
    };
}

// Expose sweep trigger for the button in the template
window.triggerSweep = function() {
    try {
        const stepsRaw = ($('#stepsList').val() || '').split(',');
        const conceptsRaw = ($('#conceptsList').val() || '').split(',');
        const stepsList = stepsRaw.map(s => parseInt(s.trim(), 10)).filter(n => !isNaN(n) && n > 0);
        const conceptsList = conceptsRaw.map(s => parseInt(s.trim(), 10)).filter(n => !isNaN(n) && n > 0);
        const selectedModels = $('.model-checkbox:checked').toArray().map(x => $(x).val());
        const testMode = $('#testMode').val();
        const numTestCases = parseInt($('#numTestCases').val(), 10) || 10;
        // Collect selected metrics directly from checkboxes
        const metrics = Array.from(document.querySelectorAll('.metric-checkbox:checked')).map(el => el.value);
        if (selectedModels.length === 0) {
            alert('Select at least one model to run the sweep.');
            return;
        }
        evaluationProgress.start('Running sweep...');
        runSweep(selectedModels, testMode, metrics, numTestCases, stepsList, conceptsList)
            .done(resp => {
                renderHeatmap(resp);
                evaluationProgress.complete();
            })
            .fail(err => {
                console.error('Sweep failed', err);
                evaluationProgress.error('Sweep failed');
            });
    } catch (e) {
        console.error('triggerSweep error', e);
        alert('Sweep failed to start. Check console for details.');
    }
};

// --- Sweep (steps x concepts) ---
function runSweep(selectedModels, testMode, metrics, numTestCases, stepsList, conceptsList) {
    const endpoint = '/evaluate/sweep';
    return $.ajax({
        url: endpoint,
        method: 'POST',
        contentType: 'application/json',
        timeout: 600000,
        data: JSON.stringify({
            models: selectedModels,
            test_mode: testMode,
            metrics: metrics,
            num_test_cases: numTestCases,
            steps_list: stepsList,
            concepts_list: conceptsList
        })
    });
}

// --- Render heatmap ---
function renderHeatmap(data) {
    const container = document.getElementById('sweepHeatmap');
    if (!container) return;
    const grid = data && data.grid ? data.grid : [];
    const steps = [...new Set(grid.map(g => g.steps))].sort((a,b)=>a-b);
    const concepts = [...new Set(grid.map(g => g.concepts))].sort((a,b)=>a-b);
    let html = '<table class="table table-sm table-bordered"><thead><tr><th>Concepts \\ Steps</th>';
    steps.forEach(s => { html += '<th>' + s + '</th>'; });
    html += '</tr></thead><tbody>';
    concepts.forEach(c => {
        html += '<tr><th>' + c + '</th>';
        steps.forEach(s => {
            const cell = grid.find(g => g.steps === s && g.concepts === c) || {};
            const val = (typeof cell.avg_cosine === 'number') ? (cell.avg_cosine * 100).toFixed(1) + '%' : '—';
            const tip = 'BERT F1: ' + ((cell.avg_bert_f1||0)*100).toFixed(1) + '%, Lat: ' + (cell.avg_latency||0).toFixed(3) + 's';
            html += '<td title="' + tip + '">' + val + '</td>';
        });
        html += '</tr>';
    });
    html += '</tbody></table>';

    // Explanatory caption
    const modelUnderTest = (data && data.model) ? (function(full){
        try {
            const marker = '/lnsp-phase-4/';
            const i = full.indexOf(marker);
            if (i !== -1) return full.substring(i + marker.length - 1);
            const art = '/artifacts/';
            const j = full.indexOf(art);
            if (j !== -1) return full.substring(j);
            const parts = full.split('/');
            return '/' + parts.slice(-2).join('/');
        } catch(_) { return full; }
    })(data.model) : 'selected model';

    html += '<div class="text-muted small mt-2">' +
        'Cells show <strong>Cosine similarity (%)</strong> (higher is better). ' +
        'Vertical axis is <strong>Concepts</strong> (number of input chunks). ' +
        'Horizontal axis is <strong>Vec2Text Steps</strong>. ' +
        'Model under test: <code>' + escapeHtml(modelUnderTest) + '</code>.' +
        '</div>';

    container.innerHTML = html;
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
        const avgBertF1 = modelResult.avg_bert && modelResult.avg_bert.f1 !== null && modelResult.avg_bert.f1 !== undefined
            ? (modelResult.avg_bert.f1 * 100).toFixed(2) : 'N/A';
        const avgBertP = modelResult.avg_bert && modelResult.avg_bert.p !== null && modelResult.avg_bert.p !== undefined
            ? (modelResult.avg_bert.p * 100).toFixed(2) : 'N/A';
        const avgBertR = modelResult.avg_bert && modelResult.avg_bert.r !== null && modelResult.avg_bert.r !== undefined
            ? (modelResult.avg_bert.r * 100).toFixed(2) : 'N/A';
        const avgCR = modelResult.avg_compression_ratio !== undefined && modelResult.avg_compression_ratio !== null
            ? modelResult.avg_compression_ratio.toFixed(2) : 'N/A';
        const avgOutLen = modelResult.avg_output_length !== undefined && modelResult.avg_output_length !== null
            ? Math.round(modelResult.avg_output_length) : 'N/A';
        const avgExpLen = modelResult.avg_expected_length !== undefined && modelResult.avg_expected_length !== null
            ? Math.round(modelResult.avg_expected_length) : 'N/A';
        
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
                                '<h6 class="card-subtitle mb-2 text-muted">BERT F1</h6>' +
                                '<h3 class="text-primary" title="P: ' + avgBertP + ' | R: ' + avgBertR + '">' + (avgBertF1) + '%</h3>' +
                            '</div>' +
                        '</div>' +
                    '</div>' +
                    '<div class="col-md-2">' +
                        '<div class="card h-100">' +
                            '<div class="card-body text-center">' +
                                '<h6 class="card-subtitle mb-2 text-muted">Compression</h6>' +
                                '<h3 class="text-primary">' + avgCR + '</h3>' +
                                '<div class="small text-muted">out ' + avgOutLen + ' / exp ' + avgExpLen + '</div>' +
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

    // Build summary/AB/export helpers
    try {
        buildSummaryTable(results);
        setupExportTopK(results);
        setupABCompare(results);
    } catch (e) {
        console.warn('Optional UI helpers failed:', e);
    }
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
