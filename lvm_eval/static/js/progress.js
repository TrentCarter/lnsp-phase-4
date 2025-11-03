class EvaluationProgress {
    constructor() {
        // Defer DOM element access until needed
        this._elements = null;

        // State
        this.originalButtonText = '';
        this.eventSource = null;
        this.currentProgress = 0;
        this.currentModel = '';
        this.currentStep = '';
        this.animationFrame = null;
        this.supportsSSE = typeof(EventSource) !== 'undefined';

        // Initialize
        this.initializeEventListeners();
    }

    // Lazy initialization of DOM elements
    get elements() {
        if (!this._elements) {
            this._elements = {
                progressBar: document.getElementById('progressBar'),
                progressText: document.getElementById('progressText'),
                statusText: document.getElementById('statusText'),
                evaluationProgress: document.getElementById('evaluationProgress'),
                evaluateBtn: document.getElementById('evaluateBtn'),
                evaluateBtnText: document.getElementById('evaluateBtnText'),
                evaluateBtnSpinner: document.getElementById('evaluateBtnSpinner'),
                resultsContainer: document.getElementById('resultsContainer')
            };

            // Store original button text
            if (this._elements.evaluateBtnText) {
                this.originalButtonText = this._elements.evaluateBtnText.textContent;
            }
        }
        return this._elements;
    }

    initializeEventListeners() {
        // Clean up any existing event source when page is unloaded
        window.addEventListener('beforeunload', () => {
            this.cleanup();
        });
    }

    start() {
        // Reset state
        this.currentProgress = 0;
        this.currentModel = '';
        this.currentStep = 'Initializing...';

        // Clear previous results
        if (this.elements.resultsContainer) {
            this.elements.resultsContainer.innerHTML = '';
        }

        // Update UI
        if (this.elements.evaluateBtn) this.elements.evaluateBtn.disabled = true;
        if (this.elements.evaluateBtnText) this.elements.evaluateBtnText.textContent = 'Evaluating...';
        if (this.elements.evaluateBtnSpinner) this.elements.evaluateBtnSpinner.style.display = 'inline-block';
        if (this.elements.evaluationProgress) this.elements.evaluationProgress.style.display = 'block';

        // Reset progress visuals to default blue animated state
        if (this.elements.progressBar) {
            this.elements.progressBar.classList.remove('bg-success', 'bg-danger');
            this.elements.progressBar.classList.add('progress-bar-striped', 'progress-bar-animated');
        }
        if (this.elements.statusText) {
            this.elements.statusText.classList.remove('text-success');
            this.elements.statusText.classList.add('text-muted');
        }
        const heading = document.getElementById('progressHeading');
        if (heading) heading.textContent = 'Evaluating models...';
        this.updateProgressBar(0);
        this.updateStatusText();

        // Start progress animation
        this.animateProgress();

        // Initialize SSE if supported
        if (this.supportsSSE) {
            this.initializeSSE();
        } else {
            console.warn('SSE not supported, falling back to polling');
            this.startPolling();
        }
    }

    updateProgress(progress, model, step) {
        if (model) this.currentModel = model;
        if (step) this.currentStep = step;
        this.currentProgress = Math.min(100, Math.max(0, progress));
    }

    updateProgressBar(percentage) {
        if (this.elements.progressBar) {
            this.elements.progressBar.style.width = percentage + '%';
            this.elements.progressBar.setAttribute('aria-valuenow', percentage);
        }
        if (this.elements.progressText) {
            this.elements.progressText.textContent = Math.round(percentage) + '%';
        }
    }

    updateStatusText() {
        let status = this.currentStep;
        if (this.currentModel) {
            status = `Evaluating ${this.currentModel}...`;
            if (this.currentStep) {
                status += ` (${this.currentStep})`;
            }
        }
        if (this.elements.statusText) {
            this.elements.statusText.textContent = status;
        }
    }

    animateProgress() {
        // Smoothly animate to the target progress
        const targetProgress = this.currentProgress;
        const currentWidth = this.elements.progressBar ? (parseFloat(this.elements.progressBar.style.width) || 0) : 0;

        if (Math.abs(targetProgress - currentWidth) < 0.1) {
            this.updateProgressBar(targetProgress);
        } else {
            const newProgress = currentWidth + (targetProgress - currentWidth) * 0.1;
            this.updateProgressBar(newProgress);
        }

        // Update status text
        this.updateStatusText();

        // Continue animation if not at 100%
        if (this.currentProgress < 100) {
            this.animationFrame = requestAnimationFrame(() => this.animateProgress());
        }
    }

    initializeSSE() {
        // Clean up any existing connection
        this.cleanup();
        
        // Create new EventSource
        this.eventSource = new EventSource('/evaluate/stream');
        
        // Handle incoming messages
        this.eventSource.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                this.handleProgressUpdate(data);
                
                // If evaluation is complete, close the connection
                if (data.status === 'complete' || data.status === 'error') {
                    this.complete(data);
                }
            } catch (e) {
                console.error('Error processing SSE message:', e);
            }
        };
        
        // Handle errors
        this.eventSource.onerror = (error) => {
            console.error('SSE Error:', error);
            this.eventSource.close();
            this.startPolling(); // Fall back to polling
        };
    }
    
    startPolling() {
        // If already polling, don't start another interval
        if (this.pollingInterval) return;
        
        this.pollingInterval = setInterval(() => {
            fetch('/api/progress')
                .then(response => response.json())
                .then(data => {
                    this.handleProgressUpdate(data);
                    
                    // If evaluation is complete, stop polling
                    if (data.status === 'complete' || data.status === 'error') {
                        this.complete(data);
                    }
                })
                .catch(error => {
                    console.error('Polling error:', error);
                    this.error('Failed to get progress updates');
                });
        }, 2000); // Poll every 2 seconds
    }
    
    handleProgressUpdate(data) {
        if (data.progress !== undefined) {
            this.updateProgress(
                data.progress,
                data.model,
                data.step
            );
        }
        
        // Update status message
        if (data.message) {
            this.currentStep = data.message;
            this.updateStatusText();
        }
        
        // Handle errors
        if (data.status === 'error' && data.error) {
            this.error(data.error);
        }
    }
    
    complete(data) {
        // Update to 100%
        this.currentProgress = 100;
        this.updateProgressBar(100);

        // Set final status message
        if (data && data.message) {
            this.currentStep = data.message;
        } else {
            this.currentStep = 'Evaluation complete!';
        }
        this.updateStatusText();

        // Switch progress bar to green and static, and heading to 'Complete'
        if (this.elements.progressBar) {
            this.elements.progressBar.classList.remove('progress-bar-striped', 'progress-bar-animated', 'bg-danger');
            this.elements.progressBar.classList.add('bg-success');
        }
        const heading = document.getElementById('progressHeading');
        if (heading) heading.textContent = 'Complete';
        if (this.elements.statusText) {
            this.elements.statusText.classList.remove('text-muted');
            this.elements.statusText.classList.add('text-success');
        }

        // Reset button state
        this.resetButton();

        // Clean up resources
        this.cleanup();

        // Return results if provided
        if (data && data.results) {
            return data.results;
        }
    }

    cleanup() {
        // Stop any active animations
        if (this.animationFrame) {
            cancelAnimationFrame(this.animationFrame);
            this.animationFrame = null;
        }
        
        // Close SSE connection if open
        if (this.eventSource) {
            this.eventSource.close();
            this.eventSource = null;
        }
        
        // Clear polling interval if active
        if (this.pollingInterval) {
            clearInterval(this.pollingInterval);
            this.pollingInterval = null;
        }
    }
    
    error(message) {
        // Show error state
        const errorMsg = message || 'Evaluation failed';
        this.currentStep = `Error: ${errorMsg}`;
        this.updateStatusText();

        if (this.elements.progressBar) {
            this.elements.progressBar.classList.remove('progress-bar-striped', 'progress-bar-animated', 'bg-success');
            this.elements.progressBar.classList.add('bg-danger');
        }

        // Log to console
        console.error('Evaluation error:', errorMsg);

        // Clean up resources
        this.cleanup();

        // Reset button state
        this.resetButton();

        // Show alert
        showAlert(message || 'An error occurred during evaluation.', 'danger');
    }

    resetButton() {
        if (this.elements.evaluateBtn) this.elements.evaluateBtn.disabled = false;
        if (this.elements.evaluateBtnText) this.elements.evaluateBtnText.textContent = this.originalButtonText;
        if (this.elements.evaluateBtnSpinner) this.elements.evaluateBtnSpinner.style.display = 'none';

        // Stop animation
        if (this.animationFrame) {
            cancelAnimationFrame(this.animationFrame);
            this.animationFrame = null;
        }
    }

    closeEventSource() {
        if (this.eventSource) {
            this.eventSource.close();
            this.eventSource = null;
        }
    }

    setupEventSource(url, onComplete) {
        this.closeEventSource();
        
        this.eventSource = new EventSource(url);
        
        this.eventSource.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                
                if (data.type === 'progress') {
                    this.updateProgress(
                        data.progress || 0,
                        data.model || '',
                        data.step || ''
                    );
                } else if (data.type === 'result') {
                    this.complete(data.results);
                    this.closeEventSource();
                    if (onComplete && data.results) {
                        onComplete(data.results);
                    }
                } else if (data.type === 'error') {
                    this.error(data.message);
                    this.closeEventSource();
                }
            } catch (e) {
                console.error('Error processing server message:', e);
                this.error('Failed to process server response');
                this.closeEventSource();
            }
        };
        
        this.eventSource.onerror = () => {
            this.error('Connection to server was lost');
            this.closeEventSource();
        };
        
        return this.eventSource;
    }
}

// Create global instance
const evaluationProgress = new EvaluationProgress();

// Helper function to show alerts
function showAlert(message, type = 'info') {
    const alertDiv = document.createElement('div');
    alertDiv.className = `alert alert-${type} alert-dismissible fade show`;
    alertDiv.role = 'alert';
    alertDiv.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
    `;
    
    const container = document.querySelector('.container-fluid');
    container.insertBefore(alertDiv, container.firstChild);
    
    // Auto-dismiss after 5 seconds
    setTimeout(() => {
        const alert = bootstrap.Alert.getOrCreateInstance(alertDiv);
        if (alert) alert.close();
    }, 5000);
}

// Export for use in other files
window.EvaluationProgress = EvaluationProgress;
window.evaluationProgress = evaluationProgress;
window.showAlert = showAlert;
