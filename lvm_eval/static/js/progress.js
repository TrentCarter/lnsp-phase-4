class EvaluationProgress {
    constructor() {
        // UI Elements
        this.progressBar = document.getElementById('progressBar');
        this.progressText = document.getElementById('progressText');
        this.statusText = document.getElementById('statusText');
        this.evaluationProgress = document.getElementById('evaluationProgress');
        this.evaluateBtn = document.getElementById('evaluateBtn');
        this.evaluateBtnText = document.getElementById('evaluateBtnText');
        this.evaluateBtnSpinner = document.getElementById('evaluateBtnSpinner');
        this.resultsContainer = document.getElementById('resultsContainer');
        
        // State
        this.originalButtonText = this.evaluateBtnText.textContent;
        this.eventSource = null;
        this.currentProgress = 0;
        this.currentModel = '';
        this.currentStep = '';
        this.animationFrame = null;
        this.supportsSSE = typeof(EventSource) !== 'undefined';
        
        // Initialize
        this.initializeEventListeners();
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
        if (this.resultsContainer) {
            this.resultsContainer.innerHTML = '';
        }
        
        // Update UI
        this.evaluateBtn.disabled = true;
        this.evaluateBtnText.textContent = 'Evaluating...';
        this.evaluateBtnSpinner.style.display = 'inline-block';
        this.evaluationProgress.style.display = 'block';
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
        this.progressBar.style.width = percentage + '%';
        this.progressBar.setAttribute('aria-valuenow', percentage);
        this.progressText.textContent = Math.round(percentage) + '%';
    }

    updateStatusText() {
        let status = this.currentStep;
        if (this.currentModel) {
            status = `Evaluating ${this.currentModel}...`;
            if (this.currentStep) {
                status += ` (${this.currentStep})`;
            }
        }
        this.statusText.textContent = status;
    }

    animateProgress() {
        // Smoothly animate to the target progress
        const targetProgress = this.currentProgress;
        const currentWidth = parseFloat(this.progressBar.style.width) || 0;
        
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
            fetch('/evaluate/progress')
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
            this.statusText.textContent = data.message;
        } else {
            this.statusText.textContent = 'Evaluation complete!';
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
        this.statusText.textContent = `Error: ${errorMsg}`;
        this.progressBar.classList.add('bg-danger');
        
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
        this.evaluateBtn.disabled = false;
        this.evaluateBtnText.textContent = this.originalButtonText;
        this.evaluateBtnSpinner.style.display = 'none';
        
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
