#!/usr/bin/env node

/**
 * n8n Workflow CLI Import Tool
 * Alternative Node.js-based import method using n8n's REST API
 */

const fs = require('fs');
const path = require('path');
const http = require('http');

const N8N_HOST = process.env.N8N_HOST || 'localhost';
const N8N_PORT = process.env.N8N_PORT || 5678;
const N8N_API_KEY = process.env.N8N_API_KEY || '';

// Workflows to import
const workflows = [
    'vec2text_test_workflow.json',
    'webhook_api_workflow.json'
];

// Colors for console output
const colors = {
    reset: '\x1b[0m',
    green: '\x1b[32m',
    red: '\x1b[31m',
    yellow: '\x1b[33m',
    blue: '\x1b[34m'
};

function log(message, color = 'reset') {
    console.log(`${colors[color]}${message}${colors.reset}`);
}

// Check if n8n is running
async function checkN8nStatus() {
    return new Promise((resolve) => {
        const options = {
            hostname: N8N_HOST,
            port: N8N_PORT,
            path: '/rest/workflows',
            method: 'GET',
            headers: {
                'Accept': 'application/json'
            }
        };

        if (N8N_API_KEY) {
            options.headers['X-N8N-API-KEY'] = N8N_API_KEY;
        }

        const req = http.request(options, (res) => {
            resolve(res.statusCode === 200 || res.statusCode === 401);
        });

        req.on('error', () => {
            resolve(false);
        });

        req.end();
    });
}

// Import workflow via REST API
async function importWorkflow(filePath) {
    const workflowData = JSON.parse(fs.readFileSync(filePath, 'utf8'));

    return new Promise((resolve, reject) => {
        const data = JSON.stringify({
            name: workflowData.name,
            nodes: workflowData.nodes,
            connections: workflowData.connections,
            settings: workflowData.settings || {},
            active: false
        });

        const options = {
            hostname: N8N_HOST,
            port: N8N_PORT,
            path: '/rest/workflows',
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Accept': 'application/json',
                'Content-Length': data.length
            }
        };

        if (N8N_API_KEY) {
            options.headers['X-N8N-API-KEY'] = N8N_API_KEY;
        }

        const req = http.request(options, (res) => {
            let responseData = '';

            res.on('data', (chunk) => {
                responseData += chunk;
            });

            res.on('end', () => {
                if (res.statusCode === 200 || res.statusCode === 201) {
                    resolve({ success: true, data: responseData });
                } else {
                    reject({ success: false, status: res.statusCode, data: responseData });
                }
            });
        });

        req.on('error', (error) => {
            reject({ success: false, error: error.message });
        });

        req.write(data);
        req.end();
    });
}

// Main execution
async function main() {
    log('🚀 n8n Workflow CLI Import Tool', 'blue');
    log('================================\n');

    // Check n8n status
    log('Checking n8n status...', 'yellow');
    const isRunning = await checkN8nStatus();

    if (!isRunning) {
        log('❌ n8n is not running or not accessible', 'red');
        log(`   Please start n8n with: N8N_SECURE_COOKIE=false n8n start`);
        log(`   Then run this script again\n`);
        process.exit(1);
    }

    log(`✅ n8n is accessible at http://${N8N_HOST}:${N8N_PORT}\n`, 'green');

    // Import workflows
    log('Importing workflows:', 'blue');
    log('===================\n');

    for (const workflowFile of workflows) {
        const filePath = path.join(__dirname, workflowFile);

        if (!fs.existsSync(filePath)) {
            log(`❌ File not found: ${workflowFile}`, 'red');
            continue;
        }

        try {
            log(`📥 Importing: ${workflowFile}`, 'yellow');
            const result = await importWorkflow(filePath);
            log(`✅ Successfully imported: ${workflowFile}`, 'green');

            // Parse and show workflow ID if available
            try {
                const responseData = JSON.parse(result.data);
                if (responseData.id) {
                    log(`   Workflow ID: ${responseData.id}`);
                }
            } catch (e) {
                // Ignore JSON parsing errors
            }
        } catch (error) {
            log(`❌ Failed to import: ${workflowFile}`, 'red');

            if (error.status === 401) {
                log('   Authentication required. Set N8N_API_KEY environment variable.', 'yellow');
            } else if (error.error) {
                log(`   Error: ${error.error}`, 'red');
            } else {
                log(`   Status: ${error.status}`, 'red');
            }
        }

        console.log(''); // Empty line for readability
    }

    log('🎉 Import process complete!', 'green');
    log('\n📌 Next steps:');
    log(`1. Open n8n at http://${N8N_HOST}:${N8N_PORT}`);
    log('2. Check your workflows in the UI');
    log('3. Activate the webhook workflow if needed');
    log('4. Test the workflows with your vec2text system\n');
}

// Run the script
main().catch(error => {
    log('❌ Unexpected error:', 'red');
    console.error(error);
    process.exit(1);
});