// Add this to your index.html in a script tag

document.addEventListener('DOMContentLoaded', function() {
    // Check if we're on Vercel deployment
    fetch('/health')
        .then(response => response.json())
        .then(data => {
            if (data.deployment_type === 'vercel-lightweight') {
                // Add Vercel deployment notice
                const noticeDiv = document.createElement('div');
                noticeDiv.className = 'vercel-notice';
                noticeDiv.innerHTML = `
                    <h3>⚠️ Vercel Lightweight Deployment</h3>
                    <p>This is a limited version of the Battery RAG System due to Vercel's 50MB function size limit.</p>
                    <p>The full system with embedding model (~90MB), vector database (>200MB), and dependencies (~150MB) 
                    exceeds Vercel's size limitations.</p>
                    
                    <div class="deployment-options">
                        <h4>For full functionality:</h4>
                        <ul>
                            <li>Run locally with <code>python app.py</code></li>
                            <li>Deploy to AWS Lambda with larger memory allocations</li>
                            <li>Deploy to Google Cloud Run</li>
                            <li>Deploy to Heroku or DigitalOcean App Platform</li>
                        </ul>
                    </div>
                `;
                
                // Insert after header
                const header = document.querySelector('.header');
                if (header && header.nextSibling) {
                    header.parentNode.insertBefore(noticeDiv, header.nextSibling);
                }
            }
        })
        .catch(error => console.error('Error checking deployment type:', error));
});
