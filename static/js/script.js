// static/js/script.js
document.addEventListener('DOMContentLoaded', () => {
    // --- Get references to DOM elements ---
    const uploadForm = document.getElementById('upload-form');
    const photoInput = document.getElementById('photo-input');
    const fileNameDisplay = document.getElementById('file-name-display');
    const resultsArea = document.getElementById('results-area');
    const resultImage = document.getElementById('result-image');
    const detectionButtons = document.getElementById('detection-buttons');
    const birdInfoArea = document.getElementById('bird-info-area');
    const loadingIndicator = document.getElementById('loading-indicator');
    const errorMessage = document.getElementById('error-message');

    // Bird Info Elements
    const birdNameEl = document.getElementById('bird-name');
    const birdGenusEl = document.getElementById('bird-genus');
    const birdLocationsEl = document.getElementById('bird-locations');
    const birdMatingEl = document.getElementById('bird-mating');
    const birdShortInfoEl = document.getElementById('bird-short-info');
    const birdExampleImageEl = document.getElementById('bird-example-image');
    const birdAudioEl = document.getElementById('bird-audio');
    const chatButton = document.getElementById('chat-button');
    const chatBirdNameSpan = document.getElementById('chat-bird-name');

    // Chat Interface Elements
    const chatInterface = document.getElementById('chat-interface');
    const chatHistory = document.getElementById('chat-history');
    const chatInput = document.getElementById('chat-input');
    const sendChatButton = document.getElementById('send-chat-button');
    const chatInterfaceBirdNameSpan = document.getElementById('chat-interface-bird-name');
    const chatLoadingIndicator = document.getElementById('chat-loading-indicator');

    let currentBirdForChat = null; // Store the bird name for the chat context
    let currentChatHistory = []; // Store conversation history {role: 'user'/'assistant', content: '...'}

    // --- Update file name display on selection ---
    photoInput.addEventListener('change', () => {
        if (photoInput.files.length > 0) {
            fileNameDisplay.textContent = photoInput.files[0].name;
        } else {
            fileNameDisplay.textContent = 'No file chosen';
        }
    });

    // --- Handle Image Upload and Prediction ---
    uploadForm.addEventListener('submit', async (event) => {
        event.preventDefault(); // Prevent default form submission
        clearState(); // Clear previous results and errors

        if (!photoInput.files || photoInput.files.length === 0) {
            displayError('Please select an image file first.');
            return;
        }

        loadingIndicator.style.display = 'flex'; // Show loading indicator

        const formData = new FormData();
        formData.append('photo', photoInput.files[0]);

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData,
                // Headers are not typically needed for FormData, browser sets Content-Type
            });

            loadingIndicator.style.display = 'none'; // Hide loading

            if (!response.ok) {
                // Try to parse error message from backend JSON response
                let errorMsg = `HTTP error! Status: ${response.status}`;
                try {
                    const errorData = await response.json();
                    errorMsg = errorData.error || errorMsg;
                } catch (e) {
                    // If response is not JSON, use status text
                    errorMsg = response.statusText || errorMsg;
                }
                throw new Error(errorMsg);
            }

            const data = await response.json();

            // Display results
            if (data.result_image_url) {
                resultImage.src = data.result_image_url;
                resultImage.alt = "Processed image showing bird detections"; // Better alt text
            } else {
                resultImage.src = ""; // Clear image if no URL
                resultImage.alt = "";
            }

            detectionButtons.innerHTML = ''; // Clear previous buttons

            if (data.detections && data.detections.length > 0) {
                data.detections.forEach(detection => {
                    const button = document.createElement('button');
                    button.classList.add('button'); // Add button class for styling
                    button.textContent = `${detection.name} (${(detection.confidence * 100).toFixed(0)}%)`;
                    button.dataset.birdName = detection.name; // Store bird name in data attribute
                    button.addEventListener('click', handleBirdButtonClick);
                    detectionButtons.appendChild(button);
                });
            } else {
                detectionButtons.innerHTML = '<p>No birds detected (or confidence too low).</p>';
            }

            resultsArea.style.display = 'block'; // Show results section

        } catch (error) {
            console.error('Upload/Prediction Error:', error);
            displayError(`Prediction failed: ${error.message}`);
            loadingIndicator.style.display = 'none'; // Ensure loading is hidden on error
        }
    });

    // --- Handle Clicking a Bird Button ---
    async function handleBirdButtonClick(event) {
        const birdName = event.target.dataset.birdName;
    
        // **** ADD LOGS HERE ****
        console.log(`Detection button clicked for: ${birdName}`); // Log 1: Check if handler runs and gets name
        // **** END LOGS ****
    
        currentBirdForChat = birdName; // This is the critical line
    
        // **** ADD LOGS HERE ****
        console.log(`currentBirdForChat variable has been set to: ${currentBirdForChat}`); // Log 2: Check if assignment happens
        // **** END LOGS ****
    
        // clearState(true); // Clear previous results but keep detections visible
    
        // Highlight the selected button (optional)
        document.querySelectorAll('#detection-buttons .button').forEach(btn => btn.classList.remove('active'));
        event.target.classList.add('active'); // You'll need CSS for '.active'

        try {
            // Fetch bird info from backend
            const response = await fetch(`/bird_info/${encodeURIComponent(birdName)}`);
            if (!response.ok) {
                 let errorMsg = `Error fetching info (${response.status})`;
                 try {
                     const errorData = await response.json();
                     errorMsg = errorData.error || errorMsg;
                 } catch (e) { /* Ignore if error is not JSON */ }
                 throw new Error(errorMsg);
            }
            const info = await response.json();

            // Populate bird info area
            birdNameEl.textContent = birdName;
            birdGenusEl.textContent = info.genus || 'N/A';
            birdLocationsEl.textContent = info.locations || 'N/A';
            birdMatingEl.textContent = info.mating_patterns || 'N/A';
            birdShortInfoEl.textContent = info.short_info || 'N/A';
            birdExampleImageEl.src = info.image_path || ''; // Use path from backend
            birdExampleImageEl.alt = `Example photo of ${birdName}`;

            // Handle audio - ensure controls are shown only if src is valid
            if (info.audio_path) {
                birdAudioEl.src = info.audio_path; // Use path from backend
                birdAudioEl.style.display = 'block'; // Or 'inline-block'
                birdAudioEl.parentElement.style.display = 'block'; // Show the 'Bird Call:' label too
            } else {
                 birdAudioEl.src = '';
                 birdAudioEl.style.display = 'none';
                 birdAudioEl.parentElement.style.display = 'none'; // Hide label if no audio
            }


            // Update chat button text and show info area
            chatBirdNameSpan.textContent = birdName;
            birdInfoArea.style.display = 'block'; // Show the info card

        } catch (error) {
            // **** ADD LOGS HERE ****
            console.error(`Error inside handleBirdButtonClick for ${birdName}:`, error); // Log 3: Check for errors within this function
            // **** END LOGS ****
            displayError(`Error fetching info for ${birdName}: ${error.message}`);
            birdInfoArea.style.display = 'none';
        }
   }

     // --- Handle Clicking "Chat about [Bird]" Button ---
     chatButton.addEventListener('click', () => {
        // **** ADD LOGS HERE ****
        console.log("Chat button clicked!"); // Check if listener fires
        console.log("Value of currentBirdForChat:", currentBirdForChat); // Check the variable
    
        if (currentBirdForChat) {
            console.log("Condition met, attempting to show chat interface."); // Check if it enters the 'if' block
            chatInterfaceBirdNameSpan.textContent = currentBirdForChat;
            chatHistory.innerHTML = ''; // Clear previous chat history visually
            currentChatHistory = []; // Clear internal history
            chatInput.value = ''; // Clear input field
            chatInterface.style.display = 'block'; // Show chat card
            addChatMessage('assistant', `Hi! Ask me anything more about the ${currentBirdForChat}.`);
            chatInput.focus(); // Focus the input field
        } else {
             // **** ADD LOG HERE ****
            console.warn("Chat button clicked, but 'currentBirdForChat' is null or empty. Cannot open chat.");
        }
    });

    // --- Handle Sending a Chat Message ---
    sendChatButton.addEventListener('click', sendChatMessageToServer);
    chatInput.addEventListener('keypress', (e) => {
        // Send message on Enter key press, but not Shift+Enter
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault(); // Prevent default Enter behavior (like adding a new line)
            sendChatMessageToServer();
        }
    });

    async function sendChatMessageToServer() {
        const userMessage = chatInput.value.trim();
        if (!userMessage || !currentBirdForChat) {
            return; // Don't send empty messages or if no bird context
        }

        addChatMessage('user', userMessage); // Display user message immediately
        chatInput.value = ''; // Clear input field
        chatInput.disabled = true; // Disable input while waiting
        sendChatButton.disabled = true;
        chatLoadingIndicator.style.display = 'flex'; // Show chat loading

        try {
            const response = await fetch('/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    bird_name: currentBirdForChat,
                    message: userMessage,
                    history: currentChatHistory // Send history context
                })
            });

             chatLoadingIndicator.style.display = 'none'; // Hide loading

            if (!response.ok) {
                 let errorMsg = `Chat API error (${response.status})`;
                 try {
                     const errorData = await response.json();
                     // Use backend's reply if it's an error message, otherwise use generic error
                     errorMsg = errorData.reply || errorData.error || errorMsg;
                 } catch(e) { /* Ignore if error is not JSON */ }
                 // Display error as an AI message
                 addChatMessage('assistant', `Sorry, I encountered an error: ${errorMsg}`);
                 // Don't throw here, just show the error message in chat
            } else {
                const data = await response.json();
                addChatMessage('assistant', data.reply); // Display AI response
            }

        } catch (error) {
            console.error('Chat Send/Receive Error:', error);
             chatLoadingIndicator.style.display = 'none'; // Hide loading on network error too
            addChatMessage('assistant', `Sorry, I couldn't connect to the chat service: ${error.message}`);
        } finally {
             chatInput.disabled = false; // Re-enable input
             sendChatButton.disabled = false;
             chatInput.focus(); // Focus back on input
        }
    }

    // --- Helper function to add messages to the chat history (visual and internal) ---
    function addChatMessage(role, message) {
        // Add to internal history (role should be 'user' or 'assistant' for OpenAI)
        currentChatHistory.push({ role: role === 'AI' ? 'assistant' : role, content: message });

        // Add to visual chat history
        const messageElement = document.createElement('p');
        // Sanitize message slightly before inserting - basic protection
        // For robust protection, use a proper sanitization library if messages can contain HTML
        const sanitizedMessage = message.replace(/</g, "&lt;").replace(/>/g, "&gt;");
        messageElement.innerHTML = `<strong>${role === 'assistant' ? 'AI' : 'You'}:</strong> ${sanitizedMessage}`;
        chatHistory.appendChild(messageElement);
        chatHistory.scrollTop = chatHistory.scrollHeight; // Scroll to the bottom
    }


    // --- Helper function to display errors ---
    function displayError(message) {
        errorMessage.textContent = message;
        errorMessage.style.display = 'block'; // Make sure error area is visible
    }

    // --- Helper function to clear state ---
    function clearState(keepDetections = false) {
        errorMessage.textContent = ''; // Clear errors
        errorMessage.style.display = 'none';
        if (!keepDetections) {
             resultsArea.style.display = 'none'; // Hide detection results
             resultImage.src = '';
             detectionButtons.innerHTML = '';
        }
        birdInfoArea.style.display = 'none';  // Hide bird info
        chatInterface.style.display = 'none'; // Hide chat
        currentBirdForChat = null;
        currentChatHistory = [];
        // Don't clear the file input/display unless starting fresh
    }
});