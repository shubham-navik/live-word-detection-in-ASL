let stream; // Variable to store the camera stream
let predictionInterval; // Variable to store the prediction interval

async function startVideo() {
    const videoElement = document.getElementById('videoElement');
    // const predictionsContainer = document.getElementById('predictionsContainer');

    try {
        stream = await navigator.mediaDevices.getUserMedia({ video: true });
        videoElement.srcObject = stream;
        await videoElement.play();
    } catch (error) {
        console.error('Error accessing webcam:', error);
    }
}

async function startPredictions() {
    const videoElement = document.getElementById('videoElement');
    const predictionsContainer = document.getElementById('predictionsContainer');

    predictionInterval = setInterval(async () => {
        if (!stream) return; // Stop sending requests if the camera stream is not active
        const image = await captureFrame(videoElement);
        const predictions  = await sendFrameForPrediction(image);
        predictionsContainer.innerHTML += JSON.stringify(predictions);
    }, 1000); // Adjust the interval as needed
}

function pausePredictions() {
    clearInterval(predictionInterval);
}

function closeCamera() {
    clearInterval(predictionInterval); // Stop the prediction interval
    if (stream) {
        const tracks = stream.getTracks();
        tracks.forEach(track => track.stop());
        stream = null; // Clear the stream variable
    }
}

// Function to capture a frame from the video stream
async function captureFrame(videoElement) {
    const canvas = document.createElement('canvas');
    const context = canvas.getContext('2d');
    canvas.width = videoElement.videoWidth;
    canvas.height = videoElement.videoHeight;
    context.drawImage(videoElement, 0, 0, canvas.width, canvas.height);
    return new Promise((resolve, reject) => {
        canvas.toBlob(blob => {
            if (!blob) {
                reject(new Error('Failed to capture frame as blob'));
            } else {
                resolve(blob);
            }
        }, 'image/jpeg');
    });
}

// Function to send a frame for prediction to the server
async function sendFrameForPrediction(frameData) {
    try {
        const response = await fetch('http://localhost:5000/predict', {
            method: 'POST',
            body: frameData,
            headers: {
                'Content-Type': 'application/octet-stream' // Adjust content type as needed
            }
        });

        if (!response.ok) {
            throw new Error('Failed to get predictions from server');
        }

        const predictions = await response.json();
        return predictions;
    } catch (error) {
        console.error('Error getting predictions from server:', error);
        return { error: 'Failed to get predictions from server' };
    }
}

// Event listeners for buttons
document.getElementById('openCamera').addEventListener('click', startVideo);
document.getElementById('startPredictions').addEventListener('click', startPredictions);
document.getElementById('pausePredictions').addEventListener('click', pausePredictions);
document.getElementById('closeCamera').addEventListener('click', closeCamera);
