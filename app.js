const video = document.getElementById('video');
const registerButton = document.getElementById('register');
const recognizeButton = document.getElementById('recognize');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const employees = {};
let faceMatcher;
let recognitionActive = false;
let personDetectionModel;

// Access the webcam with the specified resolution for mobile devices
navigator.mediaDevices.getUserMedia({ 
    video: { 
        facingMode: "user", 
        width: { ideal: 640 }, 
        height: { ideal: 480 } 
    } 
})
.then(stream => {
    video.srcObject = stream;
    console.log("Webcam access granted");
})
.catch(err => {
    console.error("Error accessing the camera: ", err);
    alert("Camera access denied. Please check your permissions.");
});

// Load models
async function loadModels() {
    const MODEL_URL = 'models';
    console.log("Loading models...");
    await faceapi.nets.ssdMobilenetv1.loadFromUri(MODEL_URL);
    await faceapi.nets.faceLandmark68Net.loadFromUri(MODEL_URL);
    await faceapi.nets.faceRecognitionNet.loadFromUri(MODEL_URL);
    console.log("Face-api.js models loaded");

    // Load COCO-SSD model
    personDetectionModel = await cocoSsd.load();
    console.log("COCO-SSD model loaded");
}

// Function to improve image quality
function improveImageQuality(imageData) {
    return new Promise((resolve) => {
        const img = new Image();
        img.src = imageData;

        img.onload = () => {
            const canvas = document.createElement('canvas');
            const ctx = canvas.getContext('2d');
            canvas.width = img.width;
            canvas.height = img.height;
            ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
            
            // Apply contrast enhancement
            let imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
            let data = imageData.data;

            // Simple contrast adjustment
            for (let i = 0; i < data.length; i += 4) {
                data[i] = Math.min(255, data[i] * 1.2);     // Red
                data[i + 1] = Math.min(255, data[i + 1] * 1.2); // Green
                data[i + 2] = Math.min(255, data[i + 2] * 1.2); // Blue
            }

            ctx.putImageData(imageData, 0, 0);
            resolve(canvas.toDataURL('image/png'));
        };

        img.onerror = (error) => {
            console.error("Image loading error:", error);
            resolve(imageData); // Fallback to original if error occurs
        };
    });
}

// Register employee
// In the register button click event
registerButton.addEventListener('click', async () => {
    //ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    const imageData='20241004_171143.jpg'
    //const imageData = canvas.toDataURL('image/png');
    console.log(imageData, 'imagedata');

    const employeeName = prompt("Enter employee name:");
    console.log("Employee Name:", employeeName);

    if (employeeName) {
        const improvedImageData = await improveImageQuality(imageData);
        console.log("Improved Image Data:", improvedImageData);

        const processedImg = await faceapi.fetchImage(improvedImageData);
        console.log("Processed Image:", processedImg);
        
        const detections = await faceapi.detectSingleFace(processedImg).withFaceLandmarks().withFaceDescriptor();
        console.log("Detections:", detections);
        
        if (detections) {
            employees[employeeName] = improvedImageData;
            displayEmployees();
            recognizeButton.disabled = false;
            console.log(`Employee registered: ${employeeName}`);
            faceMatcher = await createFaceMatcher();
        } else {
            console.warn(`No face detected for ${employeeName}`);
            alert("No face detected. Please try again.");
        }
    }
});

// Display captured employee photos
function displayEmployees() {
    const employeeDiv = document.getElementById('employees');
    employeeDiv.innerHTML = '';
    for (const [name, image] of Object.entries(employees)) {
        const img = document.createElement('img');
        img.src = image;
        img.width = 100;
        img.alt = name;
        employeeDiv.appendChild(img);
        employeeDiv.appendChild(document.createTextNode(name));
        employeeDiv.appendChild(document.createElement('br'));
    }
}

// Create labeled descriptors
async function createLabeledDescriptors() {
    const descriptors = [];
    for (const [name, imageData] of Object.entries(employees)) {
        const img = await faceapi.fetchImage(imageData);
        const detections = await faceapi.detectSingleFace(img).withFaceLandmarks().withFaceDescriptor();
        if (detections) {
            descriptors.push(new faceapi.LabeledFaceDescriptors(name, [detections.descriptor]));
        } else {
            console.warn(`No face detected for ${name}`);
        }
    }
    return descriptors;
}

// Create face matcher
async function createFaceMatcher() {
    const labeledDescriptors = await createLabeledDescriptors();
    return new faceapi.FaceMatcher(labeledDescriptors);
}

// Start recognition process
recognizeButton.addEventListener('click', async () => {
    if (!recognitionActive) {
        recognitionActive = true;
        recognizeButton.disabled = true;

        console.log("Recognition started");
        if (!faceMatcher) {
            faceMatcher = await createFaceMatcher();
            console.log("Face matcher created");
        }
        detectPerson();
    }
});

async function detectPerson() {
    console.log("Person detection started");
    let lastDetection = false;

    while (recognitionActive) {
        const predictions = await personDetectionModel.detect(video);
        const personDetected = predictions.some(prediction => prediction.class === 'person');

        if (personDetected) {
            if (!lastDetection) {
                console.log("Person detected! Starting recognition...");
                lastDetection = true;
            }
            await recognizeFaces();
        } else {
            if (lastDetection) {
                console.log("No person detected. Stopping recognition...");
                lastDetection = false;
                document.getElementById('recognized-employee').innerHTML = '';
            }
        }
        await new Promise(requestAnimationFrame);
    }
}

async function recognizeFaces() {
    const detections = await faceapi.detectAllFaces(video)
        .withFaceLandmarks()
        .withFaceDescriptors();

    const recognizedList = document.getElementById('recognized-employee');
    recognizedList.innerHTML = '';

    if (detections.length > 0) {
        const resizedDetections = faceapi.resizeResults(detections, { width: video.width, height: video.height });
        const results = resizedDetections.map(d => faceMatcher.findBestMatch(d.descriptor));
       
        results.forEach(result => {
            const [label, distance] = result.toString().split(' ');
            const name = parseFloat(distance.replace("(", "").replace(")", "")) < 0.6 ? label : 'unknown';
            const li = document.createElement('li');
            li.textContent = name;
            recognizedList.appendChild(li);
        });
    } else {
        const li = document.createElement('li');
        li.textContent = 'No face detected';
        recognizedList.appendChild(li);
    }
}

loadModels();
