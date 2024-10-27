const video = document.getElementById('video');
let employees = {};
let isRecognizing = false;

async function loadModels() {
    await faceapi.nets.tinyFaceDetector.loadFromUri('/models');
    await faceapi.nets.faceLandmark68Net.loadFromUri('/models');
    await faceapi.nets.faceRecognitionNet.loadFromUri('/models');
    console.log("Models loaded successfully.");
}

async function setupCamera() {
    const stream = await navigator.mediaDevices.getUserMedia({ video: true });
    video.srcObject = stream;
    return new Promise((resolve) => {
        video.onloadedmetadata = () => {
            resolve(video);
        };
    });
}

async function capturePhoto() {
    const canvas = document.createElement('canvas');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    return await faceapi.detectAllFaces(canvas, new faceapi.TinyFaceDetectorOptions()).withFaceLandmarks().withFaceDescriptors();
}

async function registerEmployee() {
    await setupCamera();
    const detections = await capturePhoto();

    if (detections.length > 0) {
        const name = prompt("Enter employee name:");
        if (name && !employees[name]) {
            const descriptors = detections.map(d => d.descriptor);
            employees[name] = descriptors[0]; // Store only the first descriptor
            document.getElementById('message').innerText = `Employee ${name} registered.`;
        } else if (employees[name]) {
            document.getElementById('message').innerText = "Employee already registered.";
        }
    } else {
        document.getElementById('message').innerText = "No face detected. Please try again.";
    }
}

async function recognizeEmployee() {
    await setupCamera();
    isRecognizing = true;

    const labeledDescriptors = Object.keys(employees).map(name => new faceapi.LabeledFaceDescriptors(name, [employees[name]]));

    const faceMatcher = new faceapi.FaceMatcher(labeledDescriptors, 0.4);

    setInterval(async () => {
        const detections = await capturePhoto();

        if (detections.length > 0) {
            const results = detections.map(d => faceMatcher.findBestMatch(d.descriptor));
            results.forEach((result) => {
                const label = result.toString().split(' ')[0]; // Get only the name
                document.getElementById('message').innerText = `Recognized: ${label}`;
            });
        }
    }, 1000);
}

document.getElementById('register').addEventListener('click', async () => {
    if (!isRecognizing) {
        await loadModels();
        await registerEmployee();
    } else {
        alert("Stop recognition before registering a new employee.");
    }
});

document.getElementById('recognition').addEventListener('click', async () => {
    await loadModels();
    if (!isRecognizing) {
        recognizeEmployee();
    } else {
        alert("Recognition is already in progress.");
    }
});
