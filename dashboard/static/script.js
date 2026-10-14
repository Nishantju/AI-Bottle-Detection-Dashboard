
        const cameraFeed = document.getElementById("cameraFeed");
        const feedPlaceholder = document.getElementById("feedPlaceholder");



        const mobileCamera =
            document.getElementById("mobileCamera");

        const liveIndicator = document.getElementById("liveIndicator");
        const liveLabel = document.getElementById("liveLabel");
        const startButton = document.getElementById("startButton");
        const stopButton = document.getElementById("stopButton");
        let cameraRunning = false;
        let progressInterval = null;
        let progressValue = 0;
        let cancelRequested = false;
        let mobileCaptureInterval = null;
        let mobileProcessing = false;

        function setCameraState(running) {
            cameraRunning = running;
            feedPlaceholder.hidden = running;
            liveIndicator.className = `indicator ${running ? "green" : "red"}`;
            liveLabel.textContent = running ? "LIVE" : "OFFLINE";
            startButton.disabled = running;
            stopButton.disabled = !running;
            document.getElementById("systemIndicator").className =
                `indicator ${running ? "green" : "yellow"}`;
            document.getElementById("systemText").textContent =
                running ? "System online" : "System standby";
            document.getElementById("systemCardIndicator").className =
                `indicator ${running ? "green" : "yellow"}`;
            document.getElementById("systemCardText").textContent =
                running ? "Running" : "Standby";
        }

        async function refreshStatus() {
            try {
                const response = await fetch("/status");
                if (!response.ok) {
                    throw new Error(`Status request failed (${response.status})`);
                }
                const data = await response.json();
                const detected = data.status === "Bottle Detected";
                document.getElementById("status").textContent = data.status;
                document.getElementById("confidence").textContent = `${data.confidence}%`;
                document.getElementById("count").textContent = data.count;
                document.getElementById("detectionCard").classList.toggle("detected", detected);
                document.getElementById("bottleVisual").classList.toggle("detected", detected);
                document.getElementById("statusIndicator").className =
                    `status-dot ${detected ? "green" : "yellow"}`;
                document.getElementById("statusIndicator").setAttribute(
                    "aria-label", detected ? "Bottle detected" : "No bottle detected"
                );
                document.getElementById("statusCaption").textContent =
                    detected ? "Bottle detected in frame" : "Waiting for a bottle";
                document.getElementById("confidenceIndicator").className =
                    `indicator ${detected ? "green" : "yellow"}`;
                document.getElementById("countIndicator").className =
                    `indicator ${Number(data.count) > 0 ? "green" : "yellow"}`;
            } catch (error) {
                console.error("Unable to refresh detection status:", error);
            }
        }



        async function changeCameraSource() {

            const source = document.getElementById("cameraSource").value;

            try {

                const response = await fetch(`/camera/source/${source}`);

                const data = await response.json();

                if (!response.ok) {
                    throw new Error(data.error || "Unable to change camera");
                }

                console.log("Camera source:", data.camera);

            } catch (error) {

                alert(error.message);

                // Return to webcam if changing failed
                document.getElementById("cameraSource").value = "webcam";
            }
        }




        async function startCamera() {

            try {


                const source =
                    document.getElementById("cameraSource").value;

                if (source === "mobile") {

                    try {

                        if (!navigator.mediaDevices) {
                            alert("navigator.mediaDevices is undefined");
                            return;
                        }

                        if (!navigator.mediaDevices.getUserMedia) {
                            alert("getUserMedia is not available");
                            return;
                        }



                        const stream =
                            await navigator.mediaDevices.getUserMedia({

                                video: {
                                    facingMode: "environment"
                                },

                                audio: false

                            });

                        mobileCamera.srcObject = stream;

                        mobileCamera.hidden = false;

                        cameraFeed.hidden = false;

                        feedPlaceholder.hidden = true;

                        setCameraState(true);

                        document.getElementById("cameraSource").disabled = true;


                        mobileCamera.onloadedmetadata = () => {

                            console.log("Mobile camera ready");

                            mobileCaptureInterval = setInterval(() => {

                                captureMobileFrame();

                            }, 2000);

                        };


                        return;

                    } catch (err) {

                        console.error(err);

                        alert(err.name + "\n\n" + err.message);

                        return;
                    }
                }



                const response = await fetch("/start_camera");

                if (!response.ok) {
                    throw new Error("Unable to start camera");
                }
                cameraFeed.onload = function () {
                    setCameraState(true);
                };


                document.getElementById("cameraSource").disabled = true;


                cameraFeed.onerror = function () {
                    setCameraState(false);
                    alert("Camera feed failed to load");
                };

                cameraFeed.hidden = false;
                feedPlaceholder.hidden = true;

                cameraFeed.src = "/video?" + Date.now();

                

            } catch(error) {

                console.error(error);
                alert("Camera failed to start");

            }
        }


        function captureMobileFrame() {

            if (mobileProcessing) {
                return;
            }

            const canvas = document.getElementById("captureCanvas");
            const video = document.getElementById("mobileCamera");

            if (video.videoWidth === 0 || video.videoHeight === 0) {
                return;
            }

            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;

            const ctx = canvas.getContext("2d");

            ctx.drawImage(video,0,0);

            const imageData = canvas.toDataURL(
                "image/jpeg",
                0.8
            );

            sendMobileFrame(imageData);
        }



        async function sendMobileFrame(imageData) {

            mobileProcessing = true;

            try {

                const response = await fetch("/mobile_frame", {

                    method:"POST",

                    headers:{
                        "Content-Type":"application/json"
                    },

                    body:JSON.stringify({
                        image:imageData
                    })

                });


                const result = await response.json();


                if(result.success){

                    cameraFeed.hidden=false;
                    feedPlaceholder.hidden=true;
                    cameraFeed.src=result.image;

                }


            }
            catch(error){

                console.error(error);

            }
            finally{

                mobileProcessing=false;

            }

        }



        async function stopCamera() {


            if (mobileCamera.srcObject) {

                mobileCamera.srcObject
                    .getTracks()
                    .forEach(track => track.stop());

                mobileCamera.srcObject = null;

                mobileCamera.hidden = true;
            }   



            const response = await fetch("/stop_camera");

            if (!response.ok) {
                throw new Error(`Unable to stop camera (${response.status})`);
            }

            if (mobileCaptureInterval) {

                clearInterval(mobileCaptureInterval);

                mobileCaptureInterval = null;

            }

            

            // Remove the last camera frame
            cameraFeed.removeAttribute("src");
            cameraFeed.hidden = true;

            // Show the placeholder
            feedPlaceholder.hidden = false;

            setCameraState(false);
            sessionStorage.removeItem("lastImage");
            document.getElementById("cameraSource").disabled = false;
        }
        

        async function chooseFile() {

            if (cameraRunning) {
                await stopCamera();
            }

            document.getElementById("fileInput").click();

        }


        async function uploadFile() {

            const input = document.getElementById("fileInput");

            if (input.files.length === 0) {
                return;
            }

            const formData = new FormData();
            formData.append("file", input.files[0]);

            try {

                const response = await fetch("/upload", {
                    method: "POST",
                    body: formData
                });

                const data = await response.json();

                if (data.success) {

                    await stopCamera();

                    feedPlaceholder.hidden = true;
                    cameraFeed.hidden = false;
                    cameraFeed.src = data.file + "?" + Date.now();
                    sessionStorage.setItem("lastImage", data.file);






                    localStorage.setItem(
                        "lastStatus",
                        document.getElementById("status").textContent
                    );

                    localStorage.setItem(
                        "lastConfidence",
                        document.getElementById("confidence").textContent
                    );

                    localStorage.setItem(
                        "lastCount",
                        document.getElementById("count").textContent
                    );

                    localStorage.setItem(
                        "lastCaption",
                        document.getElementById("statusCaption").textContent
                    );






                } else {

                    alert(data.message);

                }

            } catch (err) {

                console.error(err);
                alert("Upload failed.");

            }

        }

        function updateClock() {
            document.getElementById("feedClock").textContent =
                new Date().toLocaleTimeString();
        }

        async function navigatePage(url) {

            if (cameraRunning) {
                await stopCamera();
            }

            window.location.href = url;

            return false;

        }

        
        window.addEventListener("beforeunload", function () {

            cameraFeed.removeAttribute("src");

            navigator.sendBeacon("/stop_camera");


        });
        setCameraState(false);





        const lastImage = sessionStorage.getItem("lastImage");

        if (lastImage) {

            feedPlaceholder.hidden = true;
            cameraFeed.hidden = false;
            cameraFeed.src = lastImage;

            const lastStatus = localStorage.getItem("lastStatus");
            const lastConfidence = localStorage.getItem("lastConfidence");
            const lastCount = localStorage.getItem("lastCount");
            const lastCaption = localStorage.getItem("lastCaption");

            if (lastStatus) {
                document.getElementById("status").textContent = lastStatus;
            }

            if (lastConfidence) {
                document.getElementById("confidence").textContent = lastConfidence;
            }

            if (lastCount) {
                document.getElementById("count").textContent = lastCount;
            }

            if (lastCaption) {
                document.getElementById("statusCaption").textContent = lastCaption;
            }
        }







        refreshStatus();
        updateClock();
        setInterval(refreshStatus, 500);
        setInterval(updateClock, 1000);
    