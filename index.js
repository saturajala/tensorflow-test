const classifier = knnClassifier.create();
const webcamElement = document.getElementById("webcam");
let net;

async function app() {
    console.log("Loading MobileNet...");

    //Load the MobileNet model

    net = await mobilenet.load();
    console.log("Successfully loaded model!");

    /* ********* Image test: ********* */

    //Make a prediction of the image contents with our model
    /* const imgEl = document.getElementById("img");
    const result = await net.classify(imgEl);
    console.log(result); */

    /* ********* Webcam test: ********* */

    // Create an object from Tensorflow.js data API which could capture image from the web camera as Tensor
    const webcam = await tf.data.webcam(webcamElement);

    // Reads an image from the webcam and associates it with a specific class index 
    const addExample = async classId => {
        //capture image from the web camera
        const img = await webcam.capture();

        //get the intermediate activation of MobileNet 'conv_preds' and pass that to the KNN Classifier
        const activation = net.infer(img, true);

        //pass the intermediate activation to the classifier
        classifier.addExample(activation, classId);

        //dispose the tensor to release the memory
        img.dispose();

    };

    // When clicking a button, add an example for that class
    document.getElementById('class-a').addEventListener('click', () => addExample(0));
    document.getElementById('class-b').addEventListener('click', () => addExample(1));
    document.getElementById('class-c').addEventListener('click', () => addExample(2));

    while (true) {

        if (classifier.getNumClasses() > 0) {

            const img = await webcam.capture();

            // get the activation from mobilenet from the webcam
            const activation = net.infer(img, 'conv_preds');
            // get the most likely class and confidence from the classifier module
            const result = await classifier.predictClass(activation);

            const classes = ['A', 'B', 'C'];
            document.getElementById('console').innerText = `
            prediction: ${classes[result.label]} \n 
            probability: ${result.confidences[result.label]}
            `;

            //Dispose the tensor to release the memory
            img.dispose;
        }

/*         const result = await net.classify(img);

        document.getElementById('console').innerText = `
        prediction: ${result[0].className} \n 
        probability: ${result[0].probability}
        `;

        img.dispose; */

        //Wait for the next webcam frame to fire
        await tf.nextFrame();
    }
}




app();