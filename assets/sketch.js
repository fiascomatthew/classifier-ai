let classifier;

let options = {
  task: "classification",
  debug: true,
};

const trainButton = document.getElementById("trainButton");
trainButton.addEventListener("click", (e) => {
  e.preventDefault();
  setup();
});

const form = document.querySelector("form");
form.addEventListener("submit", (e) => {
  e.preventDefault();

  let sepallength = Number(form.sepal_length.value);
  let sepalwidth = Number(form.sepal_width.value);
  let petallength = Number(form.petal_length.value);
  let petalwidth = Number(form.petal_width.value);

  classifier.classify([sepallength, sepalwidth, petallength, petalwidth], handleResults);
})

async function setup() {
  ml5.setBackend("webgl");
  classifier = ml5.neuralNetwork(options);
  const data = await processCSV();

  for (let i = 0; i < data.length; i++) {
    let item = data[i];
    let inputs = [item.sepallength, item.sepalwidth, item.petallength, item.petalwidth];
    let outputs = [item.label];
    classifier.addData(inputs, outputs);
  }

  classifier.normalizeData();

  const trainingOptions = {
    epochs: 16,
    batchSize: 6,
  };

  classifier.train(trainingOptions, finishedTraining);
}

function finishedTraining() {
  console.log('hello')
}

function handleResults(results, error) {
  if (error) {
    console.error(error);
    return;
  }
  label = results[0].label;
  console.log(results, label);
}

async function processCSV() {
  const csvFile = await fetch('./data/iris_synthetic_data.csv');
  const csvData = await csvFile.text();
  const data = convertCSVtoObject(csvData);
  return data;
}

function convertCSVtoObject(data) {
  const rows = data.split('\n');
  const headers = rows.shift().split(',').map((item) => item.replace(/\s/g, ''));;

  return rows.map((row) => {
    const values = row.split(',');

    return headers.reduce((obj, key, index) => {
      obj[key] = key === 'label' ? values[index] : Number(values[index]);
      return obj;
    }, {});
  });
};