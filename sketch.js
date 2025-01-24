let classifier;
let options = {
  task: "classification",
  debug: true,
};

setup();

let sepallength = 4.4;
let sepalwidth = 3.8;
let petallength = 1.5;
let petalwidth = 0.1;
let label = "Iris-setosa";


async function setup() {
  ml5.setBackend("webgl");
  classifier = ml5.neuralNetwork(options);
  const data = await processCSV();
  console.log(data);
  for (let i = 0; i < data.length; i++) {
    let item = data[i];
    let inputs = [item.sepallength, item.sepalwidth, item.petallength, item.petalwidth];
    let outputs = [item.label];
    classifier.addData(inputs, outputs);
    classifier.normalizeData();
  }

  const trainingOptions = {
    epochs: 16,
    batchSize: 6,
  };

  console.log(classifier);
  classifier.train(trainingOptions, finishedTraining);
}

function finishedTraining() {
  classify();
}

function classify() {
  const input = [sepallength, sepalwidth, petallength, petalwidth];
  classifier.classify(input, handleResults);
}

function handleResults(results, error) {
  if (error) {
    console.error(error);
    return;
  }
  label = results[0].label;
  console.log(results);
  classify();
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