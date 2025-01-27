let classifier;

let options = {
  task: "classification",
  debug: true,
};

const trainButton = document.getElementById("trainButton");
trainButton.addEventListener("click", (e) => {
  e.preventDefault();
  setup();
  trainButton.style.display = 'none';
});

const form = document.querySelector("form");
form.addEventListener("submit", (e) => {
  e.preventDefault();

  let sepallength = Number(form.sepal_length.value);
  let sepalwidth = Number(form.sepal_width.value);
  let petallength = Number(form.petal_length.value);
  let petalwidth = Number(form.petal_width.value);

  classifier.classify([sepallength, sepalwidth, petallength, petalwidth], handleResults);
  form.reset();
})

async function setup() {
  document.getElementById('trainButton').style.display = 'none';
  document.querySelector('.train-ai__loader').style.display = 'flex';

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
  document.querySelector('.form__ai-status').style.display = 'none';
  document.querySelector('.form__description').style.display = 'block';
  document.querySelector('.form__fields').style.display = 'block';
  document.querySelector('.train-ai__loader').style.display = 'none';
}

function handleResults(results, error) {
  if (error) {
    console.error(error);
    return;
  }

  label = results[0].label;
  document.querySelector('.result__text').textContent = `It's an ${label}!`;
  document.querySelector('.result__container').style.display = 'block';
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