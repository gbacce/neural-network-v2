import * as tf from '../node_modules/@tensorflow/tfjs/dist/index';
import '../node_modules/@tensorflow/tfjs-node/dist/index';
import clientData from "./client-data.js";
import retirementTesting from "./retirement-test.json";



console.log('test');
// convert/setup our data
const trainingData = tf.tensor2d(clientData.map(item => [
    item.savings_rate, item.withdrawal_rate, item.life_expectancy, item.contribution_increase, item.portfolio_risk_number, item.investment_amount, item.retirement_year, item.birth_year,
]));
const outputData = tf.tensor2d(clientData.map(item => [
    item.retirement === "success" ? 1 : 0,
    item.retirement === "failure" ? 1 : 0
]));
const testingData = tf.tensor2d(irisTesting.map(item => [
    item.savings_rate, item.withdrawal_rate, item.life_expectancy, item.contribution_increase, item.portfolio_risk_number, item.investment_amount, item.retirement_year, item.birth_year,
]));

// build neural network
const model = tf.sequential();

model.add(tf.layers.dense({
    inputShape: [8],
    activation: "sigmoid",
    units: 9,
}));
model.add(tf.layers.dense({
    inputShape: [9],
    activation: "sigmoid",
    units: 3,
}));
model.add(tf.layers.dense({
    activation: "sigmoid",
    units: 3,
}));
model.compile({
    loss: "meanSquaredError",
    optimizer: tf.train.adam(.06),
});
// train/fit our network
const startTime = Date.now();
model.fit(trainingData, outputData, {epochs: 100})
    .then((history) => {
        console.log(history);
        model.predict(testingData).print()
    });

