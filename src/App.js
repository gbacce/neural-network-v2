import React, { Component } from 'react';
import './App.css';
import * as tf from "@tensorflow/tfjs";
import clientData from "./client-data.js";
import testingData from "./test.js";

class App extends Component {


  render() {

      const trainingData = tf.tensor2d(clientData.map(item => [
          item.savingsRate, item.withdrawalRate, item.lifeExpectancy, item.contributionIncrease, item.portfolioR, item.investmentAmount, item.retirementYear, item.birthYear,
      ]));
      const outputData = tf.tensor2d(clientData.map(item => [
          item.result === "95%+ Probability" ? 1 : 0,
          item.result !== "95%+ Probability" ? 1 : 0
      ]));
      // const testingData = tf.tensor2d(testingData.map(item => [
      //     item.savingsRate, item.withdrawalRate, item.lifeExpectancy, item.contributionIncrease, item.portfolioR, item.investmentAmount, item.retirementYear, item.birthYear,
      // ]));

      const model = tf.sequential();

      model.add(tf.layers.dense({
          inputShape: [8],
          activation: "sigmoid",
          units: 9,
      }));
      model.add(tf.layers.dense({
          inputShape: [9],
          activation: "sigmoid",
          units: 4,
      }));
      model.add(tf.layers.dense({
          activation: "sigmoid",
          units: 2,
      }));
      model.compile({
          loss: "meanSquaredError",
          optimizer: tf.train.adam(.06),
      });
      const startTime = Date.now();
      model.fit(trainingData, outputData, {epochs: 100})
          .then((history) => {
              console.log(history);
              model.predict(testingData).print()
          });
    return (
      <h1 className="App">
          ¯\_(ツ)_/¯
      </h1>
    );
  }
}

export default App;
