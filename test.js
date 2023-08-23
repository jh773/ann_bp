
const score = {
    process : 0,
    correct : 0,
    wrong : 0,
}

class neuralNetwork{
    constructor(inputnodes, hiddennodes, outputnodes, learningrate){
        this.inodes = inputnodes;
        this.hnodes = hiddennodes;
        this.onodes = outputnodes;

        this.wih = randomNormal(0.0, Math.pow(this.inodes, -0.5), [this.hnodes,this.inodes]);
        this.who = randomNormal(0.0, Math.pow(this.hnodes, -0.5), [this.onodes,this.hnodes]);

        this.lr = learningrate;
    }
    train(inputsList, targetsList,ans){
        let inputs = transpose(inputsList);
        let targets = transpose(targetsList);
        let hiddenInputs = dot(this.wih, inputs);
        let hiddenOutputs = sigmoid(hiddenInputs);
        let finalInputs = dot(this.who, hiddenOutputs);
        let finalOutputs = sigmoid(finalInputs);
        let outputErrors = subtractMatrices(targets,finalOutputs);
        let hiddenErrors = dot(transpose(this.who), outputErrors);


        const ind = finalOutputs.findIndex(x=>x==Math.max(...finalOutputs))
        
        results.push(ind==ans)

        this.who = addMatrices(
            this.who,
            numberMultiplyMatrices(
                this.lr,
                dot(
                    multiplyMatrices(
                        multiplyMatrices(outputErrors,finalOutputs),
                        numberMinusMatrices(1.0,finalOutputs)
                    ),
                    transpose(hiddenOutputs)
                )
            )
        );
        this.wih = addMatrices(
            this.wih,
            numberMultiplyMatrices(
                this.lr,
                dot(
                    multiplyMatrices(
                        multiplyMatrices(hiddenErrors,hiddenOutputs),
                        numberMinusMatrices(1.0,hiddenOutputs)
                    ),
                    transpose(inputs)
                )
            )
        );
    }
    query(inputsList){
        let inputs = transpose(inputsList);
        let hiddenInputs = dot(this.wih, inputs);
        let hiddenOutputs = sigmoid(hiddenInputs);
        let finalInputs = dot(this.who, hiddenOutputs);
        let finalOutputs = sigmoid(finalInputs);
        return finalOutputs;
    }
}

let inputNodes = 784;
let hiddenNodes = 200;
let outputNodes = 10;

let learningRate = 0.1;

let neural = new neuralNetwork(inputNodes,hiddenNodes,outputNodes,learningRate);

const { log } = require('console');
const fs = require('fs');

let results = []; 
const filePath = './mnist/mnist_test.csv';

const epochs = 1;

fs.readFile(filePath, 'utf8', function(err, data) {
    if (err) {
        console.error(err);
    } else {
        let i = 0;
        for(let k = 0; k < epochs; k++){
            for(let x of data.split('\n')){
                i++;
                if(i%100==0){
                    console.log(i,`정답률 : ${results.filter(x=>x==1).length / results.length * 100}%`);
                    // results = [];
                }
                x = x.split(',');
                let answer = x[0];
                let inputs = [numberAddMatrices(0.01,numberMultiplyMatrices(0.99,numberDivMatricesT(x.slice(1),255)))];
                let targets = new Array(outputNodes).fill(0.01);
                targets[answer] = 0.99;
                targets = [targets];
                neural.train(inputs, targets,answer);
            }
        }
    }
});



function numberAddMatrices(number,array){
    return array.map(x=>typeof x == 'object' ? numberAddMatrices(number,x) : number + x);
}
function numberMultiplyMatrices(number,array){
    return array.map(x=>typeof x == 'object' ? numberMultiplyMatrices(number,x) : number * x);
}
function numberMinusMatrices(number,array){
    return array.map(x=>typeof x == 'object' ? numberMinusMatrices(number,x) : number - x);
}
function numberDivMatrices(number,array){
    return array.map(x=>typeof x == 'object' ? numberDivMatrices(number,x) : number * x);
}
function numberMinusMatricesT(array, number){
    return array.map(x=>typeof x == 'object' ? numberMinusMatricesT(x,number) : x - number);
}
function numberDivMatricesT(array, number){
    return array.map(x=>typeof x == 'object' ? numberDivMatricesT(x,number) : x / number);
}
function addMatrices(A, B) {
    if (A.length !== B.length || A[0].length !== B[0].length) {
        throw new Error("행렬 A와 B의 차원이 일치하지 않습니다.");
    }
    
    const result = new Array(A.length)
        .fill(null)
        .map(() => new Array(A[0].length).fill(null));
    
    for (let i = 0; i < A.length; i++) {
        for (let j = 0; j < A[0].length; j++) {
            result[i][j] = A[i][j] + B[i][j];
        }
    }

    return result;
}
function subtractMatrices(A, B) {
    if (A.length !== B.length || A[0].length !== B[0].length) {
        throw new Error("행렬 A와 B의 차원이 일치하지 않습니다.");
    }
    
    const result = new Array(A.length)
        .fill(null)
        .map(() => new Array(A[0].length).fill(null));
    
    for (let i = 0; i < A.length; i++) {
        for (let j = 0; j < A[0].length; j++) {
            result[i][j] = A[i][j] - B[i][j];
        }
    }
    
    return result;
}
function multiplyMatrices(A, B) {
    if (A.length !== B.length || A[0].length !== B[0].length) {
        throw new Error(`행렬 A와 B의 차원이 일치하지 않습니다.`);
    }
    
    const result = new Array(A.length)
        .fill(null)
        .map(() => new Array(A[0].length).fill(null));
    
    for (let i = 0; i < A.length; i++) {
        for (let j = 0; j < A[0].length; j++) {
            result[i][j] = A[i][j] * B[i][j];
        }
    }
    
    return result;
}
function sigmoid(x) {
    return x.map(y=>typeof y == 'object' ? sigmoid(y) : (1/ (1+Math.exp(-y))*0.99));
}
function dot(A, B) {
    if (A[0].length !== B.length) {
        throw new Error(`행렬 A의 열 수와 행렬 B의 행 수가 일치하지 않습니다. ${A[0].length} ${B.length}`);
    }
    
    const result = new Array(A.length).fill(0).map(() => new Array(B[0].length).fill(0));
    
    for (let i = 0; i < A.length; i++) {
        for (let j = 0; j < B[0].length; j++) {
            for (let k = 0; k < A[0].length; k++) {
                result[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    
    return result;
}
function randomNormal(mean, stdDev, size) {
    const normalDistribution = (mean, stdDev) => {
        const u1 = Math.random();
        const u2 = Math.random();
        const z0 = Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
        return z0 * stdDev + mean;
    };
    
    const _recursiveRandomNormal = (mean, stdDev, size) => {
        if (size.length === 1) {
            return Array.from({ length: size[0] }, () => normalDistribution(mean, stdDev));
        } else {
            return Array.from({ length: size[0] }, () => _recursiveRandomNormal(mean, stdDev, size.slice(1)));
        }
    };
    
    const result = _recursiveRandomNormal(mean, stdDev, size);
    
    return result;
}
function transpose(matrix) {
    const rows = matrix.length;
    const cols = matrix[0].length;
  
    const transposedMatrix = new Array(cols)
        .fill(null)
        .map(() => new Array(rows).fill(null));
    
    for (let i = 0; i < rows; i++) {
        for (let j = 0; j < cols; j++) {
            transposedMatrix[j][i] = matrix[i][j];
        }
    }
    return transposedMatrix;
}

