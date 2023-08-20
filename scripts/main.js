import { world,system,Vector,MinecraftBlockTypes } from "@minecraft/server";
import { mnist_data } from './mnist'
import { trained_wih } from './trained_wih'
import { trained_who } from './trained_who'

let score = []

class neuralNetwork{
    constructor(inputnodes, hiddennodes, outputnodes, learningrate){
        this.inodes = inputnodes;
        this.hnodes = hiddennodes;
        this.onodes = outputnodes;

        this.wih = randomNormal(0.0, Math.pow(this.inodes, -0.5), [this.hnodes,this.inodes]);
        this.who = randomNormal(0.0, Math.pow(this.hnodes, -0.5), [this.onodes,this.hnodes]);

        this.lr = learningrate;
    }
    train(inputs_list, targets_list,ans){
        let inputs = transpose(inputs_list);
        let targets = transpose(targets_list);
        let hidden_inputs = dot(this.wih, inputs);
        let hidden_outputs = sigmoid(hidden_inputs);
        let final_inputs = dot(this.who, hidden_outputs);
        let final_outputs = sigmoid(final_inputs);
        let output_errors = subtractMatrices(targets,final_outputs);
        let hidden_errors = dot(transpose(this.who), output_errors);
        final_outputs.forEach((value,index)=>{
            if(index==ans){
                world.getDimension('overworld').getBlock(new Vector(0+index,-59,-1)).setType(MinecraftBlockTypes.greenWool)
            } else {
                world.getDimension('overworld').getBlock(new Vector(0+index,-59,-1)).setType(MinecraftBlockTypes.air)
            }
            if(value < 0.2){
                world.getDimension('overworld').getBlock(new Vector(0+index,-59,0)).setType(MinecraftBlockTypes.whiteWool)
            } else if(value < 0.7){
                world.getDimension('overworld').getBlock(new Vector(0+index,-59,0)).setType(MinecraftBlockTypes.yellowWool)
            } else if(value < 0.9){
                world.getDimension('overworld').getBlock(new Vector(0+index,-59,0)).setType(MinecraftBlockTypes.orangeWool)
            } else {
                world.getDimension('overworld').getBlock(new Vector(0+index,-59,0)).setType(MinecraftBlockTypes.redWool)
            }
        })

        const ind = final_outputs.findIndex(x=>x==Math.max(...final_outputs))
        
        world.getDimension('overworld').getBlock(new Vector(0+ind,-59,0)).setType(MinecraftBlockTypes.redWool)

        if(ind == ans){
            score.push(1)
            world.getDimension('overworld').getBlock(new Vector(-2,-59,0)).setType(MinecraftBlockTypes.greenWool)
        } else {
            score.push(0)
            world.getDimension('overworld').getBlock(new Vector(-2,-59,0)).setType(MinecraftBlockTypes.redWool)
        }
        if(score.length > 100) score = score.slice(1)

        world.getAllPlayers().forEach(player=>{
            player.onScreenDisplay.setActionBar(`${ind==ans ? 'a' : ''}정답 : ${ans} 예측 : ${ind} 정답률 : ${(score.filter(x=>x==1).length / score.length * 100)}%\n현재 ${train_state.pro}번 학습중`)
        })

        this.who = addMatrices(
            this.who,
            numberMultiplyMatrices(
                this.lr,
                dot(
                    multiplyMatrices(
                        multiplyMatrices(output_errors,final_outputs),
                        numberMinusMatrices(1.0,final_outputs)
                    ),
                    transpose(hidden_outputs)
                )
            )
        );
        this.wih = addMatrices(
            this.wih,
            numberMultiplyMatrices(
                this.lr,
                dot(
                    multiplyMatrices(
                        multiplyMatrices(hidden_errors,hidden_outputs),
                        numberMinusMatrices(1.0,hidden_outputs)
                    ),
                    transpose(inputs)
                )
            )
        );
    }
    query(inputs_list){
        if(use_trained_data==true){
            let inputs = transpose(inputs_list);
            let hidden_inputs = dot(trained_wih, inputs);
            let hidden_outputs = sigmoid(hidden_inputs);
            let final_inputs = dot(trained_who, hidden_outputs);
            let final_outputs = sigmoid(final_inputs);
            return final_outputs;
        } else {
            let inputs = transpose(inputs_list);
            let hidden_inputs = dot(this.wih, inputs);
            let hidden_outputs = sigmoid(hidden_inputs);
            let final_inputs = dot(this.who, hidden_outputs);
            let final_outputs = sigmoid(final_inputs);
            return final_outputs;
        }
    }
}

let inputNodes = 784;
let hiddenNodes = 200;
let outputNodes = 10;

let learningRate = 0.1;

let neural = new neuralNetwork(inputNodes,hiddenNodes,outputNodes,learningRate);
var use_trained_data = false;

var train_state = {
    pro : 0,
    running : false,
}

system.afterEvents.scriptEventReceive.subscribe(ev=>{
    let {id,message,sourceEntity} = ev;

    id = id.split(':')

    if(id[0]!='ann') return;

    if(id[1]=='train'){
        train_state.running = train_state.running==true ? false : true;
        if(train_state.running==true){
            train_state.pro = 0
            sourceEntity.sendMessage('학습을 시작합니다. 렉이 걸릴 수 있음.')
        } else {
            sourceEntity.sendMessage('학습을 종료합니다.')
        }
        
    } else if(id[1]=='use_trained_data'){
        use_trained_data = true;
    } else if(id[1]=='unuse_trained_data'){
        use_trained_data = false;
    }
})

world.afterEvents.itemUse.subscribe(ev=>{
    const { source,itemStack } = ev;
    const block = source.getBlockFromViewDirection().block

    if((block.typeId=='minecraft:white_concrete'||block.typeId=='minecraft:black_concrete')&&itemStack.typeId=="minecraft:stick"){
        write_block == true ? write_block = false : write_block = true
    }
})


var write_block = false

system.runInterval(()=>{
    if(write_block == false ) return;
    const player = world.getAllPlayers()[0];
    const block = player.getBlockFromViewDirection().block

    const loc = block.location;
    const block1 = world.getDimension('overworld').getBlock(new Vector(loc.x+1,loc.y,loc.z))
    const block2 = world.getDimension('overworld').getBlock(new Vector(loc.x-1,loc.y,loc.z))
    const block3 = world.getDimension('overworld').getBlock(new Vector(loc.x,loc.y,loc.z+1))
    const block4 = world.getDimension('overworld').getBlock(new Vector(loc.x,loc.y,loc.z-1))
    const block5 = world.getDimension('overworld').getBlock(new Vector(loc.x-1,loc.y,loc.z-1))
    const block6 = world.getDimension('overworld').getBlock(new Vector(loc.x+1,loc.y,loc.z-1))
    const block7 = world.getDimension('overworld').getBlock(new Vector(loc.x-1,loc.y,loc.z+1))
    const block8 = world.getDimension('overworld').getBlock(new Vector(loc.x+1,loc.y,loc.z+1))
    

    block.setType(MinecraftBlockTypes.blackConcrete);

    if(block1.typeId=='minecraft:white_concrete'||block1.typeId=='minecraft:light_gray_concrete'){
        block1.setType(MinecraftBlockTypes.grayConcrete)
    }
    if(block2.typeId=='minecraft:white_concrete'||block2.typeId=='minecraft:light_gray_concrete'){
        block2.setType(MinecraftBlockTypes.grayConcrete)
    }
    if(block3.typeId=='minecraft:white_concrete'||block3.typeId=='minecraft:light_gray_concrete'){
        block3.setType(MinecraftBlockTypes.grayConcrete)
    }
    if(block4.typeId=='minecraft:white_concrete'||block4.typeId=='minecraft:light_gray_concrete'){
        block4.setType(MinecraftBlockTypes.grayConcrete)
    }
    if(block5.typeId=='minecraft:white_concrete'){
        block5.setType(MinecraftBlockTypes.lightGrayConcrete)
    }
    if(block6.typeId=='minecraft:white_concrete'){
        block6.setType(MinecraftBlockTypes.lightGrayConcrete)
    }
    if(block7.typeId=='minecraft:white_concrete'){
        block7.setType(MinecraftBlockTypes.lightGrayConcrete)
    }
    if(block8.typeId=='minecraft:white_concrete'){
        block8.setType(MinecraftBlockTypes.lightGrayConcrete)
    }
})

system.runInterval(()=>{
    if (train_state.pro >= 10000 || train_state.running==false) return;

    let data = mnist_data[train_state.pro];
    let answer = data[0];
    let writ = data.slice(1)

    for(let a = 0; a < 28;a++){
        for(let b = 0; b < 28;b++){
            const value = writ[a*28+b]
            if(value == 0){
                world.getDimension('overworld').getBlock(new Vector(-2+a,-59,-3-b)).setType(MinecraftBlockTypes.whiteWool)
            } else if(value < 100) {
                world.getDimension('overworld').getBlock(new Vector(-2+a,-59,-3-b)).setType(MinecraftBlockTypes.lightGrayWool)
            } else if(value < 200) {
                world.getDimension('overworld').getBlock(new Vector(-2+a,-59,-3-b)).setType(MinecraftBlockTypes.grayWool)
            } else {
                world.getDimension('overworld').getBlock(new Vector(-2+a,-59,-3-b)).setType(MinecraftBlockTypes.blackWool)
            }
        }
    }

    let inputs = [numberAddMatrices(0.01,numberMultiplyMatrices(0.99,numberDivMatricesT(writ,255)))];
    let targets = new Array(outputNodes).fill(0.01);
    targets[answer] = 0.99;
    targets = [targets];
    neural.train(inputs, targets,answer);
    train_state.pro += 1;
},2);

system.runInterval(()=>{
    const data = []
    for(let a = 0; a < 28;a++){
        for(let b = 0; b < 28;b++){
            const block = world.getDimension('overworld').getBlock(new Vector(24-b,-60,39-a));
            if(block.typeId=="minecraft:white_concrete"){
                data.push(0);
            } else if(block.typeId=='minecraft:gray_concrete') {
                data.push(getRandomIntBetween(150,250));
            } else if(block.typeId=='minecraft:light_gray_concrete') {
                data.push(getRandomIntBetween(50,150));
            } else {
                data.push(255)
            }
        }
    }
    let inputs = [numberAddMatrices(0.01,numberMultiplyMatrices(0.99,numberDivMatricesT(data,255)))];
    const output = neural.query(inputs);
    const sorted_list = new Array(...output).sort((a,b)=>{
        return a - b;
    })
    for(let i = 0; i < 10; i++){
        world.getDimension('overworld').getBlock(new Vector(3+i,-60,11)).setType(MinecraftBlockTypes.whiteWool)
    }
    for(let i = 0; i < 3; i++){
        if(i==0){
            world.getDimension('overworld').getBlock(new Vector(3+output.findIndex(x=>x==sorted_list[i]),-60,11)).setType(MinecraftBlockTypes.redWool)
        } else if(i==1){
            world.getDimension('overworld').getBlock(new Vector(3+output.findIndex(x=>x==sorted_list[i]),-60,11)).setType(MinecraftBlockTypes.orangeWool)
        } else if(i==2){
            world.getDimension('overworld').getBlock(new Vector(3+output.findIndex(x=>x==sorted_list[i]),-60,11)).setType(MinecraftBlockTypes.yellowWool)
        }
    }
},20)

// const { log } = require('console');
// const fs = require('fs');

// const results = []; 
// const filePath = './mnist/mnist_test.csv';

// fs.readFile(filePath, 'utf8', function(err, data) {
//     if (err) {
//         console.error(err);
//     } else {
//         let i = 0;
//         for(let x of data.split('\n')){
//             i++
//             console.log(i);
//             // if(i > 200) return;
//             x = x.split(',');
//             let answer = x[0];
//             let inputs = [numberAddMatrices(0.01,numberMultiplyMatrices(0.99,numberDivMatricesT(x.slice(1),255)))];
//             let targets = new Array(outputNodes).fill(0.01);
//             targets[answer] = 0.99;
//             targets = [targets];
//             neural.train(inputs, targets,answer);
//             // console.log(neural.who);
//         }
//         // console.log('train complete!');
//         // console.log(neural.wih);
//         // console.log('==============================================================');
//         // console.log(neural.who);
//     }
// });

function getRandomIntBetween(min, max) {
    min = Math.ceil(min);
    max = Math.floor(max);
    return Math.floor(Math.random() * (max - min + 1)) + min;
  }


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
    return x.map(y=>typeof y == 'object' ? sigmoid(y) : (1/ (1+Math.exp(-y))));
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

