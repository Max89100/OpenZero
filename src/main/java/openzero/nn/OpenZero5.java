package openzero.nn;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.graph.ElementWiseVertex;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import static org.deeplearning4j.nn.conf.CNN2DFormat.NHWC;

public class OpenZero5 implements ModelInterface {

    private final int residualLayers = 5;
    private ComputationGraphConfiguration.GraphBuilder config;

    public OpenZero5() {
        this.config = new NeuralNetConfiguration.Builder()
                .seed(123)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(new Adam())
                .graphBuilder()
                .addInputs("input")
                .setInputTypes(InputType.convolutional(8,8,17,NHWC))
                .setOutputs("ValueOutput","PolicyOutput");

        String previousLayer = addConvolutionLayer(config, "input", "C1", 17, 64);

        for (int i = 0; i < residualLayers; i++) {
            previousLayer = addResidualLayer(config, previousLayer, "R" + i, 64, 64);
        }
        addPolicyHead(config, previousLayer, "PolicyHead", 64);
        addValueHead(config, previousLayer, "ValueHead", 64);
    }

    @Override
    public ComputationGraph getModel() {
        return new ComputationGraph(this.config.build());
    }

    public String addConvolutionLayer(ComputationGraphConfiguration.GraphBuilder graph, String input, String name, int nIn, int nOut) {
        //Convolution
        graph.addLayer(name, new ConvolutionLayer.Builder(3,3)
                .nIn(nIn)
                .nOut(nOut)
                .convolutionMode(ConvolutionMode.Same)
                .build(), input);

        //Batch Normalization
        String batchNorm = name + "_bn";
        graph.addLayer(batchNorm, new BatchNormalization.Builder()
                .eps(1e-5)
                .decay(0.9)
                .build(),name);

        //Activation non-linéaire
        String relu = name + "_relu";
        graph.addLayer(relu, new ActivationLayer.Builder()
                .activation(Activation.RELU)
                .build(),batchNorm);
        return relu;
    }

    public String addResidualLayer(ComputationGraphConfiguration.GraphBuilder graph, String input, String name, int nIn, int nOut) {
        //1ère Convolution
        graph.addLayer(name, new ConvolutionLayer.Builder(3,3)
                .nIn(nIn)
                .nOut(nOut)
                .convolutionMode(ConvolutionMode.Same)
                .build(), input);

        //Batch normalization
        String batchNorm1 = name + "_bn1";
        graph.addLayer(batchNorm1, new BatchNormalization.Builder()
                .eps(1e-5)
                .decay(0.9)
                .build(), name);

        //Activation non-linéaire
        String relu1 = name + "_relu1";
        graph.addLayer(relu1, new ActivationLayer.Builder()
                .activation(Activation.RELU)
                .build(), batchNorm1);

        //2ème Convolution
        String conv2 = name + "_conv2";
        graph.addLayer(conv2, new ConvolutionLayer.Builder(3,3)
                .nIn(nIn)
                .nOut(nOut)
                .convolutionMode(ConvolutionMode.Same)
                .build(), relu1);

        //Batch normalization
        String batchNorm2 = name + "_bn2";
        graph.addLayer(batchNorm2, new BatchNormalization.Builder()
                .eps(1e-5)
                .decay(0.9)
                .build(), conv2);


        //Skip connection (pour faciliter la circulation du gradient lors de la backpropagation
        String skipConnection = name + "_skipConnection";
        graph.addVertex(skipConnection, new ElementWiseVertex(ElementWiseVertex.Op.Add), relu1, batchNorm2);

        //Activation non-linéaire
        String relu2 = name + "_relu2";
        graph.addLayer(relu2, new ActivationLayer.Builder()
                .activation(Activation.RELU)
                .build(), skipConnection);

        return relu2;
    }

    public String addPolicyHead(ComputationGraphConfiguration.GraphBuilder graph, String input, String name, int nIn) {
        //Convolution 1 x 1
        graph.addLayer(name, new ConvolutionLayer.Builder(1,1)
                .nIn(nIn)
                .nOut(73) //8x8x73
                .convolutionMode(ConvolutionMode.Same)
                .build(), input);

        //Batch normalization
        String batchNorm = name + "_bn";
        graph.addLayer(batchNorm, new BatchNormalization.Builder()
                .eps(1e-5)
                .decay(0.9)
                .build(), name);

        //Activation non-linéaire
        String relu = name + "_relu";
        graph.addLayer(relu, new ActivationLayer.Builder()
                .activation(Activation.RELU)
                .build(), batchNorm);

        // 🔴 Couche dense pour préparer la sortie softmax
        String dense = name + "_dense";
        graph.addLayer(dense, new DenseLayer.Builder()
                .nIn(8 * 8 * 73)
                .nOut(4672)  // 🔴 Taille standard pour encoder tous les coups possibles
                .build(), relu);

        // Couche de sortie SOFTMAX (classification)
        graph.addLayer("PolicyOutput", new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                .nIn(4672)
                .nOut(4672)
                .activation(Activation.SOFTMAX)
                .build(), dense);

        return "PolicyOutput";
    }

    public String addValueHead(ComputationGraphConfiguration.GraphBuilder graph, String input, String name, int nIn) {
        // Convolution 1x1 pour réduire la dimensionnalité
        graph.addLayer(name, new ConvolutionLayer.Builder(1,1)
                .nIn(nIn)
                .nOut(32)
                .convolutionMode(ConvolutionMode.Same)
                .build(), input);

        // Batch normalization
        String batchNorm = name + "_bn";
        graph.addLayer(batchNorm, new BatchNormalization.Builder()
                .eps(1e-5)
                .decay(0.9)
                .build(), name);

        //Activation non-linéaire
        String relu = name + "_relu";
        graph.addLayer(relu, new ActivationLayer.Builder()
                .activation(Activation.RELU)
                .build(), batchNorm);

        // Global Average Pooling (un équivalent du flattenLayer)
        String flatten = name + "_flatten";
        graph.addLayer(flatten, new GlobalPoolingLayer(PoolingType.AVG), relu);

        // Couche dense (entièrement connectée)
        String dense1 = name + "_dense1";
        graph.addLayer(dense1, new DenseLayer.Builder()
                .nIn(32)  // Correspond au nombre de filtres après pooling
                .nOut(128)  // Augmentation de la capacité
                .activation(Activation.RELU)
                .build(), flatten);

        // Couche dense 2
        String dense2 = name + "_dense2";
        graph.addLayer(dense2, new DenseLayer.Builder()
                .nIn(128)
                .nOut(64)
                .activation(Activation.RELU)
                .build(), dense1);

        // Couche dense 3
        String dense3 = name + "_dense3";
        graph.addLayer(dense3, new DenseLayer.Builder()
                .nIn(64)
                .nOut(32)
                .activation(Activation.RELU)
                .build(), dense2);

        // Couche de sortie
        graph.addLayer("ValueOutput", new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
                .nIn(32)
                .nOut(1)
                .activation(Activation.TANH)
                .build(), dense3);

        return "ValueOutput";
    }
}
