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
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import static org.deeplearning4j.nn.conf.CNN2DFormat.NHWC;

public class OpenZero1 implements ModelInterface {

    public OpenZero1() {
    }
    @Override
    public ComputationGraph getModel() {
        ComputationGraphConfiguration config = new NeuralNetConfiguration.Builder()
                .seed(123)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(new Nesterovs(0.02, 0.9))
                .graphBuilder()
                .addInputs("input")
                .setInputTypes(InputType.convolutional(8,8,17,NHWC))
                .setOutputs("ValueOutput","PolicyOutput")
                //Première couche
                .addLayer("conv1", new ConvolutionLayer.Builder(3,3)
                        .nIn(17)
                        .nOut(32) //32 couches convolutionnelles (32 filtres)
                        .convolutionMode(ConvolutionMode.Same)
                        .build(), "input"
                )
                .addLayer("batchNorm1", new BatchNormalization.Builder()
                        .eps(1e-5)
                        .decay(0.9)
                        .build(),"conv1")
                .addLayer("relu1", new ActivationLayer.Builder()
                        .activation(Activation.RELU)
                        .build(),"batchNorm1")

                //Deuxième couche
                .addLayer("conv2", new ConvolutionLayer.Builder(3,3)
                        .nIn(32)
                        .nOut(32)
                        .convolutionMode(ConvolutionMode.Same)
                        .build(), "relu1")
                .addLayer("batchNorm2", new BatchNormalization.Builder()
                        .eps(1e-5)
                        .decay(0.9)
                        .build(), "conv2")
                .addLayer("relu2", new ActivationLayer.Builder()
                        .activation(Activation.RELU)
                        .build(), "batchNorm2")
                .addLayer("2conv2", new ConvolutionLayer.Builder(3,3)
                        .nIn(32)
                        .nOut(32)
                        .convolutionMode(ConvolutionMode.Same)
                        .build(), "relu2")
                .addLayer("2batchNorm2", new BatchNormalization.Builder()
                        .eps(1e-5)
                        .decay(0.9)
                        .build(), "2conv2")
                .addVertex("skipConnection", new ElementWiseVertex(ElementWiseVertex.Op.Add), "relu1", "2batchNorm2")
                .addLayer("2relu2", new ActivationLayer.Builder()
                        .activation(Activation.RELU)
                        .build(), "skipConnection")

                //AlphaZero répétait la couche 2 jusqu'à 40 (mdr)

                //Tête "value"
                .addLayer("ValueHead", new ConvolutionLayer.Builder(1,1)
                        .nIn(32)
                        .nOut(1)
                        .convolutionMode(ConvolutionMode.Same)
                        .build(), "2relu2")

                .addLayer("headBatchNorm", new BatchNormalization.Builder()
                        .eps(1e-5)
                        .decay(0.9)
                        .build(), "ValueHead")

                .addLayer("headRelu", new ActivationLayer.Builder()
                        .activation(Activation.RELU)
                        .build(), "headBatchNorm")

                .addLayer("Flatten", new GlobalPoolingLayer(PoolingType.AVG), "headRelu")

                .addLayer("headDenseLayer", new DenseLayer.Builder()
                        .nIn(32)
                        .nOut(64)
                        .build(), "Flatten")

                .addLayer("headRelu2", new ActivationLayer.Builder()
                        .activation(Activation.RELU)
                        .build(), "headDenseLayer")

                .addLayer("headDenseLayer2", new DenseLayer.Builder()
                        .nIn(64)
                        .nOut(32)
                        .build(), "headRelu2")

                .addLayer("headRelu3", new ActivationLayer.Builder()
                        .activation(Activation.RELU)
                        .build(), "headDenseLayer2")
                        
                .addLayer("ValueOutput", new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
                        .nIn(32)
                        .nOut(1)
                        .activation(Activation.TANH)
                        .build(), "headRelu3")


                //Tête "policy"

                .addLayer("PolicyHead", new ConvolutionLayer.Builder(1,1)
                        .nIn(32)
                        .nOut(73) //8x8x73
                        .convolutionMode(ConvolutionMode.Same)
                        .build(), "2relu2")

                .addLayer("PolicyBatchNorm", new BatchNormalization.Builder()
                        .eps(1e-5)
                        .decay(0.9)
                        .build(), "PolicyHead")

                .addLayer("PolicyRelu", new ActivationLayer.Builder()
                        .activation(Activation.RELU)
                        .build(), "PolicyBatchNorm")

                // 🔴 Flatten pour préparer la sortie softmax
                .addLayer("FlattenPolicy", new DenseLayer.Builder()
                        .nIn(8 * 8 * 73)
                        .nOut(4672)  // 🔴 Taille standard pour encoder tous les coups possibles
                        .build(), "PolicyRelu")

                .addLayer("PolicyOutput", new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nIn(4672)
                        .nOut(4672)
                        .activation(Activation.SOFTMAX)
                        .build(), "FlattenPolicy")

                .build();

        return new ComputationGraph(config);
    }




}
