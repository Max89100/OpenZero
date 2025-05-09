package openzero.pipeline;

import openzero.nn.ModelInterface;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.optimize.listeners.*;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.evaluation.regression.RegressionEvaluation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import java.io.File;
import java.net.InetAddress;
import java.util.concurrent.TimeUnit;


public class DataChessTraining {

    private ComputationGraph model;
    private final int batchSize;
    private final String datasetTrainPath;//chemin du dossier dataset
    private final String datasetTestPath;
    private String modelPath;



    public DataChessTraining(ModelInterface modelInterface, String datasetTrainPath, String datasetTestPath, int batchSize) {
        this.model = modelInterface.getModel();
        this.datasetTrainPath = datasetTrainPath;
        this.datasetTestPath = datasetTestPath;
        this.batchSize = batchSize;
        this.modelPath = "ChessData/data/models/newModel.zip";
        this.model.init();
        System.out.println("🔄 Modèle initialisé !");
        System.out.println("Nombre de paramètres : " + this.model.numParams(false));
        this.model.summary();
    }
    public DataChessTraining(String datasetTrainPath, String datasetTestPath, int batchSize) {
        this.datasetTrainPath = datasetTrainPath;
        this.datasetTestPath = datasetTestPath;
        this.batchSize = batchSize;
        this.modelPath = "ChessData/data/models/newModel.zip";
        this.model = null;
    }
    public DataChessTraining() {
        this.batchSize = 0;
        this.datasetTestPath = null;
        this.datasetTrainPath = null;
    }

    public void train(int epochs) throws Exception {
        System.out.println("Début de l'entrainement");
        HDF5DataSetIterator iterator = new HDF5DataSetIterator(datasetTrainPath,batchSize);

        //Mise en place des listeners
        String machineId = InetAddress.getLocalHost().getHostName(); // Récupère le nom de la machine
        String checkpointPath = "ChessData/data/checkpoints/"+machineId;
        File checkpointDir = new File(checkpointPath);
        if (!checkpointDir.exists()) checkpointDir.mkdirs();
        CheckpointListener checkpointListener = new CheckpointListener.Builder(checkpointPath)
                .deleteExisting(true)
                .keepLast(1)
                .saveEvery(30, TimeUnit.MINUTES)
                .build();
        CollectScoresIterationListener collectScoresIterationListener = new CollectScoresIterationListener(100);
        this.model.setListeners(checkpointListener,
                new PerformanceListener(100, true, true),
                new TimeIterationListener(100, iterator.getTotalSamples()/batchSize),
                collectScoresIterationListener
        );

        this.model.fit(iterator, epochs);
        collectScoresIterationListener.exportScores(new File("ChessData/data/scores/"+machineId+".txt"));
        System.out.println("✅ Perte de l'entraînement : " + this.model.score());
        System.out.println("✅ Modèle entraîné.");
        saveModel(this.model, modelPath);

    }

    public void evaluate() throws Exception {
        RegressionEvaluation evalRegValue = new RegressionEvaluation(); // Pour la sortie "Value" (régression)
        Evaluation evalPolicy = new Evaluation(4672); // Pour la sortie "Policy" (classification)
        HDF5DataSetIterator testIterator = new HDF5DataSetIterator(this.datasetTestPath, batchSize);

        while (testIterator.hasNext()) {
            INDArray[] batch = testIterator.nextBatch();

            MultiDataSet BatchDataSet = new org.nd4j.linalg.dataset.MultiDataSet(new INDArray[]{batch[0]}, new INDArray[]{batch[1], batch[2]});
            BatchDataSet.shuffle(); //On mélange le dataset ! Très important pour éviter les séquences (et le surentrainement !)
            INDArray tensorsFeature = BatchDataSet.getFeatures(0);
            INDArray valueLabel = BatchDataSet.getLabels(0);
            INDArray policyLabel = BatchDataSet.getLabels(1);

            INDArray[] predictions = this.model.output(false, new INDArray[]{tensorsFeature},null,null,null);

            // Évaluation
            evalRegValue.eval(valueLabel, predictions[0]); // 🔴 Régression (Value)
            evalPolicy.eval(policyLabel, predictions[1]); // 🔴 Classification (Policy)
        }
        System.out.println("=== Value Output (Régression) ===");
        System.out.println(evalRegValue.stats());
        System.out.println("=== Policy (Classification) ===");
        System.out.println(evalPolicy.stats());
        System.out.println("✅ Perte de l'évaluation : " + this.model.score());
    }

    public void testDataPipeline(String path, int batchSize) throws Exception {
        HDF5DataSetIterator iterator = new HDF5DataSetIterator(path,batchSize);
        System.out.println("Test de la pipeline et de l'intégrité du dataset");
        INDArray[] batch = iterator.nextBatch();
        // Vérifier la forme des données
        System.out.println("Tensors shape: " + batch[0].shapeInfoToString());
        System.out.println("Results shape: " + batch[1].shapeInfoToString());
        System.out.println("Moves shape: " + batch[2].shapeInfoToString());
        while (iterator.hasNext()) {
            batch = iterator.nextBatch();
        }
        System.out.println("Test effectué, rien à signaler");
        iterator.close();
    }

    public void saveModel(ComputationGraph model, String path) throws Exception {
        String machineId = InetAddress.getLocalHost().getHostName(); // Récupère le nom de la machine
        String finalPath = path.replace(".zip", "_trained_"+machineId+".zip");
        File file = new File(finalPath);
        ModelSerializer.writeModel(model, file, true);
        System.out.println("✅ Modèle sauvegardé sous : " + finalPath);
    }

    public void LoadModel(String path) throws Exception {
        this.modelPath = path;
        File file = new File(path);
        this.model = ModelSerializer.restoreComputationGraph(file);
        System.out.println("✅ Modèle chargé depuis : " + path);
        System.out.println("Nombre de paramètres : " + this.model.numParams(false));
        this.model.summary();
    }
}



