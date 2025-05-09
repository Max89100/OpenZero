package openzero.pipeline;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class ModelAverager {
    private static final String MODEL_DIR = "ChessData/data/models";
    private static final String OUTPUT_MODEL = "ChessData/data/models/merged_model.zip";

    private static List<File> getModelFiles() {
        File dir = new File(MODEL_DIR);
        if (!dir.exists()) {
            System.err.println("Le dossier " + MODEL_DIR + " n'existe pas !");
            return new ArrayList<>();
        }
        if (!dir.isDirectory()) {
            System.err.println("Le chemin " + MODEL_DIR + " n'est pas un dossier !");
            return new ArrayList<>();
        }

        File[] files = dir.listFiles((d, name) -> name.endsWith(".zip"));
        if (files == null || files.length == 0) {
            System.err.println("Aucun fichier .zip trouvé dans " + MODEL_DIR);
            return new ArrayList<>();
        }

        List<File> modelFiles = new ArrayList<>();
        for (File file : files) {
            modelFiles.add(file);
            System.out.println("Fichier trouvé : " + file.getAbsolutePath());
        }

        return modelFiles;
    }

    public void saveMergedAverageModel() throws IOException {
        List<File> modelFiles = getModelFiles();
        if (modelFiles.isEmpty()) {
            System.err.println("Aucun modèle trouvé !");
            return;
        }

        ComputationGraph mergedModel = averageComputationGraphs(modelFiles);

        // Sauvegarder le modèle fusionné
        ModelSerializer.writeModel(mergedModel, new File(OUTPUT_MODEL), true);
        System.out.println("Fusion terminée ! Modèle enregistré sous " + OUTPUT_MODEL);
    }


    public ComputationGraph averageComputationGraphs(List<File> modelFiles) throws IOException {
        File modelFile = modelFiles.get(0);
        ComputationGraph baseModel = ModelSerializer.restoreComputationGraph(modelFile);
        Map<String, INDArray> paramTable = baseModel.paramTable();

        if (modelFiles.size() == 1) return baseModel; // Rien à fusionner

        for (File model : modelFiles.subList(1, modelFiles.size())) {
            System.out.println("Fusion du modèle : " + model.getName());
            ComputationGraph currentModel = ModelSerializer.restoreComputationGraph(model);
            for (String paramName : paramTable.keySet()) {
                paramTable.get(paramName).addi(currentModel.getParam(paramName));
            }
        }

        // Calcul de la moyenne
        for (String paramName : paramTable.keySet()) {
            paramTable.get(paramName).divi(modelFiles.size());
        }

        return baseModel;
    }



    private static ComputationGraph averageComputationGraph(List<Model> models) {
        ComputationGraph baseModel = (ComputationGraph) models.get(0);
        Map<String, INDArray> paramTable = baseModel.paramTable();

        // Créer une map pour stocker la somme des poids
        Map<String, INDArray> summedParams = new HashMap<>();

        // Initialiser la somme des poids à zéro pour chaque paramètre
        for (String paramName : paramTable.keySet()) {
            summedParams.put(paramName, Nd4j.zeros(paramTable.get(paramName).shape()));
        }

        // Accumuler les poids de tous les modèles
        for (Model model : models) {
            ComputationGraph compGraph = (ComputationGraph) model;
            for (String paramName : paramTable.keySet()) {
                summedParams.get(paramName).addi(compGraph.getParam(paramName));
            }
        }

        // Calculer la moyenne des poids et les appliquer au modèle de base
        for (String paramName : paramTable.keySet()) {
            INDArray averageParam = summedParams.get(paramName).divi(models.size());
            baseModel.setParam(paramName, averageParam);
        }

        return baseModel;
    }

}
