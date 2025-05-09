package openzero.pipeline;
import lombok.Getter;
import org.bytedeco.hdf5.*;
import org.bytedeco.hdf5.global.hdf5;
import org.bytedeco.javacpp.FloatPointer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.dataset.api.MultiDataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;
import org.nd4j.linalg.factory.Nd4j;

import java.lang.Exception;
import java.util.Arrays;

import static org.bytedeco.hdf5.global.hdf5.H5set_free_list_limits;

public class HDF5DataSetIterator implements MultiDataSetIterator {
    private final int batchSize;
    private int currentIndex = 0;
    @Getter
    private final int totalSamples;
    private final H5File file;
    private final DataSet tensorsDataSet;
    private final DataSet movesDataSet;
    private final DataSet resultsDataSet;
    private FloatPointer tensorsPointer;
    private FloatPointer movesPointer;
    private FloatPointer resultsPointer;

    public HDF5DataSetIterator(String filePath, int batchSize) {
        this.batchSize = batchSize;

         this.file = new H5File(filePath, hdf5.H5F_ACC_RDONLY);
         this.tensorsDataSet = file.openDataSet("/tensors");
         this.movesDataSet = file.openDataSet("/moves");
         this.resultsDataSet = file.openDataSet("/results");

         long[] Tensorsdims = new long[tensorsDataSet.getSpace().getSimpleExtentNdims()];
         tensorsDataSet.getSpace().getSimpleExtentDims(Tensorsdims);
         long[] movesdims = new long[movesDataSet.getSpace().getSimpleExtentNdims()];
         movesDataSet.getSpace().getSimpleExtentDims(movesdims);
         long[] resultsdims = new long[resultsDataSet.getSpace().getSimpleExtentNdims()];
         resultsDataSet.getSpace().getSimpleExtentDims(resultsdims);
         totalSamples = (int)Tensorsdims[0]; //dim[0] = nombre d'échantillons, dim[1] = 8, 8 et 16 (pour le tenseur)
         System.out.println("Nombre d'exemples: " + totalSamples);
         System.out.println("Taille des tensors: " + Arrays.toString(Tensorsdims));
         System.out.println("Taille des moves: " + Arrays.toString(movesdims));
         System.out.println("Taille des results: " + Arrays.toString(resultsdims));

         int tensorDataType = tensorsDataSet.getSpace().getSimpleExtentType();
         System.out.println("Type de données de tensors : " + tensorDataType);

    }



    @Override
    public MultiDataSet next() {
        H5set_free_list_limits(0, 0, 0, 0, 0, 0);
        if (!hasNext()) {
            throw new IllegalStateException("No more batches available");
        }
        //System.out.println("Lecture depuis HDF5 : " + currentIndex + " - " + (currentIndex + batchSize));
        int startIndex = currentIndex;
        int endIndex = Math.min(currentIndex + batchSize, totalSamples);
        int actualBatchSize = endIndex - currentIndex;
        DSetMemXferPropList transferPropList = new DSetMemXferPropList(hdf5.H5P_DEFAULT);

        // Définition des dimensions de la lecture
        long[] countTensors = {actualBatchSize, 8, 8, 17};
        long[] startTensors = {startIndex, 0, 0, 0};
        long[] countMoves = {actualBatchSize, 4672};
        long[] startMoves = {startIndex, 0,0,0};
        long[] countResults = {actualBatchSize};
        long[] startResults = {startIndex};

        DataType floatDataType = PredType.IEEE_F32LE();

        // **1. Allocation des FloatPointer pour la lecture**
        if (tensorsPointer != null) tensorsPointer.deallocate();
        FloatPointer tensorsPointer = new FloatPointer((long) actualBatchSize * 8 * 8 * 17);
        if (movesPointer != null) movesPointer.deallocate();
        FloatPointer movesPointer = new FloatPointer((long) actualBatchSize * 4672);
        if (resultsPointer != null) resultsPointer.deallocate();
        FloatPointer resultsPointer = new FloatPointer(actualBatchSize);



        // **2. Lecture depuis HDF5 vers FloatPointer**
        if (tensorsDataSet.getSpace().isNull() || movesDataSet.getSpace().isNull() || resultsDataSet.getSpace().isNull()) {
            throw new RuntimeException("DataSpace HDF5 invalide !");
        }
        try {
            DataSpace tensorsSpace = tensorsDataSet.getSpace();
            tensorsSpace.selectHyperslab(hdf5.H5S_SELECT_SET, countTensors, startTensors);
            DataSpace tensorsMemSpace = new DataSpace(4, countTensors);
            tensorsDataSet.read(tensorsPointer, floatDataType, tensorsMemSpace, tensorsSpace, transferPropList);
            tensorsMemSpace.deallocate();
            tensorsSpace.deallocate();
        } catch (Exception e) {
            System.err.println("Erreur lors de l'accès au dataspace ou lors de la lecture : " + e.getMessage());
        }


        try {
            DataSpace movesSpace = movesDataSet.getSpace();
            movesSpace.selectHyperslab(hdf5.H5S_SELECT_SET, countMoves, startMoves);
            DataSpace movesMemSpace = new DataSpace(2, countMoves);
            movesDataSet.read(movesPointer, floatDataType, movesMemSpace, movesSpace, transferPropList);
            movesMemSpace.deallocate();
            movesSpace.deallocate();
        } catch (Exception e) {
            System.err.println("Erreur lors de l'accès au dataspace ou lors de la lecture : " + e.getMessage());
        }


        try {
            DataSpace resultsSpace = resultsDataSet.getSpace();
            resultsSpace.selectHyperslab(hdf5.H5S_SELECT_SET, countResults, startResults);
            DataSpace resultsMemSpace = new DataSpace(1,countResults);
            resultsDataSet.read(resultsPointer, floatDataType, resultsMemSpace, resultsSpace, transferPropList);
            resultsMemSpace.deallocate();
            resultsSpace.deallocate();
        } catch (Exception e) {
            System.err.println("Erreur lors de l'accès au dataspace ou lors de la lecture : " + e.getMessage());
        }

        // **3. Conversion des FloatPointer en float[]**
        float[] tensorsArray = new float[actualBatchSize * 8 * 8 * 17];
        float[] movesArray = new float[actualBatchSize * 4672];
        float[] resultsArray = new float[actualBatchSize];

        // **4. Libération des FloatPointer (plus besoin)**
        try {
            tensorsPointer.get(tensorsArray);
            movesPointer.get(movesArray);
            resultsPointer.get(resultsArray);
        } finally {
            tensorsPointer.deallocate();
            movesPointer.deallocate();
            resultsPointer.deallocate();
        }


        // **5. Création des INDArrays sans allocation supplémentaire**
        INDArray features = Nd4j.create(tensorsArray, new int[]{actualBatchSize, 8, 8, 17}, 'c');
        INDArray moves = Nd4j.create(movesArray, new int[]{actualBatchSize, 4672}, 'c');
        INDArray results = Nd4j.create(resultsArray, new int[]{actualBatchSize, 1}, 'c');
        currentIndex = endIndex;
        org.nd4j.linalg.dataset.MultiDataSet BatchDataSet = new org.nd4j.linalg.dataset.MultiDataSet(new INDArray[]{features}, new INDArray[]{results, moves});
        BatchDataSet.shuffle();
        return BatchDataSet;
    }

    public void close() {
        System.out.println("Fermeture du fichier HDF5");
        if (tensorsDataSet != null) tensorsDataSet.deallocate();
        if (movesDataSet != null) movesDataSet.deallocate();
        if (resultsDataSet != null) resultsDataSet.deallocate();
        if (file != null) file.deallocate();
    }

    @Override
    public boolean hasNext() {
        return currentIndex < totalSamples;
    }

    @Override
    public MultiDataSet next(int i) {
        return null;
    }

    @Override
    public void reset() {
        this.currentIndex = 0;
    }

    @Override
    public boolean resetSupported() {
        return true;
    }

    @Override
    public void setPreProcessor(MultiDataSetPreProcessor multiDataSetPreProcessor) {

    }

    @Override
    public MultiDataSetPreProcessor getPreProcessor() {
        return null;
    }



    @Override
    public boolean asyncSupported() {
        return false;
    }
























    public synchronized  INDArray[] nextBatch() {
        H5set_free_list_limits(0, 0, 0, 0, 0, 0);
        if (!hasNext()) {
            throw new IllegalStateException("No more batches available");
        }
        System.out.println("Lecture depuis HDF5 : " + currentIndex + " - " + (currentIndex + batchSize));
        int startIndex = currentIndex;
        int endIndex = Math.min(currentIndex + batchSize, totalSamples);
        int actualBatchSize = endIndex - currentIndex;
        DSetMemXferPropList transferPropList = new DSetMemXferPropList(hdf5.H5P_DEFAULT);

        // Définition des dimensions de la lecture
        long[] countTensors = {actualBatchSize, 8, 8, 17};
        long[] startTensors = {startIndex, 0, 0, 0};
        long[] countMoves = {actualBatchSize, 4672};
        long[] startMoves = {startIndex, 0,0,0};
        long[] countResults = {actualBatchSize};
        long[] startResults = {startIndex};

        DataType floatDataType = PredType.IEEE_F32LE();

        // **1. Allocation des FloatPointer pour la lecture**
        if (tensorsPointer != null) tensorsPointer.deallocate();
        FloatPointer tensorsPointer = new FloatPointer((long) actualBatchSize * 8 * 8 * 17);
        if (movesPointer != null) movesPointer.deallocate();
        FloatPointer movesPointer = new FloatPointer((long) actualBatchSize * 4672);
        if (resultsPointer != null) resultsPointer.deallocate();
        FloatPointer resultsPointer = new FloatPointer(actualBatchSize);



        // **2. Lecture depuis HDF5 vers FloatPointer**
        if (tensorsDataSet.getSpace().isNull() || movesDataSet.getSpace().isNull() || resultsDataSet.getSpace().isNull()) {
            throw new RuntimeException("DataSpace HDF5 invalide !");
        }
        try {
            DataSpace tensorsSpace = tensorsDataSet.getSpace();
            tensorsSpace.selectHyperslab(hdf5.H5S_SELECT_SET, countTensors, startTensors);
            DataSpace tensorsMemSpace = new DataSpace(4, countTensors);
            tensorsDataSet.read(tensorsPointer, floatDataType, tensorsMemSpace, tensorsSpace, transferPropList);
            tensorsMemSpace.deallocate();
            tensorsSpace.deallocate();
        } catch (Exception e) {
            System.err.println("Erreur lors de l'accès au dataspace ou lors de la lecture : " + e.getMessage());
        }


        try {
            DataSpace movesSpace = movesDataSet.getSpace();
            movesSpace.selectHyperslab(hdf5.H5S_SELECT_SET, countMoves, startMoves);
            DataSpace movesMemSpace = new DataSpace(2, countMoves);
            movesDataSet.read(movesPointer, floatDataType, movesMemSpace, movesSpace, transferPropList);
            movesMemSpace.deallocate();
            movesSpace.deallocate();
        } catch (Exception e) {
            System.err.println("Erreur lors de l'accès au dataspace ou lors de la lecture : " + e.getMessage());
        }


        try {
            DataSpace resultsSpace = resultsDataSet.getSpace();
            resultsSpace.selectHyperslab(hdf5.H5S_SELECT_SET, countResults, startResults);
            DataSpace resultsMemSpace = new DataSpace(1,countResults);
            resultsDataSet.read(resultsPointer, floatDataType, resultsMemSpace, resultsSpace, transferPropList);
            resultsMemSpace.deallocate();
            resultsSpace.deallocate();
        } catch (Exception e) {
            System.err.println("Erreur lors de l'accès au dataspace ou lors de la lecture : " + e.getMessage());
        }

        // **3. Conversion des FloatPointer en float[]**
        float[] tensorsArray = new float[actualBatchSize * 8 * 8 * 17];
        float[] movesArray = new float[actualBatchSize * 4672];
        float[] resultsArray = new float[actualBatchSize];

        // **4. Libération des FloatPointer (plus besoin)**
        try {
            tensorsPointer.get(tensorsArray);
            movesPointer.get(movesArray);
            resultsPointer.get(resultsArray);
        } finally {
            tensorsPointer.deallocate();
            movesPointer.deallocate();
            resultsPointer.deallocate();
        }


        // **5. Création des INDArrays sans allocation supplémentaire**
        INDArray features = Nd4j.create(tensorsArray, new int[]{actualBatchSize, 8, 8, 17}, 'c');
        INDArray moves = Nd4j.create(movesArray, new int[]{actualBatchSize, 4672}, 'c');
        INDArray results = Nd4j.create(resultsArray, new int[]{actualBatchSize, 1}, 'c');

        currentIndex = endIndex;
        return new INDArray[]{features, results, moves};
    }
}
