package openzero.utils;


//import openzero.MCTS.MonteCarloTreeSearch;
import com.github.bhlangonijr.chesslib.Board;
import com.github.bhlangonijr.chesslib.move.Move;
import com.github.bhlangonijr.chesslib.move.MoveException;
import com.github.bhlangonijr.chesslib.Square;
import openzero.MCTS.MonteCarloTreeSearch;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.io.File;
import java.util.*;

import static org.nd4j.linalg.indexing.NDArrayIndex.all;
import static org.nd4j.linalg.indexing.NDArrayIndex.point;

public class ChessModelInterpreter {
    private static final Map<Character, Integer> pieceMap = new HashMap<>();
    private static final Map<Map.Entry<Integer,Integer>, Integer> directionMap = new HashMap<>();
    private static final Map<Map.Entry<Integer, Integer>, Map<Integer, Character>> promotionMap = new HashMap<>();
    private static final Map<Map.Entry<Integer, Integer>, Map.Entry<Integer, Integer>> auxiliaryPromotionMap = new HashMap<>();
    private static final Map<Integer, Map.Entry<Integer, Integer>> inverseDirectionMap = new HashMap<>();
    public ComputationGraph model;



    public ChessModelInterpreter(ComputationGraph model) {
        this.model = model;
    }
    public ChessModelInterpreter(){
        this.model = null;
    }

    public void LoadModel(String path) throws Exception {
        File file = new File(path);
        this.model = ModelSerializer.restoreComputationGraph(file);
        System.out.println("✅ Modèle chargé depuis : " + path);
        System.out.println("Nombre de paramètres : " + this.model.numParams(false));
        this.model.summary();
    }

    /**
     * Convertit une position FEN en tenseur utilisable dans les CNN.
     * @param fen le string fen
     * @return le tenseur associé
     */
    public INDArray FenToTensor(String fen){
        INDArray tensor = Nd4j.zeros(8,8,17);

        // Séparation des 6 champs de la FEN
        String[] parts = fen.split(" ");
        String[] rows = parts[0].split("/");

        // Placement des pièces sur le tenseur
        for (int row = 0; row < 8; row++) {
            int col = 0;
            for (char c : rows[row].toCharArray()) {
                if (Character.isDigit(c)) {
                    col += Character.getNumericValue(c); // Sauter les cases vides
                } else {
                    int channel = pieceMap.get(c);
                    tensor.putScalar(new int[]{row, col, channel}, 1);
                    col++;
                }
            }
        }

        // Tour de jeu
        if (parts[1].equals("w")) {
            tensor.get(all(), all(), point(12)).assign(1);
        }

        // Informations de roque
        String castling = parts[2];
        if (castling.contains("K")) tensor.get(all(),all(),point(13)).assign(1);
        if (castling.contains("Q")) tensor.get(all(),all(),point(14)).assign(1);
        if (castling.contains("k")) tensor.get(all(),all(),point(15)).assign(1);
        if (castling.contains("q")) tensor.get(all(),all(),point(16)).assign(1);

        INDArray reshapedTensor = tensor.reshape(1,8,8,17); //on reshape car les CNN attendent un format NHWC (number height width channel)
        return reshapedTensor;
    }





    /**
     * Retourne les topN plus grandes probabilités dans une INDArray d'indice et une INDArray de valeurs (décroissant).
     * @param softmax
     * @param topN
     * @return
     */
    public INDArray[] argsort(INDArray softmax, int topN){
        INDArray[] sortedSoftmax = Nd4j.sortWithIndices(softmax,1,false);
        INDArray sortedIndices = sortedSoftmax[0].get(NDArrayIndex.all(),NDArrayIndex.interval(0, topN));
        INDArray sortedValues = sortedSoftmax[1].get(NDArrayIndex.all(), NDArrayIndex.interval(0, topN));
        return new INDArray[]{sortedIndices,sortedValues};
    }

    /**
     * Permet de traduire un indice du vecteur en coup sous forme de string
     * @param index l'indice du vecteur de probabilité
     * @param chessTensor le tenseur de l'échiquier fourni en entrée du modèle, nécessaire pour ajouter le suffixe de la promotion au coup.
     * @return
     */
    public String translateMoves(int index, INDArray chessTensor){

        int square = index / 73;
        int direction = index % 73;

        int fromRow = square / 8;
        int fromCol = square % 8;
        int fromRank = 8 - fromRow; // Rangées 1-8
        char fromFile = (char) ('a' + fromCol); // Colonnes a-h

        Map.Entry<Integer, Integer> directionEntry;
        String move = "";
        if(direction <= 63) {
            /**
             * Coup classique + promotion en reine par défaut
             */

            directionEntry = inverseDirectionMap.get(direction);
            int toRank = fromRank + directionEntry.getKey();
            char toFile = (char) (fromFile+directionEntry.getValue());
            move = String.format("%c%d%c%d", fromFile, fromRank, toFile, toRank);
            //promotion blanche
            if(chessTensor.getInt(0,fromRow,fromCol,0) == 1 && fromRank == 7) {
                if(promotionMap.containsKey(directionEntry)) move +="Q";
            }
            //promotion noire
            if(chessTensor.getInt(0,fromRow,fromCol,6) == 1 && fromRank == 2) {
                if(promotionMap.containsKey(directionEntry)) {
                    move +="q";
                }
            }
        }
        else {
            /**
             * Une direction supérieure à 63 concerne une sous-promotion.
             */
            int color = -1;
            //promotion blanche
            if(chessTensor.getInt(0,fromRow,fromCol,0) == 1 && fromRank == 7) {
                color = 1;
            }
            //promotion noire
            if(chessTensor.getInt(0,fromRow,fromCol,6) == 1 && fromRank == 2) {
                color = 0;
            }

            directionEntry = auxiliaryPromotionMap.get(new AbstractMap.SimpleEntry<>(direction, color));
            int toRank = fromRank + directionEntry.getKey();
            char toFile = (char) (fromFile+directionEntry.getValue());
            move = String.format("%c%d%c%d", fromFile, fromRank, toFile, toRank);
            if(promotionMap.containsKey(directionEntry)) {
                char underpromotion = promotionMap.get(directionEntry).get(direction);
                move += underpromotion;
            }
            else {
                System.out.println("Erreur : direction indiquant une promotion, mais mouvement non valide.");
            }
        }
        return move;
    }

    public List<String> getListMoves(INDArray softmaxIndices, INDArray chessTensor){
        List<String> moves = new ArrayList<>();
        for(int i=0;i<softmaxIndices.columns();i++){
            moves.add(translateMoves(softmaxIndices.getInt(i),chessTensor));
        }
        return moves;
    }

    public void printMoves(List<String> moves, INDArray softmaxValues){
        for(int i=0;i<moves.size();i++){
            System.out.println(moves.get(i)+" : "+softmaxValues.getFloat(i));
        }
    }

    public float evaluate(String fen) {
        INDArray tensorBoard = FenToTensor(fen);
        Board board = new Board();
        board.loadFromFen(fen);
        INDArray[] output = this.model.output(false,new INDArray[]{tensorBoard}, null,null,null);
        INDArray valueOutput = output[0];
        return valueOutput.getFloat(0,0);
    }
    /**
     * Le modèle initialisé prédit le coup à jouer à partir d'un format FEN
     * @param fen
     * @return
     */
    public String makeMove(String fen){
        INDArray tensorBoard = FenToTensor(fen);
        Board board = new Board();
        board.loadFromFen(fen);
        INDArray[] output = this.model.output(false,new INDArray[]{tensorBoard}, null,null,null);
        INDArray valueOutput = output[0];
        INDArray policyOutput = output[1];
        INDArray[] sortedSoftmax = argsort(policyOutput,20);
        List<String> listMoves = getListMoves(sortedSoftmax[0], tensorBoard);
        System.out.println("Evaluation de la position : "+valueOutput);
        printMoves(listMoves, sortedSoftmax[1]);

        int index = 0;
        String move;
        do {
            move = listMoves.get(index);
            index++;
        }while(!board.legalMoves().toString().contains(move));
        return move;
    }

    public String makeMoveWithMCTS(String fen, MonteCarloTreeSearch mcts){
        INDArray tensorBoard = FenToTensor(fen);
        INDArray[] output = this.model.output(false,new INDArray[]{tensorBoard}, null,null,null);
        INDArray valueOutput = output[0];
        System.out.println("Evaluation objective de la position : "+valueOutput.getFloat(0,0));
        System.out.println("Distribution des probabilités du CNN : ");
        INDArray policyOutput = output[1];
        INDArray[] sortedSoftmax = argsort(policyOutput, mcts.getTopN());
        List<String> listMoves = getListMoves(sortedSoftmax[0], tensorBoard);
        printMoves(listMoves, sortedSoftmax[1]);
        Board board = new Board();
        board.loadFromFen(fen);
        if(sortedSoftmax[1].getFloat(0) >= 0.95 && board.legalMoves().toString().contains(listMoves.get(0))) {
            return listMoves.get(0);
        }
        else {
            System.out.println("Distribution des fréquences de MCTS : ");
            INDArray[] mctsSoftmax = mcts.startMCTS(fen);
            List<String> listMovesMCTS = getListMoves(mctsSoftmax[0], tensorBoard);
            printMoves(listMovesMCTS, mctsSoftmax[1]);
            int best_move_index = Nd4j.argMax(mctsSoftmax[1],0).getInt(0);
            return listMovesMCTS.get(best_move_index);
        }

    }
























    //Dictionnaires pour les pièces noires et blanches
    static {
        pieceMap.put('P', 0); pieceMap.put('N', 1); pieceMap.put('B', 2);
        pieceMap.put('R', 3); pieceMap.put('Q', 4); pieceMap.put('K', 5);
        pieceMap.put('p', 6); pieceMap.put('n', 7); pieceMap.put('b', 8);
        pieceMap.put('r', 9); pieceMap.put('q', 10); pieceMap.put('k', 11);
    }

    //Dictionnaires des directions
    //(Tous les mouvements possibles de la reine pour les pièces)
    //Mouvement de [1...7] cases dans les 8 directions [N,NE,E,SE,S,SW,W,NW] = 0-55
    static {
        //En haut
        directionMap.put(new AbstractMap.SimpleEntry<>(1,0),0);
        directionMap.put(new AbstractMap.SimpleEntry<>(2,0),1);
        directionMap.put(new AbstractMap.SimpleEntry<>(3,0),2);
        directionMap.put(new AbstractMap.SimpleEntry<>(4,0),3);
        directionMap.put(new AbstractMap.SimpleEntry<>(5,0),4);
        directionMap.put(new AbstractMap.SimpleEntry<>(6,0),5);
        directionMap.put(new AbstractMap.SimpleEntry<>(7,0),6);

        //En bas
        directionMap.put(new AbstractMap.SimpleEntry<>(-1,0),7);
        directionMap.put(new AbstractMap.SimpleEntry<>(-2,0),8);
        directionMap.put(new AbstractMap.SimpleEntry<>(-3,0),9);
        directionMap.put(new AbstractMap.SimpleEntry<>(-4,0),10);
        directionMap.put(new AbstractMap.SimpleEntry<>(-5,0),11);
        directionMap.put(new AbstractMap.SimpleEntry<>(-6,0),12);
        directionMap.put(new AbstractMap.SimpleEntry<>(-7,0),13);

        //A droite
        directionMap.put(new AbstractMap.SimpleEntry<>(0,1),14);
        directionMap.put(new AbstractMap.SimpleEntry<>(0,2),15);
        directionMap.put(new AbstractMap.SimpleEntry<>(0,3),16);
        directionMap.put(new AbstractMap.SimpleEntry<>(0,4),17);
        directionMap.put(new AbstractMap.SimpleEntry<>(0,5),18);
        directionMap.put(new AbstractMap.SimpleEntry<>(0,6),19);
        directionMap.put(new AbstractMap.SimpleEntry<>(0,7),20);

        //A gauche
        directionMap.put(new AbstractMap.SimpleEntry<>(0,-1),21);
        directionMap.put(new AbstractMap.SimpleEntry<>(0,-2),22);
        directionMap.put(new AbstractMap.SimpleEntry<>(0,-3),23);
        directionMap.put(new AbstractMap.SimpleEntry<>(0,-4),24);
        directionMap.put(new AbstractMap.SimpleEntry<>(0,-5),25);
        directionMap.put(new AbstractMap.SimpleEntry<>(0,-6),26);
        directionMap.put(new AbstractMap.SimpleEntry<>(0,-7),27);

        //En diagonale haut-droite
        directionMap.put(new AbstractMap.SimpleEntry<>(1,1),28);
        directionMap.put(new AbstractMap.SimpleEntry<>(2,2),29);
        directionMap.put(new AbstractMap.SimpleEntry<>(3,3),30);
        directionMap.put(new AbstractMap.SimpleEntry<>(4,4),31);
        directionMap.put(new AbstractMap.SimpleEntry<>(5,5),32);
        directionMap.put(new AbstractMap.SimpleEntry<>(6,6),33);
        directionMap.put(new AbstractMap.SimpleEntry<>(7,7),34);

        //En diagonale bas-gauche
        directionMap.put(new AbstractMap.SimpleEntry<>(-1,-1),35);
        directionMap.put(new AbstractMap.SimpleEntry<>(-2,-2),36);
        directionMap.put(new AbstractMap.SimpleEntry<>(-3,-3),37);
        directionMap.put(new AbstractMap.SimpleEntry<>(-4,-4),38);
        directionMap.put(new AbstractMap.SimpleEntry<>(-5,-5),39);
        directionMap.put(new AbstractMap.SimpleEntry<>(-6,-6),40);
        directionMap.put(new AbstractMap.SimpleEntry<>(-7,-7),41);

        //En diagonale bas-droite
        directionMap.put(new AbstractMap.SimpleEntry<>(-1,1),42);
        directionMap.put(new AbstractMap.SimpleEntry<>(-2,2),43);
        directionMap.put(new AbstractMap.SimpleEntry<>(-3,3),44);
        directionMap.put(new AbstractMap.SimpleEntry<>(-4,4),45);
        directionMap.put(new AbstractMap.SimpleEntry<>(-5,5),46);
        directionMap.put(new AbstractMap.SimpleEntry<>(-6,6),47);
        directionMap.put(new AbstractMap.SimpleEntry<>(-7,7),48);

        //En diagonale haut-gauche
        directionMap.put(new AbstractMap.SimpleEntry<>(1,-1),49);
        directionMap.put(new AbstractMap.SimpleEntry<>(2,-2),50);
        directionMap.put(new AbstractMap.SimpleEntry<>(3,-3),51);
        directionMap.put(new AbstractMap.SimpleEntry<>(4,-4),52);
        directionMap.put(new AbstractMap.SimpleEntry<>(5,-5),53);
        directionMap.put(new AbstractMap.SimpleEntry<>(6,-6),54);
        directionMap.put(new AbstractMap.SimpleEntry<>(7,-7),55);
    }

    //Les 8 mouvements du cavalier = 56-63
    static {
        directionMap.put(new AbstractMap.SimpleEntry<>(2,1),56);
        directionMap.put(new AbstractMap.SimpleEntry<>(2,-1),57);
        directionMap.put(new AbstractMap.SimpleEntry<>(-2,1),58);
        directionMap.put(new AbstractMap.SimpleEntry<>(-2,-1),59);
        directionMap.put(new AbstractMap.SimpleEntry<>(1,2),60);
        directionMap.put(new AbstractMap.SimpleEntry<>(1,-2),61);
        directionMap.put(new AbstractMap.SimpleEntry<>(-1,2),62);
        directionMap.put(new AbstractMap.SimpleEntry<>(-1,-2),63);
    }

    //Les sous-promotions = 64-72
    static {
        // Promotions des Blancs
        promotionMap.put(new AbstractMap.SimpleEntry<>(1, 1), new HashMap() {{
            put(64,'N');
            put(67,'B');
            put(70,'R');
        }});
        promotionMap.put(new AbstractMap.SimpleEntry<>(1, -1), new HashMap() {{
            put(65,'N');
            put(68,'B');
            put(71,'R');
        }});
        promotionMap.put(new AbstractMap.SimpleEntry<>(1, 0), new HashMap() {{
            put(66,'N');
            put(69,'B');
            put(72,'R');
        }});

        // Promotions des Noirs
        promotionMap.put(new AbstractMap.SimpleEntry<>(-1, 1), new HashMap() {{
            put(64,'n');
            put(67,'b');
            put(70,'r');
        }});
        promotionMap.put(new AbstractMap.SimpleEntry<>(-1, -1), new HashMap() {{
            put(65,'n');
            put(68,'b');
            put(71,'r');
        }});
        promotionMap.put(new AbstractMap.SimpleEntry<>(-1, 0), new HashMap() {{
            put(66,'n');
            put(69,'b');
            put(72,'r');
        }});
    }

    static {
        //Permet de retrouver les directions en fonction d'un indice de sous-promotion
        //Pour les blancs
        auxiliaryPromotionMap.put(new AbstractMap.SimpleEntry<>(64,1), new AbstractMap.SimpleEntry<>(1,1));
        auxiliaryPromotionMap.put(new AbstractMap.SimpleEntry<>(65,1), new AbstractMap.SimpleEntry<>(1,-1));
        auxiliaryPromotionMap.put(new AbstractMap.SimpleEntry<>(66,1), new AbstractMap.SimpleEntry<>(1,0));

        auxiliaryPromotionMap.put(new AbstractMap.SimpleEntry<>(67,1), new AbstractMap.SimpleEntry<>(1,1));
        auxiliaryPromotionMap.put(new AbstractMap.SimpleEntry<>(68,1), new AbstractMap.SimpleEntry<>(1,-1));
        auxiliaryPromotionMap.put(new AbstractMap.SimpleEntry<>(69,1), new AbstractMap.SimpleEntry<>(1,0));

        auxiliaryPromotionMap.put(new AbstractMap.SimpleEntry<>(70,1), new AbstractMap.SimpleEntry<>(1,1));
        auxiliaryPromotionMap.put(new AbstractMap.SimpleEntry<>(71,1), new AbstractMap.SimpleEntry<>(1,-1));
        auxiliaryPromotionMap.put(new AbstractMap.SimpleEntry<>(72,1), new AbstractMap.SimpleEntry<>(1,0));

        //Pour les noirs
        auxiliaryPromotionMap.put(new AbstractMap.SimpleEntry<>(64,0), new AbstractMap.SimpleEntry<>(-1,1));
        auxiliaryPromotionMap.put(new AbstractMap.SimpleEntry<>(65,0), new AbstractMap.SimpleEntry<>(-1,-1));
        auxiliaryPromotionMap.put(new AbstractMap.SimpleEntry<>(66,0), new AbstractMap.SimpleEntry<>(-1,0));

        auxiliaryPromotionMap.put(new AbstractMap.SimpleEntry<>(67,0), new AbstractMap.SimpleEntry<>(-1,1));
        auxiliaryPromotionMap.put(new AbstractMap.SimpleEntry<>(68,0), new AbstractMap.SimpleEntry<>(-1,-1));
        auxiliaryPromotionMap.put(new AbstractMap.SimpleEntry<>(69,0), new AbstractMap.SimpleEntry<>(-1,0));

        auxiliaryPromotionMap.put(new AbstractMap.SimpleEntry<>(70,0), new AbstractMap.SimpleEntry<>(-1,1));
        auxiliaryPromotionMap.put(new AbstractMap.SimpleEntry<>(71,0), new AbstractMap.SimpleEntry<>(-1,-1));
        auxiliaryPromotionMap.put(new AbstractMap.SimpleEntry<>(72,0), new AbstractMap.SimpleEntry<>(-1,0));

    }

    static {
        for (Map.Entry<Map.Entry<Integer, Integer>, Integer> entry : directionMap.entrySet()) {
            inverseDirectionMap.put(entry.getValue(), entry.getKey());
        }
    }
}
