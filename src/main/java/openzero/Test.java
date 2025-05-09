package openzero;


import com.github.bhlangonijr.chesslib.Board;
import openzero.MCTS.MonteCarloTreeSearch;
import openzero.utils.ChessModelInterpreter;
import java.util.Random;
import java.util.Scanner;

public class Test {
    public static void main(String[] args) throws Exception {
        //startPlaying("src/main/resources/OpenZero5.zip");
        startInteractiveMode("src/main/resources/OpenZero5.zip");
        //testing("src/main/resources/OpenZero5.zip","src/main/resources/OpenZero5.zip");
    }
    public static void startPlaying(String path) throws Exception {
        Board board = new Board();
        board.loadFromFen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
        ChessModelInterpreter modelInterpreter = new ChessModelInterpreter();
        modelInterpreter.LoadModel(path);
        Random rand = new Random();
        int r = rand.nextInt(2);
        if(r == 1) {
            String fen = board.getFen();
            String move = modelInterpreter.makeMove(fen);
            board.doMove(move);
            System.out.println(board);
        }
        while(!board.isMated()) {
            System.out.println("Entrez un move : ");
            Scanner scanner = new Scanner(System.in);
            String move = scanner.nextLine();
            board.doMove(move);
            System.out.println(board);

            String fen = board.getFen();
            String move2 = modelInterpreter.makeMove(fen);
            board.doMove(move2);
            System.out.println(board);
        }
    }

    private static void startInteractiveMode(String path) throws Exception {
        ChessModelInterpreter interpreter = new ChessModelInterpreter();
        interpreter.LoadModel(path);

        Scanner scanner = new Scanner(System.in);
        while (true) {
            System.out.print("Entrez une position FEN (ou 'exit' pour quitter) : ");
            String fen = scanner.nextLine();
            if (fen.equalsIgnoreCase("exit")) {
                System.out.println("Fin du programme.");
                break;
            }
            try {
//                String bestMove = interpreter.makeMove(fen);
                //System.out.println("Coup prédit : " + bestMove);
                MonteCarloTreeSearch mcts = new MonteCarloTreeSearch(interpreter, 201,10);
                String move = interpreter.makeMoveWithMCTS(fen,mcts);
                System.out.println("Coup prédit : " + move);
            } catch (Exception e) {
                System.out.println("Erreur lors de la prédiction du coup : " + e.getMessage());
            }
        }
        scanner.close();
    }

    public static void testing(String path1, String path2) throws Exception {
        Board board = new Board();
        board.loadFromFen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
        ChessModelInterpreter model1 = new ChessModelInterpreter();
        model1.LoadModel(path1);
        ChessModelInterpreter model2 = new ChessModelInterpreter();
        model2.LoadModel(path2);
        System.out.println(board);
        Random rand = new Random();
        int r = rand.nextInt(2);
        if(r == 1) {System.out.println("WHITE == CNN | BLACK == CNN+MCTS");}
        else {System.out.println("BLACK == CNN | WHITE == CNN+MCTS");}
        if(r == 1) {
            String fen = board.getFen();
            String move = model1.makeMove(fen);
            board.doMove(move);
            System.out.println(board);
        }
        while(!board.isMated() || !board.isDraw() || !board.isStaleMate()) {
            String move = model2.makeMoveWithMCTS(board.getFen(), new MonteCarloTreeSearch(model2, 201,10));
            board.doMove(move);
            System.out.println(board);

            String fen = board.getFen();
            String move2 = model1.makeMove(fen);
            board.doMove(move2);
            System.out.println(board);
            System.out.println(board.getFen());
        }

    }
}
//        if (args.length < 1) {
//            System.out.println("Usage: java ChessProjectApplication <mode> [args...]");
//            System.out.println("Modes disponibles : play, merge, distribute, train");
//            return;
//        }
//
//        String mode = args[0].toLowerCase(); // Mode choisi
//
//        switch (mode) {
//            case "create":
//                String modelType = args[1];
//                DataChessTraining dt2 = new DataChessTraining();
//                switch(modelType.toLowerCase()) {
//                    case "openzero1":
//                        ComputationGraph model2 = new OpenZero1().getModel();
//                        model2.init();
//                        dt2.saveModel(model2, "ChessData/data/models/OpenZero1.zip");
//                        break;
//                    case "openzero5":
//                        ComputationGraph model3 = new OpenZero5().getModel();
//                        model3.init();
//                        dt2.saveModel(model3, "ChessData/data/models/OpenZero5.zip");
//                        break;
//                    case "openzero20":
//                        ComputationGraph model5 = new OpenZero20().getModel();
//                        model5.init();
//                        dt2.saveModel(model5, "ChessData/data/models/OpenZero20.zip");
//                    case "none":
//                        break;
//                    default:
//                        System.out.println("Erreur: Modèle inconnu.");
//                        return;
//                }
//                break;
//            case "test":
//                String testPath = args[1];
//                DataChessTraining dt = new DataChessTraining(null,null, 512);
//                dt.testDataPipeline(testPath,512);
//                break;
//            case "query":
//                String path = args[1];
//                startInteractiveMode(path);
//                break;
//            case "play":
//                String modelPathPlaying = args[1];
//                startPlaying(modelPathPlaying);
//                break;
//            case "merge":
//                mergeModels();
//                break;
//            case "distribute":
//                if (args.length < 4) {
//                    System.out.println("Usage: java ChessProjectApplication distribute <batchSize> <epochs> <modelPath>");
//                    return;
//                }
//                int batchSize = Integer.parseInt(args[1]);
//                int epochs = Integer.parseInt(args[2]);
//                String modelPath = args[3];
//                startDistributedTraining(batchSize, epochs, modelPath);
//                break;
//            case "train":
//                if (args.length < 5) {
//                    System.out.println("Usage: java ChessProjectApplication train <model> <datasetTrain> <datasetTest> <batchSize> <epochs> [<modelPath>]");
//                    return;
//                }
//                startTraining(args);
//                break;
//            case "evaluate":
//                String datasetPathEvaluate = "ChessData/data/test/" + args[1];
//                String evalModelPath = args[2];
//                DataChessTraining dtEvaluate = new DataChessTraining(null,datasetPathEvaluate, 512);
//                dtEvaluate.LoadModel(evalModelPath);
//                dtEvaluate.evaluate();
//
//            default:
//                System.out.println("Erreur : mode inconnu '" + mode + "'");
//        }
//    }
//
//    // Mode interactif : Entrée FEN → Sortie meilleur coup
//    private static void startInteractiveMode(String path) throws Exception {
//        ChessModelInterpreter interpreter = new ChessModelInterpreter();
//        interpreter.LoadModel(path);
//
//        Scanner scanner = new Scanner(System.in);
//        while (true) {
//            System.out.print("Entrez une position FEN (ou 'exit' pour quitter) : ");
//            String fen = scanner.nextLine();
//            if (fen.equalsIgnoreCase("exit")) {
//                System.out.println("Fin du programme.");
//                break;
//            }
//            try {
//                //String bestMove = interpreter.makeMove(fen);
//                //System.out.println("Coup prédit : " + bestMove);
//                MonteCarloTreeSearch mcts = new MonteCarloTreeSearch(interpreter, 201,10);
//                String move = interpreter.makeMoveWithMCTS(fen,mcts);
//                System.out.println("Coup prédit : " + move);
//            } catch (Exception e) {
//                System.out.println("Erreur lors de la prédiction du coup : " + e.getMessage());
//            }
//        }
//        scanner.close();
//    }
//
//    // Fusionne les modèles entraînés
//    private static void mergeModels() throws IOException {
//        System.out.println("Fusion des modèles...");
//        ModelAverager modelAverager = new ModelAverager();
//        modelAverager.saveMergedAverageModel();
//        System.out.println("Fusion terminée !");
//    }
//
//
//    // Entraînement distribué
//    private static void startDistributedTraining(int batchSize, int epochs, String modelPath) throws Exception {
//        String machineName = InetAddress.getLocalHost().getHostName();
//        String datasetTrainPath = "ChessData/data/trainV/dataset_" + machineName + ".h5";
//        System.out.println(datasetTrainPath);
//        DataChessTraining dt = new DataChessTraining(datasetTrainPath, "", batchSize);
//        dt.LoadModel(modelPath);
//        dt.train(epochs);
//    }
//
//    // Entraînement normal avec un modèle spécifique
//    private static void startTraining(String[] args) throws Exception {
//        String modelType = args[1];
//        String datasetTrainPath = "ChessData/data/trainV/" + args[2];
//        String datasetTestPath = "ChessData/data/test/" + args[3];
//        int batchSize = Integer.parseInt(args[4]);
//        int epochs = Integer.parseInt(args[5]);
//        String modelPath = (args.length > 6) ? args[6] : null;
//
//        DataChessTraining dt;
//        switch (modelType.toLowerCase()) {
//            case "openzero1":
//                dt = new DataChessTraining(new OpenZero1(), datasetTrainPath, datasetTestPath, batchSize);
//                break;
//            case "openzero5":
//                dt = new DataChessTraining(new OpenZero5(), datasetTrainPath, datasetTestPath, batchSize);
//                break;
//            case "openzero20":
//                dt = new DataChessTraining(new OpenZero20(), datasetTrainPath, datasetTestPath, batchSize);
//                break;
//            case "none":
//                dt = new DataChessTraining(datasetTrainPath, datasetTestPath, batchSize);
//                break;
//            default:
//                System.out.println("Erreur: Modèle inconnu.");
//                return;
//        }
//
//        if (modelPath != null) {
//            System.out.println("Chargement du modèle depuis " + modelPath);
//            dt.LoadModel(modelPath);
//        }
//
//        if (epochs > 0) dt.train(epochs);
//        dt.evaluate();
//    }
//

//}

//COMMANDE DISTRIBUTED TRAINING
//parallel-ssh -h hosts.txt -l mv023940 -i "cd Documents/MASTER/ChessProject_BACK && tmux new-session -d -s training 'mvn spring-boot:run -Dspring-boot.run.arguments=\"distribute 512 1 ChessData/data/models/modelOne.zip\"'"
//parallel-ssh -h hosts.txt -l mv023940 -i "tmux capture-pane -t training -p"
