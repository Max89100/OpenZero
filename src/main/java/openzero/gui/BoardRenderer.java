package openzero.gui;

import com.github.bhlangonijr.chesslib.Board;
import com.github.bhlangonijr.chesslib.Square;
import com.github.bhlangonijr.chesslib.move.Move;
import com.github.bhlangonijr.chesslib.move.MoveException;
import javafx.application.Platform;
import javafx.scene.control.Alert;
import javafx.scene.control.ButtonType;
import javafx.scene.control.Label;
import javafx.scene.image.Image;
import javafx.scene.image.ImageView;
import javafx.scene.layout.GridPane;
import javafx.scene.layout.StackPane;
import javafx.scene.paint.Color;
import javafx.scene.shape.Circle;
import javafx.scene.shape.Rectangle;
import lombok.Getter;
import lombok.Setter;
import openzero.MCTS.MonteCarloTreeSearch;
import openzero.utils.ChessModelInterpreter;


import java.io.FileWriter;
import java.io.IOException;
import java.util.*;
import java.util.stream.Collectors;

@Getter
@Setter
public class BoardRenderer implements MenuController.PlayerVsBotListener, MenuController.BotVsBotListener {
    private static final int TILE_SIZE = 80;
    private static final int BOARD_SIZE = 8;
    private final GridPane grid = new GridPane();
    private Board board = new Board();
    private Label statusLabel;
    private Label evalLabel;
    private Label resultLabel;
    private final Map<Square, ImageView> pieceImages = new HashMap<>();
    private Square selectedSquare = null;
    private boolean gameOver = false;
    private final Random random = new Random();
    private String modelPath = "src/main/resources/OpenZero5.zip";
    private boolean playerIsWhite;
    private int whiteWins = 0;
    private int blackWins = 0;
    private int draws = 0;
    private Thread botVsBotThread = null;
    private Thread botThread = null;
    private volatile boolean stopRequested = false;

    private ChessModelInterpreter bot;

    public BoardRenderer(Label statusLabel, Label evalLabel, Label resultLabel) {
        this.statusLabel = statusLabel;
        this.evalLabel = evalLabel;
        this.resultLabel = resultLabel;
        drawBoard();
        drawPieces();
    }

    public void resetGame() {
        stopCurrentGame();
        this.board = new Board();
        this.gameOver = false;
        this.selectedSquare = null;
        drawBoard();
        drawPieces();
        evalLabel.setText("Évaluation : -");
    }

    @Override
    public void startGame(boolean isWhite) {
        resetGame();
        playerIsWhite = isWhite;
        statusLabel.setText("Mode : Joueur vs Bot - Joueur est " + (isWhite ? "blanc" : "noir"));
        drawPieces();
        updateTurnStatus();
        if (!isWhite) makeBotMoveIfNeeded();
    }

    @Override
    public void openBotBattleDialog() {
        stopCurrentGame();
        Optional<BotBattleDialog.BotBattleSettings> result = BotBattleDialog.show();
        result.ifPresent(settings -> runBotBattleLoop(settings));
    }

    void runBotBattleLoop(BotBattleDialog.BotBattleSettings settings) {
        stopCurrentGame();
        stopRequested = false;
        botVsBotThread = new Thread(() -> {
            try (FileWriter writer = new FileWriter("bot_battle_results.txt", true)) {
                whiteWins = 0;
                blackWins = 0;
                draws = 0;

                for (int i = 1; i <= settings.gameCount && !stopRequested; i++) {
                    board = new Board();
                    boolean bot1IsWhite = resolvePlayer(settings.whitePlayer, settings.blackPlayer);
                    String whiteName = bot1IsWhite ? "Bot1 (CNN)" : "Bot2 (CNN+MCTS)";
                    String blackName = bot1IsWhite ? "Bot2 (CNN+MCTS)" : "Bot1 (CNN)";

                    int finalI = i;
                    Platform.runLater(() -> {
                        statusLabel.setText("Partie " + finalI + " | WHITE = " + whiteName + ", BLACK = " + blackName);
                        drawBoard();
                        drawPieces();
                        evalLabel.setText("Évaluation : -");
                    });

                    while (!stopRequested && !board.isMated() && !board.isDraw() && !board.isStaleMate()) {
                        String fen = board.getFen();
                        boolean isWhiteToMove = board.getSideToMove().value().equals("WHITE");
                        String move;
                        double eval;

                        if (isWhiteToMove == bot1IsWhite) {
                            move = bot.makeMove(fen);
                            eval = bot.evaluate(fen);
                        } else {
                            move = bot.makeMoveWithMCTS(fen, new MonteCarloTreeSearch(bot, 201, 10));
                            eval = bot.evaluate(fen);
                        }

                        board.doMove(move);
                        
                        double finalEval = eval;

                        Platform.runLater(() -> {
                            drawPieces();
                            evalLabel.setText(String.format("Évaluation : %.2f", finalEval));
                        });

                        try {
                            Thread.sleep(300);
                        } catch (InterruptedException e) {
                            return;
                        }
                    }

                    if (stopRequested) break;

                    String result;
                    if (board.isMated()) {
                        boolean whiteWon = board.getSideToMove().flip().value().equals("WHITE");
                        if (whiteWon == bot1IsWhite) whiteWins++;
                        else blackWins++;
                        result = (whiteWon ? whiteName : blackName) + " gagne la partie " + i;
                    } else {
                        draws++;
                        result = "Partie " + i + " nulle";
                    }

                    writer.write(result + "\n");
                    writer.flush();

                    String score = String.format("%s = %d | %s = %d | Nulle = %d",
                            "Bot1 (CNN)", whiteWins, "Bot2 (CNN+MCTS)", blackWins, draws);
                    String finalResult = result;
                    Platform.runLater(() -> {
                        resultLabel.setText(score);
                        statusLabel.setText(finalResult);
                    });
                }
            } catch (IOException e) {
                e.printStackTrace();
            }
        });
        botVsBotThread.start();
    }

    private boolean resolvePlayer(String white, String black) {
        if ("Bot1 (CNN)".equals(white) && "Bot2 (CNN+MCTS)".equals(black)) return true;
        if ("Bot2 (CNN+MCTS)".equals(white) && "Bot1 (CNN)".equals(black)) return false;
        return random.nextBoolean();
    }

    public void drawBoard() {
        grid.setGridLinesVisible(false);
        grid.getChildren().clear();
        for (int row = 0; row < BOARD_SIZE; row++) {
            for (int col = 0; col < BOARD_SIZE; col++) {
                Rectangle square = new Rectangle(TILE_SIZE, TILE_SIZE);
                square.setFill((row + col) % 2 == 0 ? Color.web("#f0d9b5") : Color.web("#b58863"));
                StackPane cell = new StackPane(square);
                final int r = row, c = col;
                cell.setOnMouseClicked(e -> handleClick(r, c));
                grid.add(cell, col, row);
            }
        }
    }

    void drawPieces() {
        for (javafx.scene.Node node : grid.getChildren()) {
            if (node instanceof StackPane stack) {
                stack.getChildren().removeIf(child -> child instanceof ImageView || child instanceof Circle);
            }
        }
        pieceImages.clear();
        for (Square square : Square.values()) {
            String piece = board.getPiece(square).toString();
            if (!piece.equals("NONE")) {
                int row = 7 - square.getRank().ordinal();
                int col = square.getFile().ordinal();
                String fileName = "/pieces/" + piece.toLowerCase() + ".png";
                var imageStream = getClass().getResourceAsStream(fileName);
                if (imageStream == null) continue;
                ImageView imageView = new ImageView(new Image(imageStream));
                imageView.setFitWidth(TILE_SIZE * 0.9);
                imageView.setFitHeight(TILE_SIZE * 0.9);
                StackPane cell = getCell(row, col);
                if (cell != null) {
                    cell.getChildren().add(imageView);
                    pieceImages.put(square, imageView);
                }
            }
        }
    }

    private void handleClick(int row, int col) {
        if (gameOver) return;
        boolean whiteToMove = board.getSideToMove().value().equals("WHITE");
        boolean isPlayersTurn = (whiteToMove && playerIsWhite) || (!whiteToMove && !playerIsWhite);
        if (!isPlayersTurn) return;
        Square clicked = getSquare(row, col);
        if (selectedSquare == null && !board.getPiece(clicked).toString().equals("NONE")) {
            selectedSquare = clicked;
            highlightLegalMoves(clicked);
        } else if (selectedSquare != null) {
            try {
                Move move = new Move(selectedSquare, clicked);
                if (isMoveInLegalMoves(move)) {
                    board.doMove(move);
                    drawPieces();
                    
                    updateTurnStatus();
                    checkGameEnd();
                    if (!gameOver) makeBotMoveIfNeeded();
                }
            } catch (MoveException ignored) {}
            selectedSquare = null;
            clearHighlights();
        }
    }

    public void makeBotMoveIfNeeded() {
        stopRequested = false;
        botThread = new Thread(() -> {
            while (!gameOver && !stopRequested && board.getSideToMove().value().equals("WHITE") != playerIsWhite) {
                try {
                    String move = bot.makeMoveWithMCTS(board.getFen(), new MonteCarloTreeSearch(bot, 201, 10));
                    board.doMove(move);
                    
                    double eval = bot.evaluate(board.getFen());

                    Platform.runLater(() -> {
                        drawPieces();
                        evalLabel.setText(String.format("Évaluation : %.2f", eval));
                        updateTurnStatus();
                        checkGameEnd();
                    });

                    Thread.sleep(100);
                } catch (Exception e) {
                    e.printStackTrace();
                    break;
                }
            }
        });
        botThread.start();
    }

    private void checkGameEnd() {
        if (board.isMated()) showEndAlert("Échec et mat !", board.getSideToMove().flip() + " a gagné.");
        else if (board.isStaleMate()) showEndAlert("Pat", "Match nul.");
        else if (board.isDraw()) showEndAlert("Nulle", "La partie est nulle.");
        else if (board.isInsufficientMaterial()) showEndAlert("Matériel insuffisant", "Aucun mat possible.");
    }

    private void showEndAlert(String title, String message) {
        gameOver = true;
        Platform.runLater(() -> {
            Alert alert = new Alert(Alert.AlertType.CONFIRMATION);
            alert.setTitle("Fin de partie");
            alert.setHeaderText(title);
            alert.setContentText(message + "\nVoulez-vous rejouer ou quitter ?");

            ButtonType replayButton = new ButtonType("Rejouer");
            ButtonType quitButton = new ButtonType("Quitter");

            alert.getButtonTypes().setAll(replayButton, quitButton);

            alert.showAndWait().ifPresent(button -> {
                if (button == replayButton) {
                    resetGame();
                } else if (button == quitButton) {
                    Platform.exit();
                    System.exit(0);
                }
            });
        });
    }


    private void highlightLegalMoves(Square from) {
        clearHighlights();
        List<Move> legalMoves = board.legalMoves().stream()
                .filter(m -> m.getFrom().equals(from))
                .collect(Collectors.toList());
        for (Move move : legalMoves) {
            Square to = move.getTo();
            int row = 7 - to.getRank().ordinal();
            int col = to.getFile().ordinal();
            StackPane cell = getCell(row, col);
            if (cell != null) {
                Circle highlight = new Circle(TILE_SIZE * 0.15, Color.rgb(22, 25, 30, 0.5));
                cell.getChildren().add(highlight);
            }
        }
    }

    private void updateTurnStatus() {
        boolean whiteToMove = board.getSideToMove().value().equals("WHITE");
        boolean isPlayersTurn = (whiteToMove && playerIsWhite) || (!whiteToMove && !playerIsWhite);
        if (statusLabel != null) {
            if (isPlayersTurn) {
                statusLabel.setText("À vous de jouer");
            } else {
                statusLabel.setText("Le bot réfléchit…");
            }
        }
    }

    private void clearHighlights() {
        for (javafx.scene.Node node : grid.getChildren()) {
            if (node instanceof StackPane stack) {
                stack.getChildren().removeIf(child -> child instanceof Circle);
            }
        }
    }

    private boolean isMoveInLegalMoves(Move move) {
        return board.legalMoves().stream().anyMatch(m -> m.equals(move));
    }

    private StackPane getCell(int row, int col) {
        for (javafx.scene.Node node : grid.getChildren()) {
            if (GridPane.getRowIndex(node) == row && GridPane.getColumnIndex(node) == col) {
                return (StackPane) node;
            }
        }
        return null;
    }

    private Square getSquare(int row, int col) {
        int rank = 8 - row;
        char file = (char) ('a' + col);
        return Square.valueOf(("" + file + rank).toUpperCase());
    }

    public void setModelPath(String modelPath) {
        this.modelPath = modelPath;
        try {
            this.bot = new ChessModelInterpreter();
            this.bot.LoadModel(modelPath);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public void stopCurrentGame() {
        stopRequested = true;
        if (botThread != null && botThread.isAlive()) botThread.interrupt();
        if (botVsBotThread != null && botVsBotThread.isAlive()) botVsBotThread.interrupt();
        gameOver = true;
    }

}
