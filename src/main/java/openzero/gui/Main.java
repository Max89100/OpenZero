package openzero.gui;


import javafx.application.Application;
import javafx.scene.Scene;
import javafx.scene.control.Button;
import javafx.scene.control.Label;
import javafx.scene.layout.BorderPane;
import javafx.scene.layout.VBox;
import javafx.stage.Stage;
import javafx.scene.image.Image;

public class Main extends Application implements MenuController.PlayerVsBotListener, MenuController.BotVsBotListener {

    private final Label statusLabel = new Label("Mode : -");
    private final Label evalLabel = new Label("\u00c9valuation : -");
    private final Label resultLabel = new Label("Résultats : -");
    private BoardRenderer boardRenderer;
    private final String modelPath = "src/main/resources/OpenZero5.zip";

    @Override
    public void start(Stage primaryStage) {
        BorderPane root = new BorderPane();

        Button stopButton = new Button("Arrêter la partie");
        stopButton.setId("stop-button");
        stopButton.setOnAction(e -> boardRenderer.stopCurrentGame());

        VBox topPane = new VBox();
        topPane.getChildren().addAll(statusLabel, evalLabel, resultLabel);
        topPane.setSpacing(5);

        this.boardRenderer = new BoardRenderer(statusLabel, evalLabel, resultLabel);
        this.boardRenderer.setModelPath(modelPath);

        MenuController menuController = new MenuController(boardRenderer, boardRenderer, primaryStage);

        root.setTop(menuController.getMenuBar());
        root.setCenter(boardRenderer.getGrid());
        root.setBottom(new VBox(stopButton));
        root.setLeft(topPane);

        Scene scene = new Scene(root, 900, 800);
        scene.getStylesheets().add(getClass().getResource("/style/dark-theme.css").toExternalForm());

        primaryStage.setTitle("OpenZero Chess");
        primaryStage.getIcons().add(new Image(getClass().getResourceAsStream("/OpenZero.png")));
        primaryStage.setScene(scene);
        primaryStage.show();
    }

    @Override
    public void startGame(boolean isWhite) {
        boardRenderer.setPlayerIsWhite(isWhite);
        boardRenderer.resetGame();

        boardRenderer.setEvalLabel(evalLabel);
        boardRenderer.setStatusLabel(statusLabel);
        boardRenderer.setResultLabel(resultLabel);

        statusLabel.setText("Mode : Joueur vs Bot - Joueur est " + (isWhite ? "blanc" : "noir"));
        evalLabel.setText("\u00c9valuation : -");
        resultLabel.setText("Résultats : -");

        boardRenderer.makeBotMoveIfNeeded();
    }

    @Override
    public void openBotBattleDialog() {
        BotBattleDialog.show().ifPresent(settings -> boardRenderer.runBotBattleLoop(settings));
    }

    public static void main(String[] args) {
        launch();
    }
}