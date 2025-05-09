package openzero.gui;

import javafx.geometry.Insets;
import javafx.scene.control.*;
import javafx.scene.layout.VBox;
import javafx.stage.Window;


public class MenuController {

    public interface PlayerVsBotListener {
        void startGame(boolean isWhite);
    }

    public interface BotVsBotListener {
        void openBotBattleDialog();
    }

    private final PlayerVsBotListener playerVsBotListener;
    private final BotVsBotListener botVsBotListener;
    private final Window ownerWindow;

    public MenuController(PlayerVsBotListener pvbListener, BotVsBotListener bvbListener, Window ownerWindow) {
        this.playerVsBotListener = pvbListener;
        this.botVsBotListener = bvbListener;
        this.ownerWindow = ownerWindow;
    }

    public MenuBar getMenuBar() {
        MenuBar menuBar = new MenuBar();
        Menu gameMenu = new Menu("Jeu");

        MenuItem joueurVsBotItem = new MenuItem("Joueur vs Bot");
        joueurVsBotItem.setOnAction(e -> showPlayerVsBotDialog());

        MenuItem botVsBotItem = new MenuItem("Bot vs Bot");
        botVsBotItem.setOnAction(e -> botVsBotListener.openBotBattleDialog());

        gameMenu.getItems().addAll(joueurVsBotItem, botVsBotItem);
        menuBar.getMenus().add(gameMenu);
        return menuBar;
    }

    private void showPlayerVsBotDialog() {
        Dialog<ButtonType> dialog = new Dialog<>();
        dialog.setTitle("Choix de la couleur");
        dialog.initOwner(ownerWindow);

        ButtonType startButton = new ButtonType("Démarrer", ButtonBar.ButtonData.OK_DONE);
        dialog.getDialogPane().getButtonTypes().addAll(startButton, ButtonType.CANCEL);

        ToggleGroup group = new ToggleGroup();
        RadioButton whiteBtn = new RadioButton("Jouer les blancs");
        RadioButton blackBtn = new RadioButton("Jouer les noirs");
        RadioButton randomBtn = new RadioButton("Aléatoire");
        whiteBtn.setToggleGroup(group);
        blackBtn.setToggleGroup(group);
        randomBtn.setToggleGroup(group);
        randomBtn.setSelected(true);

        VBox content = new VBox(10, whiteBtn, blackBtn, randomBtn);
        content.setPadding(new Insets(10));
        dialog.getDialogPane().setContent(content);

        dialog.showAndWait().ifPresent(result -> {
            if (result.getButtonData() == ButtonBar.ButtonData.OK_DONE) {
                boolean isWhite = switch (((RadioButton) group.getSelectedToggle()).getText()) {
                    case "Jouer les blancs" -> true;
                    case "Jouer les noirs" -> false;
                    default -> Math.random() < 0.5;
                };
                playerVsBotListener.startGame(isWhite);
            }
        });
    }
}
