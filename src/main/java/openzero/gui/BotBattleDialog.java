package openzero.gui;

import javafx.geometry.Insets;
import javafx.scene.control.*;
import javafx.scene.layout.GridPane;
import java.util.Optional;

public class BotBattleDialog {

    public static class BotBattleSettings {
        public int gameCount;
        public String whitePlayer;
        public String blackPlayer;
    }

    public static Optional<BotBattleSettings> show() {
        Dialog<BotBattleSettings> dialog = new Dialog<>();
        dialog.setTitle("Configurer Bot vs Bot");
        dialog.setHeaderText("Définissez les paramètres du duel entre bots");

        ButtonType startButtonType = new ButtonType("Démarrer", ButtonBar.ButtonData.OK_DONE);
        dialog.getDialogPane().getButtonTypes().addAll(startButtonType, ButtonType.CANCEL);

        GridPane grid = new GridPane();
        grid.setHgap(10);
        grid.setVgap(10);
        grid.setPadding(new Insets(20, 150, 10, 10));

        TextField gameCountField = new TextField("100");
        ComboBox<String> whiteChoice = new ComboBox<>();
        ComboBox<String> blackChoice = new ComboBox<>();

        whiteChoice.getItems().addAll("Bot1 (CNN)", "Bot2 (CNN+MCTS)", "Aléatoire");
        blackChoice.getItems().addAll("Bot1 (CNN)", "Bot2 (CNN+MCTS)", "Aléatoire");
        whiteChoice.setValue("Aléatoire");
        blackChoice.setValue("Aléatoire");

        grid.add(new Label("Nombre de parties :"), 0, 0);
        grid.add(gameCountField, 1, 0);
        grid.add(new Label("Joueur blanc :"), 0, 1);
        grid.add(whiteChoice, 1, 1);
        grid.add(new Label("Joueur noir :"), 0, 2);
        grid.add(blackChoice, 1, 2);

        dialog.getDialogPane().setContent(grid);

        dialog.setResultConverter(dialogButton -> {
            if (dialogButton == startButtonType) {
                try {
                    int count = Integer.parseInt(gameCountField.getText());
                    if (count <= 0) throw new NumberFormatException();

                    BotBattleSettings settings = new BotBattleSettings();
                    settings.gameCount = count;
                    settings.whitePlayer = whiteChoice.getValue();
                    settings.blackPlayer = blackChoice.getValue();
                    return settings;
                } catch (NumberFormatException e) {
                    Alert alert = new Alert(Alert.AlertType.ERROR);
                    alert.setTitle("Erreur de saisie");
                    alert.setHeaderText(null);
                    alert.setContentText("Veuillez entrer un nombre de parties valide (entier > 0).");
                    alert.showAndWait();
                }
            }
            return null;
        });

        return dialog.showAndWait();
    }
}
