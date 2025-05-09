package openzero.MCTS;

import com.github.bhlangonijr.chesslib.Board;
import com.github.bhlangonijr.chesslib.Side;
import lombok.Getter;
import lombok.Setter;
import openzero.utils.ChessModelInterpreter;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.ArrayList;

import static java.lang.Math.sqrt;

@Getter
@Setter
public class MonteCarloTreeSearch {
    private final ChessModelInterpreter chessModelInterpreter;
    private Board board;
    private Node root;
    private Node lastNodePlayed;
    private int simulations;
    private final int topN;


    public MonteCarloTreeSearch(ChessModelInterpreter chessModelInterpreter, int simulations, int topN) {
        this.chessModelInterpreter = chessModelInterpreter;
        this.simulations = simulations;
        this.topN = topN;
        this.lastNodePlayed = null;
        this.root = null;
    }

    public INDArray[] startMCTS(String fen) {
        this.board = new Board();
        this.board.loadFromFen(fen);
        this.root = new Node(null,fen,-1, true);

        int i = 0;
        while (i < simulations) {
            selection(this.root);
            //System.out.println("simulations "+i);
            i++;
        }
        INDArray[] mctsSoftmax = argsortMCTS(this.root);
        return mctsSoftmax;
    }


    public void selection(Node current_node) {
        //on parcourt les enfants du noeud en fonction du score UCT de chacun
        //si on arrive sur un noeud inexploré ou qui n'a pas d'enfant, expansion
        Node leaf = current_node.selectChild();
        expansion(leaf);
    }

    public void expansion(Node leaf) {
        //soit le noeud est inexploré, et on l'explore (on évalue sa position et on la retourne par backpropagating)
        //soit il a déjà été exploré, et dans ce cas là on l'étend en calculant ses enfants, et on explore un de ses enfants (puis backpropation)
        if(leaf.isFinal_state()) {
            leaf.incrementVisitCount();
            backpropagation(leaf);
        }
        else if(leaf.getVisit_count() == 0) { //Si le noeud est inexploré
            exploration(leaf);
            leaf.incrementVisitCount();
            backpropagation(leaf);
        } else if (leaf.getChildren_nodes().isEmpty() && leaf.getEvaluation() != 0) { //Si le noeud est exploré mais n'a pas d'enfants et n'est pas un état final
            for(int i = 0; i< leaf.getSortedSoftmax()[0].length(); i++) {
                //on crée tous ses gosses en fonction de son vecteur softmax
                String orignal_fen = leaf.getFen(); //on prend la position de départ du noeud feuille
                INDArray orignal_tensor = this.chessModelInterpreter.FenToTensor(orignal_fen); //on le convertit en tenseur
                int move_index = leaf.getSortedSoftmax()[0].getInt(i); //on récupère le i-ème move_index pour le i-ème noeud enfant
                String move = this.chessModelInterpreter.translateMoves(move_index,orignal_tensor); //on le traduit en string
                Board board = new Board();
                board.loadFromFen(orignal_fen);
                if(board.legalMoves().toString().contains(move)) { //Si le coup est légal
                    board.doMove(move); //on applique le move au board pour récupérer le nouveau fen
                    String new_fen = board.getFen();
                    leaf.addChildNode(createChild(leaf,new_fen,move_index,!leaf.isActive_turn())); //on ajoute l'enfant créé à la feuille
                }
            }
            if(leaf.getChildren_nodes().isEmpty()) {
                //debuggage (SI AUCUN COUPS LEGAUX DANS LE TOPN)
                String fen = leaf.getFen();
                INDArray tensor = this.chessModelInterpreter.FenToTensor(fen);
                INDArray[] output = this.chessModelInterpreter.model.output(false, new INDArray[]{tensor}, null, null, null);
                INDArray valueOutput = output[0];
                INDArray policyOutput = output[1];
                INDArray[] sortedSoftmax = this.chessModelInterpreter.argsort(policyOutput, 4672);
                for(int j=0; j< sortedSoftmax[0].length();j++) {
                    String move = this.chessModelInterpreter.translateMoves(sortedSoftmax[0].getInt(j),tensor);
                    if(board.legalMoves().toString().contains(move)) {
                        board.doMove(move); //on applique le move au board pour récupérer le nouveau fen
                        String new_fen = board.getFen();
                        leaf.addChildNode(createChild(leaf,new_fen,sortedSoftmax[0].getInt(j),!leaf.isActive_turn())); //on ajoute l'enfant créé à la feuille
                        break;
                    }
                }
            }
            //et on évalue l'un de ses gosses
            Node child_leaf = leaf.selectChild();
            //System.out.println(child_leaf.getFen());
            expansion(child_leaf);
        }
    }

    public void exploration(Node node) {
        String fen = node.getFen();
        INDArray tensor = this.chessModelInterpreter.FenToTensor(fen);
        INDArray[] output = this.chessModelInterpreter.model.output(false, new INDArray[]{tensor}, null, null, null);
        INDArray valueOutput = output[0];
        INDArray policyOutput = output[1];
        INDArray[] sortedSoftmax = this.chessModelInterpreter.argsort(policyOutput, topN);
        Board board = new Board();
        board.loadFromFen(fen);

        Side side = board.getSideToMove();
        node.setEvaluation(valueOutput.getFloat(0,0));
        if(side.value().equals("BLACK"))
            node.setEvaluation(-1 * valueOutput.getFloat(0,0));
        node.setSortedSoftmax(sortedSoftmax);
        node.setFinal_state(board.isMated() || board.isDraw() || board.isStaleMate());
        if (!node.isFinal_state()) {
            node.addTotalEvaluation(node.getEvaluation());
        }
    }

    public void backpropagation(Node leaf) {
        float evaluation = leaf.getEvaluation();
        leaf.backpropagate(evaluation);
    }



    public Node createChild(Node parent, String fen, int move_index_played , boolean active_turn) {
        Node child = new Node(parent,fen,move_index_played,active_turn);
        return child;
    }

    public Node searchChildByFen(Node parent, String fen) {
        Node child = null;
        for (Node node : parent.getChildren_nodes()) {
            if(node.getFen().equals(fen)) {
                child = node;
            }
        }
        return child;
    }


    public INDArray[] argsortMCTS(Node root) {
        INDArray[] mctsSoftmax = new INDArray[2];
        mctsSoftmax[0] = Nd4j.create(topN);
        mctsSoftmax[1] = Nd4j.create(topN);
        for(int i=0;i<root.getChildren_nodes().size();i++) {
            float frequency = (root.getChildren_nodes().get(i).getVisit_count()/ (root.getVisit_count()-1));
            mctsSoftmax[0].putScalar(i,root.getChildren_nodes().get(i).getMove_index_played());
            mctsSoftmax[1].putScalar(i,frequency);
        }
        return mctsSoftmax;
    }

}




/**
 * Cette classe permet de représenter les positions possibles dans l'arbre de MCTS.
 * Un nœud dans l'arbre = une position.
 */
@Getter
@Setter
class Node {
    private Node parent_node; // le noeud parent
    private ArrayList<Node> children_nodes; //les noeuds enfants
    private boolean final_state; //si la partie est terminée sur ce noeud
    private float evaluation; //l'évaluation de ce noeud
    private float total_evaluation; // l'évaluation totale de ce noeud et de tous ses enfants
    private float visit_count; //Combien de fois le nœud a été visité
    private int move_index_played; //le coup joué pour arriver à cette position
    private String fen; //la description de la position
    private INDArray[] sortedSoftmax; //les probabilités pour chaque coup depuis la position de ce noeud
    private boolean active_turn; //tour de jeu du noeud (true si c'est le tour de l'IA, false sinon)
    private final float exploration_rate = 2.5F; //Le paramètre C de la formule UCT

    public Node(Node parent, String fen, int move_index_played, boolean active_turn) {
        this.parent_node = parent;
        this.fen = fen;
        this.move_index_played = move_index_played;
        this.active_turn = active_turn;

        this.children_nodes = new ArrayList<Node>();
        this.visit_count = 0;
        this.total_evaluation = 0;
        this.evaluation = 0;
        this.sortedSoftmax = null;
        this.final_state = false;
    }

    /**
     * Renvoit un noeud feuille à étendre
     * @return
     */
    public Node selectChild() {
        float max_score = Float.NEGATIVE_INFINITY;
        Node selected_child = null;

        if(this.visit_count==0) {
            return this; //Si le noeud est inexploré
        } else if (this.children_nodes.isEmpty() && this.getEvaluation() !=0) {
            return this; // Si le noeud est exploré mais n'a pas d'enfants
        } else {
            incrementVisitCount(); //on incrémente le nombre de visites
            for (Node child : this.children_nodes) {
                if(child.getPUCTScore() > max_score) {
                    max_score = child.getPUCTScore();
                    selected_child = child;
                }
            }
            return selected_child.selectChild();
        }
    }

    public float getPUCTScore() {
        if (visit_count == 0) return Float.POSITIVE_INFINITY;
        float q = this.total_evaluation / this.visit_count;
        float p = getPriorProbability();
        float exploration = exploration_rate * p * (float)sqrt(parent_node.getVisit_count()) / (1 + this.visit_count);
        if (active_turn) {
            float puct = q +exploration;
            return puct;
        } else {
            float puct = -q +exploration;
            return puct;
        }
    }

    public float getPriorProbability() {
        if (sortedSoftmax == null) return 0.1f; // Valeur par défaut si pas encore évalué
        int moveIndex = this.move_index_played;
        for (int i = 0; i < parent_node.getSortedSoftmax()[0].length(); i++) {
            if (parent_node.getSortedSoftmax()[0].getInt(i) == moveIndex) {
                return parent_node.getSortedSoftmax()[1].getFloat(i); // Probabilité du coup
            }
        }
        return 0.1f; // Valeur par défaut si le coup n’est pas dans le topN
    }

    public void backpropagate(float evaluation) {
        this.addTotalEvaluation(evaluation);
        if (this.parent_node != null) {
            this.parent_node.backpropagate(-evaluation); // Inverse pour le parent
        }
    }

    public void addTotalEvaluation(float evaluation) {
        this.total_evaluation += evaluation;
    }

    public void incrementVisitCount() {
        this.visit_count++;
    }

    public void addChildNode(Node child) {
        this.children_nodes.add(child);
    }

    public void setFinal_state(boolean final_state) {
        this.final_state = final_state;
        if (this.final_state) {
            Board board = new Board();
            board.loadFromFen(this.fen);
            if(board.isDraw()) this.evaluation = 0;
            else this.evaluation = -1;
        }
    }

    public void printInfoNode() {
        System.out.println("fen :"+this.fen);
        System.out.println("evaluation :"+this.evaluation);
        System.out.println("total evaluation :"+this.total_evaluation);
        System.out.println("visit count :"+this.visit_count);
        System.out.println("move index played :"+this.move_index_played);
        System.out.println("active turn :"+this.active_turn);
        System.out.println("final_state :"+this.final_state);
    }
}
