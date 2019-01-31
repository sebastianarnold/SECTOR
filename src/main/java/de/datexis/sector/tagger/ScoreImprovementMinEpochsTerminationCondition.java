package de.datexis.sector.tagger;

import org.deeplearning4j.earlystopping.termination.ScoreImprovementEpochTerminationCondition;
import org.nd4j.shade.jackson.annotation.JsonProperty;

/**
 *
 * @author Sebastian Arnold <sarnold@beuth-hochschule.de>
 */
public class ScoreImprovementMinEpochsTerminationCondition extends ScoreImprovementEpochTerminationCondition {

  @JsonProperty
  private int minEpochs;
  
  @JsonProperty
  private int maxEpochs;
  
  public ScoreImprovementMinEpochsTerminationCondition(int minEpochs, int maxEpochsWithNoImprovement, int maxEpochs) {
    super(maxEpochsWithNoImprovement);
    this.minEpochs = minEpochs;
    this.maxEpochs = maxEpochs;
  }
  
  /**Should the early stopping training terminate at this epoch, based on the calculated score and the epoch number?
     * Returns true if training should terminated, or false otherwise
     * @param epochNum Number of the last completed epoch (starting at 0)
     * @param score Score calculate for this epoch
     * @return Whether training should be terminated at this epoch
     */
 @Override
  public boolean terminate(int epochNum, double score, boolean minimize) {
    boolean terminate = super.terminate(epochNum, score, minimize);
    if((epochNum + 1) < minEpochs) return false;
    else if((epochNum + 1) >= maxEpochs) return true;
    else return terminate;
  }

  @Override
  public String toString() {
    return "ScoreImprovementMinEpochsTerminationCondition";
  }
  

}
