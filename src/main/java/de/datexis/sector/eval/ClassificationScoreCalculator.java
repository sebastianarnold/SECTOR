package de.datexis.sector.eval;

import de.datexis.encoder.LookupCacheEncoder;
import de.datexis.sector.tagger.SectorTagger;
import de.datexis.tagger.Tagger;
import org.deeplearning4j.datasets.iterator.AsyncMultiDataSetIterator;
import org.deeplearning4j.datasets.iterator.MultiDataSetWrapperIterator;
import org.deeplearning4j.datasets.iterator.impl.MultiDataSetIteratorAdapter;
import org.deeplearning4j.earlystopping.scorecalc.base.BaseIEvaluationScoreCalculator;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.conf.BackpropType;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;

import java.util.Map;

/**
 * Score function for evaluating a MultiLayerNetwork according to an evaluation metric such
 * as accuracy, F1 score, etc.
 * Used for both MultiLayerNetwork and ComputationGraph
 *
 * @author Alex Black
 */
public class ClassificationScoreCalculator extends BaseIEvaluationScoreCalculator<Model, ClassificationEvaluation> {

  protected Tagger tagger;
  protected LookupCacheEncoder encoder;

  public ClassificationScoreCalculator(Tagger tagger, LookupCacheEncoder encoder, DataSetIterator iterator){
    super(iterator);
    this.tagger = tagger;
    this.encoder = encoder;
  }

  public ClassificationScoreCalculator(Tagger tagger, LookupCacheEncoder encoder, MultiDataSetIterator iterator){
    super(iterator);
    this.tagger = tagger;
    this.encoder = encoder;
  }

  @Override
  protected ClassificationEvaluation newEval() {
      return new ClassificationEvaluation("score calculation", encoder);
  }

  @Override
    public double calculateScore(Model network) {
    ClassificationEvaluation eval = newEval();

    if(network instanceof MultiLayerNetwork) {
      DataSetIterator i = (iter != null ? iter : new MultiDataSetWrapperIterator(iterator));
      eval = ((MultiLayerNetwork) network).doEvaluation(i, eval)[0];
    } else if(network instanceof ComputationGraph) {
      MultiDataSetIterator i = (iterator != null ? iterator : new MultiDataSetIteratorAdapter(iter));
      evaluate((ComputationGraph)network, eval, i);
      tagger.appendTrainLog("Validation score:\n" + eval.printClassificationAtKStats());
    } else {
      throw new RuntimeException("Unknown model type: " + network.getClass());
    }
    return finalScore(eval);
  }
  
  /**
   * Override evaluation to use average of forward/backward layers in a single score.
   */
  protected void evaluate(ComputationGraph net, ClassificationEvaluation evaluation, MultiDataSetIterator iterator) {

    //WorkspaceUtils.assertNoWorkspacesOpen("Expected no external workspaces open in doEvaluation");
    
    if(iterator.resetSupported() && !iterator.hasNext()) {
      iterator.reset();
    }

    MultiDataSetIterator iter = iterator.asyncSupported() ? new AsyncMultiDataSetIterator(iterator, 2, true) : iterator;

    boolean useRnnSegments = (net.getConfiguration().getBackpropType() == BackpropType.TruncatedBPTT);
    if(useRnnSegments) throw new UnsupportedOperationException("Evaluation with Truncated BPTT is not implemented.");

    /*  WorkspaceMode cMode = net.getConfiguration().getTrainingWorkspaceMode();
      net.getConfiguration().setTrainingWorkspaceMode(net.getConfiguration().getInferenceWorkspaceMode());
      MemoryWorkspace workspace
          = net.getConfiguration().getTrainingWorkspaceMode() == WorkspaceMode.NONE ? new DummyWorkspace()
          : Nd4j.getWorkspaceManager().getWorkspaceForCurrentThread(
              ComputationGraph.workspaceConfigurationExternal, ComputationGraph.WORKSPACE_EXTERNAL);*/
      
    while(iter.hasNext()) {
      
      //try (MemoryWorkspace wsB = workspace.notifyScopeEntered()) {
      
        MultiDataSet next = iter.next();

        if(next.getFeatures() == null || next.getLabels() == null) {
          break;
        }

        Map<String,INDArray> weights = SectorTagger.feedForward(net, next);
        
        INDArray predicted = null;
        if(weights.containsKey("target")) {
          predicted = weights.get("target");
        } else if(weights.containsKey("targetFW")) {
          predicted = weights.get("targetFW").dup();
          predicted.addi(weights.get("targetBW")).divi(2); // FW/BW average
          // TODO: we might add another softmax here?
        }

        //try (MemoryWorkspace wsO = Nd4j.getWorkspaceManager().scopeOutOfWorkspaces()) {
          evaluation.eval(next.getLabels(0), weights.get("target"), next.getLabelsMaskArray(0));
        //}
        
      /*} finally {
        // clear layer states to avoid leaks
        for (Layer layer : net.getLayers()) {
            layer.clear();
            layer.clearNoiseWeightParams();
        }
        for (GraphVertex vertex : net.getVertices()) {
            vertex.clearVertex();
        }
        net.clearLayerMaskArrays();
      }*/
      
    }
    
    //net.clear();
    //net.clearLayerMaskArrays();

    if(iterator.asyncSupported()) {
      ((AsyncMultiDataSetIterator) iter).shutdown();
    }
    
   // net.getConfiguration().setTrainingWorkspaceMode(cMode);
    
  }
    
  @Override
  protected double finalScore(ClassificationEvaluation e) {
      return e.getScore();
  }

  public boolean minimizeScore() {
    // false = score should be maximized
    return false;
  }

}