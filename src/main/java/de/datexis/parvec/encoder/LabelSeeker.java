package de.datexis.parvec.encoder;

import org.apache.commons.lang3.ArrayUtils;
import org.deeplearning4j.models.embeddings.inmemory.InMemoryLookupTable;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.nd4j.linalg.primitives.Pair;

import java.util.ArrayList;
import java.util.List;

/**
 * This is primitive seeker for nearest labels.
 * It's used instead of basic wordsNearest method because for ParagraphVectors
 * only labels should be taken into account, not individual words
 *
 * @author raver119@gmail.com
 */
public class LabelSeeker {
  
  private List<String> labelsUsed;
  private InMemoryLookupTable<VocabWord> lookupTable;

  public LabelSeeker(List<String> labelsUsed, InMemoryLookupTable<VocabWord> lookupTable) {
    if(labelsUsed.isEmpty()) throw new IllegalStateException("You can't have 0 labels used for ParagraphVectors");
    this.lookupTable = lookupTable;
    this.labelsUsed = labelsUsed;
  }

  /**
   * This method accepts vector, that represents any document,
   * and returns distances between this document, and previously trained categories
   * @return
   */
  public List<Pair<String, Double>> getScores(INDArray vector) {
    List<Pair<String, Double>> result = new ArrayList<>();
    for(String label : labelsUsed) {
      INDArray vecLabel = lookupTable.vector(label);
      if(vecLabel == null) throw new IllegalStateException("Label '" + label + "' has no known vector!");

      double sim = Transforms.cosineSim(vector, vecLabel);
      if(!Double.isFinite(sim)) sim = 0.;
      result.add(new Pair<String, Double>(label, sim));
    }
    return result;
  }

  public INDArray getScoresAsVector(INDArray vector) {
    List<Pair<String, Double>> resultPairs = getScores(vector);
    Double[] scores = resultPairs.stream()
        .map(Pair::getSecond)
        .toArray(Double[]::new);

    INDArray vec = Nd4j.create(ArrayUtils.toPrimitive(scores));
    
    double min = vec.minNumber().doubleValue();
    double max = vec.maxNumber().doubleValue();
    if((max - min) == 0) return Nd4j.zerosLike(vec);
    double scale = 1. / (max - min);
    
    // return array scaled from 0..1 with sum 1.
    INDArray scaled1 = vec.sub(min).muli(scale);
    double sum = scaled1.sumNumber().doubleValue();
    INDArray summax = (sum != 0.) ? scaled1.div(sum) : scaled1;
    
    // return softmax array
    //INDArray scaled2 = vec.sub(min).muli(scale * 2.).subi(1);
    //INDArray softmax = Transforms.softmax(scaled2); // TODO: softmax for lowest item should still be 0??? range -1..+1 ?
    
    return summax;
  }
}