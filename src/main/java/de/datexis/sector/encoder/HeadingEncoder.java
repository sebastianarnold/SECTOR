package de.datexis.sector.encoder;

import de.datexis.common.WordHelpers;
import de.datexis.encoder.impl.BagOfWordsEncoder;
import de.datexis.model.Span;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Wrapper for Bag-Of-Words Headings with OTHER class and minWordLength
 * @author Sebastian Arnold <sarnold@beuth-hochschule.de>
 */
public class HeadingEncoder extends BagOfWordsEncoder {

  protected final static Logger log = LoggerFactory.getLogger(HeadingEncoder.class);
  public static final String ID = "HL";
  
  public static String OTHER_CLASS = "other";
  
  public HeadingEncoder() {
    super(ID);
  }
  
  public void trainModel(List<String> headlines, int minWordFrequency, int minWordLength, WordHelpers.Language language) {
    appendTrainLog("Training " + getName() + " model...");
    setModel(null);
    totalWords = 0;
    timer.start();
    setLanguage(language);
    for(String s : headlines) {
      for(String t : WordHelpers.splitSpaces(s)) {
        String w = preprocessor.preProcess(t);
        if(!w.isEmpty()) {
          totalWords++;
          if(!wordHelpers.isStopWord(w) && w.length() >= minWordLength) {
            if(!vocab.containsWord(w)) vocab.addWord(w);
            else vocab.incrementWordCounter(w);
          }
        }
      }
    }
    int total = vocab.numWords();
    vocab.truncateVocabulary(minWordFrequency);
    vocab.addWord(preprocessor.preProcess(OTHER_CLASS));
    vocab.updateHuffmanCodes();
    timer.stop();
    appendTrainLog("trained " + vocab.numWords() + " words (" +  total + " total)", timer.getLong());
    setModelAvailable(true);
  }
  
  @Override
  public INDArray encode(String phrase) {
    if(phrase != null ) return encode(WordHelpers.splitSpaces(phrase));
    else return encodeOtherClass();
  }
  
  @Override
  public INDArray encode(Iterable<? extends Span> spans) {
    INDArray vec = super.encode(spans);
    return vec.sumNumber().doubleValue() > 0. ? vec : encodeOtherClass();
  }
  
  @Override
  protected INDArray encode(String[] words) {
    INDArray vec = super.encode(words);
    return vec.sumNumber().doubleValue() > 0. ? vec : encodeOtherClass();
  }
  
  @Override
  public INDArray encodeSubsampled(String phrase) {
    INDArray vec = super.encodeSubsampled(phrase);
    return vec.sumNumber().doubleValue() > 0. ? vec : encodeOtherClass();
  }
  
  protected INDArray encodeOtherClass() {
    INDArray vector = Nd4j.zeros(getEmbeddingVectorSize(), 1);
    /*int i = getIndex(OTHER_CLASS);
    if(i >= 0) {
      vector.put(i, 0, 1.0);
    } else {
      log.error("could not encode OTHER_CLASS");
    }*/
    return vector;
  }
  
  @Override
  public Collection<String> getNearestNeighbours(INDArray v, int maxN) {
    // find maximum entries
    INDArray[] sorted = Nd4j.sortWithIndices(v.dup(), 0, false); // index,value
    if(sorted[0].sumNumber().doubleValue() == 0.) // TODO: sortWithIndices could be run on -1 / 0 / 1 ?
      log.warn("NearestNeighbour on zero vector - please check vector alignment!");
    INDArray idx = sorted[0]; // ranked indexes
    final double max = sorted[1].getDouble(0);
    final double med = sorted[1].medianNumber().doubleValue();
    // get top n
    ArrayList<String> result = new ArrayList<>(maxN);
    int i = 0, n = 0;
    while(n < maxN) {
      String word = getWord(idx.getInt(i));
      double prob = sorted[1].getDouble(i);
      // stop after first quantile
      if(prob == 0. || prob < (max+med)/2) break;
      // skip other
      if(!word.equals(OTHER_CLASS)) {
        result.add(word);
        n++;
      }
      i++;
    }
    if(result.isEmpty()) result.add(OTHER_CLASS);
    return result;
	}
  
}
