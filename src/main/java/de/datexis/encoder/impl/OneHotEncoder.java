package de.datexis.encoder.impl;

import de.datexis.model.Document;
import de.datexis.model.Token;
import de.datexis.encoder.LookupCacheEncoder;
import de.datexis.model.Span;
import de.datexis.preprocess.MinimalLowercasePreprocessor;
import java.util.ArrayList;
import java.util.Collection;
import org.deeplearning4j.text.tokenization.tokenizer.TokenPreProcess;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.LoggerFactory;

/**
 * A one-hot encoder
 * @author sarnold
 */
public class OneHotEncoder extends LookupCacheEncoder {
  
  private static final TokenPreProcess preprocessor = new MinimalLowercasePreprocessor();
  
  public OneHotEncoder() {
    super("1H");
    log = LoggerFactory.getLogger(OneHotEncoder.class);
  }
  
  public OneHotEncoder(String id) {
    super(id);
    log = LoggerFactory.getLogger(OneHotEncoder.class);
  }
  
  @Override
  public String getName() {
    return "1-hot Encoder";
  }

  @Override
  public INDArray encode(Span span) {
    return encode(span.getText());
  }

  @Override
  public INDArray encode(String word) {
    INDArray vector = Nd4j.zeros(getEmbeddingVectorSize(), 1);
    String w = preprocessor.preProcess(word);
    int i = vocab.indexOf(w);
    if(i>=0) vector.put(i, 0, 1.0);
    return vector;
  }

  public boolean isUnknown(String word) {
    String w = preprocessor.preProcess(word);
    return !vocab.containsWord(w);
  }
    
  @Override
  public void trainModel(Collection<Document> documents) {
    trainModel(documents, 1);
  }

  /**
   * Trains the model for every word in the document
   * @param documents
   * @param minWordFrequency 
   */
  public void trainModel(Collection<Document> documents, int minWordFrequency) {
    appendTrainLog("Training " + getName() + " model...");
    setModel(null);
    timer.start();
    String w;
    totalWords = 0;
    for(Document doc : documents) {
      for(Token t : doc.getTokens()) {
        w = preprocessor.preProcess(t.getText());
        totalWords++;
        if(w.isEmpty()) continue;
        if(!vocab.containsWord(w)) {
          vocab.addWord(w);
        } else {
          vocab.incrementWordCounter(w);
        }
      }
    }
    int total = vocab.numWords();
    vocab.truncateVocabulary(minWordFrequency);
    vocab.updateHuffmanCodes();
    timer.stop();
    appendTrainLog("trained " + vocab.numWords() + " words (" +  total + " total)", timer.getLong());
    setModelAvailable(true);
  }
  
  @Override
  public Collection<String> getNearestNeighbours(INDArray v, int n) {
    // create copy
    final Double[] data = new Double[(int)v.length()];
    for(int j=0; j<v.length(); j++) {
      data[j] = v.getDouble(j);
    }
    // find maximum entries
    ArrayList<String> result = new ArrayList<>(n);
    for(int i=0; i<n; i++) {
      double max = 0.;
      int index = 0;
      for(int j=0; j<v.length(); j++) {
        if(data[j] > max) {
          index = j;
          max = data[j];
          data[j] = Double.MIN_VALUE;
        }
      }
      result.add(getWord(index));
    }
    return result;
	}
  
}
