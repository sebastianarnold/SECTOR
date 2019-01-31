package de.datexis.sector.encoder;

import de.datexis.encoder.LookupCacheEncoder;
import de.datexis.model.Document;
import de.datexis.model.Span;

import java.util.AbstractMap;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Map.Entry;

import de.datexis.preprocess.LowercasePreprocessor;
import org.deeplearning4j.models.word2vec.wordstore.VocabularyWord;
import org.deeplearning4j.text.tokenization.tokenizer.TokenPreProcess;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.LoggerFactory;

/**
 *
 * @author Sebastian Arnold <sarnold@beuth-hochschule.de>
 */
public class ClassEncoder extends LookupCacheEncoder {
  
  private static final TokenPreProcess preprocessor = new LowercasePreprocessor();
  public static final String ID = "CLS";
  
  public ClassEncoder() {
    super(ID);
    log = LoggerFactory.getLogger(ClassEncoder.class);
  }
  
  public ClassEncoder(String id) {
    super(id);
    log = LoggerFactory.getLogger(ClassEncoder.class);
  }
  
  @Override
  public String getName() {
    return "Classification Encoder";
  }

  @Override
  public INDArray encode(Span classLabel) {
    return encode(classLabel.getText());
  }

  @Override
  public long getEmbeddingVectorSize() {
    return vocab.numWords();
  }

  /*@Override
  public String getWord(int index) {
    String word = super.getWord(index);
    return (word != null ? word : "unknown");
  }*/
  
  @Override
  public INDArray encode(String classLabel) {
    return oneHot(classLabel);
  }

  public int getIndex(String word) {
    String w = preprocessor.preProcess(word);
    return vocab.indexOf(w);
  }
  
  public INDArray oneHot(String word) {
    INDArray vector = Nd4j.zeros(getEmbeddingVectorSize(), 1);
    int i = getIndex(word);
    if(i>=0) vector.put(i, 0, 1.0);
    else log.warn("could not encode class '{}'. is it contained in training set?", word);
    return vector;
  }
  
  public boolean isUnknown(String classLabel) {
    String w = preprocessor.preProcess(classLabel);
    return !vocab.containsWord(w);
  }
    
  @Override
  public void trainModel(Collection<Document> documents) {
    throw new UnsupportedOperationException("cannot train classification on Documents");
  }

  public void trainModelUsingHead(Iterable<String> classes) {
    trainModel(classes, 0);
    // stop after head of distribution (mean value reached)
    double val = 0;
    for(VocabularyWord word : vocab.words()) {
      val += word.getCount();
    }
    vocab.truncateVocabulary((int)(val / vocab.numWords()));
    vocab.updateHuffmanCodes();
    appendTrainLog("truncated to " + vocab.numWords() + " classes");
  }
  
  public void trainModel(Iterable<String> classes, int minClassFrequency) {
    appendTrainLog("Training " + getName() + " model...");
    setModel(null);
    timer.start();
    String w;
    totalWords = 0;
    for(String s : classes) {
      w = preprocessor.preProcess(s);
      totalWords++;
      if(w.isEmpty()) continue;
      if(!vocab.containsWord(w)) {
        vocab.addWord(w);
      } else {
        vocab.incrementWordCounter(w);
      }
    }
    int total = vocab.numWords();
    vocab.truncateVocabulary(minClassFrequency);
    vocab.updateHuffmanCodes();
    timer.stop();
    appendTrainLog("trained " + vocab.numWords() + " classes (" +  total + " total)", timer.getLong());
    setModelAvailable(true);
  }

  @Override
  public String getNearestNeighbour(INDArray v) {
    Collection<String> knn = getNearestNeighbours(v, 1);
    if(knn.isEmpty()) return null;
    else return knn.iterator().next();
  }
  
  @Override
  public Collection<String> getNearestNeighbours(INDArray v, int k) {
    // create copy
    final Double[] data = new Double[(int) v.length()];
    for(int j=0; j<v.length(); j++) {
      data[j] = v.getDouble(j);
    }
    // find maximum entries
    ArrayList<String> result = new ArrayList<>(k);
    for(int i=0; i<k; i++) {
      double max = Double.MIN_VALUE;
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
  
  public Collection<Entry<String,Double>> getNearestNeighbourEntries(INDArray v, int k) {
    // create copy
    final Double[] data = new Double[(int) v.length()];
    for(int j=0; j<v.length(); j++) {
      data[j] = v.getDouble(j);
    }
    // find maximum entries
    ArrayList<Entry<String,Double>> result = new ArrayList<>(k);
    for(int i=0; i<k; i++) {
      double max = Double.MIN_VALUE;
      int index = 0;
      for(int j=0; j<v.length(); j++) {
        if(data[j] > max) {
          index = j;
          max = data[j];
          data[j] = Double.MIN_VALUE;
        }
      }
      result.add(new AbstractMap.SimpleEntry<>(getWord(index),v.getDouble(index)));
    }
    return result;
  }
  
}
