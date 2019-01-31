package de.datexis.encoder.impl;

import com.fasterxml.jackson.annotation.JsonIgnore;
import de.datexis.common.WordHelpers;
import de.datexis.model.Document;
import de.datexis.model.Token;
import de.datexis.encoder.LookupCacheEncoder;
import de.datexis.model.Sentence;
import de.datexis.model.Span;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import de.datexis.preprocess.MinimalLowercaseNewlinePreprocessor;
import org.apache.commons.math3.util.Pair;
import org.deeplearning4j.models.word2vec.wordstore.VocabularyHolder;
import org.deeplearning4j.text.tokenization.tokenizer.TokenPreProcess;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.LoggerFactory;

/**
 * A Bag-Of-Words N-Hot Encoder with stopword and minFreq training
 * @author sarnold
 */
public class BagOfWordsEncoder extends LookupCacheEncoder {
  
  protected static final TokenPreProcess preprocessor = new MinimalLowercaseNewlinePreprocessor();
  protected WordHelpers wordHelpers;
  protected WordHelpers.Language language;
  
  public BagOfWordsEncoder() {
    this("BOW");
  }
  
  public BagOfWordsEncoder(String id) {
    super(id);
    log = LoggerFactory.getLogger(BagOfWordsEncoder.class);
    vocab = new VocabularyHolder.Builder().build();
  }
  
  @Override
  public String getName() {
    return "Bag-of-words Encoder";
  }

  @Override
  public void trainModel(Collection<Document> documents) {
    trainModel(documents, 1, WordHelpers.Language.EN);
  }
  
  public void trainModel(Collection<Document> documents, int minWordFrequency, WordHelpers.Language language) {
    appendTrainLog("Training " + getName() + " model...");
    setModel(null);
    totalWords = 0;
    timer.start();
    setLanguage(language);
    for(Document doc : documents) {
      for(Token t : doc.getTokens()) {
        String w = preprocessor.preProcess(t.getText());
        if(!w.isEmpty()) {
          totalWords++;
          if(!wordHelpers.isStopWord(w)) {
            if(!vocab.containsWord(w)) vocab.addWord(w);
            else vocab.incrementWordCounter(w);
          }
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
  
  public void trainModel(List<String> sentences, int minWordFrequency, WordHelpers.Language language) {
    appendTrainLog("Training " + getName() + " model...");
    setModel(null);
    totalWords = 0;
    timer.start();
    setLanguage(language);
    for(String s : sentences) {
      for(String t : WordHelpers.splitSpaces(s)) {
        String w = preprocessor.preProcess(t);
        if(!w.isEmpty()) {
          totalWords++;
          if(!wordHelpers.isStopWord(w)) {
            if(!vocab.containsWord(w)) vocab.addWord(w);
            else vocab.incrementWordCounter(w);
          }
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
  public boolean isUnknown(String word) {
    return super.isUnknown(preprocessor.preProcess(word));
  }
  
  @Override
  public int getIndex(String word) {
    return super.getIndex(preprocessor.preProcess(word));
  }
  
  @Override
  public int getFrequency(String word) {
    return super.getFrequency(preprocessor.preProcess(word));
  }
  
  @Override
  public double getProbability(String word) {
    return super.getProbability(preprocessor.preProcess(word));
  }

  public WordHelpers.Language getLanguage() {
    return language;
  }

  public void setLanguage(WordHelpers.Language language) {
    this.language = language;
    wordHelpers = new WordHelpers(language);
  }
  
  /**
   * Encode a list of Tokens into an n-hot vector
   * @param spans
   * @return 
   */
  @Override
  public INDArray encode(Iterable<? extends Span> spans) {
    INDArray vector = Nd4j.zeros(getEmbeddingVectorSize(), 1);
    int i;
    // best results were seen with no normalization and 1.0 instead of word frequency
    for(Span s : spans) {
      i = getIndex(s.getText());
      if(i>=0) vector.put(i, 0, 1.0);
    }
    return vector;
  }
  
  /**
   * Encode a list of Strings into an n-hot vector
   * @param spans
   * @return 
   */
  protected INDArray encode(String[] words) {
    INDArray vector = Nd4j.zeros(getEmbeddingVectorSize(), 1);
    int i;
    // best results were seen with no normalization and 1.0 instead of word frequency
    for(String w : words) {
      i = getIndex(w);
      if(i>=0) vector.put(i, 0, 1.0);
    }
    return vector;
  }
  
  @Override
  public INDArray encode(Span span) {
    if(span instanceof Token) return encode(Arrays.asList(span));
    else if(span instanceof Sentence) return encode(((Sentence) span).getTokens());
    else return encode(span.getText());
  }

  /**
   * Encode a phrase, splitting at spaces.
   * @param phrase
   * @return 
   */
  @Override
  public INDArray encode(String phrase) {
    return encode(WordHelpers.splitSpaces(phrase));
  }
  
  /**
   * Tokenizes the String and encodes one word out of it with given distribution.
   */
  public INDArray encodeSubsampled(String phrase) {
    INDArray vector = Nd4j.zeros(getEmbeddingVectorSize(), 1);
    String[] tokens = WordHelpers.splitSpaces(phrase);
    if(tokens.length == 1) return encode(tokens[0]);
    List<Pair<String,Double>> itemWeights = new ArrayList<>(5);
    double completeWeight = 0.0;
    String w;
    for(String t : tokens) {
      w = preprocessor.preProcess(t);
      if(!w.isEmpty() && !wordHelpers.isStopWord(w)) {
        final double weight = samplingRate(super.getProbability(w));
        if(weight == 1.) continue; // word not in vocab
        completeWeight += weight;
        itemWeights.add(new Pair(w, weight));
      }
    }
    double r = Math.random() * completeWeight;
    double countWeight = 0.0;
    for(Pair<String,Double> item : itemWeights) {
        countWeight += item.getValue();
        if(countWeight >= r) {
          int i = getIndex(item.getKey());
          if(i>=0) vector.put(i, 0, 1.0);
          return vector;
        }
    }
    return vector; // return zeroes
  }
  
  public double getConfidence(INDArray v, int i) {
    return v.getDouble(i);
  }
  
  public double getMaxConfidence(INDArray v) {
    return v.max(0).sumNumber().doubleValue();
  }

  public Set<String> asString(Iterable<Token> tokens) {
    Set<String> result = new HashSet<>();
    for(Token t : tokens) {
      if(!isUnknown(t.getText())) result.add(preprocessor.preProcess(t.getText()));
    }
    return result;
  }
  
  @Override
  public String getNearestNeighbour(INDArray v) {
    Collection<String> knn = getNearestNeighbours(v, 1);
    if(knn.isEmpty()) return null;
    else return knn.iterator().next();
  }
  
  @Override
  public Collection<String> getNearestNeighbours(INDArray v, int n) {
    // find maximum entries
    INDArray[] sorted = Nd4j.sortWithIndices(v.dup(), 0, false); // index,value
    if(sorted[0].sumNumber().doubleValue() == 0.) // TODO: sortWithIndices could be run on -1 / 0 / 1 ?
      log.warn("NearestNeighbour on zero vector - please check vector alignment!");
    INDArray idx = sorted[0]; // ranked indexes
    // get top n
    ArrayList<String> result = new ArrayList<>(n);
    for(int i=0; i<n; i++) {
      if(sorted[1].getDouble(i) > 0.) result.add(getWord(idx.getInt(i)));
    }
    return result;
	}

  public boolean keepWord(String word) {
    return(Math.random() < samplingRate(word));
  }

  /**
   * Sets words in a given target to 0 based on probabilities.
   * http://mccormickml.com/2017/01/11/word2vec-tutorial-part-2-negative-sampling/
   */
  public INDArray subsample(INDArray target) {
    INDArray result = target.dup();
    for(int i=0; i<target.length(); ++i) {
      if(target.getDouble(i) > 0) {
        if(!keepWord(getWord(i))) result.putScalar(i, 0.);
      }
    }
    return result;
  }
  
  protected double samplingRate(String word) {
    double p = getProbability(word);
    return (Math.sqrt(p / 0.001) + 1) * (0.001 / p);
  }
  
  protected double samplingRate(double p) {
    //return (Math.sqrt(p / 0.001) + 1) * (0.001 / p);
    return 0.001 / (0.001 + p);
  }

  @JsonIgnore
  public INDArray subsampleWeights() {
    INDArray vector = Nd4j.zeros(getEmbeddingVectorSize(), 1);
    String w;
    for(int i=0; i<getEmbeddingVectorSize(); ++i) {
      w = getWord(i);
      vector.put(i, 0, samplingRate(getProbability(w)));
    }
    return vector.transpose();
  }

}
