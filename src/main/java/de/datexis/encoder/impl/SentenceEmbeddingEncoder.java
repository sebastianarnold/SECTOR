package de.datexis.encoder.impl;

import de.datexis.common.Resource;
import de.datexis.model.Document;
import de.datexis.model.Token;
import de.datexis.encoder.LookupCacheEncoder;
import de.datexis.model.Sentence;
import de.datexis.model.Span;
import de.datexis.preprocess.DocumentFactory;
import de.datexis.preprocess.MinimalLowercasePreprocessor;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashSet;
import java.util.Set;
import org.deeplearning4j.text.tokenization.tokenizer.TokenPreProcess;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dimensionalityreduction.PCA;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.LoggerFactory;

/**
 * A "Simple but Tough-to-Beat Baseline for Sentence Embeddings" implemented after Arora et al. (2017)
 * @author sarnold
 */
public class SentenceEmbeddingEncoder extends LookupCacheEncoder {
  
  private static final TokenPreProcess preprocessor = new MinimalLowercasePreprocessor();
  
  /** underlying pretrained word2vec model */
  protected Word2VecEncoder vec;
  
  /** principal component */
  protected INDArray principal;
  
  /** parameter a */
  protected final double alpha = 0.0001;
  
  public SentenceEmbeddingEncoder() {
    super("EMB");
    log = LoggerFactory.getLogger(SentenceEmbeddingEncoder.class);
  }
  
  public SentenceEmbeddingEncoder(String id) {
    super(id);
    log = LoggerFactory.getLogger(SentenceEmbeddingEncoder.class);
  }
  
  public static SentenceEmbeddingEncoder create(Resource word2vecPath) {
    SentenceEmbeddingEncoder sent = new SentenceEmbeddingEncoder();
    sent.vec = Word2VecEncoder.load(word2vecPath);
    return sent;
  }
  
  @Override
  public String getName() {
    return "Simple Sentence Embedding Encoder";
  }

  @Override
  public long getEmbeddingVectorSize() {
    return vec.getEmbeddingVectorSize();
  }

  @Override
  public void trainModel(Collection<Document> documents) {
    appendTrainLog("Training " + getName() + " model...");
    setModel(null);
    timer.start();
    String w;
    int d = 0;
    totalWords = 0;
    // phase 1: gather statistics
    for(Document doc : documents) {
      d += doc.countSentences();
      for(Token t : doc.getTokens()) {
        w = preprocessor.preProcess(t.getText());
        totalWords++;
        if(!w.isEmpty()) {
          if(!vocab.containsWord(w)) {
            vocab.addWord(w);
          } else {
            vocab.incrementWordCounter(w);
          }
        }
      }
    }
    int total = vocab.numWords();
    vocab.updateHuffmanCodes();
    timer.stop();
    
    // phase 2: compute first principal component
    INDArray v = Nd4j.zeros(new long[]{d, getEmbeddingVectorSize()});
    d = 0;
    for(Document doc : documents) {
      for(Sentence s : doc.getSentences()) {
        v.getRow(d++).assign(weightedSum(s.getTokens(), alpha));
      }
    }
    this.principal = PCA.pca_factor(v, 1, false);
    
    appendTrainLog("trained " + vocab.numWords() + " words (" +  total + " total)", timer.getLong());
    setModelAvailable(true);
  }
  
  private INDArray weightedSum(Iterable<? extends Span> it, double a) {
    int i = 0;
    INDArray sum = Nd4j.create(getEmbeddingVectorSize(), 1);
    INDArray v;
    double p, f;
    for(Span s : it) {
      v = vec.encode(s.getText());
      p = getProbability(s.getText());
      f = a / (a + p);
      //if(p == 0.) f = 0;
      v.muli(f);
      sum.addi(v);
      i++;
    }
    return sum.divi(i);
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
  
  @Override
  public INDArray encode(Iterable<? extends Span> spans) {
    INDArray v = weightedSum(spans, alpha);
    INDArray u = this.principal;
    final INDArray uuTv = u.mmul(u.transpose()).mmul(v);
    return v.subi(uuTv);
  }
  
  @Override
  public INDArray encode(Span span) {
    return encode(span.getText());
  }

  @Override
  public INDArray encode(String phrase) {
    return encode(DocumentFactory.createTokensFromText(phrase));
  }

  public Set<String> asString(Iterable<Token> tokens) {
    Set<String> result = new HashSet<>();
    for(Token t : tokens) {
      if(!isUnknown(t.getText())) result.add(preprocessor.preProcess(t.getText()));
    }
    return result;
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
