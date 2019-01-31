package de.datexis.encoder;

import de.datexis.common.Resource;
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.ObjectOutputStream;
import java.io.OutputStreamWriter;
import java.util.Collection;
import java.util.List;
import java.util.stream.Collectors;
import org.deeplearning4j.models.word2vec.wordstore.VocabularyHolder;
import org.deeplearning4j.models.word2vec.wordstore.VocabularyWord;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.shade.jackson.annotation.JsonIgnore;
import org.slf4j.LoggerFactory;

/**
 * Outline for a simple cache-based 1-hot encoder
 * @author Sebastian Arnold <sarnold@beuth-hochschule.de>
 */
public abstract class LookupCacheEncoder extends Encoder {

  /** a cache of all existing n-grams */
  protected VocabularyHolder vocab;
  
  protected int totalWords = 0;
  
  public LookupCacheEncoder() {
    this("");
  }
  
  public LookupCacheEncoder(String id) {
    super(id);
    log = LoggerFactory.getLogger(LookupCacheEncoder.class);
    vocab = new VocabularyHolder.Builder().build();
  }

  public int getTotalWords() {
    return totalWords;
  }

  public void setTotalWords(int totalWords) {
    this.totalWords = totalWords;
  }
  
  @Override
  //@JsonIgnore
  public long getEmbeddingVectorSize() {
    return vocab.numWords();
  }
  
  /**
   * Return the index of a word in the vocabulary.
   */
  public int getIndex(String word) {
    return vocab.indexOf(word);
  }
  
  /**
   * @return target vector for Vocabulary word or null vector if word dies not exist
   */
  public INDArray oneHot(String word) {
    INDArray vector = Nd4j.zeros(getEmbeddingVectorSize(), 1);
    int i = getIndex(word);
    if(i>=0) vector.put(i, 0, 1.0);
    else log.warn("could not encode class '{}'. is it contained in training set?", word);
    return vector;
  }
  
  /**
   * Return the frequency of a word in the vocabulary.
   */
  public int getFrequency(String word) {
    VocabularyWord w = vocab.getVocabularyWordByString(word);
    return w != null ? w.getCount() : 0;
  }
  
  public double getProbability(String word) {
    return getFrequency(word) / (double) totalWords;
  }
  
  public double getConfidence(INDArray v, int i) {
    return v.getDouble(i);
  }
  
  public double getMaxConfidence(INDArray v) {
    return v.max(0).sumNumber().doubleValue();
  }
  
  /**
   * Return word from the vocabulary at a given index.
   */
  public String getWord(int index) {
    VocabularyWord w = vocab.getVocabularyWordByIdx(index);
    return w != null ? w.getWord() : null;
  }
  
  public boolean isUnknown(String word) {
    return !vocab.containsWord(word);
  }
  
  /**
   * Saves the model to <name>.tsv.gz
   * @param modelPath
   * @param name 
   */
  @Override
  public void saveModel(Resource modelPath, String name) {
    Resource modelFile = modelPath.resolve(name + ".tsv.gz");
    try(OutputStreamWriter out = new OutputStreamWriter(modelFile.getGZIPOutputStream())) {
      // FIXME: simply serialize vocab!
      int i = 0;
      for(VocabularyWord w : vocab.getVocabulary()) {
        i++;
        out.write(w.getHuffmanNode().getIdx() + "\t" + w.getWord() + "\t" + w.getCount() + "\n");
      }
      setModel(modelFile);
      log.info("saved " + i + " words");
    } catch(IOException ex) {
      log.error(ex.toString());
    }
    Resource streamFile = modelPath.resolve(name + ".bin");
    try(ObjectOutputStream oos = new ObjectOutputStream(streamFile.getOutputStream())) {
      oos.writeObject(vocab);
    } catch(IOException ex) {
      log.error(ex.toString());
    }
  }
  
  @Override
  public void loadModel(Resource modelFile) throws IOException {
    /* unfortunately, VocabularyHolder is in fact not serializable yet
      if(modelFile.getFileName().endsWith(".bin")) {
      vocab = SerializationUtils.readObject(modelFile.getInputStream());
      vocab.updateHuffmanCodes();
      setModel(modelFile);
      setModelAvailable(true);
      log.info("loaded " + vocab.numWords() + " words from binary " + modelFile.toString());
    } else {*/
    try(BufferedReader fr = new BufferedReader(new InputStreamReader(modelFile.getInputStream(), "UTF-8"))) {
      String line;
      int i = 1000000000; // FIXME: we are abusing this VocabCache here. But it works (TM)
                          // No, seriously, this should only be used fpr legacy models in the future. Let's save as binary files.
      while((line = fr.readLine()) != null) {
        String[] tsv = line.split("\\t");
        VocabularyWord w = new VocabularyWord(tsv[1]);
        //w.setCount(Integer.parseInt(tsv[2]));
        int huffmanIdx = Integer.parseInt(tsv[0]);
        w.setCount(i-huffmanIdx); // dirty fix to override resort of huffman tree for same counts
        vocab.addWord(w);
      }
      vocab.updateHuffmanCodes();
      setModel(modelFile);
      setModelAvailable(true);
      log.info("loaded " + vocab.numWords() + " words from " + modelFile.toString());
    }
  }
  
  @JsonIgnore
  public List<String> getWords() {
    return vocab.words().stream().map( a -> a.getWord()).collect(Collectors.toList());
  }
  
  /**
   * Return K nearest neighbours
   */
  public Collection<String> getNearestNeighbours(String word, int k) {
    throw new UnsupportedOperationException("No nearest words in LookupCache.");
  }

  public String getNearestNeighbour(INDArray v) {
    throw new UnsupportedOperationException("No nearest words in LookupCache.");
  }

  public Collection<String> getNearestNeighbours(INDArray v, int k) {
    throw new UnsupportedOperationException("No nearest words in LookupCache.");
  }
  
}
