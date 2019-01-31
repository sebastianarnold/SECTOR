package de.datexis.parvec.encoder;

import de.datexis.common.Resource;
import de.datexis.encoder.LookupCacheEncoder;
import de.datexis.model.Annotation;
import de.datexis.model.Document;
import de.datexis.model.Span;

import org.nd4j.shade.jackson.annotation.JsonIgnore;
import de.datexis.model.Dataset;
import de.datexis.model.Sentence;
import de.datexis.preprocess.DocumentFactory;
import de.datexis.preprocess.MinimalLowercasePreprocessor;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.paragraphvectors.ParagraphVectors;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.exception.ND4JIllegalStateException;
import org.nd4j.linalg.factory.Nd4j;

import java.io.IOException;
import java.lang.reflect.Field;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import org.deeplearning4j.models.embeddings.inmemory.InMemoryLookupTable;
import org.deeplearning4j.models.word2vec.wordstore.inmemory.AbstractCache;
import org.deeplearning4j.text.tokenization.tokenizer.TokenPreProcess;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class ParVecEncoder extends LookupCacheEncoder {

  protected final static Logger log = LoggerFactory.getLogger(ParVecEncoder.class);
  
  protected ParagraphVectors model;

  protected double learningRate = 0.025;
  protected double minLearningRate = 0.001;
  protected int batchSize = 16;
  protected int numEpochs = 1;
  protected int iterations = 5;
  protected int layerSize = 256;
  protected int targetSize;
  protected int windowSize = 10;
  
  protected static final TokenPreProcess preprocessor = new MinimalLowercasePreprocessor();
  protected final DefaultTokenizerFactory tokenizerFactory;
  protected List<VocabWord> labelsList;
  protected List<String> stopwords = new ArrayList<>();

  public ParVecEncoder() {
    super("PV");
    
    tokenizerFactory = new DefaultTokenizerFactory();
    tokenizerFactory.setTokenPreProcessor(preprocessor);
  }

  public void setModelParams(int layerSize, int windowSize) {
    this.layerSize = layerSize;
    this.windowSize = windowSize;
  }
  
  public void setTrainingParams(double learningRate, double minLearningRate, int batchSize, int iterations, int numEpochs) {
    this.learningRate = learningRate;
    this.minLearningRate = minLearningRate;
    this.batchSize = batchSize;
    this.iterations = iterations;
    this.numEpochs = numEpochs;
  }
  
  public void setStopWords(List<String> words) {
    this.stopwords = words;
  }
  
  @Override
  public void trainModel(Collection<Document> documents) {
    throw new UnsupportedOperationException("Please call trainModel(Dataset train)");
  }
  
  public void trainModel(Dataset train) {
    
    ParVecIterator it = new ParVecIterator(train, true);
    
    AbstractCache<VocabWord> cache = new AbstractCache<>();
    
    model =  new ParagraphVectors.Builder()
        .minWordFrequency(3)
        .iterations(iterations)
        .epochs(numEpochs)
        .layerSize(layerSize)
        .learningRate(learningRate)
        .minLearningRate(minLearningRate)
        .batchSize(batchSize)
        .windowSize(windowSize)
        .iterate(it)
        .trainWordVectors(true)
        .vocabCache(cache)
        .tokenizerFactory(tokenizerFactory)
        .stopWords(stopwords)
        .sampling(0)
        //.negativeSample(10)
        //.useUnknown(true)
        .build();
    
    log.info("training ParVec...");

    model.fit();
    
    log.info("training complete.");
    
    try {
      // get label information using reflection
      Field labelsListField = ParagraphVectors.class.getDeclaredField("labelsList");
      labelsListField.setAccessible(true);
      labelsList = (List) labelsListField.get(model);
      targetSize = labelsList.size();
    } catch(NoSuchFieldException | IllegalAccessException e) {
      log.error(e.getMessage(), e);
      throw new RuntimeException(e);
    }
    
    setModelAvailable(true);
    
  }
  
  @Override
  public INDArray encode(Span span) {
    if(span instanceof Sentence) {
      String text = ((Sentence)span).toTokenizedString()
              .trim()
              .replaceAll("\n", "*NL*")
              .replaceAll("\t", "*t*");
      try {
        return model.inferVector(text, learningRate, minLearningRate, 1).transpose();
      } catch(ND4JIllegalStateException ex) {
        //log.trace("unknown paragraph vector for '{}'", text);
        return Nd4j.zeros(layerSize).transpose();
      }
    } else {
      return encode(span.getText());
    }
  }
  
  public INDArray encode(Annotation ann, Document doc) {
    String text = doc.streamSentencesInRange(ann.getBegin(), ann.getEnd(), false)
        .map(s -> s
            .toTokenizedString()
            .trim()
            .replaceAll("\n", "*NL*")
            .replaceAll("\t", "*t*"))
        .collect(Collectors.joining(" "));
    try {
      return model.inferVector(text, learningRate, minLearningRate, 1).transpose();
    } catch(ND4JIllegalStateException ex) {
      //log.trace("unknown paragraph vector for '{}'", text);
      return Nd4j.zeros(layerSize).transpose();
    }
  }

  @Override
  public INDArray encode(String text) {
    text = DocumentFactory
        .createTokensFromText(text)
        .stream()
        .map(t -> t
            .getText()
            .trim()
            .replaceAll("\n", "*NL*")
            .replaceAll("\t", "*t*"))
        .collect(Collectors.joining(" "));
    try {
      return model.inferVector(text).transpose();
    } catch(ND4JIllegalStateException ex) {
      log.trace("unknown paragraph vector for '{}'", text);
      return Nd4j.zeros(layerSize).transpose();
    }
  }
  
  @Override
  public void saveModel(Resource modelPath, String name) {
    try {
      Resource modelFile = modelPath.resolve(name + ".zip");
      WordVectorSerializer.writeParagraphVectors(model, modelFile.getOutputStream());
      setModel(modelFile);
    } catch(IOException ex) {
      log.error(ex.toString());
    }
  }
  
  @Override
  public void loadModel(Resource modelFile) throws IOException {
    model = WordVectorSerializer.readParagraphVectors(modelFile.getInputStream());
    model.setTokenizerFactory(tokenizerFactory);
    layerSize = model.getLayerSize();
    try {
      // get label information using reflection
      Field labelsListField = ParagraphVectors.class.getDeclaredField("labelsList");
      labelsListField.setAccessible(true);
      labelsList = (List) labelsListField.get(model);
      targetSize = labelsList.size();
    } catch(NoSuchFieldException | IllegalAccessException e) {
      log.error(e.getMessage(), e);
      throw new RuntimeException(e);
    }
    log.info("Loaded ParagraphVectors with {} classes and layer size {}", targetSize, layerSize);
    setModel(modelFile);
    setModelAvailable(true);
  }

  @JsonIgnore
  @Override
  public List<String> getWords() {
    return labelsList.stream().map(VocabWord::getLabel).collect(Collectors.toList());
  }

  @Override
  public int getTotalWords() {
    return labelsList.size();
  }
  
  @Override
  public long getEmbeddingVectorSize() {
    return model.inferVector("test").length();
  }

  public long getOutputVectorSize() {
    // return number of classes!
    return targetSize;
  }
  
  public int getInputVectorSize() {
    return 0;
  }

  @Override
  public String getWord(int index) {
    if(labelsList.size() < index) return null;
    else return labelsList.get(index).getWord();
  }
  
  @Override
  public int getIndex(String word) {
    //String w = preprocessor.preProcess(word);
    return IntStream
        .range(0, labelsList.size())
        .filter(i -> word.equals(labelsList.get(i).getWord()))
        .findFirst()
        .orElse(-1);
  }
  
  @Override
  public INDArray oneHot(String word) {
    INDArray vector = Nd4j.zeros(targetSize);
    int i = getIndex(word);
    if(i>=0) vector.putScalar(i, 1.0);
    else log.warn("could not encode class '{}'. is it contained in training set?", word);
    return vector.transpose();
  }
  
  @Override
  public String getNearestNeighbour(INDArray v) {
    return getNearestNeighbours(v, 1).stream().findFirst().orElse(null);
  }

  @Override
  public Collection<String> getNearestNeighbours(INDArray v, int k) {
    /*
    // These are NN in embedding space:
    LabelSeeker seeker = new LabelSeeker(getWords(), (InMemoryLookupTable<VocabWord>) model.getLookupTable());
    return seeker.getScores(v).stream()
        .sorted(Comparator.comparing(p -> -p.getValue()))
        .limit(k)
        .map(Pair::getFirst)
        .collect(Collectors.toList());*/
    // find maximum entries
    INDArray[] sorted = Nd4j.sortWithIndices(Nd4j.toFlattened(v).dup(), 1, false); // index,value
    if(sorted[0].length() <= 1 || sorted[0].sumNumber().doubleValue() == 0.) // TODO: sortWithIndices could be run on -1 / 0 / 1 ?
      log.warn("NearestNeighbour on zero vector - please check vector alignment!");
    INDArray idx = sorted[0]; // ranked indexes
    // get top n
    ArrayList<String> result = new ArrayList<>(k);
    for(int i=0; i<k; i++) {
      result.add(getWord(idx.getInt(i)));
    }
    return result;
  }
  
  public INDArray getPredictions(INDArray v) {
    LabelSeeker seeker = new LabelSeeker(getWords(), (InMemoryLookupTable<VocabWord>) model.getLookupTable());
    return seeker.getScoresAsVector(v).transpose();
  }
  
}