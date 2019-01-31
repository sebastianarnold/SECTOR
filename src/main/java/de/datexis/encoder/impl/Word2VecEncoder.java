package de.datexis.encoder.impl;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import de.datexis.common.*;
import de.datexis.encoder.Encoder;
import de.datexis.model.*;
import de.datexis.model.Token;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Iterator;
import java.util.List;
import java.util.stream.Collectors;

import de.datexis.preprocess.LowercasePreprocessor;
import org.nd4j.linalg.primitives.Counter;
import org.deeplearning4j.models.embeddings.inmemory.InMemoryLookupTable;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import static org.deeplearning4j.models.embeddings.loader.WordVectorSerializer.fromPair;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;
import org.deeplearning4j.models.word2vec.wordstore.inmemory.AbstractCache;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.sentenceiterator.SentencePreProcessor;
import org.deeplearning4j.text.tokenization.tokenizer.TokenPreProcess;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.nd4j.linalg.primitives.Pair;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * A Word2Vec model from http://deeplearning4j.org/word2vec.html
 * @author Sebastian Arnold <sarnold@beuth-hochschule.de>
 */
@JsonIgnoreProperties(ignoreUnknown = true)
//@JsonTypeInfo(use = JsonTypeInfo.Id.CLASS, include = JsonTypeInfo.As.PROPERTY, property = "class")
public class Word2VecEncoder extends Encoder {

	private static final Logger log = LoggerFactory.getLogger(Word2VecEncoder.class);

  public static enum ModelType { TEXT, BINARY, DL4J, GOOGLE };
  
  private final static Collection<String> FILENAMES_TEXT = Arrays.asList(".txt", ".txt.gz");
  private final static Collection<String> FILENAMES_BINARY = Arrays.asList(".bin", ".bin.gz");
  private final static Collection<String> FILENAMES_DL4J = Arrays.asList(".zip");
  private final static Collection<String> FILENAMES_GOOGLE = Arrays.asList(".zip");
  
	private WordVectors vec;
	private long length;
  private String modelName;
  private TokenPreProcess preprocessor = new LowercasePreprocessor();
  
	public Word2VecEncoder() {
    super("EMB");
  }
  
  public Word2VecEncoder(String id) {
    super(id);
  }

  public static Word2VecEncoder load(Resource path) {
    Word2VecEncoder vec = new Word2VecEncoder();
    vec.loadModel(path);
    return vec;
  }
  
  /**
   * Load a dummy encoder that returns only zeros.
   */
  public static Word2VecEncoder loadDummyEncoder() {
    Word2VecEncoder vec = new Word2VecEncoder();
    Resource txt = Resource.fromJAR("encoder/word2vec.txt");
    vec.loadModel(txt);
    return vec;
  }

  @Override
  public void loadModel(Resource modelFile) {
    log.info("Loading Word2Vec model: {} with preprocessor {}", modelFile.getFileName(), getPreprocessorClass());
		try {
      switch(getModelType(modelFile.getFileName())) {
        default:
        case TEXT: vec = WordVectorSerializer.loadTxtVectors(modelFile.getInputStream(), false); break;
        case BINARY: vec = Word2VecEncoder.loadBinaryModel(modelFile.getInputStream()); break;
        case DL4J: vec = WordVectorSerializer.loadStaticModel(modelFile.toFile()); break;
        case GOOGLE: vec = WordVectorSerializer.loadStaticModel(modelFile.toFile()); break;
      }
      int size = vec.vocab().numWords();
			INDArray example = vec.getWordVectorMatrix(vec.vocab().wordAtIndex(0));
			length = example.length();
      setModel(modelFile);
      setModelAvailable(true);
      log.info("Loaded Word2Vec model '" +  modelFile.getFileName() + "' with " + size + " vectors of size " + length );
		} catch (IOException ex) {
			log.error("could not load model " + ex.toString());
		}
	}
  
  @Override
  public void saveModel(Resource modelPath, String name) {
    saveModel(modelPath, name, ModelType.BINARY);
  }
  
  public void saveModel(Resource modelPath, String name, ModelType type) {
    try {
      // TODO: we also need to save the input Token Preprocessor!
      Resource modelFile;
      ObjectSerializer.writeJSON(this, modelPath.resolve("config.json"));
      switch(type) {
        default:
        case BINARY: {
          modelFile = modelPath.resolve(name + ".bin");
          Word2VecEncoder.writeBinaryModel(vec, modelFile.getOutputStream());
        } break;
        case TEXT: {
          modelFile = modelPath.resolve(name + ".txt.gz");
          WordVectorSerializer.writeWordVectors((Word2Vec) vec, modelFile.getGZIPOutputStream());
        } break;
        case DL4J: {
          modelFile = modelPath.resolve(name+".zip");
          WordVectorSerializer.writeWord2VecModel((Word2Vec) vec, modelFile.getOutputStream());
        } break;
        case GOOGLE: {
          modelFile = null;
          log.error("Cannot write Google Model");
        } break;
      }
      setModel(modelFile);
    } catch (IOException ex) {
      ex.printStackTrace();
      log.error("Could not save model: " + ex.toString());
    }
  }

  public void setPreprocessor(TokenPreProcess preprocessor) {
    this.preprocessor = preprocessor;
  }
  
  @Override
  public void trainModel(Collection<Document> documents) {
    int batchSize = 16;
    int windowSize = 10;
    int minWordFrequency = 3;
    int layerSize = 256;
    int iterations = 5;
    int epochs = 1;
    trainModel(documents.stream().flatMap(d -> d.streamSentences()).collect(Collectors.toList()),
      batchSize, windowSize, minWordFrequency, layerSize, iterations, epochs, new ArrayList<>());
  }
  
  /*public void trainModel(Iterable<Sentence> sentences) {
    int batchSize = 1000;
    int windowSize = 5;
    int minWordFrequency = 2;
    int layerSize = 150;
    int iterations = 1;
    int epochs = 1;
    trainModel(sentences, batchSize, windowSize, minWordFrequency, layerSize, iterations, epochs, new ArrayList<String>());
  }*/
  
  public void trainModel(Iterable<Sentence> sentences, int batchSize, int windowSize, int minWordFrequency, int layerSize, int iterations, int epochs, List<String> stopWords) {
		SentenceIterator iter = new SentenceStringIterator(sentences);
    trainModel(iter, batchSize, windowSize, minWordFrequency, layerSize, iterations, epochs, stopWords);
	}
  
  public void trainModel(SentenceIterator iter, int batchSize, int windowSize, int minWordFrequency, int layerSize, int iterations, int epochs, List<String> stopWords) {
  
    //CudaEnvironment.getInstance().getConfiguration().allowMultiGPU(true).setMemoryModel(MemoryModel.DELAYED);
    // PLEASE NOTE: For CUDA FP16 precision support is available
//    DataTypeUtil.setDTypeForContext(DataBuffer.Type.HALF);

    // temp workaround for backend initialization
    Nd4j.create(1);

    /*CudaEnvironment.getInstance().getConfiguration()
        // key option enabled
        .allowMultiGPU(true)

        // we're allowing larger memory caches
        .setMaximumDeviceCache(2L * 1024L * 1024L * 1024L)

        // cross-device access is used for faster model averaging over pcie
        .allowCrossDeviceAccess(true);
    */
    
    TokenizerFactory t = new DefaultTokenizerFactory();
		t.setTokenPreProcessor(preprocessor);
    
		log.info("Building model....");
    vec = new org.deeplearning4j.models.word2vec.Word2Vec.Builder()
            .batchSize(batchSize) //# words per minibatch.
            .windowSize(windowSize)
            .minWordFrequency(minWordFrequency) // 
            .useAdaGrad(false) //
            .layerSize(layerSize) // word feature vector size
            .seed(42)
            .iterations(iterations) // # iterations to train
            .epochs(epochs)
            .stopWords(stopWords)
            .learningRate(0.025) // 
            .minLearningRate(0.001) // learning rate decays wrt # words. floor learning
            .negativeSample(10) // sample size 10 words
            .iterate(iter) //
            .tokenizerFactory(t)
            .build();

			log.info("Fitting Word2Vec model....");
      ((org.deeplearning4j.models.word2vec.Word2Vec) vec).fit();
			
	}

  public static ModelType getModelType(String filename) {
    String name = filename.toLowerCase();
    if(FILENAMES_TEXT.stream().anyMatch(ext -> name.endsWith(ext))) return ModelType.TEXT;
    else if(FILENAMES_BINARY.stream().anyMatch(ext -> name.endsWith(ext))) return ModelType.BINARY;
    else if(FILENAMES_DL4J.stream().anyMatch(ext -> name.endsWith(ext))) return ModelType.DL4J;
    else if(FILENAMES_GOOGLE.stream().anyMatch(ext -> name.endsWith(ext))) return ModelType.GOOGLE;
    else return ModelType.TEXT;
  }
  
  public Class getPreprocessorClass() {
    return preprocessor.getClass();
  }
  
	@Override
	public String getName() {
		return modelName;
	}
  
	/**
	 * Use this function to access word vectors
	 * @param word
	 * @return
	 */
	private INDArray getWordVector(String word) {
		return vec.getWordVectorMatrix(preprocessor.preProcess(word));
	}

	public boolean isUnknown(String word) {
		return !vec.hasWord(preprocessor.preProcess(word));
	}

	@Override
	public INDArray encode(Span span) {
    if(span instanceof Token) return encode(preprocessor.preProcess(span.getText()));
    else return encode(span.getText());
	}

	@Override
	public long getEmbeddingVectorSize() {
		return length;
	}

  /**
   * Encodes the word. Returns nullvector if word was not found.
   * @param word
   * @return 
   */
	@Override
	public INDArray encode(String word) {
    INDArray sum = Nd4j.zeros(getEmbeddingVectorSize(), 1);
    int len = 0;
    for(String w : WordHelpers.splitSpaces(word)) {
      if(w.trim().isEmpty()) continue;
      INDArray arr = vec.getWordVectorMatrix(preprocessor.preProcess(w));
      if(arr != null) sum.addi(arr.transpose());
      len++;
    }
    return len == 0 ? sum : sum.div(len);
	}

	public Collection<String> getNearestNeighbours(String word, int k) {
		return vec.wordsNearest(preprocessor.preProcess(word), k);
	}

	public Collection<String> getNearestNeighbours(INDArray v, int k) {
		Counter<String> distances = new Counter<>();
    for(Object s : vec.vocab().words()) {
			String word = (String) s;
			INDArray otherVec = encode(word);
			double sim = Transforms.cosineSim(v, otherVec);
			distances.incrementCount(word, sim);
		}
		distances.keepTopNElements(k);
		return distances.keySetSorted();
	}

	public String getNearestNeighbour(INDArray v) {
		Collection<String> result = getNearestNeighbours(v, 1);
    if(result.isEmpty()) return "_";
    else return result.iterator().next();
  }

  /**
   * Writes the model to DATEXIS binary format
   * @param vec
   * @param outputStream 
   */
  private static void writeBinaryModel(WordVectors vec, OutputStream outputStream) throws IOException {
    
    int words = 0;
    
    try(BufferedOutputStream buf = new BufferedOutputStream(outputStream);
         DataOutputStream writer = new DataOutputStream(buf)) {
      for(Object word : vec.vocab().words()) {
        if(word == null) continue;
        INDArray wordVector = vec.getWordVectorMatrix((String) word);
        log.trace("Write: " + word + " (size " + wordVector.length() + ")");
        writer.writeUTF((String) word);
        Nd4j.write(wordVector, writer);
        words++;
      }
      writer.flush();
    }
    
    log.info("Wrote " + words + " words with size " + vec.lookupTable().layerSize());
    
  }
  
  /**
   * Loads the model from DATEXIS bindary format
   * @param stream
   * @return 
   */
  private static WordVectors loadBinaryModel(InputStream stream) throws IOException {
    
    AbstractCache<VocabWord> cache = new AbstractCache.Builder<VocabWord>().build();
    List<INDArray> arrays = new ArrayList<>();
    int words = 0;
    
    try(BufferedInputStream buf = new BufferedInputStream(stream); 
         DataInputStream reader = new DataInputStream(buf)) {
      //for(String word = reader.readUTF(); !word.equals("_aZ92_EOF");) {
      while(reader.available() > 0) {
        String word = reader.readUTF();
        INDArray row = Nd4j.read(reader);
        VocabWord word1 = new VocabWord(1.0, word);
        word1.setIndex(cache.numWords());
        cache.addToken(word1);
        cache.addWordToIndex(word1.getIndex(), word);
        cache.putVocabWord(word);
        arrays.add(row);
        words++;
      }
      
    }

    InMemoryLookupTable<VocabWord> lookupTable = (InMemoryLookupTable<VocabWord>) new InMemoryLookupTable.Builder<VocabWord>()
            .vectorLength(arrays.get(0).columns())
            .cache(cache)
            .build();

    INDArray syn = Nd4j.vstack(arrays);

    Nd4j.clearNans(syn);
    lookupTable.setSyn0(syn);

    return fromPair(Pair.makePair((InMemoryLookupTable) lookupTable, (VocabCache) cache));
    
  }
  
 /**
  * A Simple String Iterator used for Word2Vec Training
  * @author sarnold
  */
 public class SentenceStringIterator implements SentenceIterator {

   private Iterator<Sentence> it;
   Iterable<Sentence> sentences;
   private SentencePreProcessor spp;

   public SentenceStringIterator(Iterable<Sentence> sentences) {
     this.sentences = sentences;
     reset();
   }

   @Override
   public String nextSentence() {
     return it.next().getText();
   }

   @Override
   public boolean hasNext() {
     return it.hasNext();
   }

   @Override
   public void reset() {
     it = sentences.iterator();
   }

   @Override
   public void finish() {
     it.remove();
   }

   @Override
   public SentencePreProcessor getPreProcessor() {
     return this.spp;
   }

   @Override
   public void setPreProcessor(SentencePreProcessor spp) {
     this.spp = spp;
   }

 }

}
