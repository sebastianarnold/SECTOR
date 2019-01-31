package de.datexis.sector.tagger;

import de.datexis.annotator.AnnotatorComponent;
import de.datexis.common.Resource;
import de.datexis.encoder.Encoder;
import de.datexis.encoder.EncoderSet;
import de.datexis.model.Document;
import de.datexis.model.Sentence;
import de.datexis.model.Span;
import java.util.Collection;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import org.nd4j.shade.jackson.annotation.JsonIgnore; // it is import to use the nd4j version in this class!
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.LoggerFactory;

/**
 * An Encoder that capsules SectorTagger and returns the hidden layer embedding.
 * @author Sebastian Arnold <sarnold@beuth-hochschule.de>
 */
public class SectorEncoder extends Encoder {

  protected SectorTagger tagger;
  
  public SectorEncoder() {
    this("SECTOR", new SectorTagger());
  }
  
  public SectorEncoder(String id) {
    this(id, new SectorTagger());
  }
  
  public SectorEncoder(String id, SectorTagger sector) {
    super(id);
    log = LoggerFactory.getLogger(SectorEncoder.class);
    this.tagger = sector;
    setModelFilename(tagger.getModel());
    setModelAvailable(true);
  }
  
  @JsonIgnore
  public SectorTagger getTagger() {
    return tagger;
  }
  
  public void setTagger(SectorTagger tagger) {
    this.tagger = tagger;
  }
  
  @Override
  public long getEmbeddingVectorSize() {
    return tagger.getEmbeddingLayerSize();
  }

  @Override
  public INDArray encode(Span span) {
    throw new IllegalArgumentException("SECTOR is only implemented to encode over Documents.");
  }

  @Override
  public INDArray encode(String word) {
    throw new IllegalArgumentException("SECTOR is only implemented to encode over Documents.");
  }

  @Override
  public void encodeEach(Document d, Class<? extends Span> elementClass) {
    encodeEach(Collections.singleton(d), elementClass);
  }
  
  @Override
  public void encodeEach(Collection<Document> docs, Class<? extends Span> elementClass) {
    if(elementClass != Sentence.class) throw new IllegalArgumentException("SECTOR is only implemented to encode Sentences over a Document");
    tagger.tag(docs);
  }

  /**
   * Encodes each element in the input and attaches the vectors to the element.
   * Please override this if the elements of your encoders are not independent or stateful.
   *  @param input - the Document that should be encoded
   * @param timeStepClass - the class of sub elements in the Document, e.g. Sentence.class
   */
  @Override
  public INDArray encodeMatrix(List<Document> input, int maxTimeSteps, Class<? extends Span> timeStepClass) {
    if(timeStepClass != Sentence.class) throw new IllegalArgumentException("SECTOR is only implemented to encode Sentences over a Document");
    
    SectorTaggerIterator it = new SectorTaggerIterator(DocumentSentenceIterator.Stage.ENCODE, input, tagger, tagger.getBatchSize(), false, tagger.requireSubsampling);
    INDArray result = null;// Nd4j.create(batchSize, 0, targetEncoder.getVectorSize());;
  
    // label batches of documents
    while(it.hasNext()) {
      DocumentSentenceIterator.DocumentBatch batch = it.nextDocumentBatch();
      // batch or result need padding to maxdoclength before concat
      Map<String,INDArray> weights =  tagger.encodeMatrix(batch);
      INDArray target = weights.get("target"); // target class probabilities [16xHxS]
      INDArray embedding = weights.get("embedding"); // bottleneck vectors [16xHxS]
      INDArray lstm = weights.get("BLSTM"); // LSTM layers [16xHxS]
      
      // append vectors to sentences
      int batchNum = 0; for(Document doc : batch.docs) {
        int t = 0; for(Sentence s : doc.getSentences()) {
          if(t >= maxTimeSteps) break;
          if(target != null) s.putVector(tagger.getTargetEncoder().getClass(), target.getRow(batchNum).getColumn(t));
          if(embedding != null) s.putVector(SectorEncoder.class, embedding.getRow(batchNum).getColumn(t));
          t++;
        }
        batchNum++;
      }
      
      if(maxTimeSteps > batch.maxDocLength) embedding = Nd4j.append(embedding, maxTimeSteps - batch.maxDocLength, 0, 2);
      result = (result == null) ? embedding : Nd4j.concat(0, result, embedding);
    }
    return result;
  }
  
  @Override
  public void encodeEach(Sentence input, Class<? extends Span> elementClass) {
    throw new IllegalArgumentException("SECTOR is only implemented to encode over Documents.");
  }
  
  @Override
  public void trainModel(Collection<Document> documents) {
    throw new UnsupportedOperationException("You need to train SectorTagger.");
  }

  /** overrides and re-implementations for export of nested tagger to XML/JSON */
  
  @Override
  public void loadModel(Resource file) {
    tagger.loadModel(file);
    setModelAvailable(true);
    setModel(file);
  }

  @Override
  public void saveModel(Resource dir, String name) {
    tagger.saveModel(dir, name);
    setModelFilename(tagger.getModel());
  }

  @Override
  @JsonIgnore
  public EncoderSet getEncoders() {
    return tagger.getEncoders();
  }

  @Override
  public AnnotatorComponent setEncoders(EncoderSet encs) {
    tagger.setEncoders(encs);
    return this;
  }
  
  @Override
  @JsonIgnore
  public EncoderSet getTargetEncoders() {
    return tagger.getTargetEncoders();
  }

  @Override
  public void addInputEncoder(Encoder e) {
    tagger.addInputEncoder(e);
  }

  @Override
  public void addTargetEncoder(Encoder e) {
    tagger.addTargetEncoder(e);
  }
  
  @Override
  public String getName() {
    return tagger.getName();
  }

  @Override
  public void setName(String name) {
    tagger.setName(name);
  }

  public int getBatchSize() {
    return tagger.getBatchSize();
  }
  
  public void setBatchSize(int size) {
    tagger.setBatchSize(size);
  }
  
  public int getEmbeddingLayerSize() {
    return tagger.getEmbeddingLayerSize();
  }
  
  public void setEmbeddingLayerSize(int size) {
    tagger.setEmbeddingLayerSize(size);
  }
  
  public void setMultiClass(boolean isMultiClass) {
    tagger.setRequireSubsampling(isMultiClass);
  }
  
  public boolean isMultiClass() {
    return tagger.isRequireSubsampling();
  }
  
  public void setNumEpochs(int numEpochs) {
    tagger.setNumEpochs(numEpochs);
  }
  
  public int getNumEpochs() {
    return tagger.getNumEpochs();
  }
  
  public void setRandomize(boolean rand) {
    tagger.setRandomize(rand);
  }
  
  public boolean isRandomize() {
    return tagger.isRandomize();
  }
  
}
